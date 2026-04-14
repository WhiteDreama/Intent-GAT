"""
MAPPO variants used for fair cooperative-driving baseline comparisons.

- MAPPO: decentralized actor + centralized critic, no explicit communication.
- MAPPO-IPS: same centralized critic, but the actor receives a decentralized
  top-k interaction-priority context aggregated from observable neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_project.config import Config
from marl_project.modules import IPSMeanAggregator, IntentEncoder


class CentralizedCritic(nn.Module):
    """Centralized critic over concatenated per-agent observations."""

    def __init__(self, num_agents, obs_dim_per_agent):
        super().__init__()

        self.num_agents = num_agents
        self.obs_dim = obs_dim_per_agent
        self.global_state_dim = num_agents * obs_dim_per_agent

        hidden_dim = Config.HIDDEN_DIM
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, num_agents)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, global_obs):
        global_features = self.global_encoder(global_obs)
        return self.value_head(global_features)


class MAPPOPolicy(nn.Module):
    """MAPPO actor-critic with optional IPS actor-side communication."""

    def __init__(self, input_dim, action_dim, num_agents):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.intent_dim = Config.INTENT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        self.comm_module_name = getattr(Config, "COMM_MODULE", "none").lower()
        self.use_ips = self.comm_module_name == "ips_mean"

        self.encoder = IntentEncoder(input_dim)
        if self.use_ips:
            self.actor_comm = IPSMeanAggregator()
            actor_feature_dim = self.intent_dim * 2
        else:
            self.actor_comm = None
            actor_feature_dim = self.intent_dim

        self.actor_mean = nn.Sequential(
            nn.Linear(actor_feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = CentralizedCritic(num_agents, input_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def _gather_neighbor_features(self, all_features, neighbor_indices):
        safe_indices = neighbor_indices.clone()
        safe_indices[safe_indices < 0] = 0
        return all_features[safe_indices]

    def _actor_features(self, obs_batch, return_attention=False):
        node_features = obs_batch["node_features"]
        z, pred_waypoints = self.encoder(node_features)
        attention_weights = None

        if self.use_ips:
            neighbor_zs = self._gather_neighbor_features(z, obs_batch["neighbor_indices"])
            if return_attention:
                context, attention_weights = self.actor_comm(
                    z,
                    neighbor_zs,
                    obs_batch["neighbor_rel_pos"],
                    obs_batch["neighbor_mask"],
                    return_attention=True,
                )
            else:
                context = self.actor_comm(
                    z,
                    neighbor_zs,
                    obs_batch["neighbor_rel_pos"],
                    obs_batch["neighbor_mask"],
                )
            actor_features = torch.cat([z, context], dim=1)
        else:
            actor_features = z

        return actor_features, z, pred_waypoints, attention_weights

    def forward(self, obs_batch, global_obs=None, action=None, return_attention: bool = False):
        node_features = obs_batch["node_features"]
        batch_size_agents = node_features.size(0)

        actor_features, z, pred_waypoints, attention_weights = self._actor_features(
            obs_batch,
            return_attention=return_attention,
        )

        action_mean = self.actor_mean(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        if global_obs is not None:
            num_envs = batch_size_agents // self.num_agents
            env_indices = torch.arange(0, batch_size_agents, self.num_agents, device=node_features.device)
            global_obs_env = global_obs[env_indices[:num_envs]]
            critic_values = self.critic(global_obs_env)
            value = critic_values.view(-1)[:batch_size_agents]
        else:
            value = torch.zeros(batch_size_agents, device=node_features.device)

        result = {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "pred_waypoints": pred_waypoints,
            "z": z,
        }
        if return_attention and attention_weights is not None:
            result["attention_weights"] = attention_weights

        if action is not None:
            dist = torch.distributions.Normal(action_mean, action_std)
            result["action_log_probs"] = dist.log_prob(action).sum(dim=-1)
            result["dist_entropy"] = dist.entropy().sum(dim=-1)
            if "gt_waypoints" in obs_batch:
                result["aux_loss"] = F.mse_loss(pred_waypoints, obs_batch["gt_waypoints"])
            else:
                result["aux_loss"] = torch.tensor(0.0, device=node_features.device)

        return result

    def forward_actor_critic(self, obs_batch, global_obs=None, return_attention: bool = False):
        actor_features, _, pred_waypoints, attention_weights = self._actor_features(
            obs_batch,
            return_attention=return_attention,
        )

        action_mean = self.actor_mean(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        node_features = obs_batch["node_features"]
        if global_obs is not None:
            value = self.critic(global_obs)
        else:
            batch_size = max(1, node_features.size(0) // self.num_agents)
            value = torch.zeros(batch_size, self.num_agents, device=node_features.device)

        result = {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "pred_waypoints": pred_waypoints,
        }
        if return_attention and attention_weights is not None:
            result["attention_weights"] = attention_weights
        return result

    def get_value(self, global_obs):
        return self.critic(global_obs)

    def estimate_payload_bytes(self, obs_batch):
        if not self.use_ips:
            return 0.0
        valid_counts = obs_batch["neighbor_mask"].to(dtype=torch.float32).sum(dim=-1)
        selected = torch.clamp(valid_counts, max=float(getattr(Config, "IPS_TOP_K", 3)))
        return float((selected * self.intent_dim * 4.0).mean().item())


def construct_global_obs(obs_list, num_agents):
    """Concatenate per-agent observations into a global observation."""
    del num_agents
    return torch.cat(obs_list, dim=1)


def reshape_for_centralized_critic(node_features, num_agents):
    """Reshape (B * num_agents, obs_dim) into (B, num_agents * obs_dim)."""
    batch_size = node_features.size(0) // num_agents
    obs_dim = node_features.size(1)
    reshaped = node_features.view(batch_size, num_agents, obs_dim)
    return reshaped.view(batch_size, -1)
