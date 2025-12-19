import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from marl_project.config import Config
from marl_project.modules import IntentEncoder, GraphAttention


# === [新增] 残差块定义 ===
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim), # LayerNorm 对 RL 训练极其重要
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        # 核心逻辑：输出 = 输入 + 变换后的输入
        # 这就是“跳跃连接”，防止梯度消失
        return x + self.fc(x)



class CooperativePolicy(nn.Module):
    """
    Module D: The Cooperative Policy.
    Assembles the IntentEncoder, GraphAttention, and PPO heads.
    """
    def __init__(self, input_dim, action_dim):
        super(CooperativePolicy, self).__init__()
         
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.intent_dim = Config.INTENT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        
        # --- Modules ---
        self.encoder = IntentEncoder(input_dim)
        self.gat = GraphAttention()
        
        # --- PPO Heads ---
        # Input to heads is [Ego_Z (32) + Context (32)] = 64
        self.feature_dim = self.intent_dim * 2
        
        # === [核心修改] Actor (Wide + ResNet) ===
        # 结构：投影 -> 残差 -> 残差 -> 输出
        self.actor_backbone = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            ResidualBlock(self.hidden_dim), # 加深网络
            ResidualBlock(self.hidden_dim), # 再加深
            nn.ReLU()
        )
        self.actor_head = nn.Sequential(
            nn.Linear(self.hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Actor (Mean)
        self.actor_mean = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim),
            nn.Tanh() # MetaDrive actions are usually [-1, 1]
        )
        
        # Actor (Log Std) - Learnable parameter or network
        # Using a learnable parameter is standard for PPO
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
       # === [核心修改] Critic (Wide + ResNet) ===
        self.critic_backbone = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            ResidualBlock(self.hidden_dim),
            ResidualBlock(self.hidden_dim),
            nn.ReLU()
        )
        self.critic_head = nn.Linear(self.hidden_dim, 1)

        # === 初始化逻辑 (Clean Version) ===
        self.apply(self._init_weights)
        
        # 修正 Actor 输出 (高熵)
        nn.init.orthogonal_(self.actor_head[0].weight, gain=0.01)
        nn.init.constant_(self.actor_head[0].bias, 0.0)
        
        # 修正 Critic 输出 (标准)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)
        
        # ============================================================
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
        
    def forward(self, obs_batch, action=None):
        """
        The Full Pipeline Forward Pass.
        
        Args:
            obs_batch (dict): Dictionary containing:
                - 'node_features': (B, Input_Dim)
                - 'neighbor_indices': (B, Max_Neighbors) - Indices into the current batch for neighbors
                - 'neighbor_mask': (B, Max_Neighbors) - 1 for valid, 0 for padding
                - 'gt_waypoints': (B, N, 2) - Optional, for aux loss
            action (Tensor): Optional action to evaluate (for PPO update)
            
        Returns:
            dict: containing action_mean, action_std, value, pred_waypoints, aux_loss (if gt provided)
        """
        node_features = obs_batch['node_features']
        neighbor_indices = obs_batch['neighbor_indices']
        neighbor_mask = obs_batch['neighbor_mask']
        neighbor_rel_pos = obs_batch['neighbor_rel_pos']
        
        # --- Step 1: Intent Encoding ---
        # z: (B, Intent_Dim)
        # pred_waypoints: (B, N, 2)
        z, pred_waypoints = self.encoder(node_features)
        
        # --- Step 2: Gather Neighbor Intents ---
        # neighbor_zs: (B, Max_Neighbors, Intent_Dim)
        neighbor_zs = self._gather_neighbor_intents(z, neighbor_indices)
        
        # --- Step 3: Graph Fusion (GAT) ---
        # context: (B, Intent_Dim)
        context = self.gat(z, neighbor_zs, neighbor_rel_pos, neighbor_mask)
        
        # --- Step 4: PPO Heads ---
        # Concatenate Ego Intent and Context
        features = torch.cat([z, context], dim=1)
        
        # Actor
        actor_feat = self.actor_backbone(features)
        action_mean = self.actor_head(actor_feat)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Critic
        critic_feat = self.critic_backbone(features)
        value = self.critic_head(critic_feat)
        
        
        # --- Outputs ---
        results = {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "pred_waypoints": pred_waypoints,
            "intent": z # Useful for visualization
        }
        
        # Calculate Aux Loss if GT is available
        if 'gt_waypoints' in obs_batch:
            gt_waypoints = obs_batch['gt_waypoints']
            # MSE Loss
            aux_loss = F.mse_loss(pred_waypoints, gt_waypoints)
            results["aux_loss"] = aux_loss
            
        # Calculate Action Log Prob if action is provided (for PPO update)
        if action is not None:
            dist = Normal(action_mean, action_std)
            action_log_probs = dist.log_prob(action).sum(dim=-1, keepdim=True)
            dist_entropy = dist.entropy().sum(dim=-1, keepdim=True)
            results["action_log_probs"] = action_log_probs
            results["dist_entropy"] = dist_entropy
            
        return results

    def forward_actor_critic(self, obs_batch):
        """
        Helper for PPO rollout: returns distribution and value without gradients.
        """
        with torch.no_grad():
            return self.forward(obs_batch)

    def _gather_neighbor_intents(self, all_z, neighbor_indices):
        """
        Gathers z vectors for neighbors based on indices.
        
        Args:
            all_z: (B, Intent_Dim)
            neighbor_indices: (B, Max_Neighbors) - Indices in [0, B-1], -1 for padding
            
        Returns:
            neighbor_zs: (B, Max_Neighbors, Intent_Dim)
        """
        B, MaxN = neighbor_indices.shape
        # Handle padding indices (-1)
        # We clamp -1 to 0 to avoid index out of bounds error.
        # The GAT mask will ensure these "fake" 0-th neighbors are ignored.
        safe_indices = neighbor_indices.clone()
        safe_indices[safe_indices < 0] = 0
        
        # Gather
        # We want to select rows from all_z based on safe_indices
        # all_z: (B, D)
        # safe_indices: (B, N)
        # Output: (B, N, D)
        
        # Expand all_z to be gather-able
        # We can use simple indexing if we flatten or use advanced indexing
        # neighbor_zs = all_z[safe_indices] works in PyTorch if safe_indices is LongTensor
        
        neighbor_zs = all_z[safe_indices]
        
        return neighbor_zs
