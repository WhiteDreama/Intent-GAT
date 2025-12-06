import torch
import torch.nn as nn
import torch.nn.functional as F
from marl_project.config import Config

class IntentEncoder(nn.Module):
    """
    Module B: The Intent Encoder (Local Policy).
    Encodes noisy local observations into a latent intent 'z' and predicts future waypoints.
    """
    def __init__(self, input_dim):
        super(IntentEncoder, self).__init__()
        
        self.hidden_dim = Config.HIDDEN_DIM
        self.intent_dim = Config.INTENT_DIM
        
        # Input Normalization (Critical for RL stability)
        self.input_norm = nn.LayerNorm(input_dim)

        # Shared Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Head 1: Latent Intent (z)
        # This is broadcasted to neighbors.
        self.intent_head = nn.Linear(self.hidden_dim, self.intent_dim)
        
        # Head 2: Auxiliary Task (Future Waypoints)
        # Predicts N points (x, y) relative to ego.
        # Output dim = PRED_WAYPOINTS_NUM * 2
        self.aux_head = nn.Linear(self.hidden_dim, Config.PRED_WAYPOINTS_NUM * 2)
        
    def forward(self, obs):
        """
        Args:
            obs: (Batch, Input_Dim)
            
        Returns:
            z: (Batch, Intent_Dim) - The latent intent
            pred_waypoints: (Batch, PRED_WAYPOINTS_NUM, 2) - Predicted future path
        """
        # Apply Normalization first
        obs = self.input_norm(obs)
        
        features = self.encoder(obs)
        
        # 1. Generate Latent Intent
        z = self.intent_head(features)
        # Normalize z to keep it bounded (optional but good for stability)
        z = F.normalize(z, p=2, dim=1)
        
        # 2. Auxiliary Prediction
        flat_waypoints = self.aux_head(features)
        pred_waypoints = flat_waypoints.view(-1, Config.PRED_WAYPOINTS_NUM, 2)
        
        return z, pred_waypoints


class GraphAttention(nn.Module):
    """
    Module C: The Fusion Layer (Native GAT).
    Fuses neighbor intents using an attention mechanism.
    Updated to be Spatially Aware (incorporates relative positions).
    """
    def __init__(self):
        super(GraphAttention, self).__init__()
        
        self.intent_dim = Config.INTENT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        
        # Attention Projections
        # Query comes from Ego (just intent)
        self.W_query = nn.Linear(self.intent_dim, self.hidden_dim)
        
        # Key and Value come from Neighbors (Intent + Relative Position + Relative Velocity)
        # Input dim increases by 4 (dx, dy, dvx, dvy)
        self.W_key = nn.Linear(self.intent_dim + 4, self.hidden_dim)
        self.W_value = nn.Linear(self.intent_dim + 4, self.hidden_dim)
        
        # Output projection (optional, to map back to intent dim or keep as hidden)
        # We'll map back to intent_dim to concatenate with ego z later
        self.W_out = nn.Linear(self.hidden_dim, self.intent_dim)
        
    def forward(self, ego_z, neighbor_zs, neighbor_rel_pos, mask=None):
        """
        Args:
            ego_z: (Batch, Intent_Dim)
            neighbor_zs: (Batch, Max_Neighbors, Intent_Dim)
            neighbor_rel_pos: (Batch, Max_Neighbors, 4) - Relative position (dx, dy) and velocity (dvx, dvy)
            mask: (Batch, Max_Neighbors) - 1 for valid neighbor, 0 for padding/masked
            
        Returns:
            context: (Batch, Intent_Dim) - Aggregated neighbor info
        """
        batch_size = ego_z.size(0)
        
        # 0. Prepare Neighbor Features
        # Concatenate Intent and Position: (Batch, Max_Neighbors, Intent_Dim + 4)
        neighbor_features = torch.cat([neighbor_zs, neighbor_rel_pos], dim=-1)

        # 1. Projections
        # Q: (Batch, 1, Hidden)
        Q = self.W_query(ego_z).unsqueeze(1)
        
        # K, V: (Batch, Max_Neighbors, Hidden)
        K = self.W_key(neighbor_features)
        V = self.W_value(neighbor_features)
        
        # 2. Attention Scores
        # (Batch, 1, Hidden) x (Batch, Hidden, Max_Neighbors) -> (Batch, 1, Max_Neighbors)
        scores = torch.bmm(Q, K.transpose(1, 2))
        
        # Scale scores
        scores = scores / (self.hidden_dim ** 0.5)
        
        # 3. Masking
        if mask is not None:
            # mask is (Batch, Max_Neighbors)
            # unsqueeze to (Batch, 1, Max_Neighbors)
            mask = mask.unsqueeze(1) # Fix: Ensure mask has correct dimensions for broadcasting
            # Set scores of invalid neighbors to -inf
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Attention Weights
        weights = F.softmax(scores, dim=-1)
        
        # 5. Weighted Sum
        # (Batch, 1, Max_Neighbors) x (Batch, Max_Neighbors, Hidden) -> (Batch, 1, Hidden)
        context_hidden = torch.bmm(weights, V)
        
        # 6. Output Projection
        context = self.W_out(context_hidden.squeeze(1))
        
        return context

class CooperativePolicy(nn.Module):
    """
    Module A: The Cooperative Policy (End-to-End).
    Combines IntentEncoder, GraphAttention, and Actor-Critic heads.
    """
    def __init__(self, input_dim, action_dim):
        super(CooperativePolicy, self).__init__()
        self.encoder = IntentEncoder(input_dim)
        self.fusion = GraphAttention()
        
        # Policy Head
        # Input: Ego Intent (z) + Context (aggregated neighbors)
        self.actor = nn.Sequential(
            nn.Linear(Config.INTENT_DIM * 2, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(Config.HIDDEN_DIM, action_dim)
        self.std_head = nn.Linear(Config.HIDDEN_DIM, action_dim)
        
        # Value Head
        self.critic = nn.Sequential(
            nn.Linear(Config.INTENT_DIM * 2, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 1)
        )
        
    def forward(self, obs_batch, action=None):
        """
        Args:
            obs_batch: Dict containing tensors
            action: Optional action tensor for training (calculating log_probs)
        """
        # Unpack
        node_features = obs_batch['node_features']
        neighbor_indices = obs_batch['neighbor_indices']
        neighbor_mask = obs_batch['neighbor_mask']
        neighbor_rel_pos = obs_batch['neighbor_rel_pos']
        gt_waypoints = obs_batch.get('gt_waypoints', None)
        
        # 1. Encode
        z, pred_waypoints = self.encoder(node_features)
        
        # 2. Gather Neighbor Intents
        # Handle padding index -1 by clamping to 0 (mask will zero out contribution)
        safe_indices = neighbor_indices.clamp(min=0)
        neighbor_zs = z[safe_indices] # (Batch, Max_N, Intent_Dim)
        
        # 3. Fusion
        context = self.fusion(z, neighbor_zs, neighbor_rel_pos, neighbor_mask)
        
        # 4. Concatenate
        features = torch.cat([z, context], dim=1)
        
        # 5. Actor-Critic
        actor_features = self.actor(features)
        action_mean = self.mean_head(actor_features)
        action_std = F.softplus(self.std_head(actor_features)) + 1e-5
        
        value = self.critic(features)
        
        results = {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value
        }
        
        # 6. Aux Loss
        if gt_waypoints is not None:
            recon_loss = F.mse_loss(pred_waypoints, gt_waypoints)
            results["aux_loss"] = recon_loss
            
        # 7. Log Probs (for training)
        if action is not None:
            dist = torch.distributions.Normal(action_mean, action_std)
            action_log_probs = dist.log_prob(action).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            results["action_log_probs"] = action_log_probs
            results["dist_entropy"] = dist_entropy
            
        return results
