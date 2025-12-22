import torch
import torch.nn as nn
import torch.nn.functional as F
from marl_project.config import Config

class IntentEncoder(nn.Module):
    """
    Module B: The Intent Encoder (Local Policy).
    Encodes noisy local observations into a latent intent 'z' and predicts future waypoints.
    [Upgrade] Uses 1D-CNN for Lidar processing + MLP for Ego State.
    """
    def __init__(self, input_dim):
        super(IntentEncoder, self).__init__()
        
        self.hidden_dim = Config.HIDDEN_DIM
        self.intent_dim = Config.INTENT_DIM
        self.lidar_channels = Config.LIDAR_NUM_LASERS
        
        
        # Calculate Ego State dimension (Total - Lidar)
        self.ego_dim = input_dim - self.lidar_channels
        
        # 1. Input Normalization
        # Split normalization for Ego and Lidar usually works better, 
        # but a global LayerNorm is also acceptable and simpler.
        self.input_norm = nn.LayerNorm(input_dim)
        
        
       # 2. [New] 1D-CNN for Lidar
        # Logic: Extract geometric features (corners, walls) from sequence
        # Input: (Batch, 1, 72)
        self.lidar_cnn = nn.Sequential(
            # Layer 1: Detect local patterns (e.g., 5 points window)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2, padding_mode='circular'),
            nn.ReLU(),
            # Layer 2: Aggregate features
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.ReLU(),
            # Output shape calculation:
            # 72 -> /2 -> 36 -> /2 -> 18
            # Flatten dim = 32 channels * 18 points = 576
        )
        
        # Calculate CNN output size dynamically
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, 1, self.lidar_channels)
            cnn_out = self.lidar_cnn(dummy_lidar)
            self.cnn_flat_dim = cnn_out.view(1, -1).shape[1]

        # 3. Feature Fusion (Ego + Lidar_CNN)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.ego_dim + self.cnn_flat_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        
        # Head 1: Latent Intent (z)
        self.intent_head = nn.Linear(self.hidden_dim, self.intent_dim)
        
        # Head 2: Auxiliary Task
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
         
        # [Crucial] Split Observation into Lidar and Ego State
        # Assuming Lidar is at the end of the vector (based on env_wrapper.py)
        lidar = obs[:, -self.lidar_channels:]      # (Batch, 72)
        ego_state = obs[:, :-self.lidar_channels]  # (Batch, Ego_Dim)
        
        # Pass Lidar through CNN
        # Reshape for Conv1d: (Batch, Channel=1, Length=72)
        lidar = lidar.unsqueeze(1) 
        lidar_feat = self.lidar_cnn(lidar)
        lidar_feat = lidar_feat.view(lidar_feat.size(0), -1) # Flatten
        
        # Concatenate with Ego State
        combined_feat = torch.cat([ego_state, lidar_feat], dim=1)
        
        # Encode
        features = self.fusion_mlp(combined_feat)
        
        # Heads
        z = self.intent_head(features)
        z = F.normalize(z, p=2, dim=1) # Normalize to hypersphere
        
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
        self.num_heads = getattr(Config, "NUM_ATTENTION_HEADS", 4) # Default to 4 if not in config
        self.head_dim = self.hidden_dim // self.num_heads
        
        assert self.hidden_dim % self.num_heads == 0, \
            f"Hidden dim {self.hidden_dim} must be divisible by num_heads {self.num_heads}"
        
       # Projections
        # Query: Ego Intent -> (Batch, Heads, Head_Dim)
        self.W_query = nn.Linear(self.intent_dim, self.hidden_dim)
        
        # Key & Value: Neighbor (Intent + Rel Pos) -> (Batch, Heads, Head_Dim)
        # Input dim increases by 4 (dx, dy, dvx, dvy)
        input_feat_dim = self.intent_dim + 4
        self.W_key = nn.Linear(input_feat_dim, self.hidden_dim)
        self.W_value = nn.Linear(input_feat_dim, self.hidden_dim)
        
        # Output projection
        self.W_out = nn.Linear(self.hidden_dim, self.intent_dim)
        
    def forward(self, ego_z, neighbor_zs, neighbor_rel_pos, mask=None):
        """
        Args:
            ego_z: (Batch, Intent_Dim)
            neighbor_zs: (Batch, Max_Neighbors, Intent_Dim)
            neighbor_rel_pos: (Batch, Max_Neighbors, 4) - Relative position (dx, dy) and velocity (dvx, dvy)
            mask: (Batch, Max_Neighbors) - 1 for valid neighbor, 0 for padding/masked
            
            Args:
            ego_z: (B, Intent_Dim)
            neighbor_zs: (B, N, Intent_Dim)
            neighbor_rel_pos: (B, N, 4)
            mask: (B, N)
        Returns:
            context: (Batch, Intent_Dim) - Aggregated neighbor info
        """
        batch_size = ego_z.size(0)
        num_neighbors = neighbor_zs.size(1)
        
        # 0. Prepare Inputs
        # Neighbor Features: (B, N, Intent + 4)
        neighbor_features = torch.cat([neighbor_zs, neighbor_rel_pos], dim=-1)

        # 1. Projections & Reshape for Multi-Head
        # Q: (B, Hidden) -> (B, Heads, 1, Head_Dim)
        Q = self.W_query(ego_z).view(batch_size, self.num_heads, 1, self.head_dim)
        
        # K, V: (B, N, Hidden) -> (B, Heads, N, Head_Dim)
        # Note: We transpose (1, 2) to put Heads dimension first
        K = self.W_key(neighbor_features).view(batch_size, num_neighbors, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(neighbor_features).view(batch_size, num_neighbors, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Attention Scores (Scaled Dot-Product)
        # (B, Heads, 1, Head_Dim) @ (B, Heads, Head_Dim, N) -> (B, Heads, 1, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 2.5. Physics-Aware Inductive Bias (Distance Bias)
        # Prioritize physically closer neighbors by subtracting a distance-based bias
        # neighbor_rel_pos: (B, N, 4) = [dx, dy, dvx, dvy]
        # bias: (B, N), smaller for close neighbors, larger for distant ones
        distance_scale = getattr(Config, "DISTANCE_BIAS_SCALE", 0.5)
        dx_dy = neighbor_rel_pos[..., :2]
        dists = torch.sqrt(torch.clamp((dx_dy ** 2).sum(dim=-1), min=1e-12))
        scores = scores - (distance_scale * dists).unsqueeze(1).unsqueeze(1)
        
        # 3. Masking
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) to broadcast over Heads
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            # Fill -inf where mask is 0
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            
        # 4. Weights
        weights = F.softmax(scores, dim=-1) # (B, Heads, 1, N)
        
        # 5. Weighted Sum
        # (B, Heads, 1, N) @ (B, Heads, N, Head_Dim) -> (B, Heads, 1, Head_Dim)
        context_heads = torch.matmul(weights, V)
        
        # 6. Concatenate Heads
        # (B, Heads, 1, Head_Dim) -> (B, Hidden)
        context_concat = context_heads.view(batch_size, self.hidden_dim)
        
        # 7. Output Projection
        context = self.W_out(context_concat)
        
        return context
    
