import torch
import torch.nn as nn
import torch.nn.functional as F
from marl_project.config import Config


def _topk_selection_mask(scores, mask, top_k):
    """Build a hard top-k mask from per-neighbor scores."""
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.float32)
    else:
        mask = mask.to(dtype=scores.dtype)

    valid = mask > 0
    masked_scores = scores.masked_fill(~valid, float("-inf"))
    valid_counts = valid.sum(dim=-1)
    B, N = scores.shape
    selection = torch.zeros_like(scores, dtype=scores.dtype)

    if N == 0:
        return selection, valid_counts

    for b in range(B):
        k = int(min(int(top_k), int(valid_counts[b].item())))
        if k <= 0:
            continue
        idx = torch.topk(masked_scores[b], k=k, dim=-1).indices
        selection[b, idx] = 1.0
    return selection, valid_counts


def _interaction_priority_scores(neighbor_rel_pos, mask=None):
    """Heuristic interaction score using only observable relative geometry."""
    dx = neighbor_rel_pos[..., 0]
    dy = neighbor_rel_pos[..., 1]
    dvx = neighbor_rel_pos[..., 2]
    dvy = neighbor_rel_pos[..., 3]

    dist = torch.sqrt(torch.clamp(dx * dx + dy * dy, min=1e-12))
    closing = -(dx * dvx + dy * dvy) / dist
    closing = torch.clamp(closing, min=0.0)
    ttc_like = torch.where(closing > 1e-6, dist / (closing + 1e-6), torch.full_like(dist, 1e6))

    dist_score = 1.0 / (dist + 1e-3)
    closing_score = closing
    ttc_score = 1.0 / (ttc_like + 1e-3)
    scores = dist_score + 0.5 * closing_score + ttc_score

    if mask is not None:
        scores = scores * mask.to(dtype=scores.dtype)
    return scores


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
        
    def forward(self, obs, return_features: bool = False):
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
        
        if return_features:
            return z, pred_waypoints, features
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
        
    def forward(self, ego_z, neighbor_zs, neighbor_rel_pos, mask=None, return_attention: bool = False):
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
        mask_expanded = None
        all_masked = None
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) to broadcast over Heads
            mask_float = mask.to(dtype=scores.dtype)
            mask_expanded = mask_float.unsqueeze(1).unsqueeze(1)
            # Fill -inf where mask is 0
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            # Preserve exact no-communication semantics when all neighbors are masked out.
            all_masked = (mask_float.sum(dim=-1, keepdim=True) == 0)
            
        # 4. Weights
        weights = F.softmax(scores, dim=-1) # (B, Heads, 1, N)
        if mask_expanded is not None:
            weights = weights * mask_expanded
        
        # 5. Weighted Sum
        # (B, Heads, 1, N) @ (B, Heads, N, Head_Dim) -> (B, Heads, 1, Head_Dim)
        context_heads = torch.matmul(weights, V)
        
        # 6. Concatenate Heads
        # (B, Heads, 1, Head_Dim) -> (B, Hidden)
        context_concat = context_heads.view(batch_size, self.hidden_dim)
        
        # 7. Output Projection
        context = self.W_out(context_concat)
        if all_masked is not None:
            context = context.masked_fill(all_masked.expand(-1, context.size(-1)), 0.0)
            if return_attention:
                weights = weights.masked_fill(all_masked.view(batch_size, 1, 1, 1), 0.0)
        
        if return_attention:
            return context, weights

        return context


class TarMACModule(nn.Module):
    """
    TarMAC: Targeted Multi-Agent Communication (Das et al., 2019).

    Each agent produces a *signature* (key) and a *value* (message).
    The receiver attends over incoming signatures to decide how much
    weight to give each neighbour's message.

    Key difference from Intent-GAT's GraphAttention:
      - GAT: receiver Q attends to sender K  (receiver-centric)
      - TarMAC: sender signature is matched against receiver query
               via soft attention  (sender-targeted routing)

    Interface is identical to GraphAttention so it can be swapped in
    without touching the rest of CooperativePolicy / train / evaluate.
    """

    def __init__(self):
        super(TarMACModule, self).__init__()

        self.intent_dim = Config.INTENT_DIM        # 32
        self.hidden_dim = Config.HIDDEN_DIM         # 128
        self.num_rounds = getattr(Config, "TARMAC_NUM_ROUNDS", 1)

        # --- Signature network (sender side) ---
        # Maps intent z → signature key for attention routing.
        # Pure semantic: no spatial info so the *sender* decides
        # "who should listen to me" based on content alone.
        self.signature_net = nn.Sequential(
            nn.Linear(self.intent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.intent_dim),
        )

        # --- Value / message network (sender side) ---
        # Incorporates relative geometry so the message is spatially grounded.
        # Input: intent (32) + rel_pos (4) = 36
        self.value_net = nn.Sequential(
            nn.Linear(self.intent_dim + 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.intent_dim),
        )

        # --- Query network (receiver side) ---
        # Receiver maps its own intent into a query to match against
        # neighbour signatures.
        self.query_net = nn.Sequential(
            nn.Linear(self.intent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.intent_dim),
        )

        # --- Output projection ---
        self.W_out = nn.Linear(self.intent_dim, self.intent_dim)

    def forward(self, ego_z, neighbor_zs, neighbor_rel_pos, mask=None,
                return_attention: bool = False):
        """
        Args  (same as GraphAttention):
            ego_z:            (B, Intent_Dim)           – receiver intent
            neighbor_zs:      (B, Max_Neighbors, Intent_Dim) – sender intents
            neighbor_rel_pos: (B, Max_Neighbors, 4)     – [dx, dy, dvx, dvy]
            mask:             (B, Max_Neighbors)         – 1 = valid, 0 = pad
        Returns:
            context: (B, Intent_Dim)   – aggregated message for receiver
        """
        B = ego_z.size(0)
        N = neighbor_zs.size(1)

        # ----- Sender side -----
        # Signature (key):  (B, N, D)
        sig = self.signature_net(neighbor_zs)

        # Value (message):  (B, N, D)
        neighbor_feat = torch.cat([neighbor_zs, neighbor_rel_pos], dim=-1)  # (B, N, 36)
        val = self.value_net(neighbor_feat)

        # ----- Receiver side -----
        # Query: (B, D) → (B, 1, D)
        query = self.query_net(ego_z).unsqueeze(1)

        # ----- Attention -----
        # scores: (B, 1, D) × (B, D, N) → (B, 1, N)
        scores = torch.bmm(query, sig.transpose(1, 2)) / (self.intent_dim ** 0.5)

        # Masking
        mask_expanded = None
        all_masked = None
        if mask is not None:
            mask_float = mask.to(dtype=scores.dtype)
            mask_expanded = mask_float.unsqueeze(1)  # (B, 1, N)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            # Preserve exact no-communication semantics when all neighbors are masked out.
            all_masked = (mask_float.sum(dim=-1, keepdim=True) == 0)

        weights = F.softmax(scores, dim=-1)  # (B, 1, N)
        if mask_expanded is not None:
            weights = weights * mask_expanded

        # Weighted sum of values: (B, 1, N) × (B, N, D) → (B, 1, D) → (B, D)
        context = torch.bmm(weights, val).squeeze(1)

        # Output projection
        context = self.W_out(context)
        if all_masked is not None:
            context = context.masked_fill(all_masked.expand(-1, context.size(-1)), 0.0)
            if return_attention:
                weights = weights.masked_fill(all_masked.unsqueeze(-1), 0.0)

        if return_attention:
            return context, weights.squeeze(1)  # (B, N)

        return context


class IPSMeanAggregator(nn.Module):
    """Hard top-k interaction-priority selection with mean aggregation."""

    def __init__(self):
        super().__init__()
        self.intent_dim = Config.INTENT_DIM
        self.top_k = int(getattr(Config, "IPS_TOP_K", 3))

    def forward(self, ego_z, neighbor_zs, neighbor_rel_pos, mask=None, return_attention: bool = False):
        del ego_z  # receiver intent is intentionally not used for IPS scoring

        scores = _interaction_priority_scores(neighbor_rel_pos, mask)
        selection, valid_counts = _topk_selection_mask(scores, mask, self.top_k)
        denom = selection.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = selection / denom

        context = torch.sum(weights.unsqueeze(-1) * neighbor_zs, dim=1)
        all_masked = (valid_counts == 0).unsqueeze(-1)
        context = context.masked_fill(all_masked, 0.0)

        if return_attention:
            weights = weights.masked_fill(all_masked.expand(-1, weights.size(-1)), 0.0)
            return context, weights
        return context


class Where2CommRawAggregator(nn.Module):
    """Decision-level Where2Comm-style selective communication.

    Uses a learned sender confidence gate over a higher-dimensional raw message,
    then mean-aggregates received messages and projects them back to intent_dim.
    """

    def __init__(self):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM
        self.intent_dim = Config.INTENT_DIM
        self.raw_dim = int(getattr(Config, "WHERE2COMM_RAW_DIM", 64))
        self.gate_threshold = float(getattr(Config, "WHERE2COMM_GATE_THRESHOLD", 0.5))

        self.message_proj = nn.Linear(self.hidden_dim, self.raw_dim)
        self.gate_proj = nn.Linear(self.hidden_dim, 1)
        self.out_proj = nn.Linear(self.raw_dim, self.intent_dim)

    def forward(self, ego_features, neighbor_features, neighbor_rel_pos, mask=None, return_attention: bool = False):
        del ego_features
        del neighbor_rel_pos

        if mask is None:
            mask = torch.ones(neighbor_features.shape[:2], device=neighbor_features.device, dtype=neighbor_features.dtype)
        else:
            mask = mask.to(dtype=neighbor_features.dtype)

        raw_messages = self.message_proj(neighbor_features)
        gate_logits = self.gate_proj(neighbor_features).squeeze(-1)
        gate_probs = torch.sigmoid(gate_logits)

        hard_gate = (gate_probs >= self.gate_threshold).to(dtype=neighbor_features.dtype)
        gate = hard_gate + gate_probs - gate_probs.detach()
        gate = gate * mask

        denom = gate.sum(dim=-1, keepdim=True).clamp(min=1.0)
        weights = gate / denom

        context_raw = torch.sum(weights.unsqueeze(-1) * raw_messages, dim=1)
        context = self.out_proj(context_raw)

        all_masked = (mask.sum(dim=-1, keepdim=True) == 0) | (gate.sum(dim=-1, keepdim=True) == 0)
        context = context.masked_fill(all_masked.expand(-1, context.size(-1)), 0.0)

        if return_attention:
            weights = weights.masked_fill(all_masked.expand(-1, weights.size(-1)), 0.0)
            return context, weights
        return context
