"""
Configuration file for Graph-Based MARL project.
Defines hyperparameters, structural constraints, and MetaDrive environment settings.
"""

class Config:
    """
    Central configuration class for the MARL system.
    """
    
    # --- Anti-Cheating & Perception ---
    # Critical: Agents cannot see neighbors directly via Lidar.
    # They must rely on the GNN communication mechanism.
    LIDAR_NUM_OTHERS = 0
    LIDAR_NUM_LASERS = 72
    LIDAR_DIST = 50.0  # Standard detection distance
    
    # --- Network Architecture ---
    INTENT_DIM = 32          # Dimension of the broadcasted latent vector z (Compressed bandwidth)
    MAX_NEIGHBORS = 5        # Max neighbors for graph padding
    HIDDEN_DIM = 64          # Hidden dimension for internal MLPs
    
    # --- Auxiliary Task ---
    PRED_WAYPOINTS_NUM = 5   # Number of future waypoints to predict
    AUX_LOSS_COEF = 0.5      # Weight for the auxiliary reconstruction loss
    
    # --- Sim-to-Real / Robustness ---
    NOISE_STD = 0.05         # Standard deviation for Gaussian noise on Lidar
    MASK_RATIO = 0.1         # Probability of packet loss (neighbor masking)
    COMM_RADIUS = 40.0       # Increased from 20.0 to 40.0 for better reaction time
    SAFETY_DIST = 8.0        # Distance below which a safety penalty is applied

    # --- Training ---
    # PRETRAIN_EPOCHS = 100  <-- REMOVED: We use joint training from start
    PPO_EPOCHS = 10          # <-- ADDED: Number of updates per rollout
    AUX_LOSS_COEF = 0.5      # Coefficient for the auxiliary task
    CLIP_EPSILON = 0.2       # PPO clip epsilon
    MAX_GRAD_NORM = 0.5      # Gradient clipping norm
    LR = 3e-4
    GAMMA = 0.99
    NUM_ENVS = 56            # One per physical core (Intel 8180)
    N_STEPS = 8192           # Large rollout for GPU saturation
    BATCH_SIZE = 1024
    NUM_MINIBATCHES = 8
    
    # --- Environment ---
    MAP_TYPE = "S"           # "S"=Straight (Easy), "C"=Circular, "X"=Intersection (Hard)

    @staticmethod
    def get_metadrive_config():
        """
        Returns the configuration dictionary for MultiAgentMetaDrive.
        
        Returns:
            dict: A dictionary compatible with MetaDrive's config system.
        """
        return {
            "use_render": False,
            "render_mode": "none",
            "horizon": 1000,
            "disable_model_compression": True,
            "start_seed": 42,
            "map": 3, # Keep default to avoid assertion error when using custom map_config
            "num_agents": 4,
            # --- 关键修正：传感器配置必须放在 vehicle_config 内部 ---
            "vehicle_config": {
                "lidar": {
                    "num_lasers": Config.LIDAR_NUM_LASERS,
                    "distance": Config.LIDAR_DIST,
                    "num_others": Config.LIDAR_NUM_OTHERS, # Enforce blind agents
                    "gaussian_noise": 0.0, # We implement custom noise in wrapper
                    "dropout_prob": 0.0,   # We implement custom masking in wrapper
                },
                
                # Disable unnecessary sensors for performance
                "lane_line_detector": {
                    "num_lasers": 0,
                    "distance": 0
                },
                "side_detector": {
                    "num_lasers": 0,
                    "distance": 0
                },
            },
            # ----------------------------------------------------
            
            # Environment settings
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "out_of_road_done": True,
            "horizon": 1000,
            
            # Map config
            "map_config": {
                "type": "block_sequence",
                "config": Config.MAP_TYPE, # "S"
                "lane_num": 3
            }
        }