"""
TarMAC Training Configuration — aligned with baseline_intent_gat.

This file is a COPY of the baseline_intent_gat config with ONLY two changes:
  1. EXPERIMENT_MODE = "tarmac"
  2. COMM_MODULE = "tarmac"

All other hyperparameters are IDENTICAL to ensure a fair comparison.
Generated from: logs/marl_experiment/baseline_intent_gat/config.py
"""

class Config:
    # =========================================================
    # === EXPERIMENT MODE: TarMAC external baseline ===
    # =========================================================
    EXPERIMENT_MODE = "tarmac"   # ← CHANGED from "ours"
    
    # =========================================================
    # === Seeds & Scenarios ===
    # =========================================================
    BASE_SEED = 42
    TRAIN_NUM_SCENARIOS = 5000
    
    # --- Anti-Cheating & Perception ---
    LIDAR_NUM_OTHERS = 0
    LIDAR_NUM_LASERS = 72
    LIDAR_DIST = 60.0
    
    # --- Network Architecture ---
    INTENT_DIM = 32
    MAX_NEIGHBORS = 8
    HIDDEN_DIM = 128
    NUM_ATTENTION_HEADS = 4
    
    # --- Communication Module ---
    COMM_MODULE = "tarmac"       # ← CHANGED from "gat"
    TARMAC_NUM_ROUNDS = 1
    
    # ===== Train / Eval Environment Switches =====
    TRAFFIC_DENSITY_TRAIN = 0.08    # SAME as baseline_intent_gat
    TRAFFIC_DENSITY_EVAL  = 0.08
    ALLOW_RESPAWN_TRAIN = True
    ALLOW_RESPAWN_EVAL  = False
    NUM_AGENTS = 6
    
    # === GAT Physics Bias (not used by TarMAC, kept for compatibility) ===
    DISTANCE_BIAS_SCALE = 0.5
    
    # --- Auxiliary Task ---
    PRED_WAYPOINTS_NUM = 5
    AUX_TASK_LOSS_COEF = 0.8
    
    # --- Sim-to-Real / Robustness ---
    NOISE_STD = 0.05
    MASK_RATIO = 0.02
    COMM_RADIUS = 100
    SAFETY_DIST = 12.0
    CRASH_PENALTY = -350.0
    SAFETY_PENALTY_SCALE = 2.5
    OUT_OF_ROAD_PENALTY = -350.0

    # --- Reward ---
    TARGET_SPEED_KMH = 40.0
    CURVE_SLOWDOWN_GAIN = 1.8
    MIN_TARGET_SPEED_KMH = 10.0
    SPEED_REWARD_SCALE = 0.0
    OVERSPEED_PENALTY_SCALE = 2.0
    IDLE_SPEED_KMH = 5.0
    IDLE_PENALTY = 5.0

    # --- Curriculum ---
    CURR_PHASE1_END = 0.20
    CURR_PHASE2_END = 0.50
    CURR_PHASE3_END = 0.65
    CURR_TERM_SCALE_START = 0.30
    CURR_TERM_SCALE_END = 1.00
    CURR_ENABLE_SPEED_IN_PHASE4 = True

    # --- TTC / Safety ---
    TTC_THRESHOLD_S = 2.5
    TTC_PENALTY_SCALE = 3.0
    TTC_DIST_MAX = 25.0
    TTC_CLOSING_SPEED_EPS = 0.5
    
    PROGRESS_REWARD_SCALE = 1.0
    LANE_CENTER_PENALTY_SCALE = 0.8
    LANE_WIDTH_REF = 3.6
    HEADING_PENALTY_SCALE = 0.8
    MAX_HEADING_ERROR_RAD = 0.7
    APPROACH_PENALTY_SCALE = 0.2
    SUCCESS_REWARD = 300.0

    # --- Action Smoothing / Comfort ---
    ACTION_SMOOTH_ALPHA = 0.5
    ACTION_CHANGE_PENALTY = 0.5
    ACTION_MAG_PENALTY = 0.05

    # --- Deadlock / Long Idle ---
    IDLE_LONG_STEPS = 60
    IDLE_PROGRESS_EPS = 0.15
    IDLE_LONG_PENALTY = 2.0
    IDLE_SAFE_MIN_DIST = 10.0
    IDLE_SAFE_MIN_TTC = 3.0

    # --- Training (ALL IDENTICAL to baseline_intent_gat) ---
    PPO_EPOCHS = 15
    AUX_LOSS_COEF = 1.0
    VF_COEF = 0.5
    CLIP_EPSILON = 0.15
    ENTROPY_COEF = 0.02
    MAX_GRAD_NORM = 0.5
    LR = 1e-4
    GAMMA = 0.99
    
    NUM_ENVS = 4
    N_STEPS = 2048
    BATCH_SIZE = 8192
    NUM_MINIBATCHES = 8

    # --- Early Stopping ---
    EARLY_STOP_ENABLED = False
    EARLY_STOP_METRIC = "mean_episode_reward"
    EARLY_STOP_MODE = "max"
    EARLY_STOP_WINDOW_UPDATES = 5
    EARLY_STOP_WARMUP_UPDATES = 30
    EARLY_STOP_PATIENCE_UPDATES = 30
    EARLY_STOP_MIN_DELTA = 1.0
    
    # --- Logging ---
    LOG_ROOT = "logs/marl_experiment"
    EXP_NAME = "baseline_tarmac"
    
    # --- Environment ---
    MAP_MODE = "block_num"
    MAP_BLOCK_NUM = 5            # SAME as baseline_intent_gat
    MAP_TYPE = "SCS"
    MAP_EXIT_LENGTH = 70
    DEBUG_SPAWN_ON_RESET = False
    STRICT_SPAWN_CAPACITY = True
