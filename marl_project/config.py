"""
Configuration file for Graph-Based MARL project.
Defines hyperparameters, structural constraints, and MetaDrive environment settings.
"""

class Config:
    """
    Central configuration class for the MARL system.
    """
    # =========================================================
    # === [新增] 动态地图种子基准 (Base Seed) ===
    # =========================================================
    # train.py 会基于这个值计算当前 Update 的种子：
    # current_seed = BASE_SEED + update_step
    BASE_SEED = 5000  # 建议保持 5000，如果你想换个全新世界就填 42
    
    # === [新增] 训练场景池大小 ===
    # 用于 train.py 的 make_env，决定训练集的多样性
    TRAIN_NUM_SCENARIOS = 5000
    
    # --- Anti-Cheating & Perception ---
    # Critical: Agents cannot see neighbors directly via Lidar.
    # They must rely on the GNN communication mechanism.
    LIDAR_NUM_OTHERS = 0 #它通过雷达只能看到障碍物（点），这迫使它必须依赖 GAT 通信来获取邻居信息。
    LIDAR_NUM_LASERS = 72 # 雷达检测角度数  
    LIDAR_DIST = 60.0  # 雷达的探测半径。
    
    # --- Network Architecture ---
    INTENT_DIM = 32          # (带宽) 意图向量 $z$ 的大小
    MAX_NEIGHBORS = 8        # GAT 聚合时只看最近的 8 个邻居
    HIDDEN_DIM = 128          # 神经网络中间层的神经元数量
    NUM_ATTENTION_HEADS = 4  # [新增] 注意力头数
    
    # === [新增] GAT 物理偏置参数 ===
    # 控制物理距离对注意力权重的衰减力度。
    # 0.0 = 无偏置（纯语义），0.5 = 适中，1.0 = 强物理偏置（只看近处）
    DISTANCE_BIAS_SCALE = 0.5
    
    
    # --- Auxiliary Task ---
    PRED_WAYPOINTS_NUM = 5   # (辅助任务) 未来预测的路点数量
    AUX_TASK_LOSS_COEF = 0.8      # (说明用) 辅助任务损失权重（训练时以 Training 区的 AUX_LOSS_COEF 为准）
    
    # --- Sim-to-Real / Robustness ---
    NOISE_STD = 0.05         # (模拟到真实) 雷达测量噪声的标准差
    MASK_RATIO = 0.02        # 邻居掩码的概率 (模拟通信延迟)
    COMM_RADIUS = 80       # (带宽) 通信半径，超过此距离邻居不可见
    SAFETY_DIST = 8.0        # 安全距离阈值，低于此距离会触发安全惩罚
    CRASH_PENALTY = -200.0    # 碰撞惩罚，较大地图下更温和以保持稳定性
    SAFETY_PENALTY_SCALE = 1.0  # 当过于接近时，按比例缩放安全惩罚
    OUT_OF_ROAD_PENALTY = -200.0 # 当发生出路面事件时施加的惩罚

    # --- Dense, Hierarchical Expert Driver Reward ---
    TARGET_SPEED_KMH = 50.0 # 目标速度（单位：公里每小时）
    # 在弯道/大航向误差时降低目标速度，避免“速度奖励”顶着转弯/路口把车推向碰撞
    CURVE_SLOWDOWN_GAIN = 1.8       # 0~1，越大弯道降速越强
    MIN_TARGET_SPEED_KMH = 10.0     # 弯道最低目标速度（避免奖励被压成 0）
    SPEED_REWARD_SCALE = 0.0 # 速度奖励的缩放系数
    OVERSPEED_PENALTY_SCALE = 1.0 # 超速惩罚的缩放系数
    IDLE_SPEED_KMH = 5.0 # 当速度低于此值时，会施加额外的惩罚
    IDLE_PENALTY = 5.0 # 当速度低于 IDLE_SPEED_KMH 时的惩罚系数


    # =========================================================
    # === [核心修改] 课程学习目标值 (Curriculum Targets) ===
    # =========================================================
    # train.py 会读取这些值作为“最大强度”。
    # 在训练初期，程序会自动将这些值设为 0，然后慢慢加到这里设定的值。

    # =========================================================
    # === [新增] Reward Curriculum Schedule (去重型化开关) ===
    # =========================================================
    # 用“进度比例 progress ∈ [0,1]”划分训练阶段。
    # Phase1: 仅终止 + 进度；Phase2: 打开车道/航向；Phase3: 打开 safety/approach；Phase4: 打开 TTC/comfort/idle + speed shaping
    CURR_PHASE1_END = 0.20
    CURR_PHASE2_END = 0.50
    CURR_PHASE3_END = 0.80

    # 终止项（成功/碰撞/出界）在早期缩放，降低回报方差，帮助 value 稳定学习
    # 0.3 表示：早期终止惩罚/奖励仅保留 30%
    CURR_TERM_SCALE_START = 0.30
    CURR_TERM_SCALE_END = 1.00

    # 速度 shaping 何时启用：默认只在最后阶段打开
    CURR_ENABLE_SPEED_IN_PHASE4 = True

    # --- TTC / Foreseeable Safety (Intersection-friendly) ---
    # 用邻居相对位置/速度近似 TTC，提前惩罚“即将发生的碰撞”
    TTC_THRESHOLD_S = 2.5           # 低于该 TTC 视为危险（秒）
    TTC_PENALTY_SCALE = 1.2       # TTC 惩罚强度   <--- [重要修改] 设为 0.8 或 1.0，给模型一个明确的避险信号
    TTC_DIST_MAX = 25.0             # 只对该距离内邻居计算 TTC（米）
    TTC_CLOSING_SPEED_EPS = 0.5     # 认为在接近的最小闭合速度（m/s）
    
    
    PROGRESS_REWARD_SCALE = 2.0 # 进度奖励的缩放系数
    LANE_CENTER_PENALTY_SCALE = 1.0 # 车道中心惩罚的缩放系数
    LANE_WIDTH_REF = 3.6 # 车道宽度参考值，用于计算车道中心惩罚
    HEADING_PENALTY_SCALE = 0.3 # 角度误差惩罚的缩放系数
    MAX_HEADING_ERROR_RAD = 0.7 # 最大允许的角度误差（弧度）
    APPROACH_PENALTY_SCALE = 0.2 # 接近目标奖励的缩放系数
    SUCCESS_REWARD = 300.0 # 成功完成一圈的奖励

    # --- Action Smoothing / Comfort ---
    ACTION_SMOOTH_ALPHA = 0.5  # 低通滤波系数，越大越平滑
    ACTION_CHANGE_PENALTY = 0.5  # 对动作变化幅度的惩罚系数
    ACTION_MAG_PENALTY = 0.05     # 对动作幅值的惩罚系数（鼓励低加速度/低转向）

    # --- Deadlock / Long Idle (only when low risk) ---
    # “长期不动”定义：连续 N 步 低速 + 低进度，且周围风险低（避免误罚路口让行）
    IDLE_LONG_STEPS = 60           # 触发长期不动惩罚的连续步数
    IDLE_PROGRESS_EPS = 0.15       # 每步纵向进度低于该值视为“无进度”（米）
    IDLE_LONG_PENALTY = 2.0        # 触发后每步额外惩罚
    IDLE_SAFE_MIN_DIST = 10.0      # 最近车距大于该值才允许判定“低风险”（米）
    IDLE_SAFE_MIN_TTC = 3.0        # 最小 TTC 大于该值才允许判定“低风险”（秒）

    # --- Training ---
    # PRETRAIN_EPOCHS = 100  <-- 预训练轮数
    PPO_EPOCHS = 10          # (训练) PPO 更新轮数
    AUX_LOSS_COEF = 1.0      # (训练) 辅助任务损失的权重系数
    VF_COEF = 0.5            # (训练) Value loss 权重（过大容易导致 loss 上升但策略没变好）
    CLIP_EPSILON = 0.20       # PPO clip epsilon
    ENTROPY_COEF = 0.01     # 稍微降低熵以减少噪声动作
    MAX_GRAD_NORM = 0.5      # 梯度裁剪范数
    LR = 3e-4
    GAMMA = 0.99
    
    NUM_ENVS = 48             # (训练) 并行环境数
    
    N_STEPS = 1024           # (训练) 每个环境的时间步长
    BATCH_SIZE = 49152         # (训练) 每个 PPO 更新的批量大小  注意：如果 NUM_ENVS * N_STEPS 很大，BATCH_SIZE 可以适当增大
    NUM_MINIBATCHES = 32      # (训练) 每个 PPO 更新的小批量数
    
    # --- Environment ---
    # Map switching
    # - MAP_MODE="block_num":  程序化城市（按 block 数生成）
    # - MAP_MODE="block_sequence": 按序列字符串生成（例如 "SSSSS", "X", "r" 等）
    MAP_MODE = "block_sequence"
    MAP_BLOCK_NUM = 7
    MAP_TYPE = "C"           # 当 MAP_MODE="block_sequence" 时生效
    
    
    @staticmethod
    def get_metadrive_config():
        """
        Returns the configuration dictionary for MultiAgentMetaDrive.
        Updated for Realistic Urban Traffic Simulation (Big Map Mode).
        """
        map_mode = getattr(Config, "MAP_MODE", "block_num")
        if map_mode == "block_sequence":
            map_config = {
                "type": "block_sequence",
                "config": getattr(Config, "MAP_TYPE", "SSSSS"),
                "lane_num": 2,
            }
        else:
            map_config = {
                "type": "block_num",
                "config": int(getattr(Config, "MAP_BLOCK_NUM", 3)),
                "lane_num": 3,
            }

        return {
            # === 1. 地图升级：启用程序化城市生成 (Big Map) ===
            # "type": "big" 会自动生成一个闭环的城市路网
            # 彻底解决了“连不起来”和“只有同向车流”的问题。
            "map_config": map_config,
            
            # === 2. 种子选择：锁定黄金考场 ===
            # 种子 5000 生成的地图包含丰富的 路口(X) 和 环岛(C)
            "num_scenarios": getattr(Config, "TRAIN_NUM_SCENARIOS", 5000), # 使用新的参数
            "start_seed": getattr(Config, "BASE_SEED", 5000),
            
            # === 3. 流量适配：大地图需要更多车 ===
            "num_agents": 4,          # 降低并发车流以减轻拥堵
            "traffic_density": 0.00,   # 背景干扰车 (IDM Bot)
            "traffic_mode": "respawn", # 开启无限重生，保证路口永远繁忙
            "allow_respawn": True,     # 允许 RL Agent 撞车后复活继续训练
            # 空列表触发自动填充所有可重生道路，避免类型断言错误
            "spawn_roads": [],
            
            # === 4. 时间与视觉优化 ===
            "horizon": 2000,           # 跑完一圈大概需要很久，时间给足
            # 碰撞/出界后立即把“死亡车辆”从场景移除（默认 MA 环境会 delay_done 让车留几秒）
            "delay_done": 0,
            "use_render": False,       # 训练时关闭，评估时开启
            "disable_model_compression": True, # 修复渲染材质丢失问题
            
            # === 5. 车辆配置 (保持你的 Sedan 设定) ===
            "vehicle_config": {
                "vehicle_model": "s",  # 强制使用轿车 (Sedan)，拒绝越野车
                "show_lidar": False,   # 关闭雷达线渲染，画面更干净
                "show_navi_mark": False,
                "lidar": {
                    "num_lasers": Config.LIDAR_NUM_LASERS,
                    "distance": Config.LIDAR_DIST,
                    "num_others": Config.LIDAR_NUM_OTHERS, # 保持“物理盲”设定
                    "gaussian_noise": 0.0, 
                    "dropout_prob": 0.0,   
                },
                "lane_line_detector": {"num_lasers": 0, "distance": 0},
                "side_detector": {"num_lasers": 0, "distance": 0},
            },
            
            # === 6. 终止条件 ===
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "out_of_road_done": True,
        }

