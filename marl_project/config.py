"""
Configuration file for Graph-Based MARL project.
Defines hyperparameters, structural constraints, and MetaDrive environment settings.
"""

class Config:
    """
    Central configuration class for the MARL system.
    """
    # =========================================================
    # === [论文核心] 实验模式开关 (Ablation Study) ===
    # =========================================================
    # 用于快速切换不同的基线对比实验
    # Options: "ours" | "no_comm" | "no_aux" | "lidar_only" | "oracle" | "mappo"
    EXPERIMENT_MODE = "ours"  # 设置为 "ours" 以运行完整系统（91维度，GAT通信+辅助任务）
                               # 设置为 "mappo" 以运行MAPPO基线（去中心化Actor + 中心化Critic）
    
    # =========================================================
    # === [新增] 动态地图种子基准 (Base Seed) ===
    # =========================================================
    # train.py 会基于这个值计算当前 Update 的种子：
    # current_seed = BASE_SEED + update_step
    BASE_SEED = 42  # 建议保持 5000，如果你想换个全新世界就填 42
    
    
    # === [新增] 训练场景池大小 ===
    # 用于 train.py 的 make_env，决定训练集的多样性
    TRAIN_NUM_SCENARIOS = 5000  # 保持5000场景，提供充足的训练多样性
    
    # --- Anti-Cheating & Perception ---
    # Critical: Agents cannot see neighbors directly via Lidar.
    # They must rely on the GNN communication mechanism.
    LIDAR_NUM_OTHERS = 0 #它通过雷达只能看到障碍物（点），这迫使它必须依赖 GAT 通信来获取邻居信息。
    LIDAR_NUM_LASERS = 72 # 雷达检测角度数  
    LIDAR_DIST = 60.0  # 雷达的探测半径。
    
    # --- Network Architecture ---
    INTENT_DIM = 32          # (带宽) 意图向量 $z$ 的大小 [重要：必须与训练时一致！]
    MAX_NEIGHBORS = 8        # GAT 聚合时只看最近的 8 个邻居
    HIDDEN_DIM = 128          # 神经网络中间层的神经元数量
    NUM_ATTENTION_HEADS = 4  # [新增] 注意力头数
    
    # --- Communication Module Selection ---
    # "gat"    = Intent-GAT (our method): receiver-centric multi-head graph attention
    # "tarmac" = TarMAC (Das et al. 2019): sender-targeted signature-based routing
    # "ips_mean" = hard top-k interaction-priority selection with mean aggregation
    # "where2comm_raw" = confidence-gated raw feature sharing with mean aggregation
    # "none" = decentralized actor without explicit communication
    COMM_MODULE = "gat"
    TARMAC_NUM_ROUNDS = 1      # TarMAC communication rounds (1 = single-round, matching GAT)
    IPS_TOP_K = 3
    WHERE2COMM_RAW_DIM = 64
    WHERE2COMM_GATE_THRESHOLD = 0.5
    
    # ===== Train / Eval Environment Switches =====
    TRAFFIC_DENSITY_TRAIN = 0.15    # 提高到0.15，制造更拥挤场景（无通信基线需要更高难度）
    TRAFFIC_DENSITY_EVAL  = 0.18    # 评估时更高，测试真实压力场景

    ALLOW_RESPAWN_TRAIN = True
    ALLOW_RESPAWN_EVAL  = False
    
    NUM_AGENTS = 6  # 从8降到6，平衡协作强度与任务难度

    
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
    COMM_RADIUS = 100      # (带宽) 通信半径，保持100m
    SAFETY_DIST = 12.0        # 安全距离阈值（从9增加到12，密集交通需要更大安全距离）
    CRASH_PENALTY = -500.0    # 碰撞惩罚（增强到-500，无通信时碰撞代价应该更高）
    SAFETY_PENALTY_SCALE = 3.5  # 接近惩罚（增强到3.5，无通信时必须极度保守）
    OUT_OF_ROAD_PENALTY = -500.0 # 出界惩罚（增强到-500，严厉惩罚失控）

    # --- Dense, Hierarchical Expert Driver Reward ---
    TARGET_SPEED_KMH = 30.0 # 目标速度（降到30km/h，无通信时必须更保守）
    # 在弯道/大航向误差时降低目标速度，避免"速度奖励"顶着转弯/路口把车推向碰撞
    CURVE_SLOWDOWN_GAIN = 2.0       # 提高到2.0，弯道降速更激进
    MIN_TARGET_SPEED_KMH = 5.0     # 弯道最低目标速度（降到5km/h，接近停车）
    SPEED_REWARD_SCALE = 0.0 # 速度奖励的缩放系数（保持关闭）
    OVERSPEED_PENALTY_SCALE = 3.0 # 超速惩罚（增强到3.0，严厉惩罚超速）
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
    CURR_PHASE3_END = 0.65

    # 终止项（成功/碰撞/出界）在早期缩放，降低回报方差，帮助 value 稳定学习
    # 0.3 表示：早期终止惩罚/奖励仅保留 30%
    CURR_TERM_SCALE_START = 0.30
    CURR_TERM_SCALE_END = 1.00

    # 速度 shaping 何时启用：默认只在最后阶段打开
    CURR_ENABLE_SPEED_IN_PHASE4 = True

    # --- TTC / Foreseeable Safety (Intersection-friendly) ---
    # 用邻居相对位置/速度近似 TTC，提前惩罚“即将发生的碰撞”
    TTC_THRESHOLD_S = 3.0           # 提高到3.0秒，要求更早预警
    TTC_PENALTY_SCALE = 5.0       # TTC 惩罚强度（增强到5.0，无通信必须极度保守）
    TTC_DIST_MAX = 25.0             # 只对该距离内邻居计算 TTC（米）
    TTC_CLOSING_SPEED_EPS = 0.5     # 认为在接近的最小闭合速度（m/s）
    
    
    PROGRESS_REWARD_SCALE = 1.0 # 进度奖励（从1.2降到1.0，降低激进度）
    LANE_CENTER_PENALTY_SCALE = 0.8 # 车道中心惩罚（从0.5增强到0.8，防止出界）
    LANE_WIDTH_REF = 3.6 # 车道宽度参考值，用于计算车道中心惩罚
    HEADING_PENALTY_SCALE = 0.8 # 角度误差惩罚（从0.5增强到0.8，保持车道）
    MAX_HEADING_ERROR_RAD = 0.7 # 最大允许的角度误差（弧度）
    APPROACH_PENALTY_SCALE = 0.2 # 接近目标奖励（从0.3降到0.2，不要过于激进）
    SUCCESS_REWARD = 300.0 # 成功奖励（保持300，平衡安全与完成）

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
    PPO_EPOCHS = 15          # (训练) PPO 更新轮数（从10增加到15，多车场景需要更充分学习）
    AUX_LOSS_COEF = 1.0      # (训练) 辅助任务损失的权重系数
    VF_COEF = 0.5            # (训练) Value loss 权重（提高到0.5，让价值函数更准确）
    VALUE_CLIP = 10.0        # (训练) Value prediction clipping（防止Value爆炸）
    CLIP_EPSILON = 0.2       # PPO clip epsilon（恢复到0.2，标准PPO设置）
    ENTROPY_COEF = 0.01     # 熵系数（降到0.01，让策略更确定性）
    MAX_GRAD_NORM = 0.5      # 梯度裁剪范数
    LR = 3e-4                 # 学习率（降到1e-4，更稳定的学习）
    GAMMA = 0.99
    
    NUM_ENVS = 4             # (训练) 并行环境数（从8降到4，6车场景每个环境更慢）
    
    N_STEPS = 2048           # (训练) 每个环境的时间步长（从1024增加到2048，保持总样本量）
    BATCH_SIZE = 8192         # (训练) 每个 PPO 更新的批量大小  注意：如果 NUM_ENVS * N_STEPS 很大，BATCH_SIZE 可以适当增大
    NUM_MINIBATCHES = 8      # (训练) 每个 PPO 更新的小批量数

    # --- Early Stopping (训练早停) ---
    # 说明：除了 --max_time（墙钟时间停止）外，可选启用“指标长期无提升则停止”。
    # 默认关闭，保证与旧实验完全一致。
    EARLY_STOP_ENABLED = False
    # 可选："mean_episode_reward" | "success_rate" | "-crash_rate" | "-out_of_road_rate"
    EARLY_STOP_METRIC = "mean_episode_reward"
    # "max" 表示指标越大越好；"min" 表示越小越好
    EARLY_STOP_MODE = "max"
    # 计算早停指标时的滑动窗口（按 update 计）
    EARLY_STOP_WINDOW_UPDATES = 5
    # warmup：前 N 个 update 不做早停（避免刚开始噪声大就停）
    EARLY_STOP_WARMUP_UPDATES = 30
    # patience：连续多少个 update 没有达到“最小提升”就停止
    EARLY_STOP_PATIENCE_UPDATES = 30
    # 最小提升阈值：
    # - mode=max: 需要 new >= best + min_delta 才算提升
    # - mode=min: 需要 new <= best - min_delta 才算提升
    EARLY_STOP_MIN_DELTA = 1.0
    # 1. 设置一个基础的日志目录
    LOG_ROOT = "logs/marl_experiment"
    
    # 2. 设置一个默认的实验名称 (防止你忘记传参时有个兜底)
    EXP_NAME = "final/baseline_intent_64"
    
    
    # --- Environment ---
    # Map switching
    # - MAP_MODE="block_num":  程序化城市（按 block 数生成）
    # - MAP_MODE="block_sequence": 按序列字符串生成（例如 "SSSSS", "X", "r" 等）
    MAP_MODE = "block_num"
    MAP_BLOCK_NUM = 3  # 缩小到3个block，迫使车辆频繁遭遇复杂交互
    MAP_TYPE = "SCS"           # 当 MAP_MODE="block_sequence" 时生效
    
    @staticmethod
    def apply_experiment_mode(mode: str = None):
        """
        应用特定的实验模式配置，用于消融实验。
        
        Args:
            mode: 实验模式 ("ours" | "no_comm" | "no_aux" | "lidar_only")
                  如果为None，则使用Config.EXPERIMENT_MODE
        
        用法:
            # 在train.py开头调用
            Config.apply_experiment_mode("no_comm")
            
        说明:
            - ours: 完整系统 (GAT + Aux + 物理盲雷达)
            - no_comm: IPPO基线 (无通信)
            - no_aux: 无辅助任务 (只有GAT通信)
            - lidar_only: 传统方法 (可直接看到邻居)
        """
        if mode is None:
            mode = getattr(Config, "EXPERIMENT_MODE", "ours")
        
        print(f"\n{'='*60}")
        print(f"[INFO] 应用实验模式: {mode}")
        print(f"{'='*60}")
        
        if mode == "ours":
            # 完整系统：GAT + Aux + Lidar（物理盲）
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 1.0
            Config.LIDAR_NUM_OTHERS = 0  # 保持"物理盲"设定
            Config.MASK_RATIO = 0.02  # 模拟正常的小概率通信丢包
            Config.COMM_MODULE = "gat"
            print("[OK] 完整系统: GAT通信 + 辅助任务 + 物理盲雷达")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS}")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS}")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (轻度丢包)")
            
            
        elif mode == "oracle":
            # 理想通信：有雷达 + 完美通信
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 1.0
            Config.LIDAR_NUM_OTHERS = 4
            Config.MASK_RATIO = 0.00  # 无通信，丢包参数无意义
            Config.COMM_MODULE = "gat"
            print(" 上帝视角 ")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS} ")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS}")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (无丢包)")
            
        elif mode == "no_comm":
            # IPPO基线：真正的全盲基线（无雷达 + 无通信）
            # ⚠️ 关键修复：保持MAX_NEIGHBORS=8以维持模型架构，用MASK_RATIO=1.0屏蔽通信
            Config.MAX_NEIGHBORS = 8  # 保持架构一致，避免维度不匹配
            Config.AUX_LOSS_COEF = 0.0
            Config.LIDAR_NUM_OTHERS = 0  # 真正的盲基线
            Config.MASK_RATIO = 1.0  # 100%屏蔽邻居信息，模拟无通信
            Config.COMM_MODULE = "none"
            print("🚫 无通信基线 (IPPO): 真正的全盲基线")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS} (架构保持8，用mask屏蔽)")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS} (全盲)")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (100%屏蔽=无通信)")
            
        elif mode == "no_aux":
            # 无辅助任务：只有GAT通信
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 0.0
            Config.LIDAR_NUM_OTHERS = 0
            Config.MASK_RATIO = 0.02  # 保持正常通信条件
            Config.COMM_MODULE = "gat"
            print("📡 纯通信模式: GAT通信 + 无辅助任务")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS}")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF} (禁用辅助任务)")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS}")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (轻度丢包)")
            
        elif mode == "lidar_only":
            # 传统方法：通过雷达直接看到邻居（无V2V通信）
            # ⚠️ 关键修复：保持MAX_NEIGHBORS=8以维持模型架构，用MASK_RATIO=1.0屏蔽V2V通信
            Config.MAX_NEIGHBORS = 8  # 保持架构一致，避免维度不匹配
            Config.AUX_LOSS_COEF = 0.0
            Config.LIDAR_NUM_OTHERS = 4  # 可以直接看到邻居
            Config.MASK_RATIO = 1.0  # 100%屏蔽V2V通信
            Config.COMM_MODULE = "none"
            print("👁️ 传统感知: 雷达直接观测邻居（无通信）")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS} (架构保持8，用mask屏蔽V2V)")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS} (可见邻居)")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (100%屏蔽V2V通信)")
            
        elif mode == "mappo":
            # MAPPO基线：去中心化Actor + 中心化Critic（无显式通信）
            # 与"ours"相同的观测设置，但使用中心化Critic进行训练
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 1.0  # 保持辅助任务以公平对比
            Config.LIDAR_NUM_OTHERS = 0  # 保持"物理盲"设定
            Config.MASK_RATIO = 1.0  # 100%屏蔽显式通信（MAPPO不需要显式通信）
            Config.COMM_MODULE = "none"
            print("🎯 MAPPO基线: 去中心化Actor + 中心化Critic")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS}")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF} (保持辅助任务)")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS} (物理盲)")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (无显式通信)")
            print(f"   ⚠️  注意: Critic使用全局观测(6×91=546维)进行训练")
            
        elif mode == "mappo_ips":
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 1.0
            Config.LIDAR_NUM_OTHERS = 0
            Config.MASK_RATIO = 0.02
            Config.COMM_MODULE = "ips_mean"
            print("MAPPO-IPS baseline: centralized critic + IPS top-k actor communication")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS}")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS}")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO}")
            print(f"   - COMM_MODULE: {Config.COMM_MODULE}")
            print(f"   - IPS_TOP_K: {Config.IPS_TOP_K}")

        elif mode == "where2comm":
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 1.0
            Config.LIDAR_NUM_OTHERS = 0
            Config.MASK_RATIO = 0.02
            Config.COMM_MODULE = "where2comm_raw"
            print("Where2Comm-style baseline: confidence-gated raw feature sharing")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS}")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS}")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO}")
            print(f"   - COMM_MODULE: {Config.COMM_MODULE}")
            print(f"   - WHERE2COMM_RAW_DIM: {Config.WHERE2COMM_RAW_DIM}")
            print(f"   - WHERE2COMM_GATE_THRESHOLD: {Config.WHERE2COMM_GATE_THRESHOLD}")

        elif mode == "tarmac":
            # TarMAC外部基线 (Das et al., 2019): sender-targeted attention communication
            # 与"ours"完全相同的环境/感知/辅助任务设置，仅通信模块不同
            # 训练超参与 baseline_intent_gat 完全对齐
            Config.MAX_NEIGHBORS = 8
            Config.AUX_LOSS_COEF = 1.0
            Config.LIDAR_NUM_OTHERS = 0  # 保持"物理盲"设定
            Config.MASK_RATIO = 0.02  # 与 ours 一致的轻度丢包
            Config.COMM_MODULE = "tarmac"  # 关键：切换通信模块
            # 训练环境对齐 baseline_intent_gat
            Config.TRAFFIC_DENSITY_TRAIN = 0.08
            Config.TRAFFIC_DENSITY_EVAL = 0.08
            Config.MAP_BLOCK_NUM = 5
            Config.BASE_SEED = 42
            Config.CRASH_PENALTY = -350.0
            Config.OUT_OF_ROAD_PENALTY = -350.0
            Config.SAFETY_PENALTY_SCALE = 2.5
            print("📨 TarMAC外部基线: Sender-targeted attention communication")
            print(f"   - MAX_NEIGHBORS: {Config.MAX_NEIGHBORS}")
            print(f"   - AUX_LOSS_COEF: {Config.AUX_LOSS_COEF}")
            print(f"   - LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS} (物理盲)")
            print(f"   - MASK_RATIO: {Config.MASK_RATIO} (轻度丢包)")
            print(f"   - COMM_MODULE: tarmac (替代GAT)")
            print(f"   - TRAFFIC_DENSITY: {Config.TRAFFIC_DENSITY_TRAIN}")
            print(f"   - MAP_BLOCK_NUM: {Config.MAP_BLOCK_NUM}")
            
        else:
            raise ValueError(
                f"Unknown experiment mode: {mode}. "
                f"Available: 'ours', 'oracle', 'no_comm', 'no_aux', 'lidar_only', 'mappo', 'mappo_ips', 'where2comm', 'tarmac'"
            )
        
        print(f"{'='*60}\n")

    # Spawn capacity tuning (paper-safe): increase longitudinal slots per spawn road.
    # SpawnManager uses: num_slots = floor((exit_length - ENTRANCE_LENGTH) / RESPAWN_REGION_LONGITUDE).
    # Default MetaDrive exit_length=50 often yields num_slots=2, which can be insufficient for num_agents>=8.
    MAP_EXIT_LENGTH = 70

    # Spawn diagnostics / safety
    DEBUG_SPAWN_ON_RESET = False
    STRICT_SPAWN_CAPACITY = True
    
    
    @staticmethod
    def get_metadrive_config(is_eval: bool = False):
        """
        Returns the configuration dictionary for MultiAgentMetaDrive.
        Updated for Realistic Urban Traffic Simulation (Big Map Mode).
        """
        map_mode = getattr(Config, "MAP_MODE", "block_num")
        if map_mode == "block_sequence":
            map_config = {
                "type": "block_sequence",
                "config": getattr(Config, "MAP_TYPE", "SSSSS"),
                "lane_num": 3,
                "exit_length": int(getattr(Config, "MAP_EXIT_LENGTH", 70)),
            }
        else:
            map_config = {
                "type": "block_num",
                "config": int(getattr(Config, "MAP_BLOCK_NUM", 3)),
                "lane_num": 3,
                "exit_length": int(getattr(Config, "MAP_EXIT_LENGTH", 70)),
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
            "num_agents": getattr(Config, "NUM_AGENTS", 6),          # 降低并发车流以减轻拥堵
            "traffic_density": (
                        Config.TRAFFIC_DENSITY_EVAL if is_eval
                            else Config.TRAFFIC_DENSITY_TRAIN
                        ),   # 背景干扰车 (IDM Bot)
            "traffic_mode": "respawn", # 开启无限重生，保证路口永远繁忙
            "allow_respawn": (
                        Config.ALLOW_RESPAWN_EVAL if is_eval
                            else Config.ALLOW_RESPAWN_TRAIN
                        ),     # 允许 RL Agent 撞车后复活继续训练
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

    @staticmethod
    def get_eval_config():
        """
        独立的评估配置，不受训练时的课程学习、实验模式等影响。
        
        核心差异：
        1. 固定的奖励参数（最大强度，不使用课程学习缩放）
        2. 固定的环境参数（评估专用的流量密度、重生设置等）
        3. 固定的感知设置（避免被 apply_experiment_mode 污染）
        4. 更严格的终止条件（不允许重生）
        
        返回一个包含以下字段的字典：
        - metadrive_config: MetaDrive 环境配置
        - reward_params: 奖励函数参数（固定最大值）
        - noise_std: 感知噪声标准差
        - mask_ratio: 通信掩码比例
        - comm_radius: 通信半径
        """
        # 1. 基础环境配置（评估模式）
        map_mode = getattr(Config, "MAP_MODE", "block_num")
        if map_mode == "block_sequence":
            map_config = {
                "type": "block_sequence",
                "config": getattr(Config, "MAP_TYPE", "SSSSS"),
                "lane_num": 3,
                "exit_length": 70,
            }
        else:
            map_config = {
                "type": "block_num",
                "config": 7,  # 固定使用较大地图
                "lane_num": 3,
                "exit_length": 70,
            }
        
        metadrive_config = {
            "map_config": map_config,
            "num_scenarios": 100,  # 评估通常用较少的场景
            "start_seed": 10000,  # 使用与训练不重叠的种子范围
            
            # 固定的智能体和流量设置
            "num_agents": 10,  # 可以被 CLI 参数覆盖
            "traffic_density": 0.10,  # 固定低流量密度
            "traffic_mode": "respawn",
            "allow_respawn": False,  # 评估时不允许重生（更严格）
            "spawn_roads": [],
            
            # 时间设置
            "horizon": 2000,
            "delay_done": 0,
            
            # 渲染设置（默认关闭）
            "use_render": False,
            "disable_model_compression": True,
            
            # 车辆配置（固定使用物理盲雷达）
            "vehicle_config": {
                "vehicle_model": "default",  # 评估时使用更稳定的默认模型
                "show_lidar": False,
                "show_navi_mark": False,
                "lidar": {
                    "num_lasers": 72,
                    "distance": 60.0,
                    "num_others": 0,  # 固定物理盲设置
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0,
                },
                "lane_line_detector": {"num_lasers": 0, "distance": 0},
                "side_detector": {"num_lasers": 0, "distance": 0},
            },
            
            # 终止条件
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "out_of_road_done": True,
        }
        
        # 2. 固定的奖励参数（最大强度，不使用课程学习）
        reward_params = {
            # 终止奖励/惩罚
            "SUCCESS_REWARD": 300.0,
            "CRASH_PENALTY": -200.0,
            "OUT_OF_ROAD_PENALTY": -200.0,
            
            # 密集奖励缩放系数（全部使用最大值）
            "PROGRESS_REWARD_SCALE": 1.0,
            "SPEED_REWARD_SCALE": 0.0,
            "OVERSPEED_PENALTY_SCALE": 1.0,
            "LANE_CENTER_PENALTY_SCALE": 0.5,
            "HEADING_PENALTY_SCALE": 0.5,
            "SAFETY_PENALTY_SCALE": 1.0,
            "APPROACH_PENALTY_SCALE": 0.2,
            "TTC_PENALTY_SCALE": 1.2,
            "ACTION_CHANGE_PENALTY": 0.5,
            "ACTION_MAG_PENALTY": 0.05,
            "IDLE_PENALTY": 5.0,
            "IDLE_LONG_PENALTY": 2.0,
            
            # 其他阈值参数
            "TARGET_SPEED_KMH": 50.0,
            "CURVE_SLOWDOWN_GAIN": 1.8,
            "MIN_TARGET_SPEED_KMH": 10.0,
            "IDLE_SPEED_KMH": 5.0,
            "SAFETY_DIST": 8.0,
            "TTC_THRESHOLD_S": 2.5,
            "TTC_DIST_MAX": 25.0,
            "TTC_CLOSING_SPEED_EPS": 0.5,
            "LANE_WIDTH_REF": 3.6,
            "MAX_HEADING_ERROR_RAD": 0.7,
            "IDLE_LONG_STEPS": 60,
            "IDLE_PROGRESS_EPS": 0.15,
            "IDLE_SAFE_MIN_DIST": 10.0,
            "IDLE_SAFE_MIN_TTC": 3.0,
            "ACTION_SMOOTH_ALPHA": 0.5,
        }
        
        # 3. 固定的感知参数（评估专用，不受训练时设置影响）
        perception_params = {
            "noise_std": 0.0,  # 评估时默认无噪声（可通过 CLI 覆盖）
            "mask_ratio": 0.0,  # 评估时默认无通信丢包（可通过 CLI 覆盖）
            "comm_radius": 80.0,
            "max_neighbors": 8,
        }
        
        return {
            "metadrive_config": metadrive_config,
            "reward_params": reward_params,
            "perception_params": perception_params,
        }

