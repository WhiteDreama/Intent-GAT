# -*- coding: utf-8 -*-
import os, sys
os.environ["TORCHDYNAMO_DISABLE"] = "1"          # 彻底关闭 TorchDynamo
# 可选：让 Dynamo 出错后自动回退到 eager，双保险
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import sys
import shutil
import argparse
import json
import subprocess
import time
from collections import defaultdict, deque

# Ensure project root is importable when running from repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.models.policy import CooperativePolicy
from marl_project.mappo_modules import MAPPOPolicy, reshape_for_centralized_critic
from marl_project.mp_env import MultiProcEnv


_MAX_TTC_PENALTY_SCALE = float(getattr(Config, "TTC_PENALTY_SCALE", 0.0))
_MAX_ACTION_MAG_PENALTY = float(getattr(Config, "ACTION_MAG_PENALTY", 0.0))
_MAX_ACTION_CHANGE_PENALTY = float(getattr(Config, "ACTION_CHANGE_PENALTY", 0.0))

# Reward curriculum maxima (read once from config)
_MAX_LANE_CENTER_PENALTY_SCALE = float(getattr(Config, "LANE_CENTER_PENALTY_SCALE", 0.0))
_MAX_HEADING_PENALTY_SCALE = float(getattr(Config, "HEADING_PENALTY_SCALE", 0.0))
_MAX_SAFETY_PENALTY_SCALE = float(getattr(Config, "SAFETY_PENALTY_SCALE", 0.0))
_MAX_APPROACH_PENALTY_SCALE = float(getattr(Config, "APPROACH_PENALTY_SCALE", 0.0))
_MAX_IDLE_PENALTY = float(getattr(Config, "IDLE_PENALTY", 0.0))
_MAX_IDLE_LONG_PENALTY = float(getattr(Config, "IDLE_LONG_PENALTY", 0.0))
_MAX_SPEED_REWARD_SCALE = float(getattr(Config, "SPEED_REWARD_SCALE", 0.0))
_MAX_OVERSPEED_PENALTY_SCALE = float(getattr(Config, "OVERSPEED_PENALTY_SCALE", 0.0))

# Terminal reward magnitudes (keep as-is in config, but scale them by curriculum)
_MAX_CRASH_PENALTY = float(getattr(Config, "CRASH_PENALTY", -200.0))
_MAX_OUT_OF_ROAD_PENALTY = float(getattr(Config, "OUT_OF_ROAD_PENALTY", -200.0))
_MAX_SUCCESS_REWARD = float(getattr(Config, "SUCCESS_REWARD", 300.0))


def update_curriculum(current_step: int, total_steps: int):
    """Curriculum learning schedule.

    De-heavy reward schedule:
    - Phase 1 (0-20%): only terminal + progress; disable lane/heading/safety/TTC/comfort/idle.
    - Phase 2 (20-50%): ramp lane/heading penalties.
    - Phase 3 (50-80%): ramp safety/approach penalties.
    - Phase 4 (80-100%): enable TTC + comfort + idle penalties.
    """
    if total_steps <= 0:
        progress = 1.0
    else:
        progress = float(current_step) / float(total_steps)
    progress = max(0.0, min(1.0, progress))

    # Phase boundaries (configurable)
    p1 = float(getattr(Config, "CURR_PHASE1_END", 0.2))
    p2 = float(getattr(Config, "CURR_PHASE2_END", 0.5))
    p3 = float(getattr(Config, "CURR_PHASE3_END", 0.8))
    # sanitize ordering
    if not (0.0 < p1 < p2 < p3 <= 1.0):
        p1, p2, p3 = 0.2, 0.5, 0.8

    term_start = float(getattr(Config, "CURR_TERM_SCALE_START", 0.3))
    term_end = float(getattr(Config, "CURR_TERM_SCALE_END", 1.0))
    term_start = max(0.0, min(1.0, term_start))
    term_end = max(0.0, min(1.0, term_end))
    if term_end < term_start:
        term_start, term_end = term_end, term_start

    enable_speed_phase4 = bool(getattr(Config, "CURR_ENABLE_SPEED_IN_PHASE4", True))

    # Phase 1
    if progress < p1:
        # Reduce terminal magnitudes early to lower return variance (easier value learning)
        term_scale = term_start
        Config.CRASH_PENALTY = term_scale * _MAX_CRASH_PENALTY
        Config.OUT_OF_ROAD_PENALTY = term_scale * _MAX_OUT_OF_ROAD_PENALTY
        Config.SUCCESS_REWARD = term_scale * _MAX_SUCCESS_REWARD

        Config.SPEED_REWARD_SCALE = 0.0
        Config.OVERSPEED_PENALTY_SCALE = 0.0
        Config.LANE_CENTER_PENALTY_SCALE = 0.0
        Config.HEADING_PENALTY_SCALE = 0.0
        Config.SAFETY_PENALTY_SCALE = 0.0
        Config.APPROACH_PENALTY_SCALE = 0.0
        Config.TTC_PENALTY_SCALE = 0.0
        Config.ACTION_MAG_PENALTY = 0.0
        Config.ACTION_CHANGE_PENALTY = 0.0
        Config.IDLE_PENALTY = 0.0
        Config.IDLE_LONG_PENALTY = 0.0
        return

    # Phase 2: lane/heading
    if progress < p2:
        t = (progress - p1) / max(1e-8, (p2 - p1))
        t = max(0.0, min(1.0, t))
        # Linearly restore terminal magnitudes in this phase
        term_scale = term_start + (term_end - term_start) * t
        Config.CRASH_PENALTY = term_scale * _MAX_CRASH_PENALTY
        Config.OUT_OF_ROAD_PENALTY = term_scale * _MAX_OUT_OF_ROAD_PENALTY
        Config.SUCCESS_REWARD = term_scale * _MAX_SUCCESS_REWARD

        # keep speed shaping disabled in early training
        Config.SPEED_REWARD_SCALE = 0.0
        Config.OVERSPEED_PENALTY_SCALE = 0.0
        Config.LANE_CENTER_PENALTY_SCALE = float(t) * _MAX_LANE_CENTER_PENALTY_SCALE
        Config.HEADING_PENALTY_SCALE = float(t) * _MAX_HEADING_PENALTY_SCALE
        Config.SAFETY_PENALTY_SCALE = 0.0
        Config.APPROACH_PENALTY_SCALE = 0.0
        Config.TTC_PENALTY_SCALE = 0.0
        Config.ACTION_MAG_PENALTY = 0.0
        Config.ACTION_CHANGE_PENALTY = 0.0
        Config.IDLE_PENALTY = 0.0
        Config.IDLE_LONG_PENALTY = 0.0
        return

    # Phase 3: safety/approach
    if progress < p3:
        t = (progress - p2) / max(1e-8, (p3 - p2))
        t = max(0.0, min(1.0, t))
        # Terminal magnitudes fully enabled from here
        Config.CRASH_PENALTY = _MAX_CRASH_PENALTY
        Config.OUT_OF_ROAD_PENALTY = _MAX_OUT_OF_ROAD_PENALTY
        Config.SUCCESS_REWARD = _MAX_SUCCESS_REWARD

        # optionally ramp in speed shaping slightly later (kept simple: off until final phase)
        Config.SPEED_REWARD_SCALE = 0.0
        Config.OVERSPEED_PENALTY_SCALE = 0.0
        Config.LANE_CENTER_PENALTY_SCALE = _MAX_LANE_CENTER_PENALTY_SCALE
        Config.HEADING_PENALTY_SCALE = _MAX_HEADING_PENALTY_SCALE
        Config.SAFETY_PENALTY_SCALE = float(t) * _MAX_SAFETY_PENALTY_SCALE
        Config.APPROACH_PENALTY_SCALE = float(t) * _MAX_APPROACH_PENALTY_SCALE
        Config.TTC_PENALTY_SCALE = 0.0
        Config.ACTION_MAG_PENALTY = 0.0
        Config.ACTION_CHANGE_PENALTY = 0.0
        Config.IDLE_PENALTY = 0.0
        Config.IDLE_LONG_PENALTY = 0.0
        return

    # Phase 4: TTC + comfort + idle
    Config.CRASH_PENALTY = _MAX_CRASH_PENALTY
    Config.OUT_OF_ROAD_PENALTY = _MAX_OUT_OF_ROAD_PENALTY
    Config.SUCCESS_REWARD = _MAX_SUCCESS_REWARD
    if enable_speed_phase4:
        Config.SPEED_REWARD_SCALE = _MAX_SPEED_REWARD_SCALE
        Config.OVERSPEED_PENALTY_SCALE = _MAX_OVERSPEED_PENALTY_SCALE
    else:
        Config.SPEED_REWARD_SCALE = 0.0
        Config.OVERSPEED_PENALTY_SCALE = 0.0
    Config.LANE_CENTER_PENALTY_SCALE = _MAX_LANE_CENTER_PENALTY_SCALE
    Config.HEADING_PENALTY_SCALE = _MAX_HEADING_PENALTY_SCALE
    Config.SAFETY_PENALTY_SCALE = _MAX_SAFETY_PENALTY_SCALE
    Config.APPROACH_PENALTY_SCALE = _MAX_APPROACH_PENALTY_SCALE
    Config.TTC_PENALTY_SCALE = _MAX_TTC_PENALTY_SCALE
    Config.ACTION_MAG_PENALTY = _MAX_ACTION_MAG_PENALTY
    Config.ACTION_CHANGE_PENALTY = _MAX_ACTION_CHANGE_PENALTY
    Config.IDLE_PENALTY = _MAX_IDLE_PENALTY
    Config.IDLE_LONG_PENALTY = _MAX_IDLE_LONG_PENALTY


def _vec_to_list(x, num_envs: int):
    if isinstance(x, list) or isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        # 如果是 object 类型的数组（通常是多智能体环境的返回），直接转 list
        return list(x)
    # 兼容性修改：如果 SB3 返回了未堆叠的 Dict (在某些 DummyVecEnv 版本下可能发生)
    # 我们假设它已经是我们要的格式，或者尝试拆分
    if isinstance(x, dict):
        # 只有当它是堆叠后的 Dict (key -> array of envs) 时才需要拆分
        # 但多智能体 MARL 通常是 list of dicts。
        # 这里做一个简单的处理：如果它看起来像堆叠数据，尝试拆开；否则直接报错
        sample_val = next(iter(x.values()))
        if isinstance(sample_val, (np.ndarray, list)) and len(sample_val) == num_envs:
             # 这是一个 Stacked Dict，我们需要把它转回 List of Dicts
             # (虽然你的环境因为 Key 动态变化，SB3 应该不会走到这一步，但以防万一)
             return [{k: v[i] for k, v in x.items()} for i in range(num_envs)]
        else:
             # 也许它就是单个环境的 Dict？(num_envs=1)
             if num_envs == 1:
                 return [x]
             
    # Fallback
    return [x for _ in range(int(num_envs))]

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for MARL")
    parser.add_argument("--lr", type=float, default=Config.LR, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=Config.GAMMA, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--ppo_epochs", type=int, default=Config.PPO_EPOCHS, help="PPO epochs")
    parser.add_argument("--aux_loss_coef", type=float, default=Config.AUX_LOSS_COEF, help="Aux loss coefficient")
    parser.add_argument("--exp_name", type=str, default=None, help="本次实验的名称，将决定日志保存在哪里")
    # Map switching
    parser.add_argument(
        "--map_mode",
        type=str,
        default=getattr(Config, "MAP_MODE", "block_num"),
        choices=["block_num", "block_sequence"],
        help='Map mode: "block_num" or "block_sequence"',
    )
    parser.add_argument(
        "--map_block_num",
        type=int,
        default=getattr(Config, "MAP_BLOCK_NUM", 3),
        help="Number of blocks when map_mode=block_num",
    )
    parser.add_argument(
        "--map_type",
        type=str,
        default=Config.MAP_TYPE,
        help='Block sequence string when map_mode=block_sequence (e.g., "SSSSS", "X", "r")',
    )
    
    # New args
    parser.add_argument("--aux_decay", type=int, default=1, help="Decay aux loss coef (1=True, 0=False)")
    parser.add_argument("--clip_eps", type=float, default=Config.CLIP_EPSILON, help="PPO clip epsilon")
    parser.add_argument("--grad_norm", type=float, default=Config.MAX_GRAD_NORM, help="Max gradient norm")
    parser.add_argument("--entropy_coef", type=float, default=Config.ENTROPY_COEF, help="Entropy coefficient for exploration")
    parser.add_argument("--vf_coef", type=float, default=getattr(Config, "VF_COEF", 0.5), help="Value loss coefficient")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cpu/cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_time", type=int, default=30, help="Max wall-clock seconds to train before stopping")
    parser.add_argument("--experiment_mode", type=str, default=None,
                        choices=["ours", "oracle", "no_comm", "no_aux", "lidar_only", "mappo", "mappo_ips", "where2comm", "tarmac"],
                        help="Override Config.EXPERIMENT_MODE (e.g. 'tarmac' for TarMAC baseline)")
    
    return parser.parse_args()

def update_config(args, explicit_args=None):
    """Update Config from parsed args. Only override values explicitly passed on CLI."""
    for key, value in vars(args).items():
        key_upper = key.upper()
        if hasattr(Config, key_upper):
            # Avoid clobbering defaults with None (e.g., exp_name not provided)
            if value is None:
                continue
            # If we know which args were explicitly provided, skip defaults
            if explicit_args is not None and key not in explicit_args:
                continue
            setattr(Config, key_upper, value)
            print(f"Config updated: {key_upper} = {value}")

def calculate_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    """Calculate GAE for a single agent's trajectory."""

    advantages = []
    gae = 0.0

    for i in reversed(range(len(rewards))):
        mask = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * mask - values[i]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[i]

    return torch.stack(advantages)


def _reset_with_seed(env, seed: int):
    """Reset env with a given seed.

    NOTE: In MetaDrive, reset(seed=...) expects a *scenario index* that must lie in
    [env.start_index, env.start_index + env.num_scenarios). It is NOT an arbitrary RNG seed.
    This helper maps the provided seed into the valid scenario range when possible.
    """
    start_index = getattr(env, "start_index", None)
    num_scenarios = getattr(env, "num_scenarios", None)

    # Map to valid scenario index if we can
    scenario_index = seed
    if isinstance(start_index, int) and isinstance(num_scenarios, int) and num_scenarios > 0:
        scenario_index = int(start_index + ((seed - start_index) % num_scenarios))

    try:
        return env.reset(seed=scenario_index)
    except (TypeError, AssertionError):
        # Fallback for older APIs
        try:
            env.seed(seed)
        except Exception:
            pass
        return env.reset()


def make_env(rank: int, seed: int, experiment_mode: str = "ours"):
    """Factory for creating an environment.

    IMPORTANT (Windows spawn): do NOT return a local closure here, because it is not pickleable.
    We return a top-level callable object instead.
    """
    return _EnvFactory(rank=rank, seed=seed, experiment_mode=experiment_mode)


class _EnvFactory:
    """Pickle-friendly env factory for multiprocessing spawn."""

    def __init__(self, rank: int, seed: int, experiment_mode: str = "ours"):
        self.rank = int(rank)
        self.seed = int(seed)
        self.experiment_mode = str(experiment_mode)

    def __call__(self):
        # Import inside to avoid pickling issues and heavy imports in parent.
        from marl_project.config import Config
        from marl_project.env_wrapper import GraphEnvWrapper

        # === CRITICAL: 在子进程中也应用实验模式 ===
        if hasattr(Config, 'apply_experiment_mode'):
            Config.apply_experiment_mode(self.experiment_mode)
        # =========================================

        env_cfg = Config.get_metadrive_config(is_eval=False)
        env_cfg["start_seed"] = self.seed + self.rank
        env_cfg["num_scenarios"] = int(getattr(Config, "TRAIN_NUM_SCENARIOS", 10000))
        env_cfg["use_render"] = False
        env = GraphEnvWrapper(config=env_cfg)
        try:
            env.seed(self.seed + self.rank)
        except Exception:
            pass
        return env

def train():
    # --- Setup ---
    # 统一缩进：建议每一级缩进都严格使用 4 个空格
    torch.manual_seed(42)
    np.random.seed(42)  # <--- 之前报错的就是这里
    
    args = parse_args()
    
    # === [新增] 应用实验模式配置 ===
    # CLI --experiment_mode 优先于 Config.EXPERIMENT_MODE
    if args.experiment_mode:
        Config.EXPERIMENT_MODE = args.experiment_mode
    experiment_mode = getattr(Config, "EXPERIMENT_MODE", "ours")
    if hasattr(Config, 'apply_experiment_mode'):
        Config.apply_experiment_mode(experiment_mode)
    else:
        print(f"⚠️ 警告: Config没有apply_experiment_mode方法，使用默认配置")
    # =========================================
    
    # 1. 确定实验名称
    # 优先级：命令行参数 > Config默认值
    # 注意：确保你的 Config 类里加了 EXP_NAME = "default" 之类的默认值
    exp_name = args.exp_name if args.exp_name else getattr(Config, "EXP_NAME", "default_experiment")
    
    # 2. 动态拼接路径
    # 注意：确保你的 Config 类里加了 LOG_ROOT = "logs/marl_experiment"
    log_root = getattr(Config, "LOG_ROOT", "logs/marl_experiment")
    log_dir = os.path.join(log_root, exp_name)
    
    # 3. (可选但推荐) 防止手滑覆盖重要实验
    if os.path.exists(log_dir) and os.listdir(log_dir):
        print(f"⚠️ 警告: 日志目录 {log_dir} 已存在且非空！")
        # raise ValueError("请更换实验名称或删除旧文件夹！") # 如果你想强制报错，取消注释这一行

    os.makedirs(log_dir, exist_ok=True)
    
    print(f"📂 本次训练日志将保存在: {log_dir}")
    
    # 更新配置（仅更新 CLI 显式传入的参数，避免覆盖 apply_experiment_mode 的设置）
    # Detect which args were explicitly provided on CLI (not just defaults)
    import shlex
    _explicit = set()
    for i, tok in enumerate(sys.argv[1:]):
        if tok.startswith("--"):
            _explicit.add(tok.lstrip("-").replace("-", "_"))
    update_config(args, explicit_args=_explicit)
    
    # Save config for reproducibility
    # 确保能找到 config.py 文件，如果你的 train.py 和 config.py 在同一级目录：
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    if os.path.exists(config_path):
        import shutil # 确保导入了 shutil
        shutil.copy(config_path, log_dir)
    else:
        print(f"⚠️ 警告: 找不到 config.py ({config_path})，跳过备份。")
        
    # Dump final config to hparams.json
    hparams = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=4, default=str)

    writer = SummaryWriter(log_dir)

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- Parallel envs (方案3: 自定义多进程采样器) ---
    num_envs = int(getattr(Config, "NUM_ENVS", 1))
    base_seed = int(getattr(Config, "BASE_SEED", 5000))
    # 传递实验模式到子进程
    env_fns = [make_env(i, base_seed, experiment_mode=experiment_mode) for i in range(num_envs)]

    print(f"Using MultiProcEnv with {num_envs} workers (spawn)")
    envs = MultiProcEnv(env_fns)

    # Prepare a template action space for clipping (best-effort)
    env_space = getattr(envs, "action_space", None)
    template_space = None
    if hasattr(env_space, "spaces") and len(getattr(env_space, "spaces", {})) > 0:
        for key in env_space.spaces:
            candidate = env_space.spaces[key]
            if hasattr(candidate, "low") and hasattr(candidate, "high"):
                template_space = candidate
                break
            if hasattr(candidate, "spaces"):
                for sub_key in candidate.spaces:
                    sub = candidate.spaces[sub_key]
                    if hasattr(sub, "low") and hasattr(sub, "high"):
                        template_space = sub
                        break
                if template_space is not None:
                    break
    elif hasattr(env_space, "low") and hasattr(env_space, "high"):
        template_space = env_space

    def clip_action(act: np.ndarray):
        if template_space is None:
            return act
        return np.clip(act, template_space.low, template_space.high)

    # Get dimensions from a dummy reset
    obs0_list = envs.reset(seeds=[base_seed + i for i in range(num_envs)])
    first_obs = obs0_list[0]
    sample_agent = list(first_obs.keys())[0]
    input_dim = first_obs[sample_agent]['node_features'].shape[0]
    # best-effort: infer from env_space (may be Dict)
    action_dim = None
    if hasattr(env_space, "spaces") and sample_agent in getattr(env_space, "spaces", {}):
        action_dim = env_space.spaces[sample_agent].shape[0]
    elif template_space is not None and hasattr(template_space, "shape"):
        action_dim = int(template_space.shape[0])
    else:
        # Fallback to 2-dim continuous actions (steer, throttle/brake) if not inferable
        action_dim = 2

    # 检测是否为MAPPO模式
    is_mappo = experiment_mode in {"mappo", "mappo_ips"}
    if is_mappo:
        policy = MAPPOPolicy(
            input_dim=input_dim,
            action_dim=action_dim,
            num_agents=Config.NUM_AGENTS
        ).to(device)
        print(f"\n🎯 使用MAPPO策略 (Centralized Critic)")
    else:
        policy = CooperativePolicy(input_dim, action_dim).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=Config.LR)
    
    print(f"\n{'='*70}")
    print(f"📊 模型初始化")
    print(f"{'='*70}")
    print(f"实验模式: {experiment_mode}")
    print(f"LIDAR_NUM_OTHERS: {Config.LIDAR_NUM_OTHERS}")
    print(f"环境输出维度: {input_dim}")
    print(f"动作维度: {action_dim}")
    print(f"预期维度 (oracle/lidar_only): 107, 预期维度 (ours/no_comm/no_aux): 91")
    if experiment_mode in ["oracle", "lidar_only"] and input_dim != 107:
        print(f"\n⚠️  警告: {experiment_mode}模式应该是107维，但环境输出{input_dim}维！")
        print(f"   请检查Config.apply_experiment_mode()是否正确执行")
    elif experiment_mode in ["ours", "no_comm", "no_aux", "mappo", "mappo_ips", "where2comm", "tarmac"] and input_dim != 91:
        print(f"\n⚠️  警告: {experiment_mode}模式应该是91维，但环境输出{input_dim}维！")
    print(f"{'='*70}\n")
    
    # --- Training Parameters ---
    num_updates = 3000
    steps_per_update = Config.N_STEPS
    gamma = Config.GAMMA
    gae_lambda = 0.95
    clip_epsilon = args.clip_eps
    max_grad_norm = args.grad_norm
    entropy_coef = args.entropy_coef
    vf_coef = args.vf_coef
    value_clip = getattr(Config, 'VALUE_CLIP', 10.0)  # Value clipping to prevent explosion
    
    # Checkpointing
    best_reward = -float('inf')
    best_success_rate = -float('inf')
    saved_checkpoints = deque(maxlen=3)

    # --- Early stopping (optional) ---
    early_stop_enabled = bool(getattr(Config, "EARLY_STOP_ENABLED", False))
    early_stop_metric = str(getattr(Config, "EARLY_STOP_METRIC", "mean_episode_reward"))
    early_stop_mode = str(getattr(Config, "EARLY_STOP_MODE", "max")).lower()
    early_stop_window = int(getattr(Config, "EARLY_STOP_WINDOW_UPDATES", 5))
    early_stop_warmup = int(getattr(Config, "EARLY_STOP_WARMUP_UPDATES", 30))
    early_stop_patience = int(getattr(Config, "EARLY_STOP_PATIENCE_UPDATES", 30))
    early_stop_min_delta = float(getattr(Config, "EARLY_STOP_MIN_DELTA", 0.0))

    if early_stop_window <= 0:
        early_stop_window = 1
    if early_stop_patience <= 0:
        early_stop_patience = 1
    if early_stop_warmup < 0:
        early_stop_warmup = 0

    early_stop_hist = deque(maxlen=early_stop_window)
    early_stop_best = None
    early_stop_bad = 0
    
    # --- Main Loop ---
    start_update = 1
    global_step = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            # Use weights_only to avoid pickle execution (PyTorch future default)
            state_dict = torch.load(args.resume, map_location=device, weights_only=True)

            # Drop aux_head if shape mismatches (e.g., PRED_WAYPOINTS_NUM changed)
            def _filter_mismatched(sd, model):
                filtered = {}
                model_sd = model.state_dict()
                for k, v in sd.items():
                    if k in model_sd and model_sd[k].shape != v.shape:
                        print(f"[Resume] Drop mismatched key {k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}")
                        continue
                    filtered[k] = v
                return filtered

            state_dict = _filter_mismatched(state_dict, policy)
            missing, unexpected = policy.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"[Resume] Loaded with relaxed strictness. Missing: {missing}, Unexpected: {unexpected}")
            
            # Try to parse update number from filename
            import re
            match = re.search(r"ckpt_(\d+).pth", args.resume)
            if match:
                start_update = int(match.group(1)) + 1
                global_step = (start_update - 1) * steps_per_update
                print(f"Resuming from update {start_update}, global_step {global_step}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found. Starting from scratch.")
    
    wall_start = time.time()
    print(f"\n🚀 开始训练循环: num_updates={num_updates}, steps_per_update={steps_per_update}")
    print(f"   NUM_ENVS={num_envs}, 每个update总样本={num_envs * steps_per_update * Config.NUM_AGENTS}")
    sys.stdout.flush()
    try:  # Global try/except to catch ANY crash and print traceback
      for update in range(start_update, num_updates + 1):
        if args.max_time and (time.time() - wall_start) >= args.max_time:
            print("Max wall-clock time reached, stopping.")
            break

        # Curriculum learning update (based on training progress)
        update_curriculum(current_step=update - 1, total_steps=num_updates)
        
        # === 同步 Curriculum 参数到所有 worker ===
        curriculum_params = {
            "CRASH_PENALTY": getattr(Config, "CRASH_PENALTY", -200.0),
            "OUT_OF_ROAD_PENALTY": getattr(Config, "OUT_OF_ROAD_PENALTY", -200.0),
            "SUCCESS_REWARD": getattr(Config, "SUCCESS_REWARD", 300.0),
            "TTC_PENALTY_SCALE": Config.TTC_PENALTY_SCALE,
            "ACTION_MAG_PENALTY": Config.ACTION_MAG_PENALTY,
            "ACTION_CHANGE_PENALTY": Config.ACTION_CHANGE_PENALTY,
            "LANE_CENTER_PENALTY_SCALE": getattr(Config, "LANE_CENTER_PENALTY_SCALE", 0.0),
            "HEADING_PENALTY_SCALE": getattr(Config, "HEADING_PENALTY_SCALE", 0.0),
            "SAFETY_PENALTY_SCALE": getattr(Config, "SAFETY_PENALTY_SCALE", 0.0),
            "APPROACH_PENALTY_SCALE": getattr(Config, "APPROACH_PENALTY_SCALE", 0.0),
            "IDLE_PENALTY": getattr(Config, "IDLE_PENALTY", 0.0),
            "IDLE_LONG_PENALTY": getattr(Config, "IDLE_LONG_PENALTY", 0.0),
            "SPEED_REWARD_SCALE": getattr(Config, "SPEED_REWARD_SCALE", 0.0),
            "OVERSPEED_PENALTY_SCALE": getattr(Config, "OVERSPEED_PENALTY_SCALE", 0.0),
        }
        try:
            envs.env_method("update_config_params", curriculum_params)
        except Exception as e:
            print(f"\n❌ [Update {update}] env_method('update_config_params') failed: {e}")
            import traceback; traceback.print_exc()
            break
        # ==========================================
        
        # --- 1. Rollout Phase ---
        buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "agent_ids": [],
            "old_values": [],
            "old_logprobs": [],
            "aux_preds": [],
            "gt_waypoints": []
        }
        if is_mappo:
            buffer["global_obs"] = []
        
        # Reward Tracking
        completed_episode_rewards = []
        episode_rewards = {}
        reward_components = defaultdict(float)
        reward_steps = 0
        event_counts = defaultdict(int)
        risk_stats = defaultdict(float)
        graph_stats = defaultdict(float)
        
        # Parallel reset: keep list[dict] obs for existing rollout logic
        # diversify scenario indices deterministically within each worker's scenario range
        num_scenarios = int(getattr(Config, "TRAIN_NUM_SCENARIOS", 10000))
        reset_seeds = [int(base_seed + i + ((update - 1) % max(1, num_scenarios))) for i in range(num_envs)]
        try:
            if update == start_update:
                print(f"   [Update {update}] Resetting envs with seeds {reset_seeds[:2]}...")
                sys.stdout.flush()
            obs_list = envs.reset(seeds=reset_seeds)
            if update == start_update:
                print(f"   [Update {update}] Reset OK, agents per env: {[len(o) for o in obs_list]}")
                sys.stdout.flush()
        except Exception as e:
            print(f"\n❌ [Update {update}] envs.reset() failed: {e}")
            import traceback; traceback.print_exc()
            break
        episode_rewards = [{a_id: 0.0 for a_id in obs} for obs in obs_list]
        
        for step in range(steps_per_update):
            # Progress logging (first update only, every 200 steps)
            if update == start_update and step % 200 == 0:
                print(f"   [Update {update}] Rollout step {step}/{steps_per_update}")
                sys.stdout.flush()
            # Extra diagnostic: first 3 steps of first update
            _diag = (update == start_update and step < 3)
            global_step += 1
            
            # Prepare batch for policy
            active_agents = []
            env_agent_counts = []
            for env_idx, obs_dict in enumerate(obs_list):
                env_agent_counts.append(len(obs_dict))
                active_agents.extend([f"{env_idx}:{aid}" for aid in obs_dict.keys()])
            agent_to_idx = {a_id: i for i, a_id in enumerate(active_agents)}
            
            batch_node_features = []
            batch_neighbor_indices = []
            batch_neighbor_mask = []
            batch_neighbor_rel_pos = []
            batch_gt_waypoints = []
            
            for env_idx, obs_dict in enumerate(obs_list):
                for agent_id, agent_obs in obs_dict.items():
                    batch_node_features.append(agent_obs['node_features'])
                    batch_gt_waypoints.append(agent_obs['gt_waypoints'])

                    n_indices = []
                    n_mask = []
                    n_rel_pos = []

                    raw_neighbors = agent_obs['neighbors']
                    raw_rel_pos = agent_obs['neighbor_rel_pos']

                    for i, n_id in enumerate(raw_neighbors):
                        # Neighbor ids are per-env. We must namespace them by env_idx to map into the
                        # flattened (env, agent) batch index space.
                        neighbor_key = f"{env_idx}:{n_id}"
                        if neighbor_key in agent_to_idx:
                            n_indices.append(agent_to_idx[neighbor_key])
                            n_mask.append(1.0)
                            n_rel_pos.append(raw_rel_pos[i])

                    while len(n_indices) < Config.MAX_NEIGHBORS:
                        n_indices.append(0)
                        n_mask.append(0.0)
                        n_rel_pos.append([0.0, 0.0, 0.0, 0.0])

                    n_indices = n_indices[:Config.MAX_NEIGHBORS]
                    n_mask = n_mask[:Config.MAX_NEIGHBORS]
                    n_rel_pos = n_rel_pos[:Config.MAX_NEIGHBORS]

                    batch_neighbor_indices.append(n_indices)
                    batch_neighbor_mask.append(n_mask)
                    batch_neighbor_rel_pos.append(n_rel_pos)

            # If no active agents (e.g., all done), skip this step safely
            if len(batch_node_features) == 0:
                continue
            
            # Convert to Tensor and move to device
            obs_tensor_batch = {
                "node_features": torch.tensor(np.array(batch_node_features), dtype=torch.float32),
                "neighbor_indices": torch.tensor(np.array(batch_neighbor_indices), dtype=torch.long),
                "neighbor_mask": torch.tensor(np.array(batch_neighbor_mask), dtype=torch.float32),
                "neighbor_rel_pos": torch.tensor(np.array(batch_neighbor_rel_pos), dtype=torch.float32),
                "gt_waypoints": torch.tensor(np.array(batch_gt_waypoints), dtype=torch.float32)
            }
            if _diag:
                print(f"      [DIAG step {step}] Batch: {obs_tensor_batch['node_features'].shape}"); sys.stdout.flush()

            # Graph connectivity stats (how many valid neighbors per ego)
            try:
                nmask_np = np.asarray(batch_neighbor_mask, dtype=np.float32)
                valid_n = float(np.sum(nmask_np, axis=1).mean()) if nmask_np.size > 0 else 0.0
                graph_stats["neighbors_mean_sum"] += valid_n
                graph_stats["neighbors_n"] += 1.0
            except Exception:
                pass
            
            # Inference (using forward_actor_critic for no_grad)
            if is_mappo:
                # 为MAPPO构建全局观测
                # 需要按环境分组，因为每个环境的agent数量可能不同
                global_obs_list = []
                cursor = 0
                for count in env_agent_counts:
                    env_features = obs_tensor_batch["node_features"][cursor:cursor+count]  # (N, 91)
                    # 确保每个环境固定为NUM_AGENTS个agent的观测
                    if count < Config.NUM_AGENTS:
                        # 不足的用0填充
                        pad_size = Config.NUM_AGENTS - count
                        padding = torch.zeros(pad_size, env_features.shape[1], dtype=env_features.dtype)
                        env_features = torch.cat([env_features, padding], dim=0)
                    elif count > Config.NUM_AGENTS:
                        # 超出的截断（只取前NUM_AGENTS个）
                        env_features = env_features[:Config.NUM_AGENTS]
                    # Flatten为全局观测 (NUM_AGENTS * 91,)
                    global_obs_list.append(env_features.flatten())
                    cursor += count
                
                global_obs = torch.stack(global_obs_list, dim=0)  # (num_envs, NUM_AGENTS*91)
                
                results = policy.forward_actor_critic(
                    {k: v.to(device, non_blocking=True) for k, v in obs_tensor_batch.items()},
                    global_obs=global_obs.to(device, non_blocking=True)
                )
            else:
                results = policy.forward_actor_critic({k: v.to(device, non_blocking=True) for k, v in obs_tensor_batch.items()})
            if _diag:
                print(f"      [DIAG step {step}] Forward OK"); sys.stdout.flush()
                
            # Sample Actions
            action_mean = results["action_mean"]
            action_std = results["action_std"]
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Value处理：MAPPO输出(num_envs, NUM_AGENTS)，需要提取实际agent的value
            if is_mappo:
                # 从(num_envs, NUM_AGENTS)中提取实际agent的value
                # 同时截断actions和log_probs以匹配
                values_list = []
                actions_list = []
                log_probs_list = []
                cursor = 0
                for env_idx, count in enumerate(env_agent_counts):
                    actual_count = min(count, Config.NUM_AGENTS)
                    # Value: 从(NUM_AGENTS,)中取实际数量
                    env_values = results["value"][env_idx, :actual_count]
                    values_list.append(env_values)
                    # Actions和log_probs: 从总的actions中提取这个环境的
                    actions_list.append(actions[cursor:cursor+actual_count])
                    log_probs_list.append(action_log_probs[cursor:cursor+actual_count])
                    cursor += count
                
                values = torch.cat(values_list, dim=0)  # (sum(actual_counts),)
                # 重新组装actions和log_probs，只保留实际使用的
                actions_actual = torch.cat(actions_list, dim=0)
                action_log_probs_actual = torch.cat(log_probs_list, dim=0)
            else:
                values = results["value"].squeeze(-1)  # (B*N, 1) -> (B*N,)
                actions_actual = actions
                action_log_probs_actual = action_log_probs
            
            aux_preds = results["pred_waypoints"]
            
            # Execute Step
            env_actions = []
            cursor = 0
            for env_idx, obs_dict in enumerate(obs_list):
                count = env_agent_counts[env_idx]
                env_action_dict = {}
                if is_mappo:
                    # MAPPO模式：只使用前NUM_AGENTS个agent的动作
                    actual_count = min(count, Config.NUM_AGENTS)
                    agent_ids = list(obs_dict.keys())[:actual_count]
                    for j, aid in enumerate(agent_ids):
                        # cursor现在指向actions_actual中的位置
                        actual_cursor = sum([min(env_agent_counts[i], Config.NUM_AGENTS) for i in range(env_idx)]) + j
                        raw_act = actions_actual[actual_cursor].cpu().numpy()
                        raw_act = clip_action(raw_act)
                        env_action_dict[aid] = raw_act
                    # 其余agent用0动作（保持静止）
                    for aid in list(obs_dict.keys())[actual_count:]:
                        env_action_dict[aid] = clip_action(np.zeros(action_dim))
                else:
                    for j, aid in enumerate(obs_dict.keys()):
                        raw_act = actions_actual[cursor + j].cpu().numpy()
                        raw_act = clip_action(raw_act)
                        env_action_dict[aid] = raw_act
                env_actions.append(env_action_dict)
                cursor += count

            # Parallel step
            if _diag:
                print(f"      [DIAG step {step}] Before envs.step()"); sys.stdout.flush()
            try:
                next_obs_list, rewards_list, dones_list, infos_list = envs.step(env_actions)
            except Exception as e:
                print(f"\n❌ [Update {update}, Step {step}] envs.step() failed: {e}")
                import traceback; traceback.print_exc()
                # Re-raise to break out of rollout loop
                raise

            # Track Rewards
            for env_idx, obs_dict in enumerate(obs_list):
                rewards = rewards_list[env_idx]
                dones = dones_list[env_idx]
                infos = infos_list[env_idx]
                for a_id in obs_dict.keys():
                    key = f"{env_idx}:{a_id}"
                    episode_rewards[env_idx][key] = episode_rewards[env_idx].get(key, 0.0) + rewards.get(a_id, 0.0)
                    if dones.get(a_id, False):
                        completed_episode_rewards.append(episode_rewards[env_idx][key])
                        episode_rewards[env_idx][key] = 0.0

                    info = infos.get(a_id, {}) if isinstance(infos, dict) else {}
                    breakdown = info.get("reward_breakdown") if isinstance(info, dict) else None
                    if breakdown:
                        for k, v in breakdown.items():
                            try:
                                reward_components[k] += float(v)
                            except Exception:
                                continue
                        reward_steps += 1

                    # Per-step risk stats (min_ttc/min_dist/idle_count)
                    risk_stats["step_agents"] += 1.0
                    min_ttc = info.get("min_ttc_s")
                    if min_ttc is not None:
                        risk_stats["steps_with_ttc"] += 1.0
                        risk_stats["min_ttc_sum"] += float(min_ttc)
                        ttc_thr = float(getattr(Config, "TTC_THRESHOLD_S", 2.5))
                        if float(min_ttc) < ttc_thr:
                            risk_stats["risk_ttc_steps"] += 1.0
                    min_dist = info.get("min_neighbor_dist")
                    if min_dist is not None:
                        risk_stats["steps_with_dist"] += 1.0
                        risk_stats["min_dist_sum"] += float(min_dist)
                        safety_dist = float(getattr(Config, "SAFETY_DIST", 8.0))
                        if float(min_dist) < safety_dist:
                            risk_stats["risk_dist_steps"] += 1.0
                    idle_count = info.get("idle_count")
                    if idle_count is not None:
                        risk_stats["idle_count_sum"] += float(idle_count)
                        risk_stats["idle_count_n"] += 1.0

                    # Terminal event proportions (count only when done)
                    if dones.get(a_id, False):
                        event_counts["terminal_total"] += 1
                        term = info.get("terminal_reason")
                        if term is None:
                            # Fallback if wrapper is not used
                            if info.get("arrive_dest", False):
                                term = "success"
                            elif info.get("crash", False):
                                term = "crash"
                            elif info.get("out_of_road", False):
                                term = "out_of_road"
                            elif info.get("truncated", False):
                                term = "timeout"
                            else:
                                term = "other"
                        event_counts[term] += 1
                        ev = info.get("event", {})
                        if isinstance(ev, dict) and ev.get("idle_long", False):
                            event_counts["idle_long"] += 1

            # Store in buffer
            # 对于MAPPO，需要截断obs以匹配实际使用的agent
            if is_mappo:
                obs_truncated = {}
                cursor = 0
                for key in obs_tensor_batch:
                    obs_parts = []
                    cursor_inner = 0
                    for env_idx, count in enumerate(env_agent_counts):
                        actual_count = min(count, Config.NUM_AGENTS)
                        obs_parts.append(obs_tensor_batch[key][cursor_inner:cursor_inner+actual_count])
                        cursor_inner += count
                    obs_truncated[key] = torch.cat(obs_parts, dim=0)
                buffer["obs"].append({k: v.cpu() for k, v in obs_truncated.items()})
            else:
                buffer["obs"].append({k: v.cpu() for k, v in obs_tensor_batch.items()})
            
            # 使用实际的actions和log_probs（MAPPO模式下已经截断）
            buffer["actions"].append(actions_actual.detach().cpu())
            
            # MAPPO需要特殊处理：如果某些环境的agent被截断了，需要同步active_agents
            if is_mappo:
                # 重新构建active_agents，只包含被使用的agent（截断到NUM_AGENTS）
                actual_active_agents = []
                for env_idx, count in enumerate(env_agent_counts):
                    # 对于MAPPO，每个环境最多取NUM_AGENTS个agent
                    actual_count = min(count, Config.NUM_AGENTS)
                    obs_dict = obs_list[env_idx]
                    agent_ids = list(obs_dict.keys())[:actual_count]
                    actual_active_agents.extend([f"{env_idx}:{aid}" for aid in agent_ids])
                buffer["agent_ids"].append(actual_active_agents)
            else:
                buffer["agent_ids"].append(active_agents)
            
            # Store detached tensors for PPO update
            buffer["old_values"].append(values.detach().cpu())
            buffer["old_logprobs"].append(action_log_probs_actual.detach().cpu())
            
            # MAPPO需要存储global_obs - 优化：只存储per-env，不存储per-agent（6倍显存节省）
            if is_mappo:
                # 只存储每个环境的global_obs（不复制到每个agent）
                # global_obs: (num_envs, 546)
                buffer["global_obs"].append(global_obs.detach().cpu())
                
                # 同时存储每个环境的agent数量，用于后续重建per-agent数据
                if "env_agent_counts" not in buffer:
                    buffer["env_agent_counts"] = []
                buffer["env_agent_counts"].append(env_agent_counts.copy())
            
            # aux_preds和gt_waypoints也需要截断以匹配实际使用的agent数量
            if is_mappo:
                # 截断aux_preds和gt_waypoints
                aux_preds_list = []
                gt_waypoints_list = []
                cursor = 0
                for env_idx, count in enumerate(env_agent_counts):
                    actual_count = min(count, Config.NUM_AGENTS)
                    aux_preds_list.append(aux_preds[cursor:cursor+actual_count])
                    gt_waypoints_list.append(obs_tensor_batch["gt_waypoints"][cursor:cursor+actual_count])
                    cursor += count
                buffer["aux_preds"].append(torch.cat(aux_preds_list, dim=0).detach().cpu())
                buffer["gt_waypoints"].append(torch.cat(gt_waypoints_list, dim=0).detach())
            else:
                buffer["aux_preds"].append(aux_preds.detach().cpu())
                buffer["gt_waypoints"].append(obs_tensor_batch["gt_waypoints"].detach())
            
            # Rewards and Dones
            flat_rewards = []
            flat_dones = []
            if is_mappo:
                # MAPPO模式：只收集实际使用的agent的reward和done
                for env_idx, obs_dict in enumerate(obs_list):
                    rewards = rewards_list[env_idx]
                    dones = dones_list[env_idx]
                    agent_ids = list(obs_dict.keys())[:min(len(obs_dict), Config.NUM_AGENTS)]
                    for a_id in agent_ids:
                        flat_rewards.append(rewards.get(a_id, 0.0))
                        flat_dones.append(float(dones.get(a_id, False)))
            else:
                # 原有逻辑
                for env_idx, obs_dict in enumerate(obs_list):
                    rewards = rewards_list[env_idx]
                    dones = dones_list[env_idx]
                    for a_id in obs_dict.keys():
                        flat_rewards.append(rewards.get(a_id, 0.0))
                        flat_dones.append(float(dones.get(a_id, False)))

            buffer["rewards"].append(torch.tensor(flat_rewards, dtype=torch.float32))
            buffer["dones"].append(torch.tensor(flat_dones, dtype=torch.float32))

            obs_list = next_obs_list

        # --- 2. Process Buffer (Correct GAE per Agent) ---
        
        # Flatten lists for indexing
        flat_values = torch.cat(buffer["old_values"])
        flat_rewards = torch.cat(buffer["rewards"])
        flat_dones = torch.cat(buffer["dones"])
        
        # Reconstruct trajectories by agent_id
        # We need to map the flat index back to the agent_id to group them
        
        # Create a mapping from flat_index -> agent_id
        flat_agent_ids = []
        for agents in buffer["agent_ids"]:
            flat_agent_ids.extend(agents)
            
        # Group indices by agent_id
        agent_indices = defaultdict(list)
        for idx, agent_id in enumerate(flat_agent_ids):
            agent_indices[agent_id].append(idx)
            
        # Calculate Advantages per agent trajectory
        # Initialize flat_advantages and flat_returns with zeros
        flat_advantages = torch.zeros_like(flat_rewards)
        flat_returns = torch.zeros_like(flat_rewards)
        
        for agent_id, indices in agent_indices.items():
            # Slice data for this agent
            a_rewards = flat_rewards[indices]
            a_dones = flat_dones[indices]
            a_values = flat_values[indices]
            
            # Compute GAE
            # We assume the last next_value is 0 (terminal) or we could bootstrap if we had next_obs
            # For simplicity in this refactor, we assume 0.
            next_val = 0.0
            
            # Use the helper function
            gae_tensor = calculate_gae(a_rewards, a_dones, a_values, next_val, gamma, gae_lambda)
            
            # Store back to flat arrays
            # gae_tensor is on same device as rewards
            flat_advantages[indices] = gae_tensor
            flat_returns[indices] = gae_tensor + a_values
        
        # Normalize Advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        
        # --- 3. Update Phase ---
        
        flat_actions = torch.cat(buffer["actions"])
        flat_old_logprobs = torch.cat(buffer["old_logprobs"])
        flat_old_values = torch.cat(buffer["old_values"])

        flat_obs = {
            "node_features": torch.cat([step_obs["node_features"] for step_obs in buffer["obs"]]),
            "neighbor_indices": torch.cat([step_obs["neighbor_indices"] for step_obs in buffer["obs"]]),
            "neighbor_mask": torch.cat([step_obs["neighbor_mask"] for step_obs in buffer["obs"]]),
            "neighbor_rel_pos": torch.cat([step_obs["neighbor_rel_pos"] for step_obs in buffer["obs"]]),
            "gt_waypoints": torch.cat(buffer["gt_waypoints"]),
        }
        
        # Aux Loss Scheduler
        if args.aux_decay:
            current_aux_coef = max(0.1, Config.AUX_LOSS_COEF * (1.0 - (update - 1) / num_updates))
        else:
            current_aux_coef = Config.AUX_LOSS_COEF
        
        total_ppo_loss = 0
        total_value_loss = 0
        total_aux_loss = 0
        total_entropy = 0
        total_loss = 0
        num_minibatches = 0
        
        dataset = TensorDataset(
            flat_obs["node_features"].pin_memory(),
            flat_obs["neighbor_indices"].pin_memory(),
            flat_obs["neighbor_mask"].pin_memory(),
            flat_obs["neighbor_rel_pos"].pin_memory(),
            flat_obs["gt_waypoints"].pin_memory(),
            flat_actions.pin_memory(),
            flat_old_logprobs.pin_memory(),
            flat_old_values.pin_memory(),
            flat_returns.pin_memory(),
            flat_advantages.pin_memory()
        )
        # 准备MAPPO的global_obs数据（高效扩展）
        if is_mappo:
            # Step 1: 拼接所有timesteps的global_obs
            all_global_obs = torch.cat(buffer["global_obs"], dim=0).float()  # (num_envs_total, 546)
            
            # Step 2: 获取所有timesteps的agent数量信息
            all_env_counts = [c for step_counts in buffer["env_agent_counts"] for c in step_counts]
            
            # Step 3: 高效扩展到per-agent（只在需要时复制）
            flat_global_obs_list = []
            env_idx = 0
            for count in all_env_counts:
                actual_count = min(count, Config.NUM_AGENTS)
                # 复制该环境的global_obs actual_count次
                env_global = all_global_obs[env_idx].unsqueeze(0).expand(actual_count, -1)
                flat_global_obs_list.append(env_global)
                env_idx += 1
            flat_global_obs = torch.cat(flat_global_obs_list, dim=0)  # (total_agents, 546)
            
            dataset = TensorDataset(
                flat_obs["node_features"].pin_memory(),
                flat_obs["neighbor_indices"].pin_memory(),
                flat_obs["neighbor_mask"].pin_memory(),
                flat_obs["neighbor_rel_pos"].pin_memory(),
                flat_obs["gt_waypoints"].pin_memory(),
                flat_global_obs.pin_memory(),  # 添加global_obs
                flat_actions.pin_memory(),
                flat_old_logprobs.pin_memory(),
                flat_old_values.pin_memory(),
                flat_returns.pin_memory(),
                flat_advantages.pin_memory()
            )
        
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

        for _ in range(Config.PPO_EPOCHS):
            if is_mappo:
                for (b_nodes, b_nidx, b_nmask, b_nrel, b_gtw, b_global_obs, b_actions, b_old_logprobs,
                     b_old_values, b_returns, b_advantages) in loader:
                    obs_batch = {
                        "node_features": b_nodes.to(device, non_blocking=True),
                        "neighbor_indices": b_nidx.to(device, non_blocking=True),
                        "neighbor_mask": b_nmask.to(device, non_blocking=True),
                        "neighbor_rel_pos": b_nrel.to(device, non_blocking=True),
                        "gt_waypoints": b_gtw.to(device, non_blocking=True),
                    }
                    global_obs_batch = b_global_obs.to(device, non_blocking=True)

                    results = policy(obs_batch, global_obs=global_obs_batch, action=b_actions.to(device, non_blocking=True))
                    new_logprobs = results["action_log_probs"]
                    dist_entropy = results["dist_entropy"]
                    # MAPPO的value已经是(B_agents,)形状，直接使用
                    new_values = results["value"]
                    aux_loss = results.get("aux_loss", torch.tensor(0.0, device=device))

                    ratio = torch.exp(new_logprobs - b_old_logprobs.to(device, non_blocking=True))
                    surr1 = ratio * b_advantages.to(device, non_blocking=True)
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages.to(device, non_blocking=True)
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    b_returns_gpu = b_returns.to(device, non_blocking=True)
                    b_old_values_gpu = b_old_values.to(device, non_blocking=True)
                    # Clip value predictions to prevent explosion
                    new_values = torch.clamp(new_values, -value_clip, value_clip)
                    v_clipped = b_old_values_gpu + torch.clamp(new_values - b_old_values_gpu, -clip_epsilon, clip_epsilon)
                    v_loss_unclipped = (new_values - b_returns_gpu) ** 2
                    v_loss_clipped = (v_clipped - b_returns_gpu) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = -dist_entropy.mean() * entropy_coef
                    step_loss = ppo_loss + vf_coef * value_loss + entropy_loss + current_aux_coef * aux_loss

                    optimizer.zero_grad()
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

                    total_ppo_loss += ppo_loss.item()
                    total_value_loss += value_loss.item()
                    total_aux_loss += aux_loss.item()
                    total_entropy += dist_entropy.mean().item()
                    total_loss += step_loss.item()
                    num_minibatches += 1
            else:
                for (b_nodes, b_nidx, b_nmask, b_nrel, b_gtw, b_actions, b_old_logprobs,
                     b_old_values, b_returns, b_advantages) in loader:
                    obs_batch = {
                        "node_features": b_nodes.to(device, non_blocking=True),
                        "neighbor_indices": b_nidx.to(device, non_blocking=True),
                        "neighbor_mask": b_nmask.to(device, non_blocking=True),
                        "neighbor_rel_pos": b_nrel.to(device, non_blocking=True),
                        "gt_waypoints": b_gtw.to(device, non_blocking=True),
                    }

                    results = policy(obs_batch, action=b_actions.to(device, non_blocking=True))
                    new_logprobs = results["action_log_probs"].squeeze()
                    dist_entropy = results["dist_entropy"].squeeze()
                    new_values = results["value"].squeeze()
                    aux_loss = results.get("aux_loss", torch.tensor(0.0, device=device))

                    ratio = torch.exp(new_logprobs - b_old_logprobs.to(device, non_blocking=True))
                    surr1 = ratio * b_advantages.to(device, non_blocking=True)
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * b_advantages.to(device, non_blocking=True)
                    ppo_loss = -torch.min(surr1, surr2).mean()

                    b_returns_gpu = b_returns.to(device, non_blocking=True)
                    b_old_values_gpu = b_old_values.to(device, non_blocking=True)
                    # Clip value predictions to prevent explosion
                    new_values = torch.clamp(new_values, -value_clip, value_clip)
                    v_clipped = b_old_values_gpu + torch.clamp(new_values - b_old_values_gpu, -clip_epsilon, clip_epsilon)
                    v_loss_unclipped = (new_values - b_returns_gpu) ** 2
                    v_loss_clipped = (v_clipped - b_returns_gpu) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = -dist_entropy.mean() * entropy_coef
                    step_loss = ppo_loss + vf_coef * value_loss + entropy_loss + current_aux_coef * aux_loss

                    optimizer.zero_grad()
                    step_loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    optimizer.step()

                    total_ppo_loss += ppo_loss.item()
                    total_value_loss += value_loss.item()
                    total_aux_loss += aux_loss.item()
                    total_entropy += dist_entropy.mean().item()
                    total_loss += step_loss.item()
                    num_minibatches += 1
            
        # --- Logging ---
            avg_divisor = max(1, num_minibatches)
            avg_loss = total_loss / avg_divisor
        
        mean_step_reward = flat_rewards.mean().item()
        writer.add_scalar("Reward/Mean_Step", mean_step_reward, global_step)
        denom = max(1, reward_steps)
        # Full reward breakdown (per-step average)
        for k, v in sorted(reward_components.items()):
            # total/base are still useful; skip if not numeric
            try:
                writer.add_scalar(f"RewardDecomp/{k}_per_step", float(v) / denom, global_step)
            except Exception:
                continue

        # Graph stats (per rollout step)
        g_n = max(1.0, graph_stats.get("neighbors_n", 0.0))
        writer.add_scalar("Graph/MeanValidNeighbors", float(graph_stats.get("neighbors_mean_sum", 0.0)) / g_n, global_step)

        # Terminal proportions (per-update)
        term_total = int(event_counts.get("terminal_total", 0))
        term_denom = max(1, term_total)
        success_rate = float(event_counts.get("success", 0)) / term_denom
        crash_rate = float(event_counts.get("crash", 0)) / term_denom
        out_of_road_rate = float(event_counts.get("out_of_road", 0)) / term_denom
        writer.add_scalar("Terminal/SuccessRate", success_rate, global_step)
        writer.add_scalar("Terminal/CrashRate", crash_rate, global_step)
        writer.add_scalar("Terminal/OutOfRoadRate", out_of_road_rate, global_step)
        writer.add_scalar("Terminal/TimeoutRate", float(event_counts.get("timeout", 0)) / term_denom, global_step)
        writer.add_scalar("Terminal/OtherRate", float(event_counts.get("other", 0)) / term_denom, global_step)
        writer.add_scalar("Terminal/IdleLongRate", float(event_counts.get("idle_long", 0)) / term_denom, global_step)

        # Keep legacy event counters for quick glance (now terminal counts)
        writer.add_scalar("Events/Crash", event_counts.get("crash", 0), global_step)
        writer.add_scalar("Events/OutOfRoad", event_counts.get("out_of_road", 0), global_step)

        # Risk stats (per-step)
        steps_with_ttc = max(1.0, risk_stats.get("steps_with_ttc", 0.0))
        steps_with_dist = max(1.0, risk_stats.get("steps_with_dist", 0.0))
        writer.add_scalar("Risk/FracSteps_TTC_Threat", float(risk_stats.get("risk_ttc_steps", 0.0)) / steps_with_ttc, global_step)
        writer.add_scalar("Risk/FracSteps_Dist_Threat", float(risk_stats.get("risk_dist_steps", 0.0)) / steps_with_dist, global_step)
        writer.add_scalar("Risk/MeanMinTTC", float(risk_stats.get("min_ttc_sum", 0.0)) / steps_with_ttc, global_step)
        writer.add_scalar("Risk/MeanMinDist", float(risk_stats.get("min_dist_sum", 0.0)) / steps_with_dist, global_step)
        idle_n = max(1.0, risk_stats.get("idle_count_n", 0.0))
        writer.add_scalar("Deadlock/MeanIdleCount", float(risk_stats.get("idle_count_sum", 0.0)) / idle_n, global_step)

        # Curriculum visibility
        writer.add_scalar("Curriculum/SPEED_REWARD_SCALE", float(getattr(Config, "SPEED_REWARD_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/OVERSPEED_PENALTY_SCALE", float(getattr(Config, "OVERSPEED_PENALTY_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/LANE_CENTER_PENALTY_SCALE", float(getattr(Config, "LANE_CENTER_PENALTY_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/HEADING_PENALTY_SCALE", float(getattr(Config, "HEADING_PENALTY_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/SAFETY_PENALTY_SCALE", float(getattr(Config, "SAFETY_PENALTY_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/APPROACH_PENALTY_SCALE", float(getattr(Config, "APPROACH_PENALTY_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/TTC_PENALTY_SCALE", float(getattr(Config, "TTC_PENALTY_SCALE", 0.0)), global_step)
        writer.add_scalar("Curriculum/ACTION_MAG_PENALTY", float(getattr(Config, "ACTION_MAG_PENALTY", 0.0)), global_step)
        writer.add_scalar("Curriculum/ACTION_CHANGE_PENALTY", float(getattr(Config, "ACTION_CHANGE_PENALTY", 0.0)), global_step)
        writer.add_scalar("Curriculum/IDLE_PENALTY", float(getattr(Config, "IDLE_PENALTY", 0.0)), global_step)
        writer.add_scalar("Curriculum/IDLE_LONG_PENALTY", float(getattr(Config, "IDLE_LONG_PENALTY", 0.0)), global_step)
        writer.add_scalar("Curriculum/CRASH_PENALTY", float(getattr(Config, "CRASH_PENALTY", -200.0)), global_step)
        writer.add_scalar("Curriculum/OUT_OF_ROAD_PENALTY", float(getattr(Config, "OUT_OF_ROAD_PENALTY", -200.0)), global_step)
        writer.add_scalar("Curriculum/SUCCESS_REWARD", float(getattr(Config, "SUCCESS_REWARD", 300.0)), global_step)
        
        mean_ep_reward = 0.0
        if len(completed_episode_rewards) > 0:
            mean_ep_reward = np.mean(completed_episode_rewards)
            writer.add_scalar("Reward/Mean_Episode", mean_ep_reward, global_step)

        # --- Early stopping: compute monitoring metric ---
        # Note: rates are per-update and depend on terminal_total in this rollout.
        metric_raw = None
        if early_stop_metric == "mean_episode_reward":
            metric_raw = float(mean_ep_reward)
        elif early_stop_metric == "success_rate":
            metric_raw = float(success_rate)
        elif early_stop_metric == "-crash_rate":
            metric_raw = -float(crash_rate)
        elif early_stop_metric == "-out_of_road_rate":
            metric_raw = -float(out_of_road_rate)
        else:
            # Fallback: keep training rather than crashing
            metric_raw = float(mean_ep_reward)

        early_stop_hist.append(float(metric_raw))
        metric_smooth = float(np.mean(early_stop_hist)) if len(early_stop_hist) > 0 else float(metric_raw)

        # Log early-stop tracking
        writer.add_scalar("EarlyStop/MetricRaw", float(metric_raw), global_step)
        writer.add_scalar("EarlyStop/MetricSmooth", float(metric_smooth), global_step)
        writer.add_scalar("EarlyStop/BadCount", float(early_stop_bad), global_step)
        elapsed = max(1e-8, time.time() - wall_start)
        steps_per_sec = global_step / elapsed
        try:
            gpu_util = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).decode().strip().split("\n")[0]
        except Exception:
            gpu_util = "NA"
        print(
            f"Update {update}/{num_updates} | Loss: {avg_loss:.4f} | "
            f"PPO: {total_ppo_loss/avg_divisor:.4f} | V: {total_value_loss/avg_divisor:.4f} | "
            f"Ent: {total_entropy/avg_divisor:.4f} | Aux: {total_aux_loss/avg_divisor:.4f} | "
            f"EpReward: {mean_ep_reward:.2f} | "
            f"SR: {success_rate:.3f} CR: {crash_rate:.3f} OOR: {out_of_road_rate:.3f} (term={term_total}) | "
            f"BestR: {best_reward:.2f} BestSR: {best_success_rate:.3f} | "
            f"SPS: {steps_per_sec:.2f} | GPU%: {gpu_util}"
        )
        
        writer.add_scalar("Loss/Total", avg_loss, global_step)
        writer.add_scalar("Loss/Aux", total_aux_loss/avg_divisor, global_step)
        writer.add_scalar("Loss/PPO", total_ppo_loss/avg_divisor, global_step)
        writer.add_scalar("Loss/Value", total_value_loss/avg_divisor, global_step)
        writer.add_scalar("Loss/Entropy", total_entropy/avg_divisor, global_step)
        
        # --- Save ---
        # Save Best Model
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            torch.save(policy.state_dict(), f"{log_dir}/best_model.pth")
            print(f"New best model saved with reward {best_reward:.2f}")

        # Save Best-Success Model (paper-facing)
        # Use per-update success rate; if there is no terminal in this update, success_rate will be 0.
        if success_rate > best_success_rate:
            best_success_rate = float(success_rate)
            torch.save(policy.state_dict(), f"{log_dir}/best_success_model.pth")
            print(f"New best-success model saved with success_rate {best_success_rate:.3f}")

        # Save Last 3 Checkpoints
        if update % 50 == 0:
            ckpt_path = f"{log_dir}/ckpt_{update:06d}.pth"
            torch.save(policy.state_dict(), ckpt_path)
            saved_checkpoints.append(ckpt_path)
            print(f"Model saved at update {update}")

        # --- Early stopping decision (after logging + saving) ---
        if early_stop_enabled and update >= (start_update + early_stop_warmup):
            # Initialize best on first eligible point
            if early_stop_best is None:
                early_stop_best = float(metric_smooth)
                early_stop_bad = 0
                print(
                    f"[EarlyStop] init best={early_stop_best:.6f} "
                    f"(metric={early_stop_metric}, mode={early_stop_mode}, window={early_stop_window})"
                )
            else:
                improved = False
                if early_stop_mode == "min":
                    improved = float(metric_smooth) <= float(early_stop_best) - float(early_stop_min_delta)
                else:
                    # default: max
                    improved = float(metric_smooth) >= float(early_stop_best) + float(early_stop_min_delta)

                if improved:
                    early_stop_best = float(metric_smooth)
                    early_stop_bad = 0
                    print(f"[EarlyStop] improved best={early_stop_best:.6f}")
                else:
                    early_stop_bad += 1
                    print(
                        f"[EarlyStop] no_improve={early_stop_bad}/{early_stop_patience} "
                        f"(smooth={metric_smooth:.6f}, best={early_stop_best:.6f}, min_delta={early_stop_min_delta})"
                    )

                if early_stop_bad >= early_stop_patience:
                    print(
                        f"[EarlyStop] STOP: metric={early_stop_metric}, mode={early_stop_mode}, "
                        f"best_smooth={early_stop_best:.6f}, last_smooth={metric_smooth:.6f}, "
                        f"patience={early_stop_patience}"
                    )
                    break

    except Exception as _global_exc:
        print(f"\n{'!'*60}")
        print(f"❌ 训练循环发生未捕获异常:")
        print(f"{'!'*60}")
        import traceback; traceback.print_exc()
        print(f"{'!'*60}\n")
        sys.stdout.flush()

    try:
        envs.close()
    except Exception:
        pass
    writer.close()

if __name__ == "__main__":
    train()
