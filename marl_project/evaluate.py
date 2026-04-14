"""
MetaDrive MARL Cooperative Policy Evaluation Tool
==================================================

This script evaluates trained multi-agent reinforcement learning policies in MetaDrive.
Supports single checkpoint evaluation, batch evaluation, robustness testing, and video recording.

FEATURES
--------
✓ Single/Batch Model Evaluation (--model_path / --model_glob)
✓ Model Type Auto-Detection (ours, no_comm, no_aux, lidar_only, oracle)
✓ Personalized Perception Config (each baseline uses its training config)
✓ Robustness Testing (--noise, --mask for communication loss)
✓ Mixed Traffic Sweep (--mpr_sweep for different densities)
✓ Fairness & Interaction Metrics (near-miss rate, agent variance)
✓ Top-Down Video Recording (--record_dir, --record_video)
✓ Failure Analysis Export (--save_details_json)
✓ Rendering & Visualization (--render, --top_down, --draw_connections)

BASIC USAGE
-----------
# 1. Single model evaluation (10 episodes, auto-detect model type)
python marl_project/evaluate.py --model_path logs/marl_experiment/final/baseline_intent_gat/best_success_model.pth --episodes 10

# 2. Specify model type explicitly (recommended for fair comparison)
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --episodes 50

# 3. Batch evaluation (all checkpoints in a directory)
python marl_project/evaluate.py --model_glob "logs/marl_experiment/final/baseline_intent_gat/ckpt_*.pth" --episodes 20 --save_table results.csv

ADVANCED USAGE
--------------
# 4. Robustness test: communication loss sweep
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --mask 0.10 --episodes 20 --save_json eval_mask_0.10.json

# 5. Robustness test: observation noise
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --noise 0.05 --episodes 20

# 6. Mixed traffic sweep (evaluate at different densities)
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --mpr_sweep "marl_project/json/eval_stress.json" --episodes 20

# 7. Stress test: different map types
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --map_sequence SSCC --episodes 20

# 8. Record top-down video with predicted waypoints overlay
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --episodes 3 --record_dir videos/ --record_video --top_down_overlay pred_waypoints

# 9. Interactive rendering (visualize live behavior)
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --episodes 5 --render --render_sleep 0.02 --pause_at_end

# 10. Export failure analysis (per-episode details)
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --episodes 50 --save_details_json failure_analysis.json --max_out_of_road_traces 100

BASELINE COMPARISON
-------------------
# Evaluate all baselines with personalized configs (recommended workflow for paper)
conda activate metadrive

# Ours (Intent-GAT): blind + realistic communication (MASK=0.02)
python marl_project/evaluate.py --model_path logs/marl_experiment/final/baseline_intent_gat/best_success_model.pth --model_type ours --episodes 50 --save_json eval_ours.json

# No Comm: blind + no communication (MASK=1.0)
python marl_project/evaluate.py --model_path logs/marl_experiment/final/baseline_no_comm/best_success_model.pth --model_type no_comm --episodes 50 --save_json eval_no_comm.json

# LiDAR Only: LiDAR perception + no communication
python marl_project/evaluate.py --model_path logs/marl_experiment/final/baseline_lidar_only/best_success_model.pth --model_type lidar_only --episodes 50 --save_json eval_lidar.json

# No Aux: same as ours but without auxiliary task
python marl_project/evaluate.py --model_path logs/marl_experiment/final/baseline_no_aux/best_success_model.pth --model_type no_aux --episodes 50 --save_json eval_no_aux.json

KEY PARAMETERS
--------------
--model_path          Path to single .pth checkpoint
--model_glob          Glob pattern for batch evaluation (e.g., "logs/**/*.pth")
--model_type          Model type: auto, ours, no_comm, no_aux, lidar_only, oracle
--episodes            Number of evaluation episodes (default: 10)
--start_seed          Test set starting seed (default: 10000)
--num_agents          Override environment agent count
--map_sequence        Override map (e.g., 'SSCC' or int for block_num)
--noise               Override observation noise std (default: 0.0)
--mask                Override communication mask ratio (default: model-specific)
--mpr_sweep           Mixed traffic sweep config (JSON file or string)
--device              Compute device: cuda or cpu (default: auto-detect)
--save_json           Save summary statistics to JSON file
--save_table          Save batch comparison table to CSV (for --model_glob)
--save_details_json   Export per-episode failure analysis
--render              Enable live visualization window
--top_down            Use top-down camera view
--draw_connections    Draw lines between communicating agents
--record_dir          Directory for video/screenshot recording
--record_video        Record top-down video (requires imageio + ffmpeg)
--stochastic          Sample actions from policy (default: deterministic)
--eval_full_reward    Force full reward coefficients (disable curriculum)

PERSONALIZED EVALUATION CONFIG
------------------------------
Each baseline is evaluated with the perception config it was trained with:

- ours (Intent-GAT):      LIDAR_NUM_OTHERS=0, MASK_RATIO=0.02 (realistic comm)
- no_comm (Blind):        LIDAR_NUM_OTHERS=0, MASK_RATIO=1.0  (no comm simulation)
- lidar_only (LiDAR):     LIDAR_NUM_OTHERS=4, MASK_RATIO=1.0  (physical only)
- no_aux (Ablation):      LIDAR_NUM_OTHERS=0, MASK_RATIO=0.02 (same as ours)
- oracle (Upper Bound):   LIDAR_NUM_OTHERS=4, MASK_RATIO=0.0  (perfect info)

This ensures fair comparison - each model is tested under the conditions it was trained for.

OUTPUT METRICS
--------------
Terminal Rates:   success, crash, out_of_road, timeout (%)
Return:           mean episode return (per agent)
Steps:            mean episode length (per agent)
Risk:             avg min TTC, high-risk TTC rate, avg min distance
Graph:            mean neighbors per step (communication graph)
Aux:              ADE/FDE (intent prediction error for auxiliary task)
Smoothness:       jerk_mean (action change magnitude, lower=smoother driving)
Efficiency:       avg_speed_kmh, throughput (success/total_steps), completion_efficiency
Lateral:          avg_lane_deviation_m (lane keeping quality, lower=better)
Decisiveness:     idle_rate (low-speed steps ratio, lower=more decisive)
Fairness:         worst/best agent success rate, success variance, near-miss rate
Observed MPR:     mixed penetration rate (RL vehicles / total vehicles)

SMOOTHNESS (JERK) METRIC
------------------------
Measures driving smoothness by computing action change magnitude:
  Jerk = |current_action - previous_action| (L1 norm)
  
- Lower values (→ 0): Smooth, stable driving (minimal corrections)
- Higher values: Aggressive, jerky driving (frequent large adjustments)

This metric reflects control stability and passenger comfort. In real-world
autonomous driving, excessive jerk leads to poor ride quality and mechanical wear.

TRAFFIC EFFICIENCY METRIC
-------------------------
Measures task completion rate per unit time:
  Throughput = success_count / total_steps
  Avg Speed = mean speed across all agent-steps (km/h)
  Completion Efficiency = success_rate * avg_speed

- Higher throughput: More efficient task completion
- Higher avg speed: Faster driving (but watch for safety trade-offs)

LATERAL STABILITY (LANE KEEPING) METRIC
---------------------------------------
Measures how well vehicles stay centered in their lane:
  Lane Deviation = |lateral_offset_from_lane_center| (meters)

- Lower values (→ 0): Precise lane keeping (stable driving)
- Higher values: Weaving or poor lane tracking

DECISIVENESS (IDLE RATE) METRIC
-------------------------------
Measures decision-making efficiency by tracking idle/stopped time:
  Idle Rate = Steps(Speed < 1km/h) / Total Steps

- Lower values (→ 0): Decisive driving (minimal hesitation)
- Higher values: Indecisive/hesitant behavior (excessive stopping)

This metric reflects the vehicle's ability to make timely decisions in traffic.
High idle rates may indicate overly conservative policies or difficulty in gap selection.

EXAMPLES FOR PAPER
------------------
# Benchmark evaluation (Table 1)
for model in ours no_comm no_aux lidar_only; do
    python marl_project/evaluate.py --model_path logs/marl_experiment/final/baseline_${model}/best_success_model.pth --model_type ${model} --episodes 50 --save_json eval_${model}.json
done

# Robustness: communication loss (Figure 2)
for mask in 0.0 0.02 0.05 0.10 0.20 0.50 1.0; do
    python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --mask ${mask} --episodes 20 --save_json eval_mask_${mask}.json
done

# Robustness: observation noise (Figure 3)
for noise in 0.0 0.02 0.05 0.10 0.15 0.20; do
    python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --noise ${noise} --episodes 20 --save_json eval_noise_${noise}.json
done

# Stress test: traffic density (Figure 4)
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --mpr_sweep "marl_project/json/eval_stress.json" --episodes 20 --save_json eval_density.json

# Visualization video (for presentation)
python marl_project/evaluate.py --model_path logs/.../best_success_model.pth --model_type ours --episodes 3 --record_dir videos/demo/ --record_video --video_fps 30 --top_down_overlay pred_waypoints

NOTES
-----
- Test seeds start at 10000 by default (separate from training set)
- Evaluation uses deterministic policy (mean action) unless --stochastic
- All baselines use identical reward parameters (--eval_full_reward enforced)
- MetaDrive patches auto-applied to avoid pedestrian/map crashes (disable: --disable_patches)
- Rendering may crash on some Windows setups with vehicle_model='s' (auto-fixed)

For complete documentation, see: marl_project/docs/EVALUATION_GUIDE.md
"""

import os
import sys
import json
import argparse
import glob
import re
import csv
import time
import shutil
import tempfile
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional, Sequence

import numpy as np
import torch

# 确保marl_project在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.models.policy import CooperativePolicy
from marl_project.mappo_modules import MAPPOPolicy
from marl_project.paper_viz import extract_attention_snapshot, plot_attention_dual_panel_pdf


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ============================================================================
# 工具函数：命令行参数解析与配置
# ============================================================================


def _maybe_load_json_arg(value: Optional[str]) -> Optional[Any]:
    """Parse a CLI arg that can be either a JSON string or a path to a JSON file."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Treat as file path if it exists
    try:
        if os.path.exists(s) and os.path.isfile(s):
            with open(s, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return json.loads(s)


def _normalize_mpr_sweep_configs(raw: Any) -> List[Dict[str, Any]]:
    """Validate and normalize MPR sweep configs.

    Expected schema: list of dicts
      {"name": str, "num_agents": int, "traffic_density": float}
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("--mpr_sweep must be a JSON list (or a path to a JSON file containing a list)")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"--mpr_sweep[{i}] must be an object/dict")
        if "num_agents" not in item:
            raise ValueError(f"--mpr_sweep[{i}] missing required field: num_agents")
        if "traffic_density" not in item:
            raise ValueError(f"--mpr_sweep[{i}] missing required field: traffic_density")

        name = str(item.get("name", f"mpr_{i}"))
        num_agents = int(item["num_agents"])
        traffic_density = float(item["traffic_density"])
        if num_agents <= 0:
            raise ValueError(f"--mpr_sweep[{i}].num_agents must be > 0")
        if traffic_density < 0:
            raise ValueError(f"--mpr_sweep[{i}].traffic_density must be >= 0")

        out.append({"name": name, "num_agents": num_agents, "traffic_density": traffic_density})
    return out


def _ensure_repo_root_on_path() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    if root_dir not in sys.path:
        sys.path.append(root_dir)


def _apply_metadrive_patches() -> None:
    """Best-effort runtime patches for known MetaDrive issues on some setups.

    These are intentionally optional and wrapped with broad exception handling.
    """
    # Patch 1: Pedestrian asset loading crash (seen on some Windows setups)
    try:
        from metadrive.component.traffic_participants.pedestrian import Pedestrian
        from panda3d.core import NodePath

        dummy_func = lambda *args, **kwargs: None
        Pedestrian.init_pedestrian_model = dummy_func
        Pedestrian.set_velocity = dummy_func
        Pedestrian.set_position = dummy_func
        Pedestrian.set_heading_theta = dummy_func
        Pedestrian.set_state = dummy_func
        Pedestrian.before_step = dummy_func
        Pedestrian.after_step = dummy_func
        Pedestrian.reset = dummy_func
        Pedestrian.destroy = dummy_func

        setattr(Pedestrian, "speed", 0)
        setattr(Pedestrian, "heading_theta", 0)
        setattr(Pedestrian, "velocity", [0, 0])
        setattr(Pedestrian, "position", [0, 0])
        setattr(Pedestrian, "body", None)
        setattr(Pedestrian, "config", {})
        setattr(Pedestrian, "random_seed", 0)
        setattr(Pedestrian, "name", "dummy")
        setattr(Pedestrian, "id", "dummy")
        setattr(Pedestrian, "dynamic_nodes", None)
        setattr(Pedestrian, "static_nodes", None)
        setattr(Pedestrian, "render", False)

        class _MockPhysicsNode:
            def attach_to_physics_world(self, *args, **kwargs):
                pass

            def detach_from_physics_world(self, *args, **kwargs):
                pass

            def clear(self):
                pass

            def destroy(self):
                pass

            def __len__(self):
                return 0

            def __iter__(self):
                return iter([])

        def _dummy_init(self, position, heading_theta, *args, **kwargs):
            self.origin = NodePath("dummy_pedestrian")
            self.origin.setPos(position[0], position[1], 0)
            self.id = "dummy_pedestrian_0"
            self.name = "dummy_pedestrian"
            self.config = {"show_side_detector": False, "show_lane_line_detector": False}
            self.random_seed = 0
            self.render = False
            self.dynamic_nodes = _MockPhysicsNode()
            self.static_nodes = _MockPhysicsNode()
            self.body = None
            self.speed = 0
            self.heading_theta = 0
            self.velocity = np.array([0.0, 0.0])
            self.position = np.array([0.0, 0.0])

        Pedestrian.__init__ = _dummy_init
    except Exception:
        pass

    # Patch 2: Map config assertion (counterfeit Config strategy)
    try:
        import metadrive.component.map.pg_map as pg_map_module
        import metadrive.envs.metadrive_env as metadrive_env_module
        from metadrive.utils.config import Config as MetaDriveConfig

        _original_parse_map_config = pg_map_module.parse_map_config

        def patched_parse_map_config(easy_map_config, new_map_config, default_config):
            fake_config = MetaDriveConfig(default_config)
            if "map" in fake_config:
                fake_config["map"] = easy_map_config
            return _original_parse_map_config(easy_map_config, new_map_config, fake_config)

        pg_map_module.parse_map_config = patched_parse_map_config
        if hasattr(metadrive_env_module, "parse_map_config"):
            metadrive_env_module.parse_map_config = patched_parse_map_config
    except Exception:
        pass


_ensure_repo_root_on_path()

from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.models.policy import CooperativePolicy
from marl_project.mappo_modules import MAPPOPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CooperativePolicy checkpoint")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a .pth checkpoint")
    parser.add_argument(
        "--model_glob",
        type=str,
        default=None,
        help='Glob pattern to evaluate multiple checkpoints (e.g. "logs/.../ckpt_*.pth")',
    )

    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--start_seed", type=int, default=10000, help="Test set start seed")
    parser.add_argument("--num_agents", type=int, default=None, help="Override number of agents")
    
    # Model type specification for fair evaluation
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "ours", "no_comm", "no_aux", "lidar_only", "oracle", "tarmac", "mappo", "mappo_ips", "where2comm"],
        help=(
            "Model type for personalized evaluation config. "
            "'auto' detects from model path. "
            "Each type gets its trained perception config (MAX_NEIGHBORS, LIDAR_NUM_OTHERS, etc.)"
        )
    )

    parser.add_argument("--map_sequence", type=str, default=None, help="Override map (e.g. 'SSCC' or int for block_num)")
    parser.add_argument("--noise", type=float, default=None, help="Override Config.NOISE_STD")
    parser.add_argument("--mask", type=float, default=None, help="Override Config.MASK_RATIO")
    parser.add_argument("--comm_radius", type=float, default=None, help="Override Config.COMM_RADIUS (meters)")
    parser.add_argument("--distance_bias", type=float, default=None, help="Override Config.DISTANCE_BIAS_SCALE")
    parser.add_argument("--max_neighbors_override", type=int, default=None, help="Override Config.MAX_NEIGHBORS (must be <= 8)")
    parser.add_argument(
        "--comm_mode",
        type=str,
        default="iid",
        choices=["iid", "burst", "staleness"],
        help="Communication loss semantics: iid mask, burst loss, or staleness",
    )
    parser.add_argument(
        "--burst_len",
        type=int,
        default=1,
        help="Burst length for --comm_mode burst (burst_len=1 is equivalent to iid)",
    )
    parser.add_argument(
        "--stale_steps",
        type=int,
        default=0,
        help="How many steps to keep stale neighbor features for --comm_mode staleness",
    )

    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--top_down", action="store_true", help="Use top-down render mode when rendering")
    parser.add_argument("--draw_connections", action="store_true", help="Draw lines between communicating agents")

    # Top-down overlay controls
    parser.add_argument(
        "--top_down_overlay",
        type=str,
        default="pred_waypoints",
        choices=["pred_waypoints", "history"],
        help=(
            "What to draw in pygame top-down view. "
            "'pred_waypoints' draws aux predicted future waypoints; 'history' draws vehicle history trails."
        ),
    )

    # Visualization quality-of-life
    parser.add_argument(
        "--render_sleep",
        type=float,
        default=None,
        help="Sleep seconds per step when --render is enabled (helps keep window visible). Default: 1/60.",
    )
    parser.add_argument(
        "--pause_on_start",
        type=float,
        default=0.0,
        help="If >0, pause this many seconds after first reset (keeps window open).",
    )
    parser.add_argument(
        "--pause_at_end",
        action="store_true",
        help="If set, wait for Enter before exiting (keeps render window open).",
    )

    # Evaluation protocol controls
    parser.add_argument(
        "--eval_full_reward",
        action="store_true",
        help="Force all reward shaping coefficients to their configured maxima (avoid curriculum affecting eval).",
    )

    parser.add_argument("--stochastic", action="store_true", help="Sample actions from policy (default is deterministic mean)")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max env steps per episode")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument("--disable_patches", action="store_true", help="Disable MetaDrive runtime patches")

    parser.add_argument("--save_json", type=str, default=None, help="Save evaluation summary to JSON file")
    parser.add_argument(
        "--save_table",
        type=str,
        default=None,
        help="Save batch comparison table as CSV (only meaningful with --model_glob)",
    )

    # Mixed traffic / MPR sweep (evaluation only)
    parser.add_argument(
        "--mpr_sweep",
        type=str,
        default=None,
        help=(
            "Run a mixed-traffic sweep by overriding env config per run (supports both --model_path and --model_glob). "
            "Provide a JSON string or (recommended) a path to a JSON file containing a list of dicts: "
            "[{name,num_agents,traffic_density}, ...]. The summary/CSV will also include best-effort observed MPR (RL/(RL+BG))."
        ),
    )

    # Failure analysis export
    parser.add_argument(
        "--save_details_json",
        type=str,
        default=None,
        help="Save per-episode analysis JSON (termination distribution, avg speed, out_of_road traces)",
    )
    parser.add_argument(
        "--last_n_steps",
        type=int,
        default=30,
        help="How many last steps to keep per-agent for failure trace export",
    )
    parser.add_argument(
        "--max_out_of_road_traces",
        type=int,
        default=50,
        help="Max number of out_of_road traces to store in details export",
    )

    # Recording (top-down video / screenshots)
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="If set, record top-down video and/or screenshots into this directory",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Write per-episode MP4 video in record_dir (requires imageio + ffmpeg)",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Video FPS when recording",
    )
    parser.add_argument(
        "--screenshot_every",
        type=int,
        default=0,
        help="If >0, save a PNG every N steps (top-down) into record_dir",
    )

    # TTC threshold override for evaluation
    parser.add_argument(
        "--ttc_threshold",
        type=float,
        default=None,
        help="Override TTC_THRESHOLD_S for high-risk TTC calculation (default: use Config value)",
    )

    # Paper-quality attention snapshot capture (inside evaluate loop)
    parser.add_argument("--paper_capture", action="store_true", help="Enable paper snapshot capture during evaluation")
    parser.add_argument("--paper_capture_dir", type=str, default="logs/paper_attention_capture", help="Output directory for paper snapshots")
    parser.add_argument("--paper_max_captures", type=int, default=1, help="Maximum captures per evaluation run")
    parser.add_argument("--paper_trigger_step_min", type=int, default=50, help="Capture only when step > this threshold")
    parser.add_argument("--paper_trigger_speed_kmh", type=float, default=15.0, help="Capture only when ego speed > this threshold")
    parser.add_argument("--paper_focus_radius_m", type=float, default=42.0, help="Focus crop radius in meters for spatial panel")
    parser.add_argument("--paper_temporal_window", type=int, default=80, help="Temporal window size for attention evolution panel")
    parser.add_argument("--paper_export_texture_view", action="store_true", help="Export additional texture-like top-down view frame")
    parser.add_argument("--paper_top_k", type=int, default=3, help="Auto select top-k strongest polarization frames")

    return parser.parse_args()


def _detect_model_type(model_path: str) -> str:
    """Auto-detect model type from path for personalized evaluation config.
    
    Returns:
        Model type: 'ours', 'no_comm', 'no_aux', 'lidar_only', or 'oracle'
    """
    path_lower = model_path.lower()
    
    # Check for explicit markers in path
    if "where2comm" in path_lower:
        return "where2comm"
    elif "mappo_ips" in path_lower or "baseline_mappo_ips" in path_lower:
        return "mappo_ips"
    elif "mappo" in path_lower:
        return "mappo"
    elif "tarmac" in path_lower:
        return "tarmac"
    elif "no_comm" in path_lower or "ippo" in path_lower:
        return "no_comm"
    elif "no_aux" in path_lower:
        return "no_aux"
    elif "lidar_only" in path_lower or "lidar" in path_lower:
        return "lidar_only"
    elif "oracle" in path_lower:
        return "oracle"
    elif "intent_gat" in path_lower or "baseline_intent" in path_lower:
        return "ours"
    
    # Default to ours if uncertain
    print(f"[Eval] Warning: Could not detect model type from path '{model_path}', defaulting to 'ours'")
    return "ours"


def _get_personalized_perception_config(model_type: str) -> Dict[str, Any]:
    """Get personalized perception config for different model types.
    
    Each baseline should be evaluated with the same perception config it was trained with.
    
    Args:
        model_type: 'ours', 'no_comm', 'no_aux', 'lidar_only', or 'oracle'
    
    Returns:
        Dict with perception parameters: max_neighbors, lidar_num_others, mask_ratio, noise_std
    """
    configs = {
        # 1. Ours (Intent-GAT): 核心方法
        # 特征: 物理盲 (Lidar=0) + 有通信 (Mask=0.02)
        "ours": {
            "max_neighbors": None,     # ⚠️ 不修改（保持模型架构）
            "lidar_num_others": 0,     # 【关键】物理盲，看不到邻居，必须靠通信
            "mask_ratio": 0.02,        # 【修改】模拟真实通信丢包 (Realism)
            "noise_std": 0.0,          # 评估时无噪声（除非压力测试）
            "comm_module": "gat",
            "description": "Intent-GAT (物理盲 + 真实通信)"
        },
        
        # 2. Blind Baseline (No Comm): 最弱基线
        # 特征: 物理盲 (Lidar=0) + 100% mask模拟无通信
        "no_comm": {
            "max_neighbors": None,     # ⚠️ 不修改！保持模型架构，用mask模拟无通信
            "lidar_num_others": 0,     # 【修改】真正的盲基线（无雷达）
            "mask_ratio": 1.0,         # 【关键】100% mask模拟无通信
            "noise_std": 0.0,
            "comm_module": "none",
            "description": "全盲基线 (无雷达 + 100% mask模拟无通信)"
        },
        
        # 3. LiDAR Baseline: 强力基线
        # 特征: 有雷达 (Lidar=4) + 无通信 (Mask=1.0)
        "lidar_only": {
            "max_neighbors": None,     # ⚠️ 不修改
            "lidar_num_others": 4,     # 【关键】有雷达，能看到邻居
            "mask_ratio": 1.0,         # 【关键】100% 丢包，强制切断通信
            "noise_std": 0.0,
            "comm_module": "none",
            "description": "LiDAR基线 (有雷达 + 100% mask)"
        },
        
        # 4. Ablation (No Aux): 消融实验
        # 特征: 输入和 Ours 一模一样，只是训练时没有辅助任务
        "no_aux": {
            "max_neighbors": None,     # ⚠️ 不修改
            "lidar_num_others": 0,     # 【修改】必须和 Ours 一致 (物理盲)
            "mask_ratio": 0.02,        # 【修改】必须和 Ours 一致
            "noise_std": 0.0,
            "comm_module": "gat",
            "description": "消融实验 (无辅助任务)"
        },
        
        # 5. Oracle: 上帝视角 (上限)
        # 特征: 有雷达 + 有完美通信
        "oracle": {
            "max_neighbors": None,     # ⚠️ 不修改
            "lidar_num_others": 4,
            "mask_ratio": 0.0,         # 完美通信
            "noise_std": 0.0,
            "comm_module": "gat",
            "description": "Oracle (全知全能)"
        },

        "mappo": {
            "max_neighbors": None,
            "lidar_num_others": 0,
            "mask_ratio": 1.0,
            "noise_std": 0.0,
            "comm_module": "none",
            "description": "MAPPO baseline (centralized critic, no explicit communication)",
        },

        "mappo_ips": {
            "max_neighbors": None,
            "lidar_num_others": 0,
            "mask_ratio": 0.02,
            "noise_std": 0.0,
            "comm_module": "ips_mean",
            "description": "MAPPO-IPS baseline (centralized critic + IPS top-k actor communication)",
        },

        "where2comm": {
            "max_neighbors": None,
            "lidar_num_others": 0,
            "mask_ratio": 0.02,
            "noise_std": 0.0,
            "comm_module": "where2comm_raw",
            "description": "Where2Comm-style baseline (confidence-gated raw feature sharing)",
        },
        
        # 6. TarMAC: 外部基线 (Das et al., 2019)
        # 特征: 与 Ours 完全相同的环境/感知设置，仅通信模块不同
        "tarmac": {
            "max_neighbors": None,     # ⚠️ 不修改
            "lidar_num_others": 0,     # 【关键】物理盲，与 Ours 一致
            "mask_ratio": 0.02,        # 【关键】与 Ours 一致的轻度丢包
            "noise_std": 0.0,
            "comm_module": "tarmac",   # 切换通信模块
            "description": "TarMAC外部基线 (物理盲 + TarMAC通信)"
        },
    }
    
    if model_type not in configs:
        print(f"[Eval] Warning: Unknown model type '{model_type}', using 'ours' config")
        model_type = "ours"
    
    return configs[model_type]


def _apply_overrides(args: argparse.Namespace) -> None:
    if args.noise is not None:
        Config.NOISE_STD = float(args.noise)
    if args.mask is not None:
        Config.MASK_RATIO = float(args.mask)
    if getattr(args, "comm_radius", None) is not None:
        Config.COMM_RADIUS = float(args.comm_radius)
        print(f"[Eval] COMM_RADIUS overridden to {Config.COMM_RADIUS:.1f}m")
    if getattr(args, "distance_bias", None) is not None:
        Config.DISTANCE_BIAS_SCALE = float(args.distance_bias)
        print(f"[Eval] DISTANCE_BIAS_SCALE overridden to {Config.DISTANCE_BIAS_SCALE:.3f}")
    if getattr(args, "max_neighbors_override", None) is not None:
        Config.MAX_NEIGHBORS = min(8, max(1, int(args.max_neighbors_override)))
        print(f"[Eval] MAX_NEIGHBORS overridden to {Config.MAX_NEIGHBORS}")
    if not hasattr(Config, "COMM_BURST_LEN"):
        setattr(Config, "COMM_BURST_LEN", 1)
    if not hasattr(Config, "COMM_STALE_STEPS"):
        setattr(Config, "COMM_STALE_STEPS", 0)
    setattr(Config, "COMM_BURST_LEN", max(1, int(getattr(args, "burst_len", 1) or 1)))
    setattr(Config, "COMM_STALE_STEPS", max(0, int(getattr(args, "stale_steps", 0) or 0)))
    if getattr(args, "ttc_threshold", None) is not None:
        Config.TTC_THRESHOLD_S = float(args.ttc_threshold)
        print(f"[Eval] TTC_THRESHOLD_S overridden to {Config.TTC_THRESHOLD_S:.2f}s")


def _capture_reward_maxima_from_config() -> Dict[str, float]:
    """Capture the current Config reward-related coefficients as maxima.

    This makes evaluation independent from any curriculum-scaled values that might have been
    applied earlier in the process.
    """
    keys = [
        # terminal magnitudes
        "CRASH_PENALTY",
        "OUT_OF_ROAD_PENALTY",
        "SUCCESS_REWARD",
        # shaping scales
        "TTC_PENALTY_SCALE",
        "ACTION_MAG_PENALTY",
        "ACTION_CHANGE_PENALTY",
        "LANE_CENTER_PENALTY_SCALE",
        "HEADING_PENALTY_SCALE",
        "SAFETY_PENALTY_SCALE",
        "APPROACH_PENALTY_SCALE",
        "IDLE_PENALTY",
        "IDLE_LONG_PENALTY",
        "SPEED_REWARD_SCALE",
        "OVERSPEED_PENALTY_SCALE",
    ]
    maxima: Dict[str, float] = {}
    for k in keys:
        if hasattr(Config, k):
            try:
                maxima[k] = float(getattr(Config, k))
            except Exception:
                # if non-float, keep as-is in a safe way
                try:
                    maxima[k] = float(getattr(Config, k, 0.0))
                except Exception:
                    maxima[k] = 0.0
        else:
            maxima[k] = 0.0
    return maxima


def _apply_reward_params(params: Dict[str, float]) -> None:
    for k, v in params.items():
        if hasattr(Config, k):
            try:
                setattr(Config, k, float(v))
            except Exception:
                setattr(Config, k, v)


def _maybe_fix_vehicle_model_for_render(eval_config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Best-effort workaround for Windows/Panda3D crashes when loading some vehicle glTF assets.

    On some Windows setups, using onscreen rendering with certain vehicle models (notably "s") can
    trigger a hard crash (access violation) inside Panda3D's model loader. During evaluation, when
    the user explicitly requests --render, we switch to a safer default visual model.
    """
    if not bool(getattr(args, "render", False)):
        return

    vc = eval_config.get("vehicle_config")
    if not isinstance(vc, dict):
        return

    vm = vc.get("vehicle_model")
    if vm == "s":
        vc["vehicle_model"] = "default"
        print('[Eval][Render] Detected vehicle_model="s" which may crash onscreen rendering on Windows; '
              'switching to vehicle_model="default" for visualization.')


def _maybe_get_imageio():
    try:
        import imageio.v2 as imageio  # type: ignore

        return imageio
    except Exception:
        try:
            import imageio  # type: ignore

            return imageio
        except Exception:
            return None


def _ensure_mp4_writer_available(imageio_module, record_dir: str) -> None:
    """Fail fast with a clear error when MP4 recording backend is unavailable."""
    if imageio_module is None:
        raise RuntimeError(
            "--record_video was requested, but imageio is not installed in the current Python environment. "
            "Install imageio plus an MP4 backend such as imageio-ffmpeg or pyav."
        )

    probe_path = None
    writer = None
    try:
        fd, probe_path = tempfile.mkstemp(prefix="mp4_probe_", suffix=".mp4", dir=record_dir)
        os.close(fd)
        writer = imageio_module.get_writer(probe_path, fps=1)
        writer.close()
        writer = None
    except Exception as e:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        raise RuntimeError(
            "--record_video was requested, but this Python environment can not open an MP4 writer. "
            "Install an MP4 backend, for example `pip install imageio-ffmpeg` or `pip install av`. "
            f"Original error: {e}"
        ) from e
    finally:
        if probe_path and os.path.exists(probe_path):
            try:
                os.remove(probe_path)
            except Exception:
                pass


def _vehicle_state_from_env(env: "GraphEnvWrapper", agent_id: str) -> Dict[str, float]:
    veh = env.agents[agent_id]
    pos = np.asarray(getattr(veh, "position", [0.0, 0.0]), dtype=np.float32)
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "heading": float(getattr(veh, "heading_theta", getattr(veh, "heading", 0.0))),
        "length": float(getattr(veh, "LENGTH", getattr(veh, "length", 4.8))),
        "width": float(getattr(veh, "WIDTH", getattr(veh, "width", 2.0))),
    }


def _extract_attn_vector(results: Dict[str, Any], ego_idx: int) -> np.ndarray:
    attn = results.get("attention_weights", None)
    if attn is None:
        return np.zeros((0,), dtype=np.float32)
    attn_np = attn[ego_idx].detach().cpu().numpy().astype(np.float32)
    if attn_np.ndim == 3:
        attn_np = attn_np.mean(axis=0)
    if attn_np.ndim == 2:
        attn_np = attn_np.squeeze(0)
    if attn_np.ndim != 1:
        attn_np = np.asarray(attn_np).reshape(-1)
    return attn_np


def _ego_alpha_map_from_obs(
    obs_dict: Dict[str, Dict[str, Any]],
    active_agents: Sequence[str],
    results: Dict[str, Any],
    ego_id: str,
) -> Dict[str, float]:
    if ego_id not in active_agents:
        return {}
    ego_idx = list(active_agents).index(ego_id)
    attn_vec = _extract_attn_vector(results, ego_idx)
    raw_neighbors = list(obs_dict.get(ego_id, {}).get("neighbors", []))
    alpha_map: Dict[str, float] = {}
    for i, nid in enumerate(raw_neighbors):
        if i >= len(attn_vec):
            break
        alpha_map[str(nid)] = float(np.clip(attn_vec[i], 0.0, 1.0))
    return alpha_map


def _pick_conflict_pair_from_info(info: Dict[str, Any], alpha_map: Dict[str, float]) -> Tuple[Optional[str], Optional[str], float]:
    neigh_ids = list(info.get("neighbors_true", []) or [])
    rel_list = list(info.get("neighbor_rel_pos_true", []) or [])
    if len(neigh_ids) == 0 or len(rel_list) == 0:
        return None, None, 0.0

    danger_id: Optional[str] = None
    safe_id: Optional[str] = None
    best_danger_score = -1e9
    best_safe_score = -1e9

    for idx, nid in enumerate(neigh_ids):
        if idx >= len(rel_list):
            continue
        rel = rel_list[idx]
        if len(rel) < 4:
            continue
        dx, dy, dvx, dvy = [float(x) for x in rel[:4]]
        dist = float(np.hypot(dx, dy))
        closing = float(-(dx * dvx + dy * dvy) / max(1e-6, dist))
        lateral_ratio = abs(dy) / max(1.0, abs(dx))
        rel_speed = float(np.hypot(dvx, dvy))
        alpha = float(alpha_map.get(str(nid), 0.0))

        danger_score = 1.2 * closing + 0.8 * lateral_ratio + 0.7 * rel_speed + 1.8 * alpha
        safe_score = (1.0 if closing < 0.8 else -1.0) + 0.8 * max(0.0, dist - 8.0) + 1.4 * (1.0 - alpha)

        if danger_score > best_danger_score:
            best_danger_score = danger_score
            danger_id = str(nid)
        if safe_score > best_safe_score:
            best_safe_score = safe_score
            safe_id = str(nid)

    return danger_id, safe_id, float(best_danger_score)


def _pick_conflict_pair_from_obs(
    obs_dict: Dict[str, Dict[str, Any]],
    ego_id: str,
    alpha_map: Dict[str, float],
) -> Tuple[Optional[str], Optional[str], float]:
    ego_obs = obs_dict.get(ego_id, {})
    neigh_ids = list(ego_obs.get("neighbors", []) or [])
    rel_list = list(ego_obs.get("neighbor_rel_pos", []) or [])
    if len(neigh_ids) == 0 or len(rel_list) == 0:
        return None, None, 0.0

    danger_id: Optional[str] = None
    safe_id: Optional[str] = None
    best_danger_score = -1e9
    best_safe_score = -1e9

    for idx, nid in enumerate(neigh_ids):
        if idx >= len(rel_list):
            continue
        nid_s = str(nid)
        if nid_s not in alpha_map:
            continue
        rel = rel_list[idx]
        if len(rel) < 4:
            continue
        dx, dy, dvx, dvy = [float(x) for x in rel[:4]]
        dist = float(np.hypot(dx, dy))
        closing = float(-(dx * dvx + dy * dvy) / max(1e-6, dist))
        lateral_ratio = abs(dy) / max(1.0, abs(dx))
        rel_speed = float(np.hypot(dvx, dvy))
        alpha = float(alpha_map.get(nid_s, 0.0))

        danger_score = 1.0 * closing + 0.9 * lateral_ratio + 0.6 * rel_speed + 2.0 * alpha
        safe_score = 1.7 * (1.0 - alpha) + 0.6 * max(0.0, dist - 6.0) + (0.5 if closing < 1.0 else -0.5)

        if danger_score > best_danger_score:
            best_danger_score = danger_score
            danger_id = nid_s
        if safe_score > best_safe_score:
            best_safe_score = safe_score
            safe_id = nid_s

    return danger_id, safe_id, float(best_danger_score)


def _sorted_series_arrays(series: Sequence[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    if not series:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    ordered = sorted([(float(s), float(v)) for s, v in series], key=lambda x: x[0])
    steps = np.asarray([s for s, _ in ordered], dtype=np.float32)
    vals = np.asarray([v for _, v in ordered], dtype=np.float32)
    return steps, vals


def _series_delta(series: Sequence[Tuple[float, float]]) -> float:
    _, vals = _sorted_series_arrays(series)
    if vals.size < 2:
        return 0.0
    return float(vals[-1] - vals[0])


def _series_monotonicity_score(series: Sequence[Tuple[float, float]], *, expect: str) -> float:
    _, vals = _sorted_series_arrays(series)
    if vals.size < 2:
        return 0.0
    diffs = np.diff(vals)
    good = np.sum(diffs >= -1e-4) if expect == "up" else np.sum(diffs <= 1e-4)
    score = float(good) / float(max(1, diffs.size))
    delta = _series_delta(series)
    directional = float(np.clip(delta if expect == "up" else -delta, 0.0, 1.0))
    return 0.55 * score + 0.45 * directional


def _series_spike_penalty(series: Sequence[Tuple[float, float]]) -> float:
    _, vals = _sorted_series_arrays(series)
    if vals.size < 4:
        return 0.0
    diffs = np.abs(np.diff(vals))
    max_jump = float(np.max(diffs))
    median_jump = float(np.median(diffs))
    excess = max(0.0, max_jump - max(0.10, 2.5 * median_jump))
    return float(np.clip(excess / 0.35, 0.0, 1.0))


def _ttc_alignment_score(
    danger_series: Sequence[Tuple[float, float]],
    safe_series: Sequence[Tuple[float, float]],
    ttc_series: Sequence[Tuple[float, float]],
) -> float:
    danger_delta = _series_delta(danger_series)
    safe_delta = _series_delta(safe_series)
    ttc_delta = _series_delta(ttc_series)
    risk_rising = float(np.clip(-ttc_delta, 0.0, 6.0) / 6.0)
    if risk_rising <= 1e-6:
        return 0.0
    danger_score = float(np.clip(danger_delta, 0.0, 1.0))
    safe_score = float(np.clip(-safe_delta, 0.0, 1.0))
    return risk_rising * (0.6 * danger_score + 0.4 * safe_score)


def _label_layout_clarity_score(spatial_meta: Dict[str, Any]) -> float:
    layout = ((spatial_meta or {}).get("spatial_axis", {}) or {}).get("label_layout", {}) or {}
    boxes: List[Tuple[float, float, float, float]] = []
    for key in ("ego", "danger", "safe"):
        box = layout.get(key, {}).get("label_box")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            boxes.append(tuple(float(v) for v in box))
    if len(boxes) < 2:
        return 0.0
    min_sep = 1e9
    for i in range(len(boxes)):
        ax0, ax1, ay0, ay1 = boxes[i]
        acx = 0.5 * (ax0 + ax1)
        acy = 0.5 * (ay0 + ay1)
        for j in range(i + 1, len(boxes)):
            bx0, bx1, by0, by1 = boxes[j]
            bcx = 0.5 * (bx0 + bx1)
            bcy = 0.5 * (by0 + by1)
            min_sep = min(min_sep, float(np.hypot(acx - bcx, acy - bcy)))
    return float(np.clip(min_sep / 12.0, 0.0, 1.0))


def _spatial_intuition_score(spatial_meta: Dict[str, Any]) -> float:
    layout = ((spatial_meta or {}).get("spatial_axis", {}) or {}).get("label_layout", {}) or {}
    danger_xy = layout.get("danger", {}).get("xy")
    safe_xy = layout.get("safe", {}).get("xy")
    if not (isinstance(danger_xy, (list, tuple)) and len(danger_xy) >= 2 and isinstance(safe_xy, (list, tuple)) and len(safe_xy) >= 2):
        return 0.0
    danger_dist = float(np.hypot(float(danger_xy[0]), float(danger_xy[1])))
    safe_dist = float(np.hypot(float(safe_xy[0]), float(safe_xy[1])))
    # Prefer candidates where the chosen danger is not dramatically farther than the safe reference.
    gap = max(0.0, danger_dist - safe_dist - 4.0)
    return float(1.0 - np.clip(gap / 18.0, 0.0, 1.0))


def _paper_candidate_score(
    *,
    polarization_score: float,
    conflict_score: float,
    danger_series: Sequence[Tuple[float, float]],
    safe_series: Sequence[Tuple[float, float]],
    ttc_series: Sequence[Tuple[float, float]],
    spatial_meta: Dict[str, Any],
) -> Dict[str, float]:
    visible_score = 1.0 if bool(spatial_meta.get("danger_visible")) and bool(spatial_meta.get("safe_visible")) else 0.0
    triplet_only_score = 1.0 if len(list(spatial_meta.get("visible_neighbor_ids", []) or [])) == 2 else 0.0
    red_up = _series_monotonicity_score(danger_series, expect="up")
    blue_down = _series_monotonicity_score(safe_series, expect="down")
    trend_score = 0.55 * red_up + 0.45 * blue_down
    spike_penalty = max(_series_spike_penalty(danger_series), _series_spike_penalty(safe_series))
    ttc_score = _ttc_alignment_score(danger_series, safe_series, ttc_series)
    clarity_score = _label_layout_clarity_score(spatial_meta)
    intuition_score = _spatial_intuition_score(spatial_meta)
    base_score = float(np.clip(polarization_score / 3.0, 0.0, 1.0))
    conflict_norm = float(np.clip(conflict_score / 3.0, 0.0, 1.0))

    total = (
        0.18 * visible_score
        + 0.08 * triplet_only_score
        + 0.20 * trend_score
        + 0.18 * ttc_score
        + 0.16 * clarity_score
        + 0.10 * intuition_score
        + 0.06 * base_score
        + 0.08 * conflict_norm
        - 0.12 * spike_penalty
    )
    return {
        "paper_score": float(total),
        "visible_score": float(visible_score),
        "triplet_only_score": float(triplet_only_score),
        "trend_score": float(trend_score),
        "ttc_alignment_score": float(ttc_score),
        "label_clarity_score": float(clarity_score),
        "spatial_intuition_score": float(intuition_score),
        "spike_penalty": float(spike_penalty),
        "conflict_norm": float(conflict_norm),
        "polarization_norm": float(base_score),
    }


def _save_rgb_png(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
        return
    except Exception:
        pass

    imageio = _maybe_get_imageio()
    if imageio is not None:
        imageio.imwrite(path, rgb)


def _sort_key_for_ckpt(path: str) -> Tuple[int, str]:
    m = re.search(r"ckpt_(\d+)\.pth$", os.path.basename(path))
    if m:
        try:
            return int(m.group(1)), path
        except Exception:
            pass
    return 10**18, path


def _format_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    return f"{100.0 * float(x):5.2f}%"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        ("ckpt", 10),
        ("success", 9),
        ("crash", 9),
        ("out", 9),
        ("timeout", 9),
        ("ret", 10),
        ("steps", 8),
        ("neigh", 7),
    ]
    line = " ".join([h[0].ljust(h[1]) for h in headers])
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        ckpt = str(r.get("ckpt", "-"))[:10].ljust(10)
        success = _format_pct(r.get("success"))
        crash = _format_pct(r.get("crash"))
        out = _format_pct(r.get("out_of_road"))
        timeout = _format_pct(r.get("timeout"))
        ret = r.get("return_mean")
        steps = r.get("steps_mean")
        neigh = r.get("neighbors_mean")
        ret_s = (f"{float(ret):9.2f}" if ret is not None else "-".rjust(10))
        steps_s = (f"{float(steps):7.1f}" if steps is not None else "-".rjust(8))
        neigh_s = (f"{float(neigh):6.2f}" if neigh is not None else "-".rjust(7))
        print(f"{ckpt} {success} {crash} {out} {timeout} {ret_s} {steps_s} {neigh_s}")


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fieldnames = [
        "model_path",
        "ckpt",
        "sweep",
        "episodes",
        "start_seed",
        "num_agents",
        "traffic_density",
        "observed_mpr_mean",
        "observed_rl_vehicles_mean",
        "observed_bg_vehicles_mean",
        "success",
        "crash",
        "out_of_road",
        "timeout",
        "return_mean",
        "return_std",
        "steps_mean",
        "neighbors_mean",
        "ade",
        "fde",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _count_observed_mixed_traffic(env: "GraphEnvWrapper") -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Best-effort observed mixed traffic composition.

    Returns:
        (rl_vehicle_count, bg_vehicle_count, observed_mpr)

    Notes:
        - RL vehicles are vehicles controlled by env agents.
        - BG vehicles are other VEHICLE objects in the scene (traffic).
        - This is used for reporting only; exact counts depend on MetaDrive internals.
    """
    try:
        engine = getattr(env, "engine", None)
        if engine is None or (not hasattr(engine, "get_objects")):
            return None, None, None

        try:
            from metadrive.constants import MetaDriveType
        except Exception:
            return None, None, None

        # RL-controlled objects
        rl_objs = []
        try:
            if hasattr(env, "agents_including_just_terminated"):
                rl_objs = list(getattr(env, "agents_including_just_terminated", {}).values())
            else:
                rl_objs = list(getattr(env, "agents", {}).values())
        except Exception:
            rl_objs = list(getattr(env, "agents", {}).values()) if hasattr(env, "agents") else []
        rl_ids = {id(o) for o in rl_objs if o is not None}

        all_objs = engine.get_objects(
            lambda obj: hasattr(obj, "metadrive_type") and obj.metadrive_type == MetaDriveType.VEHICLE
        )
        if not isinstance(all_objs, dict):
            return None, None, None

        rl_cnt = 0
        bg_cnt = 0
        for o in all_objs.values():
            if id(o) in rl_ids:
                rl_cnt += 1
            else:
                bg_cnt += 1
        denom = rl_cnt + bg_cnt
        mpr = (float(rl_cnt) / float(denom)) if denom > 0 else None
        return rl_cnt, bg_cnt, mpr
    except Exception:
        return None, None, None



def evaluate_single_model(
    model_path: str,
    args: argparse.Namespace,
    eval_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Evaluate a single checkpoint.

    Returns:
        (summary, details) where details may be None unless requested.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # === [关键修复] 应用个性化感知配置 ===
    # 根据model_type自动检测或使用用户指定的类型
    model_type_str = str(getattr(args, "model_type", "auto"))
    if model_type_str == "auto":
        model_type_str = _detect_model_type(model_path)
    
    # 获取该模型类型的个性化配置
    perception_cfg = _get_personalized_perception_config(model_type_str)
    
    # 应用到Config全局变量（因为GraphEnvWrapper从Config读取）
    # ⚠️ 注意：max_neighbors为None时不修改（避免模型架构不匹配）
    if perception_cfg["max_neighbors"] is not None:
        Config.MAX_NEIGHBORS = int(perception_cfg["max_neighbors"])
    if perception_cfg["lidar_num_others"] is not None:
        Config.LIDAR_NUM_OTHERS = int(perception_cfg["lidar_num_others"])
    if perception_cfg["mask_ratio"] is not None:
        Config.MASK_RATIO = float(perception_cfg["mask_ratio"])
    if perception_cfg["noise_std"] is not None:
        Config.NOISE_STD = float(perception_cfg["noise_std"])
    # Apply comm_module switch (for TarMAC external baseline)
    if "comm_module" in perception_cfg and perception_cfg["comm_module"]:
        Config.COMM_MODULE = str(perception_cfg["comm_module"])

    # Ensure CLI overrides (e.g., --mask/--noise/--ttc_threshold) take precedence
    # over model-type personalized defaults during evaluation.
    _apply_overrides(args)
    
    print(f"\n[Eval] 应用 {model_type_str} 个性化配置: {perception_cfg['description']}")
    print(f"       MAX_NEIGHBORS={Config.MAX_NEIGHBORS} (保持模型架构), LIDAR_NUM_OTHERS={Config.LIDAR_NUM_OTHERS}")
    print(f"       MASK_RATIO={Config.MASK_RATIO:.2f}, NOISE_STD={Config.NOISE_STD:.2f}")

    # Capture maxima *after* any user overrides to Config (e.g. different map/reward settings)
    reward_maxima = _capture_reward_maxima_from_config()
    if bool(getattr(args, "eval_full_reward", False)):
        _apply_reward_params(reward_maxima)

    # Build env config
    eval_config = Config.get_metadrive_config()
    eval_config["start_seed"] = int(args.start_seed)
    eval_config["num_scenarios"] = int(args.episodes)
    eval_config["comm_mode"] = str(getattr(args, "comm_mode", "iid") or "iid").lower()
    eval_config["comm_burst_len"] = max(1, int(getattr(args, "burst_len", 1) or 1))
    eval_config["comm_stale_steps"] = max(0, int(getattr(args, "stale_steps", 0) or 0))

    sweep_name = None
    if isinstance(eval_overrides, dict) and eval_overrides:
        sweep_name = eval_overrides.get("sweep")
        if "num_agents" in eval_overrides and eval_overrides["num_agents"] is not None:
            eval_config["num_agents"] = int(eval_overrides["num_agents"])
        if "traffic_density" in eval_overrides and eval_overrides["traffic_density"] is not None:
            eval_config["traffic_density"] = float(eval_overrides["traffic_density"])

    paper_capture_enabled = bool(getattr(args, "paper_capture", False))
    paper_capture_dir = str(getattr(args, "paper_capture_dir", "logs/paper_attention_capture"))
    paper_max_captures = max(1, int(getattr(args, "paper_max_captures", 1)))
    paper_top_k = max(1, int(getattr(args, "paper_top_k", 3)))
    if paper_max_captures < paper_top_k:
        paper_max_captures = paper_top_k
    paper_trigger_step_min = int(getattr(args, "paper_trigger_step_min", 50))
    paper_trigger_speed_kmh = float(getattr(args, "paper_trigger_speed_kmh", 15.0))
    paper_focus_radius_m = float(getattr(args, "paper_focus_radius_m", 42.0))
    paper_temporal_window = max(20, int(getattr(args, "paper_temporal_window", 80)))
    paper_export_texture_view = bool(getattr(args, "paper_export_texture_view", False))

    want_record = bool(args.record_dir) and (bool(args.record_video) or int(args.screenshot_every) > 0)
    use_render = bool(args.render) or want_record
    eval_config["use_render"] = use_render

    if use_render:
        eval_config["window_size"] = (800, 600)
        eval_config["force_render_fps"] = 60

    if bool(args.render):
        # Make the onscreen window more obvious.
        eval_config["show_fps"] = True
        eval_config["show_interface"] = True

    _maybe_fix_vehicle_model_for_render(eval_config, args)

    # CLI override still applies for non-sweep runs
    if args.num_agents is not None and not (isinstance(eval_overrides, dict) and ("num_agents" in eval_overrides)):
        eval_config["num_agents"] = int(args.num_agents)

    # Map override
    if args.map_sequence is not None:
        map_value: Any = args.map_sequence
        try:
            if str(map_value).isdigit():
                map_value = int(map_value)
        except Exception:
            pass
        eval_config["map"] = map_value
        if "map_config" in eval_config:
            eval_config.pop("map_config")

    env = GraphEnvWrapper(config=eval_config)

    paper_capture_count = 0
    paper_candidates: List[Dict[str, Any]] = []
    if paper_capture_enabled:
        os.makedirs(paper_capture_dir, exist_ok=True)
        print(
            f"[PaperCapture] enabled | step>{paper_trigger_step_min}, speed>{paper_trigger_speed_kmh:.1f}km/h, max={paper_max_captures}, topk={paper_top_k}"
        )

    obs_dict, _ = env.reset(seed=int(args.start_seed))
    if not obs_dict:
        env.close()
        raise RuntimeError("Empty observation on reset. Check env config.")

    # Render one frame immediately so the window has a chance to appear.
    if bool(args.render):
        try:
            env.render()
        except Exception:
            pass
        pause_s = float(getattr(args, "pause_on_start", 0.0) or 0.0)
        if pause_s > 0:
            time.sleep(pause_s)

    sample_agent = list(obs_dict.keys())[0]
    input_dim = int(obs_dict[sample_agent]["node_features"].shape[0])

    if hasattr(env.action_space, "spaces"):
        action_dim = int(env.action_space[sample_agent].shape[0])
        low = env.action_space[sample_agent].low
        high = env.action_space[sample_agent].high
    else:
        action_dim = int(env.action_space.shape[0])
        low = env.action_space.low
        high = env.action_space.high

    is_mappo = model_type_str in {"mappo", "mappo_ips"}
    if is_mappo:
        policy = MAPPOPolicy(input_dim=input_dim, action_dim=action_dim, num_agents=Config.NUM_AGENTS).to(device)
    else:
        policy = CooperativePolicy(input_dim, action_dim).to(device)
    print(f"[Eval] Loading model from {model_path}")
    _load_checkpoint_into_policy(policy, model_path, device)
    policy.eval()

    # Metrics
    terminal_counts = defaultdict(int)
    total_finished = 0

    ep_agent_rewards: List[float] = []
    ep_agent_steps: List[int] = []

    risk_stats = defaultdict(float)
    graph_stats = defaultdict(float)
    aux_metrics = {"ade_sum": 0.0, "fde_sum": 0.0, "count": 0}

    # Observed mixed-traffic stats aggregated over env steps
    obs_mix = {"steps": 0, "rl_sum": 0.0, "bg_sum": 0.0, "mpr_sum": 0.0, "mpr_n": 0}

    # === [新增] 平滑度统计 (Smoothness / Jerk Metrics) ===
    # 衡量动作变化的剧烈程度，反映驾驶平滑性
    smoothness_stats = {
        "jerk_sum": 0.0,           # 累积动作变化幅度
        "jerk_count": 0,           # 统计次数
        "prev_actions": {},        # 上一步每个agent的动作
    }

    # === [新增] 通行效率统计 (Traffic Efficiency / Throughput) ===
    # 衡量单位时间内的任务完成量
    efficiency_stats = {
        "total_steps": 0,          # 总步数
        "speed_sum": 0.0,          # 累积速度
        "speed_count": 0,          # 速度统计次数
    }

    # === [新增] 横向稳定性统计 (Lateral Stability / Lane Keeping) ===
    # 衡量车辆偏离车道中心的程度
    lateral_stats = {
        "lane_deviation_sum": 0.0,  # 累积车道中心偏移量
        "lane_deviation_count": 0,  # 统计次数
    }

    # === [新增] 决策果断性统计 (Decisiveness / Idle Rate) ===
    # 衡量车辆停止或极低速行驶的时间占比
    decisiveness_stats = {
        "idle_steps": 0,             # 低速步数 (speed < 1 km/h)
        "total_steps": 0,            # 总步数
    }

    # Failure analysis
    want_details = bool(args.save_details_json)
    last_n = max(1, int(getattr(args, "last_n_steps", 30)))
    max_traces = max(0, int(getattr(args, "max_out_of_road_traces", 50)))
    details: Optional[Dict[str, Any]] = None
    if want_details:
        details = {
            "model_path": model_path,
            "episodes": int(args.episodes),
            "start_seed": int(args.start_seed),
            "eval_full_reward": bool(getattr(args, "eval_full_reward", False)),
            "noise": float(getattr(Config, "NOISE_STD", 0.0)),
            "mask": float(getattr(Config, "MASK_RATIO", 0.0)),
            "aggregate_terminal_counts": {},
            "episode_summaries": [],
        }

    # === [新增] 公平性与交互强度统计 (Fairness & Interaction Metrics) ===
    # 用于论文多智能体分析
    fairness_stats = {
        "agent_successes": defaultdict(list),      # 每个agent的成功记录
        "agent_step_stats": defaultdict(lambda: {
            "has_neighbor": 0, 
            "near_miss": 0, 
            "stopped": 0, 
            "total": 0
        }),
    }

    # Recording
    imageio = _maybe_get_imageio() if want_record else None
    if want_record and imageio is None:
        print("[Eval] Recording requested but imageio is not available; skipping recording.")

    if want_record:
        os.makedirs(args.record_dir, exist_ok=True)
        if bool(args.record_video):
            _ensure_mp4_writer_available(imageio, args.record_dir)

    for ep in range(int(args.episodes)):
        # 重置本回合的上一步动作记录
        smoothness_stats["prev_actions"].clear()
        seed = int(args.start_seed) + ep
        obs_dict, _ = env.reset(seed=seed)

        paper_alpha_history = deque(maxlen=paper_temporal_window)
        paper_pos_history: Dict[str, deque] = {}

        per_agent_return = defaultdict(float)
        per_agent_steps_local = defaultdict(int)

        # per-agent recent history for failure trace
        recent = defaultdict(lambda: deque(maxlen=last_n))
        out_of_road_traces: List[Dict[str, Any]] = []

        # episode-level aggregation
        ep_term_counts = defaultdict(int)
        speed_sum = 0.0
        speed_n = 0.0

        video_writer = None
        if want_record and imageio is not None and bool(args.record_video):
            video_path = os.path.join(args.record_dir, f"ep_{ep:04d}_seed_{seed}.mp4")
            try:
                video_writer = imageio.get_writer(video_path, fps=int(args.video_fps))
            except Exception as e:
                print(f"[Eval] Failed to open video writer ({video_path}): {e}")
                video_writer = None

        ep_done = False
        step_idx = 0
        topdown_error_printed = False
        while not ep_done:
            if args.max_steps is not None and step_idx >= int(args.max_steps):
                break

            for aid, veh in getattr(env, "agents", {}).items():
                if aid not in paper_pos_history:
                    paper_pos_history[aid] = deque(maxlen=paper_temporal_window)
                try:
                    pos = np.asarray(getattr(veh, "position", [0.0, 0.0]), dtype=np.float32)
                    paper_pos_history[aid].append([float(pos[0]), float(pos[1])])
                except Exception:
                    pass

            # Render / record frame
            # NOTE:
            # - MetaDrive's render(mode="top_down") is typically an offscreen top-down frame (for recording/analysis)
            #   and may NOT refresh the onscreen window.
            # - To ensure the visualization window shows up when --render is set, always drive onscreen render().
            frame = None
            if use_render:
                # Always refresh onscreen window when requested.
                if bool(args.render):
                    try:
                        env.render()
                    except Exception:
                        pass

                # Optionally capture a top-down frame for recording / screenshots.
                if bool(args.top_down) or want_record:
                    try:
                        # Prepare aux predicted waypoints overlay (world coords) if requested.
                        pred_wp_world = None
                        if getattr(args, "top_down_overlay", "pred_waypoints") == "pred_waypoints":
                            try:
                                # results is computed later in the loop; use the last computed one if present.
                                pred_wp_world = locals().get("_last_pred_wp_world", None)
                            except Exception:
                                pred_wp_world = None

                        # God-view top-down with interactive zoom/pan + history trails.
                        # Note: engine.render_topdown() passes kwargs to TopDownRenderer __init__ on first call
                        # and to TopDownRenderer.render() every call.
                        frame = env.render(
                            mode="top_down",
                            center_on_map=True,
                            # A slightly larger window helps show more context in multi-agent scenes.
                            screen_size=(1000, 1000),
                            # Choose overlay type
                            num_stack=(60 if getattr(args, "top_down_overlay", "pred_waypoints") == "history" else 1),
                            history_smooth=(2 if getattr(args, "top_down_overlay", "pred_waypoints") == "history" else 0),
                            draw_vehicle_trails=(getattr(args, "top_down_overlay", "pred_waypoints") == "history"),
                            vehicle_trail_width=2,
                            draw_pred_waypoints=(getattr(args, "top_down_overlay", "pred_waypoints") == "pred_waypoints"),
                            pred_waypoints_world=pred_wp_world,
                            pred_waypoints_color=(255, 0, 0),
                            pred_waypoints_radius=3,
                            pred_waypoints_line_width=2,
                        )
                    except Exception as e:
                        frame = None
                        if bool(args.top_down) and not topdown_error_printed:
                            topdown_error_printed = True
                            print(f"[Eval][TopDown] Top-down render failed (pygame window may not appear): {e!r}")

                if args.draw_connections:
                    _draw_communication_lines(env, obs_dict)

                # When visualizing (not recording), slow down so the window doesn't flash and close immediately.
                if bool(args.render) and not want_record:
                    sleep_s = getattr(args, "render_sleep", None)
                    if sleep_s is None:
                        sleep_s = 1.0 / 60.0
                    try:
                        time.sleep(float(sleep_s))
                    except Exception:
                        pass

            if want_record and imageio is not None:
                if video_writer is not None and frame is not None:
                    try:
                        video_writer.append_data(frame)
                    except Exception:
                        pass
                if int(args.screenshot_every) > 0 and (step_idx % int(args.screenshot_every) == 0) and frame is not None:
                    png_path = os.path.join(args.record_dir, f"ep_{ep:04d}_seed_{seed}_step_{step_idx:05d}.png")
                    try:
                        imageio.imwrite(png_path, frame)
                    except Exception:
                        pass

            obs_batch, active_agents = _build_batch_from_obs(obs_dict)
            if not active_agents:
                break

            # Graph stats: neighbors per ego
            try:
                graph_stats["steps"] += 1.0
                graph_stats["neighbors_mean"] += float(np.mean([len(obs_dict[a].get("neighbors", [])) for a in active_agents]))
            except Exception:
                pass

            obs_tensor = {k: v.to(device) for k, v in obs_batch.items()}

            with torch.no_grad():
                results = policy(obs_tensor, return_attention=paper_capture_enabled)

                if args.stochastic:
                    dist = torch.distributions.Normal(results["action_mean"], results["action_std"])
                    actions = dist.sample().cpu().numpy()
                else:
                    actions = results["action_mean"].cpu().numpy()

                # Aux waypoint prediction error
                try:
                    pred_wp = results["pred_waypoints"]
                    gt_wp = obs_tensor["gt_waypoints"]
                    err = torch.norm(pred_wp - gt_wp, dim=-1)
                    aux_metrics["ade_sum"] += float(err.mean().item()) * len(active_agents)
                    aux_metrics["fde_sum"] += float(err[:, -1].mean().item()) * len(active_agents)
                    aux_metrics["count"] += int(len(active_agents))
                except Exception:
                    pass

            alpha_maps_all: Dict[str, Dict[str, float]] = {}
            if paper_capture_enabled:
                for aid in active_agents:
                    alpha_maps_all[aid] = _ego_alpha_map_from_obs(
                        obs_dict=obs_dict,
                        active_agents=active_agents,
                        results=results,
                        ego_id=aid,
                    )

            # Convert predicted waypoints (ego-local) to world coords for visualization.
            # GT/pred waypoints are in ego-local coordinates per env_wrapper._get_future_waypoints().
            pred_waypoints_world_map = None
            _last_pred_wp_world = None
            should_decode_pred_wp = bool(
                paper_capture_enabled
                or (use_render and (bool(args.top_down) or want_record) and getattr(args, "top_down_overlay", "pred_waypoints") == "pred_waypoints")
            )
            if should_decode_pred_wp:
                try:
                    pred_wp_np = results.get("pred_waypoints").detach().cpu().numpy()  # (B, N, 2)
                    pred_map = {}
                    for i, aid in enumerate(active_agents):
                        if not hasattr(env, "agents") or aid not in getattr(env, "agents", {}):
                            continue
                        veh = env.agents[aid]
                        pos = np.asarray(getattr(veh, "position", [0.0, 0.0]), dtype=np.float32)
                        heading = float(getattr(veh, "heading_theta", 0.0))
                        c, s = np.cos(heading), np.sin(heading)
                        pts_world = []
                        for j in range(pred_wp_np.shape[1]):
                            lx = float(pred_wp_np[i, j, 0])
                            ly = float(pred_wp_np[i, j, 1])
                            dx = lx * c - ly * s
                            dy = lx * s + ly * c
                            pts_world.append([float(pos[0] + dx), float(pos[1] + dy)])
                        pred_map[str(aid)] = pts_world
                    pred_waypoints_world_map = pred_map
                    if use_render and (bool(args.top_down) or want_record) and getattr(args, "top_down_overlay", "pred_waypoints") == "pred_waypoints":
                        _last_pred_wp_world = {aid: {"points": pts} for aid, pts in pred_map.items()}
                except Exception:
                    pred_waypoints_world_map = None
                    _last_pred_wp_world = None
        
            action_dict = {}
            for i, agent_id in enumerate(active_agents):
                action_dict[agent_id] = np.clip(actions[i], low, high)

            next_obs, rewards, dones, infos = env.step(action_dict)

            # === [新增] 计算平滑度 (Jerk) ===
            # 对每个agent计算动作变化幅度
            for agent_id, action in action_dict.items():
                if agent_id in smoothness_stats["prev_actions"]:
                    prev_action = smoothness_stats["prev_actions"][agent_id]
                    # 计算动作变化的L1范数（Manhattan距离）
                    jerk = np.abs(action - prev_action).sum()
                    smoothness_stats["jerk_sum"] += jerk
                    smoothness_stats["jerk_count"] += 1
                # 更新上一步动作
                smoothness_stats["prev_actions"][agent_id] = action.copy()

            # Observed RL vs BG vehicles (best-effort)
            rl_cnt, bg_cnt, mpr = _count_observed_mixed_traffic(env)
            if rl_cnt is not None and bg_cnt is not None:
                obs_mix["steps"] += 1
                obs_mix["rl_sum"] += float(rl_cnt)
                obs_mix["bg_sum"] += float(bg_cnt)
            if mpr is not None:
                obs_mix["mpr_sum"] += float(mpr)
                obs_mix["mpr_n"] += 1

            # Accumulate per-agent return
            for aid in active_agents:
                per_agent_return[aid] += float(rewards.get(aid, 0.0))
                per_agent_steps_local[aid] += 1

            # Step-level info collection
            for aid in active_agents:
                info = infos.get(aid, {})
                if not isinstance(info, dict):
                    continue

                # Speed stats
                if "speed_kmh" in info:
                    try:
                        speed_val = float(info["speed_kmh"])
                        speed_sum += speed_val
                        speed_n += 1.0
                        # 通行效率统计
                        efficiency_stats["speed_sum"] += speed_val
                        efficiency_stats["speed_count"] += 1
                        # 决策果断性统计 (Idle Rate)
                        decisiveness_stats["total_steps"] += 1
                        if speed_val < 1.0:  # 低于1 km/h视为停止/极低速
                            decisiveness_stats["idle_steps"] += 1
                    except Exception:
                        pass

                # 横向稳定性统计 (Lane Keeping)
                if "abs_lane_lat" in info:
                    try:
                        lateral_stats["lane_deviation_sum"] += float(info["abs_lane_lat"])
                        lateral_stats["lane_deviation_count"] += 1
                    except Exception:
                        pass

                # Risk stats
                if "min_ttc_s" in info:
                    risk_stats["ttc_sum"] += float(info["min_ttc_s"])
                    risk_stats["ttc_count"] += 1.0
                    if float(info["min_ttc_s"]) < float(getattr(Config, "TTC_THRESHOLD_S", 2.5)):
                        risk_stats["ttc_risk_steps"] += 1.0

                if "min_neighbor_dist" in info:
                    risk_stats["dist_sum"] += float(info["min_neighbor_dist"])
                    risk_stats["dist_count"] += 1.0
                    if float(info["min_neighbor_dist"]) < float(getattr(Config, "SAFETY_DIST", 8.0)):
                        risk_stats["dist_risk_steps"] += 1.0

                # === [新增] 交互强度统计 (Interaction Intensity) ===
                # 收集每步的交互指标
                fairness_stats["agent_step_stats"][aid]["has_neighbor"] += info.get("has_neighbor", 0)
                fairness_stats["agent_step_stats"][aid]["near_miss"] += info.get("near_miss", 0)
                fairness_stats["agent_step_stats"][aid]["stopped"] += info.get("is_stopped", 0)
                fairness_stats["agent_step_stats"][aid]["total"] += 1

                # Failure trace buffer
                if want_details:
                    recent[aid].append(
                        {
                            "step": int(step_idx),
                            "speed_kmh": float(info.get("speed_kmh", 0.0)),
                            "min_ttc_s": info.get("min_ttc_s"),
                            "min_neighbor_dist": info.get("min_neighbor_dist"),
                            "abs_lane_lat": info.get("abs_lane_lat"),
                            "heading_err_rad": info.get("heading_err_rad"),
                            "reward_total": (info.get("reward_breakdown", {}) or {}).get("total"),
                        }
                    )

            # Terminal reasons
            for aid, done in dones.items():
                if aid == "__all__":
                    continue
                if bool(done):
                    total_finished += 1
                    info = infos.get(aid, {})
                    if isinstance(info, dict):
                        reason = info.get("terminal_reason")
                        if reason is None:
                            if info.get("arrive_dest", False):
                                reason = "success"
                            elif info.get("crash", False):
                                reason = "crash"
                            elif info.get("out_of_road", False):
                                reason = "out_of_road"
                            elif info.get("truncated", False):
                                reason = "timeout"
                            else:
                                reason = "other"
                    else:
                        reason = "other"

                    reason = str(reason)
                    terminal_counts[reason] += 1
                    ep_term_counts[reason] += 1

                    # === [新增] 公平性指标：记录每个agent的成功情况 ===
                    is_success = (reason == "success")
                    fairness_stats["agent_successes"][aid].append(int(is_success))

                    ep_agent_rewards.append(float(per_agent_return.get(aid, 0.0)))
                    ep_agent_steps.append(int(per_agent_steps_local.get(aid, 0)))

                    # Save out_of_road traces
                    if want_details and reason == "out_of_road" and len(out_of_road_traces) < max_traces:
                        out_of_road_traces.append(
                            {
                                "agent_id": aid,
                                "seed": seed,
                                "last_steps": list(recent.get(aid, [])),
                            }
                        )

            # Paper capture hook: avoid step 0, require step>threshold and ego speed>threshold.
            if (
                paper_capture_enabled
                and paper_capture_count < paper_max_captures
                and step_idx > int(paper_trigger_step_min)
            ):
                ego_for_capture = None
                best_ttc = float("inf")
                for aid in active_agents:
                    info_i = infos.get(aid, {}) if isinstance(infos.get(aid, {}), dict) else {}
                    speed_i = float(info_i.get("speed_kmh", 0.0) or 0.0)
                    ttc_i = float(info_i.get("min_ttc_s", 999.0) or 999.0)
                    if speed_i > float(paper_trigger_speed_kmh) and ttc_i < best_ttc:
                        best_ttc = ttc_i
                        ego_for_capture = aid

                if ego_for_capture is None:
                    ego_info = {}
                else:
                    ego_info = infos.get(ego_for_capture, {}) if isinstance(infos.get(ego_for_capture, {}), dict) else {}

                ego_speed = float(ego_info.get("speed_kmh", 0.0) or 0.0)
                min_ttc_now = float(ego_info.get("min_ttc_s", 999.0) or 999.0)
                alpha_map_ego = dict(alpha_maps_all.get(ego_for_capture, {})) if ego_for_capture is not None else {}

                paper_alpha_history.append(
                    {
                        "step": int(step_idx),
                        "ego_id": str(ego_for_capture) if ego_for_capture is not None else None,
                        "alpha_map": dict(alpha_map_ego),
                        "min_ttc_s": min_ttc_now,
                    }
                )

                if ego_for_capture is not None and ego_speed > float(paper_trigger_speed_kmh):
                    danger_id, safe_id, conflict_score = _pick_conflict_pair_from_obs(obs_dict, ego_for_capture, alpha_map_ego)
                    danger_alpha = float(alpha_map_ego.get(str(danger_id), 0.0)) if danger_id is not None else 0.0
                    safe_alpha = float(alpha_map_ego.get(str(safe_id), 1.0)) if safe_id is not None else 1.0
                    polarized = (
                        (danger_alpha >= 0.40)
                        and (safe_alpha <= 0.30)
                        and ((danger_alpha - safe_alpha) >= 0.20)
                        and (danger_id is not None)
                        and (safe_id is not None)
                        and (min_ttc_now <= 5.0)
                    )

                    if polarized and conflict_score > 0.8:
                        try:
                            ego_state = _vehicle_state_from_env(env, ego_for_capture)
                            snapshot = extract_attention_snapshot(
                                env,
                                ego_id=str(ego_for_capture),
                                alpha_map=alpha_map_ego,
                                history_by_agent=paper_pos_history,
                                pred_waypoints_world=(pred_waypoints_world_map or {}).get(str(ego_for_capture)),
                            )

                            danger_series = []
                            safe_series = []
                            ttc_series = []
                            for rec in list(paper_alpha_history):
                                if rec.get("ego_id") != str(ego_for_capture):
                                    continue
                                s = float(rec["step"])
                                amap = rec.get("alpha_map", {})
                                ttc_series.append((s, float(rec.get("min_ttc_s", 999.0))))
                                if danger_id is not None and str(danger_id) in amap:
                                    danger_series.append((s, float(amap[str(danger_id)])))
                                if safe_id is not None and str(safe_id) in amap:
                                    safe_series.append((s, float(amap[str(safe_id)])))

                            polarization_score = float((danger_alpha - safe_alpha) * (ego_speed / 20.0) * (1.0 / max(0.25, min_ttc_now)))

                            stem = os.path.join(
                                paper_capture_dir,
                                f"paper_ep{ep:03d}_step{step_idx:04d}_{ego_for_capture}",
                            )
                            dual_pdf = stem + "_dual_panel.pdf"
                            dual_png = stem + "_dual_panel.png"
                            title = (
                                f"Conflict Snapshot | ep={ep} step={step_idx} | ego={ego_for_capture} "
                                f"| danger={danger_id} α={danger_alpha:.2f} | safe={safe_id} α={safe_alpha:.2f}"
                            )

                            spatial_meta = plot_attention_dual_panel_pdf(
                                snapshot=snapshot,
                                output_pdf=dual_pdf,
                                output_png=dual_png,
                                title=title,
                                danger_series=danger_series,
                                safe_series=safe_series,
                                ttc_series=ttc_series,
                                danger_label=str(danger_id),
                                safe_label=str(safe_id),
                                focus_radius_m=float(paper_focus_radius_m),
                                ttc_threshold=float(getattr(Config, "TTC_THRESHOLD_S", 3.0)),
                            )
                            candidate_metrics = _paper_candidate_score(
                                polarization_score=float(polarization_score),
                                conflict_score=float(conflict_score),
                                danger_series=danger_series,
                                safe_series=safe_series,
                                ttc_series=ttc_series,
                                spatial_meta=spatial_meta,
                            )

                            texture_png = None
                            if paper_export_texture_view:
                                texture = frame
                                if texture is None:
                                    try:
                                        texture = env.render(
                                            mode="top_down",
                                            center_on_map=True,
                                            screen_size=(1200, 1200),
                                            num_stack=1,
                                            draw_vehicle_trails=False,
                                            draw_pred_waypoints=False,
                                        )
                                    except Exception:
                                        texture = None
                                if texture is not None:
                                    texture_png = stem + "_texture_view.png"
                                    _save_rgb_png(texture_png, np.asarray(texture, dtype=np.uint8))

                            detail = {
                                "episode": int(ep),
                                "step": int(step_idx),
                                "seed": int(seed),
                                "trigger": {
                                    "step_gt": int(paper_trigger_step_min),
                                    "speed_gt_kmh": float(paper_trigger_speed_kmh),
                                    "ego_speed_kmh": ego_speed,
                                    "ego_min_ttc_s": min_ttc_now,
                                    "danger_id": str(danger_id),
                                    "safe_id": str(safe_id),
                                    "danger_alpha": danger_alpha,
                                    "safe_alpha": safe_alpha,
                                    "conflict_score": float(conflict_score),
                                    "polarization_score": polarization_score,
                                },
                                "ego_state": ego_state,
                                "attention_map": dict(alpha_map_ego),
                                "spatial_panel": dict(spatial_meta),
                                "candidate_metrics": dict(candidate_metrics),
                                "danger_series": [{"step": float(s), "alpha": float(a)} for s, a in danger_series],
                                "safe_series": [{"step": float(s), "alpha": float(a)} for s, a in safe_series],
                                "ttc_series": [{"step": float(s), "ttc_s": float(t)} for s, t in ttc_series],
                                "outputs": {
                                    "dual_pdf": dual_pdf,
                                    "dual_png": dual_png,
                                    "texture_png": texture_png,
                                },
                            }
                            json_path = stem + "_paper_detail.json"
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(_to_jsonable(detail), f, ensure_ascii=False, indent=2)

                            print(f"[PaperCapture] dual panel saved: {dual_pdf}")
                            print(f"[PaperCapture] detail json: {json_path}")
                            print(f"[PaperCapture] polarization_score={polarization_score:.4f}")
                            print(f"[PaperCapture] paper_score={float(candidate_metrics.get('paper_score', 0.0)):.4f}")

                            paper_candidates.append(
                                {
                                    "polarization_score": polarization_score,
                                    "paper_score": float(candidate_metrics.get("paper_score", 0.0)),
                                    "candidate_metrics": dict(candidate_metrics),
                                    "conflict_score": float(conflict_score),
                                    "episode": int(ep),
                                    "step": int(step_idx),
                                    "ego_id": str(ego_for_capture),
                                    "danger_id": str(danger_id),
                                    "safe_id": str(safe_id),
                                    "dual_pdf": dual_pdf,
                                    "dual_png": dual_png,
                                    "texture_png": texture_png,
                                    "detail_json": json_path,
                                }
                            )
                            paper_capture_count += 1
                        except Exception as e:
                            print(f"[PaperCapture] capture failed: {e}")

            obs_dict = next_obs
            ep_done = bool(dones.get("__all__", False)) or (len(obs_dict) == 0)
            step_idx += 1
            efficiency_stats["total_steps"] += 1  # 累计总步数

        if video_writer is not None:
            try:
                video_writer.close()
            except Exception:
                pass

        if want_details and details is not None:
            details["episode_summaries"].append(
                {
                    "episode": int(ep),
                    "seed": int(seed),
                    "terminal_counts": dict(ep_term_counts),
                    "avg_speed_kmh": (float(speed_sum) / max(1.0, float(speed_n))),
                    "out_of_road_traces": out_of_road_traces,
                }
            )

    env.close()

    if paper_capture_enabled and paper_candidates:
        ranked = sorted(
            paper_candidates,
            key=lambda x: (
                float(x.get("paper_score", 0.0)),
                float(x.get("polarization_score", 0.0)),
                float(x.get("conflict_score", 0.0)),
            ),
            reverse=True,
        )
        topk = ranked[: min(len(ranked), int(paper_top_k))]
        topk_records: List[Dict[str, Any]] = []
        for i, item in enumerate(topk, start=1):
            rank_prefix = os.path.join(
                paper_capture_dir,
                f"TOP{i:02d}_ep{int(item['episode']):03d}_step{int(item['step']):04d}_{item['ego_id']}",
            )
            ranked_pdf = rank_prefix + "_dual_panel.pdf"
            ranked_png = rank_prefix + "_dual_panel.png"
            ranked_tex = rank_prefix + "_texture_view.png"
            ranked_json = rank_prefix + "_paper_detail.json"

            if item.get("dual_pdf") and os.path.exists(str(item["dual_pdf"])):
                shutil.copy2(str(item["dual_pdf"]), ranked_pdf)
            if item.get("dual_png") and os.path.exists(str(item["dual_png"])):
                shutil.copy2(str(item["dual_png"]), ranked_png)
            if item.get("texture_png") and os.path.exists(str(item["texture_png"])):
                shutil.copy2(str(item["texture_png"]), ranked_tex)
            if item.get("detail_json") and os.path.exists(str(item["detail_json"])):
                shutil.copy2(str(item["detail_json"]), ranked_json)

            topk_records.append(
                {
                    "rank": int(i),
                    **item,
                    "ranked_outputs": {
                        "dual_pdf": ranked_pdf,
                        "dual_png": ranked_png,
                        "texture_png": ranked_tex if os.path.exists(ranked_tex) else None,
                        "detail_json": ranked_json,
                    },
                }
            )

        topk_summary_path = os.path.join(paper_capture_dir, "paper_topk_summary.json")
        with open(topk_summary_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable({"topk": topk_records}), f, ensure_ascii=False, indent=2)
        print(f"[PaperCapture] Top-{len(topk_records)} summary: {topk_summary_path}")

    if bool(args.render) and bool(getattr(args, "pause_at_end", False)):
        try:
            input("[Eval][Render] Press Enter to exit...")
        except Exception:
            # Non-interactive terminal
            pass

    success = terminal_counts.get("success", 0)
    crash = terminal_counts.get("crash", 0)

    # === [新增] 计算平滑度统计 ===
    jerk_mean = (
        smoothness_stats["jerk_sum"] / max(1, smoothness_stats["jerk_count"])
        if smoothness_stats["jerk_count"] > 0
        else 0.0
    )
    out_of_road = terminal_counts.get("out_of_road", 0)
    timeout = terminal_counts.get("timeout", 0)

    ttc_avg = float(risk_stats["ttc_sum"]) / max(1.0, float(risk_stats["ttc_count"]))
    dist_avg = float(risk_stats["dist_sum"]) / max(1.0, float(risk_stats["dist_count"]))
    ttc_risk_rate = float(risk_stats["ttc_risk_steps"]) / max(1.0, float(risk_stats["ttc_count"]))
    dist_risk_rate = float(risk_stats["dist_risk_steps"]) / max(1.0, float(risk_stats["dist_count"]))

    neighbors_mean = None
    if float(graph_stats.get("steps", 0.0)) > 0:
        neighbors_mean = float(graph_stats.get("neighbors_mean", 0.0)) / float(graph_stats["steps"])

    ade = aux_metrics["ade_sum"] / max(1, aux_metrics["count"])
    fde = aux_metrics["fde_sum"] / max(1, aux_metrics["count"])

    reward_mean = float(np.mean(ep_agent_rewards)) if ep_agent_rewards else float("nan")
    reward_std = float(np.std(ep_agent_rewards)) if ep_agent_rewards else float("nan")
    steps_mean = float(np.mean(ep_agent_steps)) if ep_agent_steps else float("nan")

    observed_rl_mean = (obs_mix["rl_sum"] / obs_mix["steps"]) if obs_mix["steps"] > 0 else None
    observed_bg_mean = (obs_mix["bg_sum"] / obs_mix["steps"]) if obs_mix["steps"] > 0 else None
    observed_mpr_mean = (obs_mix["mpr_sum"] / obs_mix["mpr_n"]) if obs_mix["mpr_n"] > 0 else None

    # === [新增] 计算通行效率指标 ===
    # Throughput: 成功数 / 总步数 (越高越好)
    throughput = (float(success) / max(1.0, float(efficiency_stats["total_steps"]))) if efficiency_stats["total_steps"] > 0 else 0.0
    # 平均速度 (km/h)
    avg_speed_kmh = (efficiency_stats["speed_sum"] / max(1.0, float(efficiency_stats["speed_count"]))) if efficiency_stats["speed_count"] > 0 else 0.0
    # 完成效率: 成功率 * 平均速度 (综合指标)
    completion_efficiency = (success / max(1, total_finished)) * avg_speed_kmh if total_finished > 0 else 0.0

    # === [新增] 计算横向稳定性指标 ===
    # 平均车道偏移量 (米，越小越好)
    avg_lane_deviation = (
        lateral_stats["lane_deviation_sum"] / max(1.0, float(lateral_stats["lane_deviation_count"]))
        if lateral_stats["lane_deviation_count"] > 0
        else 0.0
    )

    # === [新增] 计算决策果断性指标 ===
    # Idle Rate: 低速步数占比 (越低越果断)
    idle_rate = (
        float(decisiveness_stats["idle_steps"]) / max(1.0, float(decisiveness_stats["total_steps"]))
        if decisiveness_stats["total_steps"] > 0
        else 0.0
    )

    # === [新增] 计算公平性与交互强度指标 ===
    # 1. 公平性指标（Fairness Metrics）
    fairness_output = {}
    if fairness_stats["agent_successes"]:
        per_agent_sr = {
            aid: np.mean(eps) if eps else 0.0
            for aid, eps in fairness_stats["agent_successes"].items()
        }
        if per_agent_sr:
            fairness_output["worst_agent_success"] = float(min(per_agent_sr.values()))
            fairness_output["best_agent_success"] = float(max(per_agent_sr.values()))
            fairness_output["success_variance"] = float(np.var(list(per_agent_sr.values())))
            fairness_output["success_std"] = float(np.std(list(per_agent_sr.values())))
    
    # 2. 交互强度指标（Interaction Intensity）
    total_steps = sum(s["total"] for s in fairness_stats["agent_step_stats"].values())
    if total_steps > 0:
        total_has_neighbor = sum(s["has_neighbor"] for s in fairness_stats["agent_step_stats"].values())
        total_near_miss = sum(s["near_miss"] for s in fairness_stats["agent_step_stats"].values())
        total_stopped = sum(s["stopped"] for s in fairness_stats["agent_step_stats"].values())
        
        fairness_output["interaction_rate"] = float(total_has_neighbor) / float(total_steps)
        fairness_output["near_miss_rate"] = float(total_near_miss) / float(total_steps)
        fairness_output["stop_rate"] = float(total_stopped) / float(total_steps)

    summary = {
        "model_path": model_path,
        "sweep": sweep_name,
        "episodes": int(args.episodes),
        "start_seed": int(args.start_seed),
        "num_agents": int(eval_config.get("num_agents", (args.num_agents or 0)) or 0),
        "traffic_density": float(eval_config.get("traffic_density", 0.0) or 0.0),
        "observed": {
            "rl_vehicles_mean": observed_rl_mean,
            "bg_vehicles_mean": observed_bg_mean,
            "mpr_mean": observed_mpr_mean,
            "steps": int(obs_mix["steps"]),
        },
        "deterministic": (not bool(args.stochastic)),
        "eval_full_reward": bool(getattr(args, "eval_full_reward", False)),
        "robustness": {
            "comm_mode": str(eval_config.get("comm_mode", "iid")),
            "mask_ratio": float(getattr(Config, "MASK_RATIO", 0.0) or 0.0),
            "noise_std": float(getattr(Config, "NOISE_STD", 0.0) or 0.0),
            "burst_len": int(eval_config.get("comm_burst_len", 1) or 1),
            "stale_steps": int(eval_config.get("comm_stale_steps", 0) or 0),
        },
        "terminal_counts": dict(terminal_counts),
        "rates": {
            "success": (success / total_finished) if total_finished else 0.0,
            "crash": (crash / total_finished) if total_finished else 0.0,
            "out_of_road": (out_of_road / total_finished) if total_finished else 0.0,
            "timeout": (timeout / total_finished) if total_finished else 0.0,
        },
        "return": {
            "mean": reward_mean,
            "std": reward_std,
        },
        "steps": {
            "mean": steps_mean,
        },
        "risk": {
            "avg_min_ttc_s": ttc_avg,
            "high_risk_ttc_rate": ttc_risk_rate,
            "avg_min_dist_m": dist_avg,
            "high_risk_dist_rate": dist_risk_rate,
        },
        "graph": {
            "neighbors_mean": neighbors_mean,
        },
        "aux": {
            "ade": ade,
            "fde": fde,
        },
        "smoothness": {
            "jerk_mean": jerk_mean,  # 新增：平滑度指标（动作变化幅度）
        },
        "efficiency": {
            "throughput": throughput,              # 成功数/总步数
            "avg_speed_kmh": avg_speed_kmh,        # 平均速度
            "completion_efficiency": completion_efficiency,  # 成功率*平均速度
        },
        "lateral_stability": {
            "avg_lane_deviation_m": avg_lane_deviation,  # 平均车道中心偏移量
        },
        "decisiveness": {
            "idle_rate": idle_rate,  # 低速步数占比 (Speed < 1km/h)
        },
        "fairness": fairness_output,  # 新增：公平性与交互强度指标
    }

    if want_details and details is not None:
        details["aggregate_terminal_counts"] = dict(terminal_counts)
    return summary, details


def _build_batch_from_obs(obs_dict: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    active_agents = list(obs_dict.keys())
    agent_to_idx = {a_id: i for i, a_id in enumerate(active_agents)}

    batch_node_features = []
    batch_neighbor_indices = []
    batch_neighbor_mask = []
    batch_neighbor_rel_pos = []
    batch_gt_waypoints = []

    for agent_id in active_agents:
        agent_obs = obs_dict[agent_id]
        batch_node_features.append(agent_obs["node_features"])
        batch_gt_waypoints.append(
            agent_obs.get(
                "gt_waypoints",
                np.zeros((Config.PRED_WAYPOINTS_NUM, 2), dtype=np.float32),
            )
        )

        n_indices: List[int] = []
        n_mask: List[float] = []
        n_rel_pos: List[List[float]] = []

        raw_neighbors = agent_obs.get("neighbors", [])
        raw_rel_pos = agent_obs.get("neighbor_rel_pos", [])

        for i, n_id in enumerate(raw_neighbors):
            if n_id in agent_to_idx:
                n_indices.append(agent_to_idx[n_id])
                n_mask.append(1.0)
                if i < len(raw_rel_pos):
                    n_rel_pos.append(raw_rel_pos[i])
                else:
                    n_rel_pos.append([0.0, 0.0, 0.0, 0.0])

        while len(n_indices) < int(getattr(Config, "MAX_NEIGHBORS", 8)):
            # Use -1 for padding so "no neighbor" never aliases a real batch item.
            n_indices.append(-1)
            n_mask.append(0.0)
            n_rel_pos.append([0.0, 0.0, 0.0, 0.0])

        max_n = int(getattr(Config, "MAX_NEIGHBORS", 8))
        batch_neighbor_indices.append(n_indices[:max_n])
        batch_neighbor_mask.append(n_mask[:max_n])
        batch_neighbor_rel_pos.append(n_rel_pos[:max_n])

    obs_tensor_batch = {
        "node_features": torch.tensor(np.array(batch_node_features), dtype=torch.float32),
        "neighbor_indices": torch.tensor(np.array(batch_neighbor_indices), dtype=torch.long),
        "neighbor_mask": torch.tensor(np.array(batch_neighbor_mask), dtype=torch.float32),
        "neighbor_rel_pos": torch.tensor(np.array(batch_neighbor_rel_pos), dtype=torch.float32),
        "gt_waypoints": torch.tensor(np.array(batch_gt_waypoints), dtype=torch.float32),
    }

    return obs_tensor_batch, active_agents



def _draw_communication_lines(env: GraphEnvWrapper, obs_dict: Dict[str, Dict[str, Any]]) -> None:
    if not hasattr(env, "engine") or env.engine is None:
        return
    try:
        if hasattr(env.engine, "clear_lines"):
            env.engine.clear_lines()
    except Exception:
        pass

    for agent_id, obs in obs_dict.items():
        if agent_id not in getattr(env, "agents", {}):
            continue
        start_pos = env.agents[agent_id].position
        for n_id in obs.get("neighbors", []):
            if n_id in getattr(env, "agents", {}):
                end_pos = env.agents[n_id].position
                try:
                    env.engine.add_line(start_pos, end_pos, (0, 1, 0, 0.8), 2.0)
                except Exception:
                    pass


def _safe_torch_load_state_dict(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    # Prefer weights_only=True when available to avoid pickle execution
    try:
        return torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(model_path, map_location=device)


def _load_checkpoint_into_policy(policy: torch.nn.Module, model_path: str, device: torch.device) -> None:
    state_dict = _safe_torch_load_state_dict(model_path, device)

    model_sd = policy.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state_dict.items():
        if k in model_sd and hasattr(model_sd[k], "shape") and hasattr(v, "shape") and model_sd[k].shape != v.shape:
            dropped += 1
            continue
        if k in model_sd:
            filtered[k] = v

    missing, unexpected = policy.load_state_dict(filtered, strict=False)
    if dropped or missing or unexpected:
        print(f"[Eval] Loaded with relaxed strictness. dropped={dropped}, missing={len(missing)}, unexpected={len(unexpected)}")


def evaluate() -> None:
    args = parse_args()

    # 修复 Windows 终端 UTF-8 编码问题
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    if not args.disable_patches:
        _apply_metadrive_patches()

    # === [关键] 应用CLI覆盖（如--noise, --mask等） ===
    # 注意：这会覆盖评估配置中的默认值
    _apply_overrides(args)
    
    # 打印评估配置信息
    print("\n" + "="*60)
    print("📊 评估配置 (Evaluation Configuration)")
    print("="*60)
    print(f"模型类型检测: {args.model_type if hasattr(args, 'model_type') else 'auto'}")
    if args.model_path or args.model_glob:
        print(f"模型路径: {args.model_path or args.model_glob}")
    print(f"评估种子起点 (start_seed): {args.start_seed}")
    print(f"评估回合数 (episodes): {args.episodes}")
    print(f"确定性策略 (deterministic): {not args.stochastic}")
    print("\n📌 个性化配置说明:")
    print("  - 每个模型使用其训练时的感知配置进行评估")
    print("  - 'ours': GAT通信 + 物理盲雷达")
    print("  - 'no_comm': 无通信 + 雷达可见邻居")
    print("  - 'lidar_only': 无V2V通信 + 雷达可见邻居")
    print("  - 'oracle': 全局信息 + 无雷达")
    print("  - 所有模型使用相同的奖励参数（最大值）")
    print("  - 详细配置将在每个模型评估时打印")
    print("="*60 + "\n")

    # Parse MPR sweep configs if requested
    mpr_sweep_cfgs: List[Dict[str, Any]] = []
    try:
        mpr_sweep_cfgs = _normalize_mpr_sweep_configs(_maybe_load_json_arg(getattr(args, "mpr_sweep", None)))
    except Exception as e:
        raise ValueError(f"Invalid --mpr_sweep: {e}")
    # Note: --mpr_sweep supports both --model_path and --model_glob.

    # Resolve model list
    model_paths: List[str] = []
    if args.model_glob:
        model_paths = sorted(glob.glob(args.model_glob), key=_sort_key_for_ckpt)
        if not model_paths:
            raise FileNotFoundError(f"No checkpoints matched --model_glob: {args.model_glob}")
    elif args.model_path:
        model_paths = [args.model_path]
    else:
        raise ValueError("Must provide either --model_path or --model_glob")

    # Evaluate
    summaries: List[Dict[str, Any]] = []
    details_list: List[Dict[str, Any]] = []

    print(f"[Eval] Device: {torch.device(args.device if torch.cuda.is_available() else 'cpu')}")
    for mpth in model_paths:
        if not os.path.exists(mpth):
            raise FileNotFoundError(f"Checkpoint not found: {mpth}")
        if mpr_sweep_cfgs:
            print(f"[Eval] Running MPR sweep with {len(mpr_sweep_cfgs)} configs")
            for cfg in mpr_sweep_cfgs:
                overrides = {
                    "sweep": str(cfg.get("name", "mpr")),
                    "num_agents": int(cfg["num_agents"]),
                    "traffic_density": float(cfg["traffic_density"]),
                }
                print(
                    f"\n[Eval][Sweep] {overrides['sweep']}: num_agents={overrides['num_agents']} "
                    f"traffic_density={overrides['traffic_density']:.3f}"
                )
                summary, details = evaluate_single_model(mpth, args, eval_overrides=overrides)
                summaries.append(summary)
                if details is not None:
                    details_list.append(details)

                rates = summary.get("rates", {})
                total = sum(summary.get("terminal_counts", {}).values())
                print("\n" + "=" * 44)
                print(
                    f"EVALUATION REPORT | sweep={overrides['sweep']} | ckpt={os.path.basename(mpth)} | AgentsFinished={total}"
                )
                print("=" * 44)
                print(f"Success Rate:      {rates.get('success', 0.0):.2%}")
                print(f"Crash Rate:        {rates.get('crash', 0.0):.2%}")
                print(f"Out of Road Rate:  {rates.get('out_of_road', 0.0):.2%}")
                print(f"Timeout Rate:      {rates.get('timeout', 0.0):.2%}")
                print(f"Smoothness (Jerk): {summary.get('smoothness', {}).get('jerk_mean', float('nan')):.4f}")
                print(f"High TTC Rate:     {summary.get('risk', {}).get('high_risk_ttc_rate', 0.0):.2%}")
                print(f"Avg Speed:         {summary.get('efficiency', {}).get('avg_speed_kmh', float('nan')):.2f} km/h")
                print(f"Lane Deviation:    {summary.get('lateral_stability', {}).get('avg_lane_deviation_m', float('nan')):.3f} m")
                print(f"Idle Rate:         {summary.get('decisiveness', {}).get('idle_rate', 0.0):.2%} (Speed < 1km/h)")
                obs = summary.get("observed", {}) if isinstance(summary.get("observed", {}), dict) else {}
                if obs.get("mpr_mean") is not None:
                    print(f"Observed MPR(mean): {float(obs.get('mpr_mean')):.2%} (RL/(RL+BG))")
                print("=" * 44)
        else:
            summary, details = evaluate_single_model(mpth, args)
            summaries.append(summary)
            if details is not None:
                details_list.append(details)

            # Print per-model report (compact)
            rates = summary.get("rates", {})
            total = sum(summary.get("terminal_counts", {}).values())
            print("\n" + "=" * 44)
            print(f"EVALUATION REPORT | ckpt={os.path.basename(mpth)} | AgentsFinished={total}")
            print("=" * 44)
            print(f"Success Rate:      {rates.get('success', 0.0):.2%}")
            print(f"Crash Rate:        {rates.get('crash', 0.0):.2%}")
            print(f"Out of Road Rate:  {rates.get('out_of_road', 0.0):.2%}")
            print(f"Timeout Rate:      {rates.get('timeout', 0.0):.2%}")
            print(f"Smoothness (Jerk): {summary.get('smoothness', {}).get('jerk_mean', float('nan')):.4f}")
            print(f"High TTC Rate:     {summary.get('risk', {}).get('high_risk_ttc_rate', 0.0):.2%}")
            print(f"Avg Speed:         {summary.get('efficiency', {}).get('avg_speed_kmh', float('nan')):.2f} km/h")
            print(f"Lane Deviation:    {summary.get('lateral_stability', {}).get('avg_lane_deviation_m', float('nan')):.3f} m")
            print(f"Idle Rate:         {summary.get('decisiveness', {}).get('idle_rate', 0.0):.2%} (Speed < 1km/h)")
            print("-" * 44)
            print(f"Return (per agent): mean={summary.get('return', {}).get('mean', float('nan')):.2f} std={summary.get('return', {}).get('std', float('nan')):.2f}")
            print(f"Steps  (per agent): mean={summary.get('steps', {}).get('mean', float('nan')):.1f}")
            neigh = summary.get("graph", {}).get("neighbors_mean")
            if neigh is not None:
                print(f"Graph neighbors/step: mean={float(neigh):.2f}")
            print("-" * 44)
            print(f"Avg Min TTC:       {summary.get('risk', {}).get('avg_min_ttc_s', float('nan')):.2f} s")
            print(f"High Risk TTC %:   {summary.get('risk', {}).get('high_risk_ttc_rate', 0.0):.2%} (Steps < {float(getattr(Config, 'TTC_THRESHOLD_S', 2.5))}s)")
            print(f"Avg Min Dist:      {summary.get('risk', {}).get('avg_min_dist_m', float('nan')):.2f} m")
            print(f"High Risk Dist %:  {summary.get('risk', {}).get('high_risk_dist_rate', 0.0):.2%} (Steps < {float(getattr(Config, 'SAFETY_DIST', 8.0))}m)")
            print("-" * 44)
            print(f"Intent Pred ADE:   {summary.get('aux', {}).get('ade', float('nan')):.3f} m")
            print(f"Intent Pred FDE:   {summary.get('aux', {}).get('fde', float('nan')):.3f} m")
            obs = summary.get("observed", {}) if isinstance(summary.get("observed", {}), dict) else {}
            if obs.get("mpr_mean") is not None:
                print(f"Observed MPR(mean): {float(obs.get('mpr_mean')):.2%} (RL/(RL+BG))")
            print("=" * 44)

    # Batch comparison table (always build rows if requested or using model_glob)
    want_table = bool(args.model_glob) or bool(args.save_table)
    if want_table:
        rows = []
        for s in summaries:
            bn = os.path.basename(s["model_path"]) if s.get("model_path") else "-"
            ckpt_num, _ = _sort_key_for_ckpt(bn)
            rates = s.get("rates", {})
            rows.append(
                {
                    "model_path": s.get("model_path"),
                    "ckpt": (ckpt_num if ckpt_num < 10**18 else bn.replace(".pth", "")),
                    "sweep": s.get("sweep"),
                    "episodes": s.get("episodes"),
                    "start_seed": s.get("start_seed"),
                    "num_agents": s.get("num_agents"),
                    "traffic_density": s.get("traffic_density"),
                    "comm_mode": (s.get("robustness", {}) or {}).get("comm_mode"),
                    "mask_ratio": (s.get("robustness", {}) or {}).get("mask_ratio"),
                    "noise_std": (s.get("robustness", {}) or {}).get("noise_std"),
                    "burst_len": (s.get("robustness", {}) or {}).get("burst_len"),
                    "stale_steps": (s.get("robustness", {}) or {}).get("stale_steps"),
                    "observed_mpr_mean": (s.get("observed", {}) or {}).get("mpr_mean"),
                    "observed_rl_vehicles_mean": (s.get("observed", {}) or {}).get("rl_vehicles_mean"),
                    "observed_bg_vehicles_mean": (s.get("observed", {}) or {}).get("bg_vehicles_mean"),
                    "success": rates.get("success"),
                    "crash": rates.get("crash"),
                    "out_of_road": rates.get("out_of_road"),
                    "timeout": rates.get("timeout"),
                    "return_mean": s.get("return", {}).get("mean"),
                    "return_std": s.get("return", {}).get("std"),
                    "steps_mean": s.get("steps", {}).get("mean"),
                    "neighbors_mean": s.get("graph", {}).get("neighbors_mean"),
                    "ade": s.get("aux", {}).get("ade"),
                    "fde": s.get("aux", {}).get("fde"),
                }
            )
        rows = sorted(rows, key=lambda r: (r.get("ckpt", 10**18), str(r.get("model_path", ""))))
        _print_table(rows)
        if args.save_table:
            _write_csv(args.save_table, rows)
            print(f"[Eval] Saved comparison table to {args.save_table}")

    # Save JSON outputs
    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        if len(summaries) == 1:
            payload: Any = summaries[0]
        else:
            payload = {"runs": summaries}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        print(f"[Eval] Saved summary to {args.save_json}")

    if args.save_details_json and details_list:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_details_json)) or ".", exist_ok=True)
        payload: Any = details_list[0] if len(details_list) == 1 else {"runs": details_list}
        with open(args.save_details_json, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)
        print(f"[Eval] Saved details to {args.save_details_json}")


if __name__ == "__main__":
    evaluate()
