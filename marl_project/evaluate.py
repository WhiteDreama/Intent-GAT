"""
模型评估主程序 (Evaluation Entrypoint)

【功能说明】
1. 加载训练好的模型权重 (.pth)。
2. 在 MetaDrive 环境中运行指定次数的测试。
3. 生成三种类型的数据供下游工具分析：
   - Summary JSON: 总体评分 (供 print_report.py 使用)
   - Details JSON: 详细轨迹 (供 analysis_tools.py 使用)
   - CSV Table:    批量对比表 (供 plot_benchmark.py 使用)
4. 支持实时渲染和视频录制。

【终端用法示例】

1. 单个模型测试 (生成详细数据用于分析):
   python marl_project/evaluate.py --model_path logs/exp/ckpt_000200.pth --episodes 10 --save_json logs/summary.json --save_details_json logs/details.json

2. 批量模型对比 (生成 CSV 用于画趋势图):
   python marl_project/evaluate.py --model_glob "logs/exp/ckpt_*.pth" --episodes 5 --save_table logs/benchmark.csv

3. 可视化模式 (观看驾驶效果):
   python marl_project/evaluate.py --model_path logs/exp/best.pth --render --top_down --episodes 3

4. 强制指定地图 (例如只跑直道):
   python marl_project/evaluate.py --model_path logs/exp/best.pth --map_sequence "S"

【关键参数】
--model_path: 指定单个模型路径。
--model_glob: 指定通配符路径 (如 "ckpt_*.pth")，用于批量评估。
--save_details_json: ⚠️ 必须开启此项才能使用 analysis_tools.py。
--save_table: ⚠️ 必须开启此项才能使用 plot_benchmark.py。
"""

import os
import sys
import json
import argparse
import glob
import re
import csv
import time
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch


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
    parser.add_argument("--start_seed", type=int, default=6000, help="Test set start seed")
    parser.add_argument("--num_agents", type=int, default=None, help="Override number of agents")

    parser.add_argument("--map_sequence", type=str, default=None, help="Override map (e.g. 'SSCC' or int for block_num)")
    parser.add_argument("--noise", type=float, default=None, help="Override Config.NOISE_STD")
    parser.add_argument("--mask", type=float, default=None, help="Override Config.MASK_RATIO")

    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--top_down", action="store_true", help="Use top-down render mode when rendering")
    parser.add_argument("--draw_connections", action="store_true", help="Draw lines between communicating agents")

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

    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    if args.noise is not None:
        Config.NOISE_STD = float(args.noise)
    if args.mask is not None:
        Config.MASK_RATIO = float(args.mask)


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
        "episodes",
        "start_seed",
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


def evaluate_single_model(model_path: str, args: argparse.Namespace) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Evaluate a single checkpoint.

    Returns:
        (summary, details) where details may be None unless requested.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Capture maxima *after* any user overrides to Config (e.g. different map/reward settings)
    reward_maxima = _capture_reward_maxima_from_config()
    if bool(getattr(args, "eval_full_reward", False)):
        _apply_reward_params(reward_maxima)

    # Build env config
    eval_config = Config.get_metadrive_config()
    eval_config["start_seed"] = int(args.start_seed)
    eval_config["num_scenarios"] = int(args.episodes)

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

    if args.num_agents is not None:
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

    # Recording
    imageio = _maybe_get_imageio() if want_record else None
    if want_record and imageio is None:
        print("[Eval] Recording requested but imageio is not available; skipping recording.")

    if want_record:
        os.makedirs(args.record_dir, exist_ok=True)

    for ep in range(int(args.episodes)):
        seed = int(args.start_seed) + ep
        obs_dict, _ = env.reset(seed=seed)

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
        while not ep_done:
            if args.max_steps is not None and step_idx >= int(args.max_steps):
                break

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
                        frame = env.render(mode="top_down")
                    except Exception:
                        frame = None

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
                results = policy(obs_tensor)

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

            action_dict = {}
            for i, agent_id in enumerate(active_agents):
                action_dict[agent_id] = np.clip(actions[i], low, high)

            next_obs, rewards, dones, infos = env.step(action_dict)

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
                        speed_sum += float(info["speed_kmh"])
                        speed_n += 1.0
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

            obs_dict = next_obs
            ep_done = bool(dones.get("__all__", False)) or (len(obs_dict) == 0)
            step_idx += 1

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

    if bool(args.render) and bool(getattr(args, "pause_at_end", False)):
        try:
            input("[Eval][Render] Press Enter to exit...")
        except Exception:
            # Non-interactive terminal
            pass

    success = terminal_counts.get("success", 0)
    crash = terminal_counts.get("crash", 0)
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

    summary = {
        "model_path": model_path,
        "episodes": int(args.episodes),
        "start_seed": int(args.start_seed),
        "num_agents": args.num_agents,
        "deterministic": (not bool(args.stochastic)),
        "eval_full_reward": bool(getattr(args, "eval_full_reward", False)),
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
            n_indices.append(0)
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


def _load_checkpoint_into_policy(policy: CooperativePolicy, model_path: str, device: torch.device) -> None:
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

    if not args.disable_patches:
        _apply_metadrive_patches()

    _apply_overrides(args)

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
                    "episodes": s.get("episodes"),
                    "start_seed": s.get("start_seed"),
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
            payload = {"models": summaries}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Saved summary to {args.save_json}")

    if args.save_details_json and details_list:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_details_json)) or ".", exist_ok=True)
        payload: Any = details_list[0] if len(details_list) == 1 else {"models": details_list}
        with open(args.save_details_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Eval] Saved details to {args.save_details_json}")


if __name__ == "__main__":
    evaluate()
