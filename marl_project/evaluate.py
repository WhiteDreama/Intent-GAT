import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
from metadrive.constants import TerminationState

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.models.policy import CooperativePolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Deterministic evaluation for MARL CooperativePolicy")
    parser.add_argument("--start_seed", type=int, default=5000, help="Test set start seed")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth checkpoint")
    # Map switching
    parser.add_argument(
        "--map_mode",
        type=str,
        default=None,
        choices=["block_num", "block_sequence"],
        help='Override Config.MAP_MODE ("block_num" or "block_sequence")',
    )
    parser.add_argument(
        "--map_block_num",
        type=int,
        default=None,
        help="Override Config.MAP_BLOCK_NUM when map_mode=block_num",
    )
    parser.add_argument(
        "--map_type",
        type=str,
        default=None,
        help='Override Config.MAP_TYPE when map_mode=block_sequence (e.g., "SSSSS", "X", "r")',
    )
    parser.add_argument("--noise", type=float, default=None, help="Override Config.NOISE_STD for robustness test")
    parser.add_argument("--mask", type=float, default=None, help="Override Config.MASK_RATIO for packet loss test")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--top_down", action="store_true", help="Use global top-down view instead of follow view")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def apply_overrides(args):
    if args.map_mode is not None:
        Config.MAP_MODE = args.map_mode
    if args.map_block_num is not None:
        Config.MAP_BLOCK_NUM = args.map_block_num
    if args.map_type is not None:
        Config.MAP_TYPE = args.map_type
    if args.noise is not None:
        Config.NOISE_STD = args.noise
    if args.mask is not None:
        Config.MASK_RATIO = args.mask


def build_batch_from_obs(obs_dict):
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
        batch_gt_waypoints.append(agent_obs.get("gt_waypoints", np.zeros((Config.PRED_WAYPOINTS_NUM, 2), dtype=np.float32)))

        n_indices = []
        n_mask = []
        n_rel_pos = []

        raw_neighbors = agent_obs.get("neighbors", [])
        raw_rel_pos = agent_obs.get("neighbor_rel_pos", [])

        for i, n_id in enumerate(raw_neighbors):
            if n_id in agent_to_idx:
                n_indices.append(agent_to_idx[n_id])
                n_mask.append(1.0)
                n_rel_pos.append(raw_rel_pos[i])
        while len(n_indices) < Config.MAX_NEIGHBORS:
            n_indices.append(0)
            n_mask.append(0.0)
            n_rel_pos.append([0.0, 0.0, 0.0, 0.0])

        batch_neighbor_indices.append(n_indices[: Config.MAX_NEIGHBORS])
        batch_neighbor_mask.append(n_mask[: Config.MAX_NEIGHBORS])
        batch_neighbor_rel_pos.append(n_rel_pos[: Config.MAX_NEIGHBORS])

    obs_tensor_batch = {
        "node_features": torch.tensor(np.array(batch_node_features), dtype=torch.float32),
        "neighbor_indices": torch.tensor(np.array(batch_neighbor_indices), dtype=torch.long),
        "neighbor_mask": torch.tensor(np.array(batch_neighbor_mask), dtype=torch.float32),
        "neighbor_rel_pos": torch.tensor(np.array(batch_neighbor_rel_pos), dtype=torch.float32),
        "gt_waypoints": torch.tensor(np.array(batch_gt_waypoints), dtype=torch.float32),
    }
    return obs_tensor_batch, active_agents


def evaluate():
    args = parse_args()
    apply_overrides(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    print(f"[Eval] Overrides -> MAP_TYPE: {Config.MAP_TYPE}, NOISE_STD: {Config.NOISE_STD}, MASK_RATIO: {Config.MASK_RATIO}")
    
    # 在 evaluate() 初始化环境前，构造完整配置并传入
    eval_config = Config.get_metadrive_config()
    eval_config["start_seed"] = args.start_seed
    eval_config["num_scenarios"] = args.episodes  # 保证不重复
    eval_config["use_render"] = args.render
    env = GraphEnvWrapper(config=eval_config)
    obs_dict, _ = env.reset()
    if not obs_dict:
        print("[Eval] Error: empty observation on reset.")
        return

    sample_agent = list(obs_dict.keys())[0]
    input_dim = obs_dict[sample_agent]["node_features"].shape[0]
    action_dim = env.action_space[sample_agent].shape[0]

    policy = CooperativePolicy(input_dim, action_dim).to(device)
    if not os.path.exists(args.model_path):
        print(f"[Eval] Checkpoint not found: {args.model_path}")
        return
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)

    def _filter_mismatched(sd, model):
        filtered = {}
        model_sd = model.state_dict()
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                print(f"[Eval] Drop mismatched key {k}: ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}")
                continue
            filtered[k] = v
        return filtered

    state_dict = _filter_mismatched(state_dict, policy)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[Eval] Loaded with relaxed strictness. Missing: {missing}, Unexpected: {unexpected}")
    policy.eval()
    print(f"[Eval] Loaded model from {args.model_path}")

    success_cnt = 0
    crash_cnt = 0
    out_of_road_cnt = 0
    timeout_cnt = 0
    other_cnt = 0
    idle_long_cnt = 0
    finished_agents = 0
    total_steps = 0
    risk_stats = defaultdict(float)
    metrics = {"ade_sum": 0.0, "fde_sum": 0.0, "count": 0}
    prev_actions = {}

    for ep in range(args.episodes):
        obs_dict, _ = env.reset()
        ep_done = False
        while not ep_done:
            if args.render:
                render_kwargs = {"mode": "top_down"} if args.top_down else {}
                env.render(**render_kwargs)  # global top-down if requested
            obs_batch, active_agents = build_batch_from_obs(obs_dict)
            obs_batch = {k: v.to(device) for k, v in obs_batch.items()}

            with torch.no_grad():
                results = policy(obs_batch)
                actions = results["action_mean"].cpu().numpy()
                pred_wp = results["pred_waypoints"]  # (B, 5, 2)
                gt_wp = obs_batch["gt_waypoints"]    # (B, 5, 2)

                # Euclidean distance per step: (B, 5)
                distance_error = torch.norm(pred_wp - gt_wp, dim=-1)
                ade_batch = distance_error.mean(dim=-1)  # (B,)
                fde_batch = distance_error[:, -1]        # (B,)

                metrics["ade_sum"] += ade_batch.sum().item()
                metrics["fde_sum"] += fde_batch.sum().item()
                metrics["count"] += obs_batch["node_features"].size(0)

            action_dict = {}
            # Representative action space template
            env_space = env.action_space
            template_space = None
            if hasattr(env_space, "spaces") and len(env_space.spaces) > 0:
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

            def clip_action(act):
                if template_space is None:
                    return act
                return np.clip(act, template_space.low, template_space.high)

            for i, agent_id in enumerate(active_agents):
                raw_act = actions[i]
                prev = prev_actions.get(agent_id)
                if prev is not None:
                    smoothed = Config.ACTION_SMOOTH_ALPHA * prev + (1 - Config.ACTION_SMOOTH_ALPHA) * raw_act
                else:
                    smoothed = raw_act
                smoothed = clip_action(smoothed)
                action_dict[agent_id] = smoothed
                prev_actions[agent_id] = smoothed

            next_obs, rewards, dones, infos = env.step(action_dict)

            # Per-step risk stats from wrapper infos
            if isinstance(infos, dict):
                for aid, info in infos.items():
                    if aid == "__all__" or not isinstance(info, dict):
                        continue
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

            # Drop history for finished agents to avoid stale smoothing
            for aid, done_flag in dones.items():
                if done_flag:
                    prev_actions.pop(aid, None)

            total_steps += 1
            obs_dict = next_obs
            for aid in list(dones.keys()):
                if aid == "__all__":
                    continue
                if dones[aid]:
                    finished_agents += 1
                    info = infos.get(aid, {})
                    term = info.get("terminal_reason") if isinstance(info, dict) else None
                    if term is None:
                        arrived = info.get(TerminationState.SUCCESS, False) or info.get("arrive_dest", False)
                        crashed = info.get(TerminationState.CRASH, False) or info.get(TerminationState.CRASH_VEHICLE, False) or info.get("crash", False)
                        out_road = info.get(TerminationState.OUT_OF_ROAD, False) or info.get("out_of_road", False)
                        truncated = info.get("truncated", False)
                        if arrived:
                            term = "success"
                        elif crashed:
                            term = "crash"
                        elif out_road:
                            term = "out_of_road"
                        elif truncated:
                            term = "timeout"
                        else:
                            term = "other"

                    if term == "success":
                        success_cnt += 1
                    elif term == "crash":
                        crash_cnt += 1
                    elif term == "out_of_road":
                        out_of_road_cnt += 1
                    elif term == "timeout":
                        timeout_cnt += 1
                    else:
                        other_cnt += 1

                    ev = info.get("event", {}) if isinstance(info, dict) else {}
                    if isinstance(ev, dict) and ev.get("idle_long", False):
                        idle_long_cnt += 1

            ep_done = dones.get("__all__", False) or (not obs_dict)

        print(f"[Eval] Episode {ep + 1}/{args.episodes} done.")

    total_agents = max(finished_agents, 1)
    sr = success_cnt / total_agents
    cr = crash_cnt / total_agents
    oor = out_of_road_cnt / total_agents
    tr = timeout_cnt / total_agents
    intent_bytes = Config.INTENT_DIM * 4
    lidar_bytes = 72 * 4

    steps_with_ttc = max(1.0, risk_stats.get("steps_with_ttc", 0.0))
    steps_with_dist = max(1.0, risk_stats.get("steps_with_dist", 0.0))
    mean_min_ttc = float(risk_stats.get("min_ttc_sum", 0.0)) / steps_with_ttc
    mean_min_dist = float(risk_stats.get("min_dist_sum", 0.0)) / steps_with_dist
    frac_ttc_threat = float(risk_stats.get("risk_ttc_steps", 0.0)) / steps_with_ttc
    frac_dist_threat = float(risk_stats.get("risk_dist_steps", 0.0)) / steps_with_dist
    idle_n = max(1.0, risk_stats.get("idle_count_n", 0.0))
    mean_idle = float(risk_stats.get("idle_count_sum", 0.0)) / idle_n
    idle_long_rate = float(idle_long_cnt) / total_agents

    print("\n========== Evaluation Summary ==========")
    print(f"Episodes: {args.episodes}")
    print(f"Agents Finished: {finished_agents}")
    print(f"Success Rate: {sr:.3f}")
    print(f"Collision Rate: {cr:.3f}")
    print(f"OutOfRoad Rate: {oor:.3f}")
    print(f"Timeout Rate: {tr:.3f}")
    print(f"IdleLong Rate: {idle_long_rate:.3f}")
    print(f"Mean min_ttc_s: {mean_min_ttc:.3f} | TTC threat frac: {frac_ttc_threat:.3f}")
    print(f"Mean min_dist_m: {mean_min_dist:.3f} | Dist threat frac: {frac_dist_threat:.3f}")
    print(f"Mean idle_count: {mean_idle:.2f}")
    ade_mean = metrics["ade_sum"] / max(metrics["count"], 1)
    fde_mean = metrics["fde_sum"] / max(metrics["count"], 1)
    print(f"Intent Prediction ADE: {ade_mean:.3f} meters")
    print(f"Intent Prediction FDE: {fde_mean:.3f} meters")
    print(f"Bandwidth per agent per step: {intent_bytes} bytes (vs. lidar baseline {lidar_bytes} bytes)")
    print("========================================\n")


if __name__ == "__main__":
    evaluate()
