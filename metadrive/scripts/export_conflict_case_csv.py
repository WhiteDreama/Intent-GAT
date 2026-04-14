#!/usr/bin/env python3
"""
Export a representative conflict-case CSV for Figure 6.

This pipeline is intentionally independent from run_packet_loss_sweep and does
not modify marl_project/evaluate.py main logic.

Implementation note:
- TTC is always computed from simulator/world-state positions and velocities.
- actor_visible_to_ego is retained as an auxiliary semantic field.
- The fixed critical actor is selected from the anchor rollout ("ours") and
  reused for all compared methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.evaluate import (
    _build_batch_from_obs,
    _get_personalized_perception_config,
    _load_checkpoint_into_policy,
)
from marl_project.models.policy import CooperativePolicy


# =========================
# Config Section
# =========================
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "representative_conflict_case"
ANCHOR_METHOD = "ours"
METHOD_ORDER = ["ours", "no_aux", "dense_comm", "lidar_only"]
PLOT_METHODS = ["ours", "no_aux", "dense_comm"]
TTC_THRESHOLD = 2.5
ONSET_CONSECUTIVE_STEPS = 3
DT = 0.1
WINDOW_S = (-3.0, 3.0)
TOP_K = 10
PREVIEW_EXPORT_TOP_K = 10
EPISODES_TO_SCAN = 5
START_SEED = 10000
NUM_AGENTS = 6
MAP_SEQUENCE = "X"
DEFAULT_MAX_STEPS = 400
DEFAULT_DEVICE = "cpu"
BRAKE_ACCEL_THRESHOLD = -0.5
POST_BRAKE_WINDOW_S = 1.5
SCREEN_SMOOTH_WINDOW = 5
REACTION_ACCEL_THRESHOLD = -0.5
REACTION_CONSECUTIVE_STEPS = 2
CENTRAL_WINDOW = (-1.5, 2.0)
VALID_TTC_COUNT_WINDOW = (-0.5, 1.0)
DESCENT_SUPPORT_WINDOW = (-1.0, 1.5)
DESCENT_PRE_WINDOW = (-1.0, -0.2)
DESCENT_POST_WINDOW = (0.0, 1.5)
OURS_TTC_DROP_MIN = 1.5
BASELINE_TTC_DROP_MIN = 1.0
POST_TTC_WINDOW = (0.0, 2.5)
TTC_DROP_PRE_WINDOW = (-2.5, -0.5)
TTC_DROP_POST_WINDOW = (0.0, 1.5)
SMOOTHNESS_WINDOW = (0.0, 2.0)
OURS_MIN_ACCEL_RANGE = (-3.0, -0.8)
MIN_ACTOR_IN_WORLD_RATE = 0.9
MIN_TTC_VALID_RATE = 0.8
MIN_VALID_TTC_POINTS_FOCUS = 5
REACTION_ALLOWANCE_S = 0.2
LATE_TTC_REBOUND_START_S = 1.5
LATE_TTC_REBOUND_JUMP_S = 1.5
LATE_TTC_REBOUND_MAX_STEP_GAP = 2

METHOD_DISPLAY = {
    "ours": "Intent-GAT (Ours)",
    "no_aux": "No-Aux",
    "dense_comm": "Dense Comm",
    "lidar_only": "LiDAR-Only",
}
PREVIEW_STYLE = {
    "ours": {"color": "#000000", "linestyle": "-", "marker": "o"},
    "no_aux": {"color": "#444444", "linestyle": "--", "marker": "s"},
    "dense_comm": {"color": "#6a6a6a", "linestyle": "-.", "marker": "^"},
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    model_path: Path
    model_type: str


def build_method_specs() -> Dict[str, MethodSpec]:
    return {
        "ours": MethodSpec(
            name="ours",
            display_name="Intent-GAT (Ours)",
            model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_intent_gat" / "best_success_model.pth",
            model_type="ours",
        ),
        "no_aux": MethodSpec(
            name="no_aux",
            display_name="No-Aux",
            model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_no_aux" / "best_success_model.pth",
            model_type="no_aux",
        ),
        "dense_comm": MethodSpec(
            name="dense_comm",
            display_name="Dense Comm",
            model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_Dense_Comm" / "best_success_model.pth",
            model_type="oracle",
        ),
        "lidar_only": MethodSpec(
            name="lidar_only",
            display_name="LiDAR-Only",
            model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_lidar_only" / "best_success_model.pth",
            model_type="lidar_only",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export representative conflict-case CSVs for Figure 6.")
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--episodes", type=int, default=EPISODES_TO_SCAN)
    parser.add_argument("--start_seed", type=int, default=START_SEED)
    parser.add_argument("--num_agents", type=int, default=NUM_AGENTS)
    parser.add_argument("--map_sequence", type=str, default=MAP_SEQUENCE)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--target_episode_id", type=int, default=None)
    parser.add_argument("--target_ego_id", type=str, default=None)
    parser.add_argument("--allow_rejected_target", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_map(map_sequence: str) -> None:
    if str(map_sequence).isdigit():
        Config.MAP_MODE = "block_num"
        Config.MAP_BLOCK_NUM = int(map_sequence)
    else:
        Config.MAP_MODE = "block_sequence"
        Config.MAP_TYPE = str(map_sequence)


def apply_method_globals(spec: MethodSpec, map_sequence: str, num_agents: int) -> None:
    configure_map(map_sequence)
    Config.NUM_AGENTS = int(num_agents)
    Config.COMM_MODULE = "gat"
    personalized = _get_personalized_perception_config(spec.model_type)
    if personalized.get("lidar_num_others") is not None:
        Config.LIDAR_NUM_OTHERS = int(personalized["lidar_num_others"])
    if personalized.get("max_neighbors") is not None:
        Config.MAX_NEIGHBORS = int(personalized["max_neighbors"])
    if personalized.get("mask_ratio") is not None:
        Config.MASK_RATIO = float(personalized["mask_ratio"])
    if personalized.get("noise_std") is not None:
        Config.NOISE_STD = float(personalized["noise_std"])
    if personalized.get("comm_module"):
        Config.COMM_MODULE = str(personalized["comm_module"])


def build_env_config(seed: int, num_agents: int) -> Dict[str, Any]:
    env_config = Config.get_metadrive_config(is_eval=True)
    env_config["start_seed"] = int(seed)
    env_config["num_scenarios"] = 1
    env_config["num_agents"] = int(num_agents)
    env_config["allow_respawn"] = False
    env_config["delay_done"] = 0
    env_config["use_render"] = False
    env_config["comm_mode"] = "iid"
    env_config["comm_stale_steps"] = 0
    env_config["comm_burst_len"] = 1
    return env_config


def make_env_and_policy(spec: MethodSpec, seed: int, num_agents: int, map_sequence: str, device: torch.device):
    apply_method_globals(spec, map_sequence=map_sequence, num_agents=num_agents)
    env = GraphEnvWrapper(config=build_env_config(seed=seed, num_agents=num_agents))
    obs_dict, _ = env.reset(seed=int(seed))
    if not obs_dict:
        env.close()
        raise RuntimeError(f"Empty observation on reset for method={spec.name}, seed={seed}")

    sample_agent = list(obs_dict.keys())[0]
    input_dim = int(obs_dict[sample_agent]["node_features"].shape[0])
    if hasattr(env.action_space, "spaces"):
        action_dim = int(env.action_space[sample_agent].shape[0])
    else:
        action_dim = int(env.action_space.shape[0])

    policy = CooperativePolicy(input_dim, action_dim).to(device)
    _load_checkpoint_into_policy(policy, str(spec.model_path), device)
    policy.eval()
    return env, policy, obs_dict


def safe_float_pair(vec: Any) -> Tuple[float, float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return 0.0, 0.0
    return float(arr[0]), float(arr[1])


def canonical_actor_id(obj: BaseVehicle, object_name_to_agent: Dict[str, str]) -> str:
    name = str(getattr(obj, "name", ""))
    return object_name_to_agent.get(name, name)


def get_action_bounds(env: GraphEnvWrapper, sample_agent: str) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(env.action_space, "spaces"):
        space = env.action_space[sample_agent]
    else:
        space = env.action_space
    return np.asarray(space.low, dtype=np.float32), np.asarray(space.high, dtype=np.float32)


def iter_world_vehicle_objects(env: GraphEnvWrapper) -> List[BaseVehicle]:
    vehicles: List[BaseVehicle] = []
    seen: set[int] = set()
    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    candidates: List[Any] = []
    if traffic_manager is not None and hasattr(traffic_manager, "vehicles"):
        try:
            candidates.extend(list(traffic_manager.vehicles))
        except Exception:
            pass
    candidates.extend(list(getattr(env, "agents", {}).values()))
    for obj in candidates:
        if isinstance(obj, BaseVehicle) and id(obj) not in seen:
            seen.add(id(obj))
            vehicles.append(obj)
    return vehicles


def visible_actor_ids_for_ego(
    env: GraphEnvWrapper,
    ego_id: str,
    obs_dict: Dict[str, Dict[str, Any]],
    object_name_to_agent: Dict[str, str],
) -> List[str]:
    visible = set(obs_dict.get(ego_id, {}).get("neighbors_true", []) or [])
    ego_vehicle = getattr(env, "agents", {}).get(ego_id)
    if ego_vehicle is not None and hasattr(ego_vehicle, "lidar"):
        try:
            objs = ego_vehicle.lidar.get_surrounding_objects(
                ego_vehicle,
                radius=int(getattr(Config, "LIDAR_DIST", 60.0)),
            )
            for obj in objs:
                if isinstance(obj, BaseVehicle):
                    actor_id = canonical_actor_id(obj, object_name_to_agent)
                    if actor_id and actor_id != ego_id:
                        visible.add(actor_id)
        except Exception:
            pass
    return sorted(visible)


def collect_episode_snapshot(
    env: GraphEnvWrapper,
    obs_dict: Dict[str, Dict[str, Any]],
    step_idx: int,
) -> Dict[str, Any]:
    object_name_to_agent = {str(getattr(v, "name", aid)): aid for aid, v in getattr(env, "agents", {}).items()}
    actors: Dict[str, Dict[str, Any]] = {}
    for obj in iter_world_vehicle_objects(env):
        actor_id = canonical_actor_id(obj, object_name_to_agent)
        actors[actor_id] = {
            "position": safe_float_pair(getattr(obj, "position", (0.0, 0.0))),
            "velocity": safe_float_pair(getattr(obj, "velocity", (0.0, 0.0))),
            "speed_ms": float(np.linalg.norm(np.asarray(getattr(obj, "velocity", (0.0, 0.0)), dtype=np.float32))),
        }

    visible_by_ego = {
        ego_id: visible_actor_ids_for_ego(env, ego_id, obs_dict, object_name_to_agent)
        for ego_id in obs_dict.keys()
    }

    return {
        "step": int(step_idx),
        "sim_time_s": float(step_idx) * DT,
        "actors": actors,
        "visible_by_ego": visible_by_ego,
    }


def compute_pairwise_ttc(
    ego_state: Optional[Dict[str, Any]],
    actor_state: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], bool]:
    if ego_state is None or actor_state is None:
        return None, False
    ego_pos = np.asarray(ego_state["position"], dtype=np.float32)
    ego_vel = np.asarray(ego_state["velocity"], dtype=np.float32)
    actor_pos = np.asarray(actor_state["position"], dtype=np.float32)
    actor_vel = np.asarray(actor_state["velocity"], dtype=np.float32)
    rel_pos = actor_pos - ego_pos
    dist = float(np.linalg.norm(rel_pos))
    if dist <= 1e-4:
        return None, False
    rel_vel = actor_vel - ego_vel
    closing = -float(np.dot(rel_pos / dist, rel_vel))
    if closing <= 1e-5:
        return None, False
    return dist / closing, True


def detect_conflict_onset(ttc_series: Sequence[Optional[float]], threshold: float, consecutive_steps: int) -> Optional[int]:
    run = 0
    for idx, value in enumerate(ttc_series):
        if value is not None and math.isfinite(value) and value < threshold:
            run += 1
            if run >= consecutive_steps:
                return idx - consecutive_steps + 1
        else:
            run = 0
    return None


def rollout_episode(
    spec: MethodSpec,
    seed: int,
    episode_id: int,
    num_agents: int,
    map_sequence: str,
    max_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    seed_everything(seed)
    env, policy, obs_dict = make_env_and_policy(spec, seed=seed, num_agents=num_agents, map_sequence=map_sequence, device=device)
    scenario_key = f"seed{seed}_ep{episode_id:03d}"
    initial_rl_agent_ids = list(obs_dict.keys())
    low, high = get_action_bounds(env, initial_rl_agent_ids[0])

    snapshots: List[Dict[str, Any]] = [collect_episode_snapshot(env, obs_dict, step_idx=0)]
    step_idx = 0
    done_all = False

    try:
        while not done_all and step_idx < int(max_steps):
            batch, active_agents = _build_batch_from_obs(obs_dict)
            if not active_agents:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                results = policy(batch)
            actions = results["action_mean"].detach().cpu().numpy()
            action_dict = {
                agent_id: np.clip(actions[i], low, high)
                for i, agent_id in enumerate(active_agents)
            }
            step_out = env.step(action_dict)
            if len(step_out) == 5:
                obs_dict, _, dones, truncated, _ = step_out
                done_all = bool(dones.get("__all__", False)) or bool(truncated.get("__all__", False))
            elif len(step_out) == 4:
                obs_dict, _, dones, _ = step_out
                done_all = bool(dones.get("__all__", False))
            else:
                raise RuntimeError(f"Unexpected env.step() return length: {len(step_out)}")
            step_idx += 1
            snapshots.append(collect_episode_snapshot(env, obs_dict, step_idx=step_idx))
    finally:
        env.close()

    return {
        "method": spec.name,
        "seed": int(seed),
        "episode_id": int(episode_id),
        "scenario_key": scenario_key,
        "initial_rl_agent_ids": initial_rl_agent_ids,
        "snapshots": snapshots,
    }


def compute_actor_ttc_series(trace: Dict[str, Any], ego_id: str, actor_id: str) -> List[Optional[float]]:
    series: List[Optional[float]] = []
    for snap in trace["snapshots"]:
        actors = snap["actors"]
        ttc, valid = compute_pairwise_ttc(actors.get(ego_id), actors.get(actor_id))
        series.append(float(ttc) if valid and ttc is not None else None)
    return series


def get_actor_step_states(trace: Dict[str, Any], actor_id: str) -> Dict[int, Dict[str, Any]]:
    step_states: Dict[int, Dict[str, Any]] = {}
    for snap in trace["snapshots"]:
        actor_state = snap["actors"].get(actor_id)
        if actor_state is not None:
            step_states[int(snap["step"])] = actor_state
    return step_states


def build_scenario_actor_key(trace: Dict[str, Any], ego_id: str) -> str:
    # Use a scenario-level semantic key instead of a rollout-specific UUID.
    return f"{trace['scenario_key']}|ego={ego_id}|critical_actor"


def build_actor_match_signature(
    trace: Dict[str, Any],
    actor_id: str,
    onset_step: int,
    max_steps: int = 20,
) -> Dict[str, Any]:
    actor_states = get_actor_step_states(trace, actor_id)
    if not actor_states:
        return {"steps": [], "states": {}}

    first_seen = min(actor_states.keys())
    last_allowed = min(first_seen + int(max_steps) - 1, int(onset_step) + 5)
    selected_steps = [step for step in sorted(actor_states.keys()) if first_seen <= step <= last_allowed]
    return {
        "steps": selected_steps,
        "states": {step: actor_states[step] for step in selected_steps},
    }


def match_actor_runtime_id(
    trace: Dict[str, Any],
    ego_id: str,
    signature: Dict[str, Any],
) -> Optional[str]:
    signature_steps: List[int] = list(signature.get("steps", []))
    signature_states: Dict[int, Dict[str, Any]] = dict(signature.get("states", {}))
    if not signature_steps or not signature_states:
        return None

    candidate_ids = set()
    for snap in trace["snapshots"]:
        candidate_ids.update(snap["actors"].keys())
    candidate_ids.discard(ego_id)

    best_actor_id: Optional[str] = None
    best_score: Optional[float] = None

    for candidate_id in sorted(candidate_ids):
        candidate_states = get_actor_step_states(trace, candidate_id)
        overlap_steps = [step for step in signature_steps if step in candidate_states]
        if len(overlap_steps) < 5:
            continue

        pos_dists: List[float] = []
        vel_dists: List[float] = []
        for step in overlap_steps:
            anchor_state = signature_states[step]
            cand_state = candidate_states[step]
            pos_dists.append(
                float(
                    np.linalg.norm(
                        np.asarray(anchor_state["position"], dtype=np.float32)
                        - np.asarray(cand_state["position"], dtype=np.float32)
                    )
                )
            )
            vel_dists.append(
                float(
                    np.linalg.norm(
                        np.asarray(anchor_state["velocity"], dtype=np.float32)
                        - np.asarray(cand_state["velocity"], dtype=np.float32)
                    )
                )
            )

        first_step = overlap_steps[0]
        first_anchor = signature_states[first_step]
        first_cand = candidate_states[first_step]
        first_pos_dist = float(
            np.linalg.norm(
                np.asarray(first_anchor["position"], dtype=np.float32)
                - np.asarray(first_cand["position"], dtype=np.float32)
            )
        )

        mean_pos = float(np.mean(pos_dists))
        mean_vel = float(np.mean(vel_dists))
        overlap_bonus = 0.15 * float(len(signature_steps) - len(overlap_steps))
        score = mean_pos + 0.35 * mean_vel + 0.50 * first_pos_dist + overlap_bonus

        if best_score is None or score < best_score:
            best_score = score
            best_actor_id = candidate_id

    return best_actor_id


def resolve_runtime_actor_ids(
    traces_by_method: Dict[str, Dict[str, Any]],
    ego_id: str,
    anchor_actor_id: str,
    onset_step: int,
) -> Dict[str, Optional[str]]:
    anchor_trace = traces_by_method[ANCHOR_METHOD]
    signature = build_actor_match_signature(anchor_trace, actor_id=anchor_actor_id, onset_step=onset_step)

    resolved: Dict[str, Optional[str]] = {}
    for method_name, trace in traces_by_method.items():
        if method_name == ANCHOR_METHOD:
            resolved[method_name] = anchor_actor_id
            continue
        if get_actor_step_states(trace, anchor_actor_id):
            resolved[method_name] = anchor_actor_id
            continue
        resolved[method_name] = match_actor_runtime_id(trace, ego_id=ego_id, signature=signature)
    return resolved


def select_anchor_critical_actor(trace: Dict[str, Any], ego_id: str, ttc_threshold: float) -> Optional[Dict[str, Any]]:
    actor_ids = set()
    for snap in trace["snapshots"]:
        actor_ids.update(snap["actors"].keys())
    actor_ids.discard(ego_id)

    eligible: List[Dict[str, Any]] = []
    for actor_id in sorted(actor_ids):
        ttc_series = compute_actor_ttc_series(trace, ego_id=ego_id, actor_id=actor_id)
        below = [v for v in ttc_series if v is not None and v < float(ttc_threshold)]
        if not below:
            continue
        min_ttc = min(below)
        eligible.append(
            {
                "actor_id": actor_id,
                "min_ttc": float(min_ttc),
                "ttc_series": ttc_series,
            }
        )
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item["min_ttc"], str(item["actor_id"])))
    return eligible[0]


def compute_ego_accel(speed_series: Sequence[Optional[float]], dt: float) -> List[Optional[float]]:
    accels: List[Optional[float]] = []
    prev_speed: Optional[float] = None
    for speed in speed_series:
        if speed is None or prev_speed is None:
            accels.append(None)
        else:
            accels.append((float(speed) - float(prev_speed)) / float(dt))
        prev_speed = None if speed is None else float(speed)
    return accels


def build_pair_rows(
    trace: Dict[str, Any],
    ego_id: str,
    scenario_actor_key: str,
    runtime_actor_id: Optional[str],
    anchor_actor_id: str,
    onset_step: int,
    critical_actor_source_method: str,
    ttc_threshold: float,
) -> List[Dict[str, Any]]:
    speed_series: List[Optional[float]] = []
    ttc_series: List[Optional[float]] = []
    actor_in_world_series: List[bool] = []
    actor_visible_series: List[bool] = []
    ttc_valid_series: List[bool] = []

    for snap in trace["snapshots"]:
        actors = snap["actors"]
        ego_state = actors.get(ego_id)
        actor_state = actors.get(runtime_actor_id) if runtime_actor_id else None
        actor_in_world = actor_state is not None
        actor_visible = bool(runtime_actor_id) and runtime_actor_id in set(snap["visible_by_ego"].get(ego_id, []))
        ttc_value, ttc_valid = compute_pairwise_ttc(ego_state, actor_state)
        speed_series.append(None if ego_state is None else float(ego_state["speed_ms"]))
        ttc_series.append(float(ttc_value) if ttc_valid and ttc_value is not None else None)
        actor_in_world_series.append(bool(actor_in_world))
        actor_visible_series.append(bool(actor_visible))
        ttc_valid_series.append(bool(ttc_valid))

    accel_series = compute_ego_accel(speed_series, dt=DT)
    rows: List[Dict[str, Any]] = []
    for idx, snap in enumerate(trace["snapshots"]):
        rows.append(
            {
                "seed": int(trace["seed"]),
                "episode_id": int(trace["episode_id"]),
                "scenario_key": trace["scenario_key"],
                "method": trace["method"],
                "ego_id": ego_id,
                "critical_actor_id": scenario_actor_key,
                "critical_actor_anchor_id": anchor_actor_id,
                "matched_actor_id": runtime_actor_id or "",
                "critical_actor_source_method": critical_actor_source_method,
                "step": int(snap["step"]),
                "sim_time_s": float(snap["sim_time_s"]),
                "time_s": (int(snap["step"]) - int(onset_step)) * DT,
                "ttc": ttc_series[idx],
                "ego_speed_ms": speed_series[idx],
                "ego_accel": accel_series[idx],
                "onset_step": int(onset_step),
                "ttc_threshold": float(ttc_threshold),
                "actor_in_world": int(actor_in_world_series[idx]),
                "actor_visible_to_ego": int(actor_visible_series[idx]),
                "ttc_valid": int(ttc_valid_series[idx]),
            }
        )
    return rows


def rows_in_window(rows: Sequence[Dict[str, Any]], window_s: Tuple[float, float]) -> List[Dict[str, Any]]:
    left, right = window_s
    return [row for row in rows if left <= float(row["time_s"]) <= right]


def min_value(rows: Sequence[Dict[str, Any]], key: str, tmin: float, tmax: float) -> Optional[float]:
    vals = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and tmin <= float(row["time_s"]) <= tmax
    ]
    return min(vals) if vals else None


def mean_value(rows: Sequence[Dict[str, Any]], key: str, tmin: float, tmax: float) -> Optional[float]:
    vals = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and tmin <= float(row["time_s"]) <= tmax
    ]
    return float(np.mean(vals)) if vals else None


def std_value(rows: Sequence[Dict[str, Any]], key: str, tmin: float, tmax: float) -> Optional[float]:
    vals = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and tmin <= float(row["time_s"]) <= tmax
    ]
    return float(np.std(vals)) if vals else None


def valid_ttc_count(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(int(row["ttc_valid"]) for row in rows)


def valid_ttc_count_in_window(rows: Sequence[Dict[str, Any]], tmin: float, tmax: float) -> int:
    return sum(
        int(row["ttc_valid"])
        for row in rows
        if tmin <= float(row["time_s"]) <= tmax
    )


def actor_presence_ratio(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(row["actor_in_world"]) for row in rows) / float(len(rows))


def first_brake_latency_s(rows: Sequence[Dict[str, Any]], accel_threshold: float) -> Optional[float]:
    candidates = []
    for row in rows:
        accel = row.get("ego_accel")
        if accel is None:
            continue
        t = float(row["time_s"])
        if 0.0 <= t <= POST_BRAKE_WINDOW_S and float(accel) <= accel_threshold:
            candidates.append(t)
    return min(candidates) if candidates else None


def centered_moving_average(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if int(window) <= 1:
        return list(values)
    half = int(window) // 2
    smoothed: List[Optional[float]] = []
    for idx in range(len(values)):
        chunk = [v for v in values[max(0, idx - half): min(len(values), idx + half + 1)] if v is not None]
        smoothed.append(None if not chunk else float(np.mean(chunk)))
    return smoothed


def values_in_window(rows: Sequence[Dict[str, Any]], key: str, tmin: float, tmax: float) -> List[float]:
    vals: List[float] = []
    for row in rows:
        t = float(row["time_s"])
        value = row.get(key)
        if value is None or not (tmin <= t <= tmax):
            continue
        vals.append(float(value))
    return vals


def rate_in_window(rows: Sequence[Dict[str, Any]], key: str, tmin: float, tmax: float) -> float:
    window_rows = [row for row in rows if tmin <= float(row["time_s"]) <= tmax]
    if not window_rows:
        return 0.0
    return float(np.mean([float(row.get(key, 0) or 0) for row in window_rows]))


def max_consecutive_below_threshold(rows: Sequence[Dict[str, Any]], key: str, threshold: float, tmin: float, tmax: float) -> int:
    run = 0
    best = 0
    for row in rows:
        t = float(row["time_s"])
        if not (tmin <= t <= tmax):
            continue
        value = row.get(key)
        if value is not None and float(value) < float(threshold):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def mean_abs_first_difference(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None]
    if len(finite) < 2:
        return None
    diffs = np.abs(np.diff(np.asarray(finite, dtype=np.float32)))
    return float(np.mean(diffs)) if diffs.size > 0 else None


def detect_late_ttc_rebound(rows: Sequence[Dict[str, Any]]) -> bool:
    # Reject cases where TTC reappears with an abrupt late jump after the
    # interesting interaction window, which usually indicates an actor-validity
    # artifact rather than a meaningful conflict-resolution trend.
    for idx in range(1, len(rows)):
        row = rows[idx]
        t = float(row["time_s"])
        if t <= float(LATE_TTC_REBOUND_START_S):
            continue
        current_ttc = row.get("ttc")
        if current_ttc is None or not int(row.get("ttc_valid", 0) or 0):
            continue
        for back in (1, 2):
            prev_idx = idx - back
            if prev_idx < 0:
                continue
            prev_row = rows[prev_idx]
            prev_ttc = prev_row.get("ttc")
            if prev_ttc is None or not int(prev_row.get("ttc_valid", 0) or 0):
                continue
            step_gap = int(row["step"]) - int(prev_row["step"])
            if step_gap < 1 or step_gap > int(LATE_TTC_REBOUND_MAX_STEP_GAP):
                continue
            validity_changed = any(
                int(rows[k].get("ttc_valid", 0) or 0) != int(rows[k - 1].get("ttc_valid", 0) or 0)
                for k in range(prev_idx + 1, idx + 1)
            )
            if not validity_changed:
                continue
            if float(current_ttc) - float(prev_ttc) > float(LATE_TTC_REBOUND_JUMP_S):
                return True
    return False


def detect_reaction_onset_time_s(
    rows: Sequence[Dict[str, Any]],
    smoothed_accel: Sequence[Optional[float]],
    threshold: float,
    consecutive_steps: int,
) -> Optional[float]:
    run = 0
    start_idx: Optional[int] = None
    for idx, row in enumerate(rows):
        t = float(row["time_s"])
        value = smoothed_accel[idx] if idx < len(smoothed_accel) else None
        if t >= 0.0 and value is not None and float(value) < float(threshold):
            if run == 0:
                start_idx = idx
            run += 1
            if run >= int(consecutive_steps) and start_idx is not None:
                return float(rows[start_idx]["time_s"])
        else:
            run = 0
            start_idx = None
    return None


def compute_method_diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    accel_raw = [None if row.get("ego_accel") is None else float(row["ego_accel"]) for row in rows]
    smoothed_accel = centered_moving_average(accel_raw, SCREEN_SMOOTH_WINDOW)
    smoothed_pairs = [
        (float(row["time_s"]), smoothed_accel[idx])
        for idx, row in enumerate(rows)
        if idx < len(smoothed_accel)
    ]
    smoothed_vals_post = [v for t, v in smoothed_pairs if v is not None and SMOOTHNESS_WINDOW[0] <= t <= SMOOTHNESS_WINDOW[1]]
    pre_ttc_vals = values_in_window(rows, "ttc", *TTC_DROP_PRE_WINDOW)
    post_ttc_vals = values_in_window(rows, "ttc", *TTC_DROP_POST_WINDOW)
    post_ttc_local_vals = values_in_window(rows, "ttc", *POST_TTC_WINDOW)
    descent_pre_vals = values_in_window(rows, "ttc", *DESCENT_PRE_WINDOW)
    descent_post_vals = values_in_window(rows, "ttc", *DESCENT_POST_WINDOW)
    smoothed_post_series = [v for t, v in smoothed_pairs if SMOOTHNESS_WINDOW[0] <= t <= SMOOTHNESS_WINDOW[1]]
    descent_pre_mean = float(np.mean(descent_pre_vals)) if descent_pre_vals else None
    descent_post_mean = float(np.mean(descent_post_vals)) if descent_post_vals else None
    ttc_descent_mag: Optional[float] = None
    if descent_pre_mean is not None and descent_post_mean is not None:
        ttc_descent_mag = max(0.0, float(descent_pre_mean) - float(descent_post_mean))

    return {
        "pre_ttc_mean": float(np.mean(pre_ttc_vals)) if pre_ttc_vals else None,
        "post_ttc_mean": float(np.mean(post_ttc_vals)) if post_ttc_vals else None,
        "post_ttc_min": min(post_ttc_vals) if post_ttc_vals else None,
        "min_ttc_post_window": min(post_ttc_local_vals) if post_ttc_local_vals else None,
        "actor_in_world_rate": rate_in_window(rows, "actor_in_world", *CENTRAL_WINDOW),
        "ttc_valid_rate": rate_in_window(rows, "ttc_valid", *CENTRAL_WINDOW),
        "valid_ttc_points_focus": valid_ttc_count_in_window(rows, *VALID_TTC_COUNT_WINDOW),
        "below_threshold_run_steps": max_consecutive_below_threshold(rows, "ttc", TTC_THRESHOLD, -0.5, 1.5),
        "descent_pre_mean": descent_pre_mean,
        "descent_post_mean": descent_post_mean,
        "ttc_drop": ttc_descent_mag,
        "smoothed_min_accel": min(smoothed_vals_post) if smoothed_vals_post else None,
        "reaction_time": detect_reaction_onset_time_s(
            rows,
            smoothed_accel,
            threshold=REACTION_ACCEL_THRESHOLD,
            consecutive_steps=REACTION_CONSECUTIVE_STEPS,
        ),
        "smoothness_std": float(np.std(smoothed_vals_post)) if smoothed_vals_post else None,
        "smoothness_madiff": mean_abs_first_difference(smoothed_post_series),
        "late_ttc_rebound_flag": detect_late_ttc_rebound(rows),
    }


def bounded_reaction_latency_score(reaction_time: Optional[float]) -> float:
    if reaction_time is None:
        return 0.0
    target = 0.55
    sigma = 0.40
    return float(math.exp(-((float(reaction_time) - target) ** 2) / (2.0 * sigma * sigma)))


def moderation_score(smoothed_min_accel: Optional[float]) -> float:
    if smoothed_min_accel is None:
        return 0.0
    value = float(smoothed_min_accel)
    if value < float(OURS_MIN_ACCEL_RANGE[0]) or value > float(OURS_MIN_ACCEL_RANGE[1]):
        return 0.0
    center = -2.25
    sigma = 0.95
    return float(math.exp(-((value - center) ** 2) / (2.0 * sigma * sigma)))


def compute_candidate_diagnostics(method_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    per_method = {method: compute_method_diagnostics(rows) for method, rows in method_rows.items()}
    ours_diag = per_method.get(ANCHOR_METHOD, {})

    coverage_components = []
    for method in PLOT_METHODS:
        diag = per_method.get(method, {})
        coverage_components.append(
            0.5 * float(diag.get("actor_in_world_rate", 0.0) or 0.0)
            + 0.5 * float(diag.get("ttc_valid_rate", 0.0) or 0.0)
        )
        coverage_components.append(
            min(1.0, float(diag.get("valid_ttc_points_focus", 0.0) or 0.0) / 5.0)
        )

    baseline_separations: List[float] = []
    ours_reaction = ours_diag.get("reaction_time")
    ours_min_accel = ours_diag.get("smoothed_min_accel")
    for method in ("no_aux", "dense_comm"):
        diag = per_method.get(method, {})
        reaction_gap_score = 0.0
        if ours_reaction is not None and diag.get("reaction_time") is not None:
            reaction_gap = float(diag["reaction_time"]) - float(ours_reaction)
            reaction_gap_score = max(0.0, min(1.0, reaction_gap / 0.8))
        shape_gap_score = 0.0
        if ours_min_accel is not None and diag.get("smoothed_min_accel") is not None:
            shape_gap = abs(float(diag["smoothed_min_accel"]) - float(ours_min_accel))
            shape_gap_score = min(1.0, shape_gap / 3.0)
        baseline_separations.append(0.6 * reaction_gap_score + 0.4 * shape_gap_score)
    baseline_separations.sort(reverse=True)
    cross_method_separation_score = float(np.mean(baseline_separations[:2])) if baseline_separations else 0.0
    meaningful_ttc_descent_methods = []
    for method in PLOT_METHODS:
        drop = per_method.get(method, {}).get("ttc_drop")
        if drop is None:
            continue
        threshold = OURS_TTC_DROP_MIN if method == ANCHOR_METHOD else BASELINE_TTC_DROP_MIN
        if float(drop) > float(threshold):
            meaningful_ttc_descent_methods.append(method)

    late_ttc_rebound_flag = any(
        bool(per_method.get(method, {}).get("late_ttc_rebound_flag", False))
        for method in PLOT_METHODS
    )

    return {
        "per_method": per_method,
        "coverage_score": float(np.mean(coverage_components)) if coverage_components else 0.0,
        "cross_method_separation_score": cross_method_separation_score,
        "meaningful_ttc_descent_methods": meaningful_ttc_descent_methods,
        "meaningful_ttc_descent_count": len(meaningful_ttc_descent_methods),
        "late_ttc_rebound_flag": late_ttc_rebound_flag,
    }


def hard_filter_candidate(
    anchor_rows: Sequence[Dict[str, Any]],
    method_rows: Dict[str, List[Dict[str, Any]]],
    diagnostics: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    anchor_window = rows_in_window(anchor_rows, WINDOW_S)
    if not anchor_window:
        reasons.append("empty_anchor_window")
    if valid_ttc_count(anchor_rows) < 4:
        reasons.append("too_few_valid_ttc")

    per_method = diagnostics["per_method"]
    ours_diag = per_method.get(ANCHOR_METHOD, {})
    ours_ttc_drop = ours_diag.get("ttc_drop")
    if ours_ttc_drop is None:
        reasons.append("ours_ttc_drop_missing")
    elif float(ours_ttc_drop) <= float(OURS_TTC_DROP_MIN):
        reasons.append("ours_ttc_drop_low")

    baseline_has_descent = False
    for baseline in ("no_aux", "dense_comm"):
        baseline_drop = per_method.get(baseline, {}).get("ttc_drop")
        if baseline_drop is not None and float(baseline_drop) > float(BASELINE_TTC_DROP_MIN):
            baseline_has_descent = True
            break
    if not baseline_has_descent:
        reasons.append("no_baseline_ttc_descent")

    # Representative figure should not show ours with extreme or too-weak braking.
    ours_smoothed_min_accel = ours_diag.get("smoothed_min_accel")
    if ours_smoothed_min_accel is None:
        reasons.append("ours_missing_accel")
    else:
        if float(ours_smoothed_min_accel) < float(OURS_MIN_ACCEL_RANGE[0]):
            reasons.append("ours_extreme_braking")
        if float(ours_smoothed_min_accel) > float(OURS_MIN_ACCEL_RANGE[1]):
            reasons.append("ours_too_passive")

    # Ours should not react clearly later than the main baselines.
    ours_reaction_time = ours_diag.get("reaction_time")
    if ours_reaction_time is None:
        reasons.append("ours_no_reaction_onset")
    for baseline in ("no_aux", "dense_comm"):
        baseline_rt = per_method.get(baseline, {}).get("reaction_time")
        if ours_reaction_time is not None and baseline_rt is not None:
            if float(ours_reaction_time) > float(baseline_rt) + float(REACTION_ALLOWANCE_S):
                reasons.append(f"ours_reacts_later_than_{baseline}")

    if bool(diagnostics.get("late_ttc_rebound_flag", False)):
        reasons.append("late_ttc_rebound_artifact")

    # The shared reference actor must remain meaningful for the three plotted methods.
    for method in PLOT_METHODS:
        rows = method_rows.get(method, [])
        window_rows = rows_in_window(rows, WINDOW_S)
        usable = bool(window_rows) and sum(1 for r in window_rows if r.get("ego_speed_ms") is not None) >= 5
        if not usable:
            reasons.append(f"missing_method_{method}")
            continue
        diag = per_method.get(method, {})
        if float(diag.get("actor_in_world_rate", 0.0) or 0.0) < float(MIN_ACTOR_IN_WORLD_RATE):
            reasons.append(f"{method}_actor_in_world_low")
        if float(diag.get("ttc_valid_rate", 0.0) or 0.0) < float(MIN_TTC_VALID_RATE):
            reasons.append(f"{method}_ttc_valid_low")
        if int(diag.get("valid_ttc_points_focus", 0) or 0) < int(MIN_VALID_TTC_POINTS_FOCUS):
            reasons.append(f"{method}_valid_ttc_points_low")

    return (len(reasons) == 0), reasons


def score_candidate(
    anchor_rows: Sequence[Dict[str, Any]],
    method_rows: Dict[str, List[Dict[str, Any]]],
    diagnostics: Dict[str, Any],
) -> Dict[str, float]:
    per_method = diagnostics["per_method"]
    ours_diag = per_method.get(ANCHOR_METHOD, {})

    ours_ttc_drop = ours_diag.get("ttc_drop")
    if ours_ttc_drop is None:
        ttc_drop_score = 0.0
    else:
        # Prefer a clear TTC descent from pre-onset to post-onset.
        ttc_drop_score = float(1.0 - math.exp(-max(0.0, float(ours_ttc_drop)) / 1.8))

    # Prefer clean sustained threshold crossing instead of a one-step dip.
    below_run = float(ours_diag.get("below_threshold_run_steps", 0) or 0.0)
    onset_clarity_score = min(1.0, below_run / 8.0)

    # Prefer earlier reaction, but do not over-reward immediate aggressive braking.
    ours_reaction_time = ours_diag.get("reaction_time")
    ours_reaction_latency_score = bounded_reaction_latency_score(ours_reaction_time)

    # Prefer low oscillation in the smoothed acceleration trace.
    smoothness_metric = ours_diag.get("smoothness_madiff")
    if smoothness_metric is None:
        ours_smoothness_score = 0.0
    else:
        ours_smoothness_score = float(1.0 / (1.0 + float(smoothness_metric)))

    # Prefer meaningful braking without very harsh spikes.
    ours_moderation_score = moderation_score(ours_diag.get("smoothed_min_accel"))

    # Prefer visible separation from the two communication baselines and high actor/TTC coverage.
    cross_method_separation_score = float(diagnostics.get("cross_method_separation_score", 0.0) or 0.0)
    coverage_score = float(diagnostics.get("coverage_score", 0.0) or 0.0)
    baseline_support = 0.0
    for method in ("no_aux", "dense_comm"):
        drop = per_method.get(method, {}).get("ttc_drop")
        if drop is not None and float(drop) > float(BASELINE_TTC_DROP_MIN):
            baseline_support += 1.0
    descent_support_score = min(1.0, baseline_support / 2.0)
    late_rebound_penalty = 0.0 if bool(diagnostics.get("late_ttc_rebound_flag", False)) else 1.0

    # Weights are intentionally explicit and easy to tune later.
    weights = {
        "ttc_drop_score": 0.25,
        "onset_clarity_score": 0.14,
        "ours_reaction_latency_score": 0.16,
        "ours_smoothness_score": 0.12,
        "ours_moderation_score": 0.16,
        "cross_method_separation_score": 0.08,
        "coverage_score": 0.07,
        "descent_support_score": 0.02,
    }
    total = (
        weights["ttc_drop_score"] * ttc_drop_score
        + weights["onset_clarity_score"] * onset_clarity_score
        + weights["ours_reaction_latency_score"] * ours_reaction_latency_score
        + weights["ours_smoothness_score"] * ours_smoothness_score
        + weights["ours_moderation_score"] * ours_moderation_score
        + weights["cross_method_separation_score"] * cross_method_separation_score
        + weights["coverage_score"] * coverage_score
        + weights["descent_support_score"] * descent_support_score
    )
    total *= late_rebound_penalty
    return {
        "ttc_drop_score": float(ttc_drop_score),
        "onset_clarity_score": float(onset_clarity_score),
        "ours_reaction_latency_score": float(ours_reaction_latency_score),
        "ours_smoothness_score": float(ours_smoothness_score),
        "ours_moderation_score": float(ours_moderation_score),
        "cross_method_separation_score": float(cross_method_separation_score),
        "coverage_score": float(coverage_score),
        "descent_support_score": float(descent_support_score),
        "reaction_latency_s": None if ours_reaction_time is None else float(ours_reaction_time),
        "late_rebound_penalty": float(late_rebound_penalty),
        "anchor_score": float(total),
        "total_score": float(total),
    }


def build_candidate_for_seed(
    seed: int,
    episode_id: int,
    traces_by_method: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    anchor_trace = traces_by_method.get(ANCHOR_METHOD)
    if not anchor_trace:
        return []

    candidate_records: List[Dict[str, Any]] = []
    for ego_id in anchor_trace["initial_rl_agent_ids"]:
        critical = select_anchor_critical_actor(anchor_trace, ego_id=ego_id, ttc_threshold=TTC_THRESHOLD)
        if not critical:
            continue
        onset_step = detect_conflict_onset(critical["ttc_series"], TTC_THRESHOLD, ONSET_CONSECUTIVE_STEPS)
        if onset_step is None:
            continue
        scenario_actor_key = build_scenario_actor_key(anchor_trace, ego_id=ego_id)
        matched_actor_ids = resolve_runtime_actor_ids(
            traces_by_method=traces_by_method,
            ego_id=ego_id,
            anchor_actor_id=str(critical["actor_id"]),
            onset_step=int(onset_step),
        )

        method_rows: Dict[str, List[Dict[str, Any]]] = {}
        for method_name, trace in traces_by_method.items():
            method_rows[method_name] = build_pair_rows(
                trace,
                ego_id=ego_id,
                scenario_actor_key=scenario_actor_key,
                runtime_actor_id=matched_actor_ids.get(method_name),
                anchor_actor_id=str(critical["actor_id"]),
                onset_step=int(onset_step),
                critical_actor_source_method=ANCHOR_METHOD,
                ttc_threshold=TTC_THRESHOLD,
            )

        diagnostics = compute_candidate_diagnostics(method_rows)
        ok, reasons = hard_filter_candidate(method_rows[ANCHOR_METHOD], method_rows, diagnostics)
        metrics = score_candidate(method_rows[ANCHOR_METHOD], method_rows, diagnostics)
        per_method = diagnostics["per_method"]
        candidate = {
            "seed": int(seed),
            "episode_id": int(episode_id),
            "scenario_key": anchor_trace["scenario_key"],
            "ego_id": ego_id,
            "critical_actor_id": scenario_actor_key,
            "critical_actor_anchor_id": str(critical["actor_id"]),
            "critical_actor_source_method": ANCHOR_METHOD,
            "onset_step": int(onset_step),
            "valid_methods": sorted(method_rows.keys()),
            "filter_reasons": reasons,
            "rejection_reason": "" if ok else ";".join(reasons),
            "passed_hard_filter": bool(ok),
            "diagnostics": diagnostics,
            "scores": metrics,
            "late_ttc_rebound_flag": bool(diagnostics.get("late_ttc_rebound_flag", False)),
            "preview_path": "",
            "_method_rows": method_rows,
        }
        for method in METHOD_ORDER:
            diag = per_method.get(method, {})
            candidate[f"{method}_matched_actor_id"] = matched_actor_ids.get(method)
            candidate[f"{method}_min_ttc"] = diag.get("min_ttc_post_window")
            candidate[f"{method}_reaction_time"] = diag.get("reaction_time")
            candidate[f"{method}_smoothed_min_accel"] = diag.get("smoothed_min_accel")
            candidate[f"{method}_actor_in_world_rate"] = diag.get("actor_in_world_rate")
            candidate[f"{method}_ttc_valid_rate"] = diag.get("ttc_valid_rate")
            candidate[f"{method}_valid_ttc_points_focus"] = diag.get("valid_ttc_points_focus")
            candidate[f"{method}_ttc_drop"] = diag.get("ttc_drop")
        candidate_records.append(candidate)
    return candidate_records


def sanitize_candidate_for_export(candidate: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(candidate)
    clean.pop("_method_rows", None)
    return clean


def export_case_csvs(
    candidate: Dict[str, Any],
    output_root: Path,
    method_rows: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Path, Path, int, int]:
    all_rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        rows = method_rows.get(method, [])
        for row in rows:
            row = dict(row)
            row["rank"] = candidate.get("rank", "")
            row["anchor_score"] = float(candidate["scores"]["anchor_score"])
            row["window_left_s"] = float(WINDOW_S[0])
            row["window_right_s"] = float(WINDOW_S[1])
            all_rows.append(row)

    fieldnames = list(all_rows[0].keys()) if all_rows else []
    windowed_rows = rows_in_window(all_rows, WINDOW_S)
    full_path = output_root / "conflict_case_top1_full.csv"
    win_path = output_root / "conflict_case_top1_windowed.csv"
    write_csv(full_path, all_rows, fieldnames)
    write_csv(win_path, windowed_rows, fieldnames)

    episode_tag = f"seed{candidate['seed']}_ep{candidate['episode_id']:03d}"
    write_csv(output_root / f"conflict_case_{episode_tag}_full.csv", all_rows, fieldnames)
    write_csv(output_root / f"conflict_case_{episode_tag}_windowed.csv", windowed_rows, fieldnames)
    return full_path, win_path, len(all_rows), len(windowed_rows)


def configure_preview_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 2.6,
            "savefig.bbox": "tight",
        }
    )


def save_preview_plot(
    method_rows: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
) -> None:
    configure_preview_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_ttc, ax_accel) = plt.subplots(2, 1, sharex=True, figsize=(3.15, 3.7), constrained_layout=True)

    for method in PLOT_METHODS:
        rows = method_rows.get(method, [])
        if not rows:
            continue
        style = PREVIEW_STYLE[method]
        times = [float(row["time_s"]) for row in rows]
        ttc_vals = [np.nan if row.get("ttc") is None else min(6.0, float(row["ttc"])) for row in rows]
        accel_vals = centered_moving_average(
            [None if row.get("ego_accel") is None else float(row["ego_accel"]) for row in rows],
            SCREEN_SMOOTH_WINDOW,
        )
        accel_plot = [np.nan if v is None else float(v) for v in accel_vals]
        ax_ttc.plot(
            times,
            ttc_vals,
            label=METHOD_DISPLAY[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            markevery=3,
        )
        ax_accel.plot(
            times,
            accel_plot,
            label=METHOD_DISPLAY[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            markevery=3,
        )

    ax_ttc.axhline(TTC_THRESHOLD, color="#777777", linestyle="--", linewidth=0.9)
    ax_ttc.axvline(0.0, color="#999999", linestyle="--", linewidth=0.8)
    ax_accel.axhline(0.0, color="#777777", linestyle="--", linewidth=0.9)
    ax_accel.axvline(0.0, color="#999999", linestyle="--", linewidth=0.8)
    ax_ttc.set_ylabel("TTC (s)")
    ax_accel.set_ylabel("Accel")
    ax_accel.set_xlabel("Time (s)")
    ax_ttc.set_xlim(*WINDOW_S)
    ax_ttc.set_ylim(0.0, 6.0)

    for ax in (ax_ttc, ax_accel):
        ax.grid(True, which="major", color="#b5b5b5", alpha=0.20, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = ax_ttc.get_legend_handles_labels()
    if handles:
        ax_ttc.legend(handles, labels, loc="upper right", framealpha=0.95)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def export_preview_candidates(
    output_root: Path,
    candidates: Sequence[Dict[str, Any]],
    max_preview: int,
) -> None:
    preview_root = output_root / "preview_candidates"
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_plot_root = output_root / "previews"
    preview_plot_root.mkdir(parents=True, exist_ok=True)
    preview_pool = list(candidates[: max(0, int(max_preview))])
    for idx, candidate in enumerate(preview_pool, start=1):
        method_rows = candidate.get("_method_rows")
        if not isinstance(method_rows, dict):
            continue
        preview_dir = preview_root / f"top{idx:02d}_{candidate['scenario_key']}_{candidate['ego_id']}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        clean_candidate = sanitize_candidate_for_export(candidate)
        export_case_csvs(clean_candidate, preview_dir, method_rows)
        dump_json(preview_dir / "candidate_meta.json", {
            "rank": idx if candidate.get("passed_hard_filter", False) else "",
            "seed": candidate["seed"],
            "episode_id": candidate["episode_id"],
            "scenario_key": candidate["scenario_key"],
            "ego_id": candidate["ego_id"],
            "critical_actor_id": candidate["critical_actor_id"],
            "critical_actor_anchor_id": candidate.get("critical_actor_anchor_id"),
            "rejection_reason": candidate.get("rejection_reason", ""),
            "scores": candidate.get("scores", {}),
        })
        preview_png = preview_plot_root / f"rank{idx:02d}.png"
        save_preview_plot(method_rows, preview_png)
        candidate["preview_path"] = str(preview_png.relative_to(output_root))


def write_candidate_summary(output_root: Path, candidates: Sequence[Dict[str, Any]], top_k: int) -> None:
    fieldnames = [
        "rank",
        "seed",
        "episode_id",
        "scenario_key",
        "ego_id",
        "critical_actor_id",
        "critical_actor_anchor_id",
        "critical_actor_source_method",
        "onset_step",
        "valid_methods",
        "rejection_reason",
        "late_ttc_rebound_flag",
        "preview_path",
        "anchor_score",
        "total_score",
        "ttc_drop_score",
        "onset_clarity_score",
        "ours_reaction_latency_score",
        "ours_smoothness_score",
        "ours_moderation_score",
        "cross_method_separation_score",
        "coverage_score",
        "descent_support_score",
        "reaction_latency_s",
        "ours_min_ttc",
        "no_aux_min_ttc",
        "dense_comm_min_ttc",
        "lidar_only_min_ttc",
        "ours_matched_actor_id",
        "no_aux_matched_actor_id",
        "dense_comm_matched_actor_id",
        "lidar_only_matched_actor_id",
        "ours_reaction_time",
        "no_aux_reaction_time",
        "dense_comm_reaction_time",
        "lidar_only_reaction_time",
        "ours_smoothed_min_accel",
        "no_aux_smoothed_min_accel",
        "dense_comm_smoothed_min_accel",
        "lidar_only_smoothed_min_accel",
        "ours_actor_in_world_rate",
        "no_aux_actor_in_world_rate",
        "dense_comm_actor_in_world_rate",
        "lidar_only_actor_in_world_rate",
        "ours_ttc_valid_rate",
        "no_aux_ttc_valid_rate",
        "dense_comm_ttc_valid_rate",
        "lidar_only_ttc_valid_rate",
        "ours_valid_ttc_points_focus",
        "no_aux_valid_ttc_points_focus",
        "dense_comm_valid_ttc_points_focus",
        "lidar_only_valid_ttc_points_focus",
        "ours_ttc_drop",
        "no_aux_ttc_drop",
        "dense_comm_ttc_drop",
        "lidar_only_ttc_drop",
    ]
    summary_rows: List[Dict[str, Any]] = []
    for rank, cand in enumerate(candidates, start=1):
        scores = cand.get("scores", {})
        summary_rows.append(
            {
                "rank": rank if cand.get("passed_hard_filter", False) else "",
                "seed": cand["seed"],
                "episode_id": cand["episode_id"],
                "scenario_key": cand["scenario_key"],
                "ego_id": cand["ego_id"],
                "critical_actor_id": cand["critical_actor_id"],
                "critical_actor_anchor_id": cand.get("critical_actor_anchor_id"),
                "critical_actor_source_method": cand["critical_actor_source_method"],
                "onset_step": cand["onset_step"],
                "valid_methods": ",".join(cand.get("valid_methods", [])),
                "rejection_reason": cand.get("rejection_reason", ""),
                "late_ttc_rebound_flag": int(bool(cand.get("late_ttc_rebound_flag", False))),
                "preview_path": cand.get("preview_path", ""),
                "anchor_score": scores.get("anchor_score"),
                "total_score": scores.get("total_score"),
                "ttc_drop_score": scores.get("ttc_drop_score"),
                "onset_clarity_score": scores.get("onset_clarity_score"),
                "ours_reaction_latency_score": scores.get("ours_reaction_latency_score"),
                "ours_smoothness_score": scores.get("ours_smoothness_score"),
                "ours_moderation_score": scores.get("ours_moderation_score"),
                "cross_method_separation_score": scores.get("cross_method_separation_score"),
                "coverage_score": scores.get("coverage_score"),
                "descent_support_score": scores.get("descent_support_score"),
                "reaction_latency_s": scores.get("reaction_latency_s"),
                "ours_min_ttc": cand.get("ours_min_ttc"),
                "no_aux_min_ttc": cand.get("no_aux_min_ttc"),
                "dense_comm_min_ttc": cand.get("dense_comm_min_ttc"),
                "lidar_only_min_ttc": cand.get("lidar_only_min_ttc"),
                "ours_matched_actor_id": cand.get("ours_matched_actor_id"),
                "no_aux_matched_actor_id": cand.get("no_aux_matched_actor_id"),
                "dense_comm_matched_actor_id": cand.get("dense_comm_matched_actor_id"),
                "lidar_only_matched_actor_id": cand.get("lidar_only_matched_actor_id"),
                "ours_reaction_time": cand.get("ours_reaction_time"),
                "no_aux_reaction_time": cand.get("no_aux_reaction_time"),
                "dense_comm_reaction_time": cand.get("dense_comm_reaction_time"),
                "lidar_only_reaction_time": cand.get("lidar_only_reaction_time"),
                "ours_smoothed_min_accel": cand.get("ours_smoothed_min_accel"),
                "no_aux_smoothed_min_accel": cand.get("no_aux_smoothed_min_accel"),
                "dense_comm_smoothed_min_accel": cand.get("dense_comm_smoothed_min_accel"),
                "lidar_only_smoothed_min_accel": cand.get("lidar_only_smoothed_min_accel"),
                "ours_actor_in_world_rate": cand.get("ours_actor_in_world_rate"),
                "no_aux_actor_in_world_rate": cand.get("no_aux_actor_in_world_rate"),
                "dense_comm_actor_in_world_rate": cand.get("dense_comm_actor_in_world_rate"),
                "lidar_only_actor_in_world_rate": cand.get("lidar_only_actor_in_world_rate"),
                "ours_ttc_valid_rate": cand.get("ours_ttc_valid_rate"),
                "no_aux_ttc_valid_rate": cand.get("no_aux_ttc_valid_rate"),
                "dense_comm_ttc_valid_rate": cand.get("dense_comm_ttc_valid_rate"),
                "lidar_only_ttc_valid_rate": cand.get("lidar_only_ttc_valid_rate"),
                "ours_valid_ttc_points_focus": cand.get("ours_valid_ttc_points_focus"),
                "no_aux_valid_ttc_points_focus": cand.get("no_aux_valid_ttc_points_focus"),
                "dense_comm_valid_ttc_points_focus": cand.get("dense_comm_valid_ttc_points_focus"),
                "lidar_only_valid_ttc_points_focus": cand.get("lidar_only_valid_ttc_points_focus"),
                "ours_ttc_drop": cand.get("ours_ttc_drop"),
                "no_aux_ttc_drop": cand.get("no_aux_ttc_drop"),
                "dense_comm_ttc_drop": cand.get("dense_comm_ttc_drop"),
                "lidar_only_ttc_drop": cand.get("lidar_only_ttc_drop"),
            }
        )
    write_csv(output_root / "candidate_summary.csv", summary_rows, fieldnames)
    dump_json(
        output_root / "candidate_summary.json",
        {
            "top_k": int(top_k),
            "candidates": [row for row in summary_rows if row.get("rejection_reason", "") == ""][: max(1, int(top_k))],
            "all_candidates": summary_rows,
        },
    )


def select_export_candidate(
    candidates: Sequence[Dict[str, Any]],
    target_episode_id: Optional[int],
    target_ego_id: Optional[str] = None,
    allow_rejected_target: bool = False,
) -> Dict[str, Any]:
    passed = [cand for cand in candidates if cand.get("passed_hard_filter", False)]
    if not passed:
        if not (allow_rejected_target and target_episode_id is not None):
            raise RuntimeError("No valid representative conflict candidates were found.")
    if target_episode_id is None and target_ego_id is None:
        chosen = dict(passed[0], rank=1)
        chosen["_method_rows"] = passed[0].get("_method_rows")
        return chosen
    pool = list(passed)
    if allow_rejected_target and target_episode_id is not None:
        pool = list(candidates)
    filtered = pool
    if target_episode_id is not None:
        filtered = [cand for cand in filtered if int(cand["episode_id"]) == int(target_episode_id)]
    if target_ego_id is not None:
        filtered = [cand for cand in filtered if str(cand["ego_id"]) == str(target_ego_id)]
    if not filtered:
        raise RuntimeError(
            f"No candidate found for target_episode_id={target_episode_id}, target_ego_id={target_ego_id}"
        )
    filtered.sort(key=lambda c: c["scores"]["anchor_score"], reverse=True)
    chosen = dict(filtered[0])
    chosen["_method_rows"] = filtered[0].get("_method_rows")
    if chosen.get("passed_hard_filter", False):
        chosen["rank"] = next((cand.get("rank", 1) for cand in passed if cand is filtered[0]), chosen.get("rank", 1))
    return chosen


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    specs = build_method_specs()

    candidates: List[Dict[str, Any]] = []
    traces_cache: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for episode_offset in range(int(args.episodes)):
        seed = int(args.start_seed) + episode_offset
        episode_id = episode_offset
        traces_by_method: Dict[str, Dict[str, Any]] = {}
        for method in METHOD_ORDER:
            traces_by_method[method] = rollout_episode(
                specs[method],
                seed=seed,
                episode_id=episode_id,
                num_agents=int(args.num_agents),
                map_sequence=str(args.map_sequence),
                max_steps=int(args.max_steps),
                device=device,
            )
        traces_cache[(seed, episode_id)] = traces_by_method
        candidates.extend(build_candidate_for_seed(seed=seed, episode_id=episode_id, traces_by_method=traces_by_method))

    passed_candidates = [cand for cand in candidates if cand.get("passed_hard_filter", False)]
    passed_candidates.sort(key=lambda c: c["scores"]["anchor_score"], reverse=True)
    rejected_candidates = [cand for cand in candidates if not cand.get("passed_hard_filter", False)]
    rejected_candidates.sort(key=lambda c: c["scores"]["anchor_score"], reverse=True)
    ranked_candidates = passed_candidates + rejected_candidates

    for rank, cand in enumerate(passed_candidates, start=1):
        cand["rank"] = rank
    for cand in rejected_candidates:
        cand["rank"] = ""

    export_preview_candidates(output_root, ranked_candidates, max_preview=min(int(args.top_k), PREVIEW_EXPORT_TOP_K))
    write_candidate_summary(output_root, ranked_candidates, top_k=args.top_k)
    print(f"[ConflictCase] total scanned: {len(candidates)}")
    print(f"[ConflictCase] hard-pass count: {len(passed_candidates)}")
    top10_pool = ranked_candidates[:10]
    if top10_pool:
        top10 = ", ".join(f"{cand['scenario_key']}:{cand['ego_id']}" for cand in top10_pool)
        print(f"[ConflictCase] top-10 episode ids: {top10}")
    else:
        print("[ConflictCase] top-10 episode ids: none")

    top3_pool = ranked_candidates[:3]
    for idx, cand in enumerate(top3_pool, start=1):
        print(
            f"[ConflictCase] top-{idx}: {cand['scenario_key']} ego={cand['ego_id']} "
            f"| reaction ours/no_aux/dense={cand.get('ours_reaction_time')}/{cand.get('no_aux_reaction_time')}/{cand.get('dense_comm_reaction_time')} "
            f"| minTTC ours/no_aux/dense={cand.get('ours_min_ttc')}/{cand.get('no_aux_min_ttc')}/{cand.get('dense_comm_min_ttc')} "
            f"| minAccel ours/no_aux/dense={cand.get('ours_smoothed_min_accel')}/{cand.get('no_aux_smoothed_min_accel')}/{cand.get('dense_comm_smoothed_min_accel')}"
        )

    if passed_candidates:
        top1 = passed_candidates[0]
        coverage_bits = []
        for method in PLOT_METHODS:
            coverage_bits.append(
                f"{method}:in={float(top1.get(f'{method}_actor_in_world_rate') or 0.0):.2f}/ttc={float(top1.get(f'{method}_ttc_valid_rate') or 0.0):.2f}"
            )
        print(
            "[ConflictCase] top-1 diagnostics: "
            f"critical_actor={top1['critical_actor_id']}, "
            f"ours_min_ttc={top1.get('ours_min_ttc')}, "
            f"ours_reaction_time={top1.get('ours_reaction_time')}, "
            f"ours_smoothed_min_accel={top1.get('ours_smoothed_min_accel')}, "
            f"coverage={' | '.join(coverage_bits)}"
        )

    if not passed_candidates and not (bool(args.allow_rejected_target) and (args.target_episode_id is not None or args.target_ego_id is not None)):
        print("[ConflictCase] No valid representative conflict candidates were found in this scan.")
        print("[ConflictCase] candidate_summary.json has been written; increase --episodes for a real export.")
        return
    chosen = select_export_candidate(
        ranked_candidates,
        target_episode_id=args.target_episode_id,
        target_ego_id=args.target_ego_id,
        allow_rejected_target=bool(args.allow_rejected_target),
    )
    if not chosen.get("passed_hard_filter", False):
        print("[ConflictCase] Exporting a targeted candidate that did not pass the current hard filter.")
        print(f"[ConflictCase] rejection_reason: {chosen.get('rejection_reason', '')}")

    export_rows = chosen.get("_method_rows")
    if not isinstance(export_rows, dict):
        traces_for_export = traces_cache[(int(chosen["seed"]), int(chosen["episode_id"]))]
        export_rows = {
            method: build_pair_rows(
                trace,
                ego_id=chosen["ego_id"],
                scenario_actor_key=chosen["critical_actor_id"],
                runtime_actor_id=chosen.get(f"{method}_matched_actor_id"),
                anchor_actor_id=chosen.get("critical_actor_anchor_id", ""),
                onset_step=chosen["onset_step"],
                critical_actor_source_method=ANCHOR_METHOD,
                ttc_threshold=TTC_THRESHOLD,
            )
            for method, trace in traces_for_export.items()
        }
    clean_chosen = sanitize_candidate_for_export(chosen)
    full_path, win_path, full_rows, win_rows = export_case_csvs(clean_chosen, output_root, export_rows)

    print(f"[ConflictCase] chosen critical actor: {chosen['critical_actor_id']}")
    print(f"[ConflictCase] critical actor source method: {chosen['critical_actor_source_method']}")
    print(f"[ConflictCase] onset step: {chosen['onset_step']}")
    print(f"[ConflictCase] valid methods: {', '.join(chosen['valid_methods'])}")
    print(f"[ConflictCase] exported full trace: {full_path} ({full_rows} rows)")
    print(f"[ConflictCase] exported windowed trace: {win_path} ({win_rows} rows)")


if __name__ == "__main__":
    main()
