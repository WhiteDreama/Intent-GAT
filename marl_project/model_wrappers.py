"""
model_wrappers.py

Template for adapting Waymo samples into the MetaDrive-trained model input
format (91-D vector: 19-D ego-state + 72-D pseudo-LiDAR), and a wrapper to run
the model and return denormalized predictions. Replace TODOs with real logic.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from math import cos, sin


def compute_pseudo_lidar(sample: dict, num_rays: int = 72, lidar_range: float = 50.0,
                         default_length: float = 4.5, default_width: float = 2.0) -> np.ndarray:
    """Compute a pseudo 2D LiDAR (hit fraction per beam) from surrounding boxes.

    - Looks for candidate agent lists in common sample keys and extracts
      (center_x, center_y, heading, length, width) for each agent.
    - Transforms agent centers into the ego frame if `sample['ego_init']` exists
      (ego_init = (ego_x, ego_y, ego_yaw)). The ego frame used here matches the
      one used by the miner/evaluator: ego at (0,0) facing +X.
    - Rays originate at (0,0) and extend along directions evenly spaced over
      2*pi. For each ray, the distance to the nearest intersection with any
      box is computed; the returned value is clipped(distance / lidar_range, 0,1).

    The function is defensive about missing fields and uses sensible defaults.
    """
    # Helper: extract agent list from various possible keys
    candidates = None
    for key in ("surrounding_tracks", "agents", "other_agents", "neighbors", "neighbors_list", "agents_list", "tracks"):
        if key in sample and sample[key] is not None:
            candidates = sample[key]
            break
    # Some datasets may encode agents in sample['meta']
    if candidates is None and isinstance(sample.get("meta"), dict):
        for k in ("agents", "other_agents", "tracks"):
            if k in sample["meta"]:
                candidates = sample["meta"][k]
                break

    # If nothing found, return all-ones (no hits)
    if candidates is None:
        return np.ones(num_rays, dtype=float)

    # Extract ego origin if present (global -> ego transform)
    ego_init = sample.get("ego_init")
    if ego_init is not None and len(ego_init) >= 3:
        ego_x, ego_y, ego_yaw = float(ego_init[0]), float(ego_init[1]), float(ego_init[2])
        cos_ey, sin_ey = np.cos(-ego_yaw), np.sin(-ego_yaw)
        def to_ego(px, py):
            dx = px - ego_x
            dy = py - ego_y
            # rotate by -ego_yaw
            ex = cos_ey * dx - sin_ey * dy
            ey = sin_ey * dx + cos_ey * dy
            return float(ex), float(ey)
    else:
        # Assume coordinates already in ego frame
        def to_ego(px, py):
            return float(px), float(py)
        ego_yaw = 0.0

    # Helper: extract center, heading, length, width from an agent entry
    def extract_agent_info(agent) -> tuple[float, float, float, float, float] | None:
        try:
            # If agent is an array-like of coords
            if isinstance(agent, (list, tuple, np.ndarray)) and len(agent) >= 2:
                cx, cy = float(agent[0]), float(agent[1])
                heading = float(agent[2]) if len(agent) > 2 else 0.0
                return cx, cy, heading, default_length, default_width

            if isinstance(agent, dict):
                # center extraction
                if "center" in agent:
                    c = agent["center"]
                    cx, cy = float(c[0]), float(c[1])
                elif "position" in agent:
                    p = agent["position"]
                    cx, cy = float(p[0]), float(p[1])
                elif "x" in agent and "y" in agent:
                    cx, cy = float(agent["x"]), float(agent["y"])
                elif "bbox" in agent:
                    # bbox could be [x_min,y_min,x_max,y_max] or corners
                    b = agent["bbox"]
                    b = np.asarray(b, dtype=float)
                    if b.size >= 4:
                        xmin, ymin, xmax, ymax = b.ravel()[:4]
                        cx, cy = float((xmin + xmax) / 2.0), float((ymin + ymax) / 2.0)
                    else:
                        return None
                else:
                    return None

                # heading
                heading = 0.0
                for hkey in ("heading", "yaw", "theta", "rotation"):
                    if hkey in agent:
                        heading = float(agent[hkey])
                        break

                # sizes
                length = None
                width = None
                for lkey in ("length", "l", "size_x", "size_long", "dim_x"):
                    if lkey in agent:
                        length = float(agent[lkey])
                        break
                for wkey in ("width", "w", "size_y", "size_short", "dim_y"):
                    if wkey in agent:
                        width = float(agent[wkey])
                        break

                if length is None or width is None:
                    # try to infer from bbox if present
                    if "bbox" in agent:
                        b = np.asarray(agent["bbox"], dtype=float)
                        if b.size >= 4:
                            xmin, ymin, xmax, ymax = b.ravel()[:4]
                            length = abs(xmax - xmin)
                            width = abs(ymax - ymin)

                if length is None:
                    length = default_length
                if width is None:
                    width = default_width

                return cx, cy, heading, length, width
        except Exception:
            return None
        return None

    # Build list of boxes in ego frame: each box as list of 4 corner (x,y) tuples
    boxes = []
    for a in candidates:
        info = extract_agent_info(a)
        if info is None:
            continue
        cx, cy, yaw, length, width = info
        # transform center to ego frame
        ex, ey = to_ego(cx, cy)
        yaw_ego = float(yaw - ego_yaw)

        # compute corners in world (ego) frame
        hl = length / 2.0
        hw = width / 2.0
        # local corners relative to center
        local_corners = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]], dtype=float)
        cosa = np.cos(yaw_ego)
        sina = np.sin(yaw_ego)
        rot = np.array([[cosa, -sina], [sina, cosa]], dtype=float)
        world_corners = (local_corners @ rot.T) + np.array([ex, ey])
        boxes.append(world_corners.tolist())

    if len(boxes) == 0:
        return np.ones(num_rays, dtype=float)

    # Ray casting utilities
    angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
    ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (num_rays,2)

    def ray_segment_intersection(rx, ry, ax, ay, bx, by):
        # Solve o + t*r = a + u*(b-a) where o=(0,0), r=(rx,ry)
        dx = bx - ax
        dy = by - ay
        det = rx * (-dy) - ry * (-dx)
        if abs(det) < 1e-8:
            return None
        # Solve for t and u using Cramer's rule
        # [rx  -dx][t] = [ax]
        # [ry  -dy][u]   [ay]
        t = (ax * (-dy) - ay * (-dx)) / det
        u = (rx * ay - ry * ax) / det
        # u between 0 and 1 indicates intersection on segment
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    lidar = np.ones(num_rays, dtype=float)
    for i in range(num_rays):
        rx, ry = float(ray_dirs[i, 0]), float(ray_dirs[i, 1])
        min_t = float(lidar_range) + 1.0
        for box in boxes:
            # box is list of four corners
            for j in range(len(box)):
                ax, ay = box[j]
                bx, by = box[(j + 1) % len(box)]
                t = ray_segment_intersection(rx, ry, ax, ay, bx, by)
                if t is None:
                    continue
                # t is distance along unit ray
                if t < min_t:
                    min_t = t
        if min_t <= float(lidar_range):
            lidar[i] = float(np.clip(min_t / float(lidar_range), 0.0, 1.0))
        else:
            lidar[i] = 1.0

    return lidar

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - torch optional
    torch = None
    Tensor = Any


class WaymoToMetaDriveAdapter:
    """Adapter converting a single Waymo `sample` dict into a (91,) vector.

    Expected sample keys provided by miner: 'gt_trajectory_xy', 'gt_yaw',
    'gt_velocity_ms', and optionally 'meta' and 'ego_init'.
    """

    def __init__(self):
        # Any normalization constants or lookup tables can be set here
        self.MAX_SPEED_KMH = 40.0

    def _get_current_speed_ms(self, sample: dict) -> float:
        # Prefer an explicit field, fallback to gt_velocity_ms[0]
        if "current_speed_ms" in sample:
            return float(sample["current_speed_ms"])
        vel = sample.get("gt_velocity_ms")
        if vel is not None:
            return float(np.asarray(vel, dtype=float).flat[0])
        # If no speed available, default to 0.0
        return 0.0

    def construct_ego_state(self, sample: dict) -> np.ndarray:
        """Build a 19-D ego-state vector normalized to [0,1].

        Minimal implementation populates `speed_norm` per spec and fills the
        remaining slots with zeros as placeholders for later fields.
        """
        speed_ms = self._get_current_speed_ms(sample)
        speed_kmh = speed_ms * 3.6
        speed_norm = (speed_kmh + 1.0) / (self.MAX_SPEED_KMH + 1.0)
        speed_norm = float(np.clip(speed_norm, 0.0, 1.0))

        # Placeholder slots for other kinematic features.
        # TODO: replace zeros with computed/normalized steering, heading diff,
        # lateral offset, acceleration, etc. Ensure each feature is in [0,1].
        placeholders = np.zeros(18, dtype=float)

        ego19 = np.concatenate(([speed_norm], placeholders), axis=0)
        if ego19.shape != (19,):
            raise RuntimeError("ego-state vector must be shape (19,)")
        return ego19

    def construct_pseudo_lidar(self, sample: dict) -> np.ndarray:
        """Produce a 72-D pseudo-LiDAR hit-fractions vector.

        This uses 2D raycasting against axis-aligned or rotated bounding boxes
        provided in `sample`. The function is robust to a variety of sample
        schemas: it will look for common keys containing surrounding agent
        lists and attempt to extract (x,y,heading,length,width). If sizes or
        headings are missing, sensible defaults are used.
        """
        # Delegate to module-level helper for clarity/testability
        return compute_pseudo_lidar(sample, num_rays=72, lidar_range=50.0)

    def adapt(self, sample: dict) -> np.ndarray:
        """Return a (91,) NumPy array ready for the model.

        Steps: assemble 19-D ego, 72-D lidar, concat and final validation.
        """
        ego19 = self.construct_ego_state(sample)
        lidar72 = self.construct_pseudo_lidar(sample)
        obs91 = np.concatenate([ego19, lidar72], axis=0)
        if obs91.shape != (91,):
            raise RuntimeError(f"Adapted observation must be shape (91,), got {obs91.shape}")
        return obs91


def run_model(sample: dict, model: Callable, device: Optional[str] = None) -> np.ndarray:
    """Adapter wrapper to run the provided PyTorch `model` on a single sample.

    Args:
        sample: dict saved by miner containing GT and meta fields.
        model: callable or nn.Module. If nn.Module, it should accept a FloatTensor
               shaped (B,91) and return normalized waypoints in [-1,1].
        device: device string (e.g., 'cpu' or 'cuda') or None to use CPU.

    Returns:
        NumPy array of shape (T,2) representing normalized waypoints in [-1,1].

    Notes:
    - This function does not load checkpoints; it only prepares the observation
      and calls the provided model callable. Move the model to the correct
      device outside this function if desired.
    """
    adapter = WaymoToMetaDriveAdapter()
    obs91 = adapter.adapt(sample)  # (91,)

    # Convert to torch tensor
    if torch is None:
        raise RuntimeError("PyTorch not available in this environment")

    tensor = torch.from_numpy(obs91.astype(np.float32)).unsqueeze(0)  # (1,91)
    if device is not None:
        tensor = tensor.to(device)

    # Call model. Accept both nn.Module and generic callable.
    model_out = model(tensor)

    # Convert output to NumPy
    if isinstance(model_out, Tensor):
        out_np = model_out.detach().cpu().numpy()
    elif isinstance(model_out, np.ndarray):
        out_np = model_out
    else:
        # Try to coerce
        try:
            out_np = np.asarray(model_out)
        except Exception:
            raise RuntimeError("Model output could not be converted to NumPy array")

    # Normalize shapes to (T,2)
    if out_np.ndim == 3 and out_np.shape[0] == 1 and out_np.shape[2] == 2:
        out_np = out_np.squeeze(0)
    elif out_np.ndim == 2 and out_np.shape[0] == 1 and out_np.shape[1] % 2 == 0:
        # (1, N) -> (N/2, 2)
        N = out_np.shape[1]
        T = N // 2
        out_np = out_np.reshape(1, T, 2).squeeze(0)
    elif out_np.ndim == 2 and out_np.shape[1] == 2:
        # already (T,2)
        pass
    elif out_np.ndim == 1 and out_np.size % 2 == 0:
        T = out_np.size // 2
        out_np = out_np.reshape(T, 2)
    else:
        raise RuntimeError(f"Unexpected model output shape: {out_np.shape}")

    # At this point, out_np should be (T,2) in normalized units [-1,1]
    return out_np


__all__ = ["WaymoToMetaDriveAdapter", "run_model"]


def get_waymo_eval_callable(model_class_path: str, checkpoint_path: str, init_kwargs: dict, device: str = "cpu"):
    """Dynamically load model class, instantiate, load checkpoint, and return a callable.

    Args:
        model_class_path: import path to model class, e.g. 'marl_project.algo.intent_gat.IntentGATPolicy'
        checkpoint_path: path to .pth checkpoint file
        init_kwargs: dict of kwargs to pass to model constructor
        device: 'cpu' or 'cuda' or torch device string

    Returns:
        Callable[[dict], np.ndarray] that accepts a miner `sample` dict and returns
        a NumPy array (T,2) of normalized waypoints in [-1,1].
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to load the model")

    import importlib

    module_name, class_name = model_class_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    ModelClass = getattr(mod, class_name)

    # Instantiate model
    model = ModelClass(**init_kwargs)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Common checkpoint key patterns
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dicts" in ckpt and isinstance(ckpt["state_dicts"], dict):
            # try first value
            state = list(ckpt["state_dicts"].values())[0]
        else:
            # assume ckpt itself is the state_dict
            state = ckpt
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # Try to load by matching keys prefix (common when saved with module prefix)
        new_state = {}
        for k, v in state.items():
            new_key = k
            if new_key.startswith("module."):
                new_key = new_key[len("module."):]
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)

    model.to(device)
    model.eval()

    def eval_fn(sample: dict) -> np.ndarray:
        # Prefer calling the full policy which expects an obs_batch dict.
        with torch.no_grad():
            # Adapt sample to (1,91) tensor
            adapter = WaymoToMetaDriveAdapter()
            obs91 = adapter.adapt(sample)
            tensor = torch.from_numpy(obs91.astype(np.float32)).unsqueeze(0).to(device)

            # Build minimal obs_batch expected by CooperativePolicy.forward
            try:
                from marl_project.config import Config
                max_neighbors = getattr(Config, "MAX_NEIGHBORS", 8)
            except Exception:
                max_neighbors = 8

            node_features = tensor  # (1,91)
            neighbor_indices = torch.full((1, max_neighbors), -1, dtype=torch.long, device=device)
            neighbor_mask = torch.zeros((1, max_neighbors), dtype=torch.uint8, device=device)
            neighbor_rel_pos = torch.zeros((1, max_neighbors, 4), dtype=torch.float32, device=device)

            obs_batch = {
                "node_features": node_features,
                "neighbor_indices": neighbor_indices,
                "neighbor_mask": neighbor_mask,
                "neighbor_rel_pos": neighbor_rel_pos,
            }

            try:
                out = model(obs_batch)
                # Expect dict with key 'pred_waypoints'
                if isinstance(out, dict) and "pred_waypoints" in out:
                    pred = out["pred_waypoints"]
                else:
                    # Fallback: model may return tuple (z, pred_waypoints)
                    if isinstance(out, (list, tuple)) and len(out) >= 2:
                        pred = out[1]
                    else:
                        # Last resort: call generic run_model
                        pred = run_model(sample, model, device=device)

                if isinstance(pred, torch.Tensor):
                    pred_np = pred.detach().cpu().numpy()
                else:
                    pred_np = np.asarray(pred)

                # If shape is (1, T, 2) -> squeeze
                if pred_np.ndim == 3 and pred_np.shape[0] == 1:
                    pred_np = pred_np.squeeze(0)

                return pred_np

            except Exception:
                # If dict-based call fails, fallback to generic run_model wrapper
                return run_model(sample, model, device=device)

    return eval_fn
