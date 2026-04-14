import numpy as np
import torch
from metadrive.envs.marl_envs import MultiAgentMetaDrive
from marl_project.config import Config

class GraphEnvWrapper(MultiAgentMetaDrive):
    """
    Wrapper for MultiAgentMetaDrive that implements:
    1. Sim-to-Real Noise Injection (Lidar)
    2. Dynamic Graph Construction (Euclidean Distance)
    3. Packet Loss Simulation (Masking)
    4. Auxiliary Task Data Extraction (Future Waypoints)
    """
    
    def __init__(self, config=None):
        # Merge default config with user config
        meta_config = Config.get_metadrive_config()
        comm_mode = "iid"
        comm_burst_len = int(getattr(Config, "COMM_BURST_LEN", 1) or 1)
        comm_stale_steps = int(getattr(Config, "COMM_STALE_STEPS", 0) or 0)
        if config:
            comm_mode = str(config.get("comm_mode", comm_mode) or comm_mode)
            comm_burst_len = int(config.get("comm_burst_len", comm_burst_len) or comm_burst_len)
            comm_stale_steps = int(config.get("comm_stale_steps", comm_stale_steps) or comm_stale_steps)
            meta_config.update(config)

        # Remove wrapper-only keys before passing config to MetaDrive core.
        meta_config.pop("comm_mode", None)
        meta_config.pop("comm_burst_len", None)
        meta_config.pop("comm_stale_steps", None)
            
        super().__init__(meta_config)
        # Track previous actions for smoothing/reward shaping
        self.prev_actions = {}
        # Track previous longitudinal position for progress shaping
        self._prev_longs = {}
        # Track previous lane id to avoid progress jumps on lane switch
        self._prev_lane_ids = {}
        # Track long-idle steps (deadlock) per agent
        self._idle_counts = {}
        # Communication robustness semantics
        self._comm_mode = str(comm_mode or "iid").lower()
        self._comm_burst_len = max(1, int(comm_burst_len or 1))
        self._comm_stale_steps = max(0, int(comm_stale_steps or 0))
        self._comm_burst_remaining = {}
        self._stale_neighbor_cache = {}

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _get_rel_denoms() -> tuple:
        """Return (pos_denom, vel_denom) used for neighbor_rel_pos normalization."""
        pos_denom = getattr(Config, "REL_POS_NORM_POS_DENOM", None)
        if pos_denom is None:
            pos_denom = float(getattr(Config, "COMM_RADIUS", 80.0))
        pos_denom = max(1e-6, float(pos_denom))
        vel_denom = max(1e-6, float(getattr(Config, "REL_POS_NORM_VEL_DENOM", 15.0)))
        return pos_denom, vel_denom

    @staticmethod
    def _compute_neighbor_rel(
        ego_pos,
        ego_vel,
        ego_heading: float,
        n_pos,
        n_vel,
        *,
        pos_denom: float,
        vel_denom: float,
        normalize: bool,
    ):
        """Compute ego-local (dx,dy,dvx,dvy) from world coords.

        - ego-local frame: x forward, y left
        - normalize=True: divide by pos_denom/vel_denom for model input stability
        """
        ego_pos = np.asarray(ego_pos, dtype=np.float32)
        ego_vel = np.asarray(ego_vel, dtype=np.float32)
        n_pos = np.asarray(n_pos, dtype=np.float32)
        n_vel = np.asarray(n_vel, dtype=np.float32)

        c, s = np.cos(float(ego_heading)), np.sin(float(ego_heading))
        rel_pos = n_pos - ego_pos
        local_x = float(rel_pos[0] * c + rel_pos[1] * s)
        local_y = float(-rel_pos[0] * s + rel_pos[1] * c)
        rel_vel = n_vel - ego_vel
        local_vx = float(rel_vel[0] * c + rel_vel[1] * s)
        local_vy = float(-rel_vel[0] * s + rel_vel[1] * c)

        if normalize:
            return [local_x / float(pos_denom), local_y / float(pos_denom), local_vx / float(vel_denom), local_vy / float(vel_denom)]
        return [local_x, local_y, local_vx, local_vy]

    @staticmethod
    def _debug_graph_checks_enabled() -> bool:
        return bool(getattr(Config, "DEBUG_GRAPH_CHECKS", False))

    @staticmethod
    def _debug_graph_eps() -> float:
        return float(getattr(Config, "DEBUG_GRAPH_EPS", 1e-3))

    @staticmethod
    def _safe_speed_kmh(vehicle):
        if vehicle is None:
            return 0.0
        if hasattr(vehicle, "speed_km_h"):
            try:
                return float(vehicle.speed_km_h)
            except Exception:
                pass
        if hasattr(vehicle, "velocity"):
            try:
                v = np.asarray(vehicle.velocity, dtype=np.float32)
                return float(np.linalg.norm(v) * 3.6)
            except Exception:
                pass
        return 0.0

    def _prune_link_states(self, agent_id, neighbors_true):
        nset = set(neighbors_true)
        stale_keys = [k for k in self._stale_neighbor_cache.keys() if k[0] == agent_id and k[1] not in nset]
        for k in stale_keys:
            self._stale_neighbor_cache.pop(k, None)
        burst_keys = [k for k in self._comm_burst_remaining.keys() if k[0] == agent_id and k[1] not in nset]
        for k in burst_keys:
            self._comm_burst_remaining.pop(k, None)

    def _sample_link_drop(self, agent_id, neighbor_id, mask_ratio: float) -> bool:
        """Return True if this link is dropped for current step under current communication mode."""
        key = (agent_id, neighbor_id)
        if self._comm_mode == "burst":
            remaining = int(self._comm_burst_remaining.get(key, 0) or 0)
            if remaining > 0:
                self._comm_burst_remaining[key] = remaining - 1
                return True
            if np.random.random() < float(mask_ratio):
                self._comm_burst_remaining[key] = max(0, int(self._comm_burst_len) - 1)
                return True
            return False
        return bool(np.random.random() < float(mask_ratio))

    def _build_masked_neighbors_and_rel(
        self,
        agent_id,
        neighbors_true,
        agent_positions,
        agent_velocities,
        ego_pos,
        ego_vel,
        ego_heading: float,
        pos_denom: float,
        vel_denom: float,
    ):
        mask_ratio = float(getattr(Config, "MASK_RATIO", 0.0) or 0.0)
        masked_neighbors = []
        neighbor_rel_pos = []
        neighbor_rel_pos_true = []

        self._prune_link_states(agent_id, neighbors_true)

        for n_id in neighbors_true:
            if n_id not in agent_positions:
                continue

            rel_true = self._compute_neighbor_rel(
                ego_pos,
                ego_vel,
                ego_heading,
                agent_positions[n_id],
                agent_velocities[n_id],
                pos_denom=pos_denom,
                vel_denom=vel_denom,
                normalize=False,
            )
            neighbor_rel_pos_true.append(rel_true)

            link_drop = self._sample_link_drop(agent_id, n_id, mask_ratio)
            cache_key = (agent_id, n_id)

            if not link_drop:
                rel_norm = self._compute_neighbor_rel(
                    ego_pos,
                    ego_vel,
                    ego_heading,
                    agent_positions[n_id],
                    agent_velocities[n_id],
                    pos_denom=pos_denom,
                    vel_denom=vel_denom,
                    normalize=True,
                )
                masked_neighbors.append(n_id)
                neighbor_rel_pos.append(rel_norm)
                if self._comm_mode == "staleness" and self._comm_stale_steps > 0:
                    self._stale_neighbor_cache[cache_key] = {"rel": rel_norm, "age": 0}
                continue

            if self._comm_mode == "staleness" and self._comm_stale_steps > 0:
                cached = self._stale_neighbor_cache.get(cache_key)
                if isinstance(cached, dict):
                    cached_age = int(cached.get("age", 0) or 0)
                    if cached_age < self._comm_stale_steps:
                        rel_cached = list(cached.get("rel", [0.0, 0.0, 0.0, 0.0]))
                        masked_neighbors.append(n_id)
                        neighbor_rel_pos.append(rel_cached)
                        cached["age"] = cached_age + 1

        return masked_neighbors, neighbor_rel_pos, neighbor_rel_pos_true

    def _lane_metrics(self, vehicle):
        """Return (abs_lat, heading_err_rad, long, lane_id) or (None, None, None, None) if unavailable."""
        try:
            lane = getattr(vehicle, "lane", None)
            pos = getattr(vehicle, "position", None)
            if lane is None or pos is None:
                return None, None, None, None

            long, lat = lane.local_coordinates(pos)
            abs_lat = float(abs(lat))

            lane_heading = None
            if hasattr(lane, "heading_theta_at"):
                lane_heading = lane.heading_theta_at(long)
            elif hasattr(lane, "heading_at"):
                lane_heading = lane.heading_at(long)

            heading_err = None
            if lane_heading is not None and hasattr(vehicle, "heading_theta"):
                heading_err = float(abs(self._wrap_to_pi(float(vehicle.heading_theta) - float(lane_heading))))

            lane_id = None
            # MetaDrive lanes often have lane_index / index; fall back to object id
            if hasattr(lane, "lane_index"):
                lane_id = tuple(getattr(lane, "lane_index"))
            elif hasattr(lane, "index"):
                lane_id = tuple(getattr(lane, "index")) if isinstance(getattr(lane, "index"), (list, tuple)) else getattr(lane, "index")
            else:
                lane_id = id(lane)

            return abs_lat, heading_err, float(long), lane_id
        except Exception:
            return None, None, None, None

    def _get_spawn_manager(self):
        """Best-effort locate SpawnManager across MetaDrive versions.

        Some versions expose it as engine.spawn_manager, others nest it under
        engine.agent_manager.spawn_manager.
        """
        eng = getattr(self, "engine", None)
        if eng is None:
            return None
        sm = getattr(eng, "spawn_manager", None)
        if sm is not None:
            return sm
        am = getattr(eng, "agent_manager", None)
        sm = getattr(am, "spawn_manager", None) if am is not None else None
        return sm

    def _debug_spawn_capacity_after_reset(self, *, tag: str = "reset") -> dict:
        """Collect + optionally print spawn capacity diagnostics.

        Returns a dict with computed fields for reuse in error messages.
        """
        sm = self._get_spawn_manager()
        eng = getattr(self, "engine", None)
        gcfg = getattr(eng, "global_config", {}) if eng is not None else {}

        def _cfg_get(cfg, key, default=None):
            try:
                if hasattr(cfg, "get"):
                    return cfg.get(key, default)
            except Exception:
                pass
            return default

        map_cfg = _cfg_get(gcfg, "map_config", {})
        map_mode = str(getattr(Config, "MAP_MODE", "block_num"))
        info = {
            "tag": str(tag),
            "map_mode": map_mode,
            "map_type": (str(getattr(Config, "MAP_TYPE", "")) if map_mode == "block_sequence" else ""),
            "block_num": (int(getattr(Config, "MAP_BLOCK_NUM", 0)) if map_mode == "block_num" else None),
            "lane_num": _cfg_get(map_cfg, "lane_num", None),
            "exit_length": _cfg_get(map_cfg, "exit_length", None),
            "spawn_roads_cfg": _cfg_get(gcfg, "spawn_roads", None),
            "num_agents_cfg": _cfg_get(gcfg, "num_agents", None),
        }

        if sm is None:
            info["error"] = "spawn_manager_not_found"
            return info

        spawn_roads = getattr(sm, "spawn_roads", None)
        if spawn_roads is None:
            spawn_roads = []
        try:
            spawn_roads = list(spawn_roads)
        except Exception:
            spawn_roads = [spawn_roads]

        # Infer slot count from SpawnManager logic
        try:
            exit_length = float(getattr(sm, "exit_length"))
            respawn_long = float(getattr(sm, "RESPAWN_REGION_LONGITUDE"))
            num_slots = int(np.floor(exit_length / max(1e-6, respawn_long)))
        except Exception:
            exit_length, respawn_long, num_slots = None, None, None

        lane_num = info.get("lane_num")
        try:
            if lane_num is None:
                lane_num = int(getattr(sm, "lane_num"))
        except Exception:
            pass

        capacity = None
        if lane_num is not None and num_slots is not None:
            capacity = int(lane_num) * int(len(spawn_roads)) * int(max(0, num_slots))

        # Spawn lanes: approximate by counting unique spawn_lane_index in configs
        spawn_lane_count = None
        try:
            cfgs = getattr(sm, "available_agent_configs", None)
            if cfgs is not None:
                lane_set = set()
                for c in cfgs:
                    try:
                        lane_idx = c["config"]["spawn_lane_index"]
                        lane_set.add(tuple(lane_idx))
                    except Exception:
                        continue
                spawn_lane_count = len(lane_set)
        except Exception:
            pass

        road_ids = []
        for r in spawn_roads:
            try:
                road_ids.append(f"{getattr(r, 'start_node', '?')}->{getattr(r, 'end_node', '?')}")
            except Exception:
                road_ids.append(str(r))

        # Prefer exit_length from map_config for reporting reproducibility
        if info.get("exit_length") is None:
            info["exit_length"] = exit_length

        info.update(
            {
                "len_spawn_roads": int(len(spawn_roads)),
                "spawn_roads": road_ids,
                "spawn_lane_count": spawn_lane_count,
                "lane_num": lane_num,
                "exit_length_internal": exit_length,
                "respawn_region_longitude": respawn_long,
                "num_slots": num_slots,
                "capacity": capacity,
            }
        )

        # One-line summary (paper-friendly): only print when debug enabled or on failure.
        def _summary_line() -> str:
            mm = info.get("map_mode")
            mt = info.get("map_type")
            bn = info.get("block_num")
            ln = info.get("lane_num")
            el = info.get("exit_length")
            na = info.get("num_agents_cfg")
            sr = info.get("len_spawn_roads")
            ns = info.get("num_slots")
            cap = info.get("capacity")
            return (
                f"[SpawnCapacity:{tag}] map_mode={mm} map_type={mt} block_num={bn} "
                f"lane_num={ln} exit_length={el} num_agents={na} spawn_roads={sr} "
                f"num_slots={ns} capacity={cap}"
            )

        should_print = bool(getattr(Config, "DEBUG_SPAWN_ON_RESET", False))
        if should_print:
            print(_summary_line())
            print(f"[SpawnDebug:{tag}] spawn_roads={road_ids}")

        # Strict safety check: fail fast if capacity is insufficient.
        strict = bool(getattr(Config, "STRICT_SPAWN_CAPACITY", True))
        try:
            num_agents = int(info.get("num_agents_cfg") or 0)
        except Exception:
            num_agents = 0

        if strict and capacity is not None and num_agents > 0 and capacity < num_agents:
            # Always print one-line summary + details on failure.
            print(_summary_line())
            print(
                f"[SpawnDebug:{tag}] ERROR: capacity({capacity}) < num_agents({num_agents}). "
                "Refusing to continue (to avoid sampling spawn slots with replacement)."
            )
            print(f"[SpawnDebug:{tag}] spawn_roads={road_ids}")
            raise RuntimeError(
                "Spawn capacity insufficient for num_agents. "
                f"capacity={capacity}, num_agents={num_agents}, lane_num={lane_num}, "
                f"num_slots={num_slots}, len_spawn_roads={len(spawn_roads)}, "
                f"exit_length={info.get('exit_length')}, respawn_region_longitude={respawn_long}, "
                f"spawn_roads_cfg={info.get('spawn_roads_cfg')}"
            )

        return info
        
    def step(self, actions):
        """
        Step the environment and process observations.
        Returns a 4-tuple (obs, rewards, dones, infos) so it can be used inside SubprocVecEnv workers.
        """
        # === 动作平滑处理 (Action Smoothing) ===
        # 说明：只在环境侧做一次平滑，训练侧不要重复滤波。
        alpha = float(getattr(Config, "ACTION_SMOOTH_ALPHA", 0.0))

        smoothed_actions = {}
        if actions is None:
            actions = {}
        for agent_id, action in actions.items():
            act = np.asarray(action, dtype=np.float32)
            if alpha > 1e-8:
                prev_act = self.prev_actions.get(agent_id)
                if prev_act is not None:
                    prev_act = np.asarray(prev_act, dtype=np.float32)
                    act = alpha * prev_act + (1.0 - alpha) * act
            smoothed_actions[agent_id] = act

        obs, rewards, dones, truncated, infos = super().step(smoothed_actions)

        # Remove crashed traffic vehicles immediately (so collision counterpart disappears too)
        try:
            traffic_manager = getattr(self.engine, "traffic_manager", None)
            traffic_vehicles = getattr(traffic_manager, "_traffic_vehicles", None)
            if traffic_manager is not None and isinstance(traffic_vehicles, list) and traffic_vehicles:
                to_remove = [
                    v for v in list(traffic_vehicles)
                    if getattr(v, "crash_vehicle", False) or getattr(v, "crash_object", False)
                    or getattr(v, "crash_building", False) or getattr(v, "crash_sidewalk", False)
                    or getattr(v, "crash_human", False)
                ]
                for v in to_remove:
                    try:
                        traffic_manager.clear_objects([v.id])
                    except Exception:
                        try:
                            v.destroy()
                        except Exception:
                            pass
                    try:
                        traffic_vehicles.remove(v)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # Process observations for each active agent
        processed_obs = {}
        
        # 1. Get positions and velocities for graph construction
        agent_positions = {}
        agent_velocities = {}
        for agent_id in obs.keys():
            if agent_id in self.agents:
                agent_positions[agent_id] = self.agents[agent_id].position
                agent_velocities[agent_id] = self.agents[agent_id].velocity
            else:
                # Fallback for done agents or missing agents
                agent_positions[agent_id] = np.array([0.0, 0.0])
                agent_velocities[agent_id] = np.array([0.0, 0.0])
        
        for agent_id, raw_obs in obs.items():
            # --- Sim-to-Real: Noise Injection ---
            # Lidar data is typically at the end of the observation vector in MetaDrive
            # We assume the standard observation structure: [Ego State, Lidar]
            # Lidar size is Config.LIDAR_NUM_LASERS
            
            noisy_obs = raw_obs.copy()
            lidar_start_idx = -Config.LIDAR_NUM_LASERS
            
            # Add Gaussian noise to Lidar channels only
            noise = np.random.normal(0, Config.NOISE_STD, size=Config.LIDAR_NUM_LASERS)
            noisy_obs[lidar_start_idx:] += noise
            
            # Clip to valid range [0, 1] if necessary, though MetaDrive lidar is usually normalized
            noisy_obs[lidar_start_idx:] = np.clip(noisy_obs[lidar_start_idx:], 0.0, 1.0)
            
            # --- Topology: Graph Construction ---
            # Define a deterministic neighbor set: nearest top-K within COMM_RADIUS.
            # Communication perturbation semantics are applied after top-K selection.
            neighbors_true = self._get_neighbors_within_radius_sorted(agent_id, agent_positions)
            K = int(getattr(Config, "MAX_NEIGHBORS", 0) or 0)
            if K < 0:
                K = 0
            neighbors_true = neighbors_true[:K] if K > 0 else []

            # Calculate relative positions and velocities for GAT
            masked_neighbors = []
            neighbor_rel_pos = []
            neighbor_rel_pos_true = []
            if agent_id in self.agents:
                ego_pos = self.agents[agent_id].position
                ego_vel = self.agents[agent_id].velocity
                ego_heading = float(self.agents[agent_id].heading_theta)
                pos_denom, vel_denom = self._get_rel_denoms()
                masked_neighbors, neighbor_rel_pos, neighbor_rel_pos_true = self._build_masked_neighbors_and_rel(
                    agent_id,
                    neighbors_true,
                    agent_positions,
                    agent_velocities,
                    ego_pos,
                    ego_vel,
                    ego_heading,
                    pos_denom,
                    vel_denom,
                )

                # Optional sanity checks (debug only)
                if self._debug_graph_checks_enabled():
                    eps = self._debug_graph_eps()
                    assert len(masked_neighbors) == len(neighbor_rel_pos)
                    assert len(neighbors_true) == len(neighbor_rel_pos_true)
                    assert len(neighbors_true) <= K
                    if neighbors_true:
                        n0 = neighbors_true[0]
                        recompute = self._compute_neighbor_rel(
                            ego_pos,
                            ego_vel,
                            ego_heading,
                            agent_positions[n0],
                            agent_velocities[n0],
                            pos_denom=pos_denom,
                            vel_denom=vel_denom,
                            normalize=False,
                        )
                        diff = np.max(np.abs(np.asarray(recompute, dtype=np.float32) - np.asarray(neighbor_rel_pos_true[0], dtype=np.float32)))
                        assert float(diff) <= float(eps)
                    if masked_neighbors:
                        n1 = masked_neighbors[0]
                        recompute = self._compute_neighbor_rel(
                            ego_pos,
                            ego_vel,
                            ego_heading,
                            agent_positions[n1],
                            agent_velocities[n1],
                            pos_denom=pos_denom,
                            vel_denom=vel_denom,
                            normalize=True,
                        )
                        diff = np.max(np.abs(np.asarray(recompute, dtype=np.float32) - np.asarray(neighbor_rel_pos[0], dtype=np.float32)))
                        assert float(diff) <= float(eps)
            else:
                for _ in masked_neighbors:
                    neighbor_rel_pos.append([0.0, 0.0, 0.0, 0.0])
                for _ in neighbors_true:
                    neighbor_rel_pos_true.append([0.0, 0.0, 0.0, 0.0])
            
            # --- Auxiliary Task: Ground Truth Waypoints ---
            gt_waypoints = self._get_future_waypoints(agent_id)
            
            processed_obs[agent_id] = {
                "node_features": noisy_obs,
                "neighbors": masked_neighbors, # effective neighbors after packet loss (ordered)
                "neighbors_true": neighbors_true, # top-K neighbors before packet loss (ordered)
                "neighbor_rel_pos": neighbor_rel_pos, # normalized, aligned with `neighbors`
                "neighbor_rel_pos_true": neighbor_rel_pos_true, # unnormalized, aligned with `neighbors_true`
                "gt_waypoints": gt_waypoints,   # Shape (PRED_WAYPOINTS_NUM, 2)
                "raw_position": agent_positions[agent_id], # Useful for visualization/debugging
                "agent_id": agent_id # Explicitly track agent_id
            }

        # --- Reward Shaping ---
        new_prev_actions = {}
        new_prev_longs = {}
        new_idle_counts = {}
        for agent_id in rewards.keys():
            base_reward = rewards[agent_id]

            info_i = infos.get(agent_id, {}) if isinstance(infos, dict) else {}
            done_i = bool(dones.get(agent_id, False))

            event_crash = bool(info_i.get("crash", False))
            event_out_of_road = bool(info_i.get("out_of_road", False))
            event_success = bool(info_i.get("arrive_dest", False))
            truncated_i = bool(truncated.get(agent_id, False)) if isinstance(truncated, dict) else False

            
            # --------- Terminal (Safety, highest priority) ---------
            crash_penalty = 0.0
            out_of_road_penalty = 0.0
            success_bonus = 0.0
            if done_i and event_crash:
                crash_penalty = float(getattr(Config, "CRASH_PENALTY", -20.0))
                rewards[agent_id] += crash_penalty
            if done_i and event_out_of_road:
                out_of_road_penalty = float(getattr(Config, "OUT_OF_ROAD_PENALTY", -20.0))
                rewards[agent_id] += out_of_road_penalty
            if done_i and event_success:
                success_bonus = float(getattr(Config, "SUCCESS_REWARD", 10.0))
                rewards[agent_id] += success_bonus

            # --------- Efficiency (dense) ---------
            speed_kmh = 0.0
            speed_reward = 0.0
            overspeed_penalty = 0.0
            idle_penalty = 0.0
            idle_long_penalty = 0.0
            dyn_target_speed = None
            vehicle = self.agents.get(agent_id) if hasattr(self, "agents") else None
            speed_kmh = self._safe_speed_kmh(vehicle)

            # Pre-compute lane metrics for curve-aware target speed & progress shaping
            progress_reward = 0.0
            abs_lat = None
            heading_err = None
            long_now = None
            lane_id_now = None
            delta_long = None
            if vehicle is not None:
                abs_lat, heading_err, long_now, lane_id_now = self._lane_metrics(vehicle)
                if long_now is not None:
                    prev_long = self._prev_longs.get(agent_id)
                    prev_lane = self._prev_lane_ids.get(agent_id)
                    # If lane changed, reset progress accumulator to avoid jumps
                    if prev_lane is not None and lane_id_now is not None and prev_lane != lane_id_now:
                        prev_long = None
                    if prev_long is not None:
                        delta_long = float(long_now - prev_long)
                        progress_scale = float(getattr(Config, "PROGRESS_REWARD_SCALE", 0.02))
                        progress_reward = progress_scale * max(0.0, delta_long)
                        rewards[agent_id] += progress_reward
                    new_prev_longs[agent_id] = float(long_now)
                    if lane_id_now is not None:
                        self._prev_lane_ids[agent_id] = lane_id_now
            if agent_id in self._prev_longs and agent_id not in new_prev_longs:
                new_prev_longs[agent_id] = self._prev_longs[agent_id]

            base_target_speed = float(getattr(Config, "TARGET_SPEED_KMH", 35.0))
            target_speed = base_target_speed
            # Dynamic target speed in curves: scale down by heading error ratio
            if heading_err is not None:
                max_heading_err = float(getattr(Config, "MAX_HEADING_ERROR_RAD", 0.7))
                if max_heading_err > 1e-6:
                    curve_gain = float(getattr(Config, "CURVE_SLOWDOWN_GAIN", 0.6))
                    ratio = float(np.clip(heading_err / max_heading_err, 0.0, 1.0))
                    factor = float(np.clip(1.0 - curve_gain * ratio, 0.2, 1.0))
                    min_target = float(getattr(Config, "MIN_TARGET_SPEED_KMH", 12.0))
                    target_speed = max(min_target, base_target_speed * factor)
                    dyn_target_speed = float(target_speed)

            speed_scale = float(getattr(Config, "SPEED_REWARD_SCALE", 1.0))
            overspeed_scale = float(getattr(Config, "OVERSPEED_PENALTY_SCALE", 0.5))
            idle_speed = float(getattr(Config, "IDLE_SPEED_KMH", 1.0))
            idle_cost = float(getattr(Config, "IDLE_PENALTY", 0.2))

            if target_speed > 1e-6:
                speed_reward = speed_scale * float(np.clip(speed_kmh / target_speed, 0.0, 1.0))
                if speed_kmh > target_speed:
                    overspeed_penalty = -overspeed_scale * float(
                        np.clip((speed_kmh - target_speed) / target_speed, 0.0, 1.0)
                    )
            if speed_kmh < idle_speed:
                idle_penalty = -idle_cost

            rewards[agent_id] += speed_reward + overspeed_penalty + idle_penalty

            # --------- Control (lane + heading + comfort) ---------
            lane_center_penalty = 0.0
            heading_penalty = 0.0
            if abs_lat is not None:
                lane_scale = float(getattr(Config, "LANE_CENTER_PENALTY_SCALE", 0.5))
                lane_width_ref = float(getattr(Config, "LANE_WIDTH_REF", 3.6))
                if lane_width_ref > 1e-6:
                    lane_center_penalty = -lane_scale * float(np.clip(abs_lat / lane_width_ref, 0.0, 1.0))
                    rewards[agent_id] += lane_center_penalty
            if heading_err is not None:
                heading_scale = float(getattr(Config, "HEADING_PENALTY_SCALE", 0.2))
                max_heading_err = float(getattr(Config, "MAX_HEADING_ERROR_RAD", 0.7))
                if max_heading_err > 1e-6:
                    heading_penalty = -heading_scale * float(np.clip(heading_err / max_heading_err, 0.0, 1.0))
                    rewards[agent_id] += heading_penalty

            # ============================================================
            # [修改点 2] Comfort: 动作舒适度 - 支持动态参数 (Curriculum)
            # ============================================================
            smooth_penalty = 0.0
            mag_penalty = 0.0
            if smoothed_actions and agent_id in smoothed_actions:
                act = np.asarray(smoothed_actions[agent_id])
                
                # [Curriculum Hook] 动态读取动作幅值惩罚 (初期为0，后期变大)
                mag_scale = float(getattr(Config, "ACTION_MAG_PENALTY", 0.02))
                mag_penalty = -mag_scale * float(np.linalg.norm(act))
                
                prev = self.prev_actions.get(agent_id)
                if prev is not None:
                    delta = act - prev
                    # [Curriculum Hook] 动态读取动作平滑惩罚 (初期为0，后期变大)
                    change_scale = float(getattr(Config, "ACTION_CHANGE_PENALTY", 0.05))
                    smooth_penalty = -change_scale * float(np.linalg.norm(delta))
                    
                rewards[agent_id] += smooth_penalty + mag_penalty
                new_prev_actions[agent_id] = act
            elif agent_id in self.prev_actions:
                new_prev_actions[agent_id] = self.prev_actions[agent_id]
                
                
            # --------- Safety shield (dense) ---------
            safety_penalty = 0.0
            approach_penalty = 0.0
            ttc_penalty = 0.0
            min_dist = None
            min_ttc = None
            safety_dist = float(getattr(Config, "SAFETY_DIST", 8.0))   # 8.0m 警戒线
            critical_dist = 5.0                                        # 5.0m 死亡线
            safety_scale = float(getattr(Config, "SAFETY_PENALTY_SCALE", 1.0))
            valid_rels = np.zeros((0, 4), dtype=np.float32)
            
            # IMPORTANT: safety signals should NOT depend on packet loss masking.
            if agent_id in processed_obs and processed_obs[agent_id].get("neighbor_rel_pos_true"):
                rels = np.asarray(processed_obs[agent_id]["neighbor_rel_pos_true"], dtype=np.float32)  # (N,4)
                # 过滤掉 padding 的全0数据
                valid_rels = rels[np.abs(rels).sum(axis=1) > 0.01]
                if valid_rels.size > 0:
                    dists = np.linalg.norm(valid_rels[:, :2], axis=1)
                    min_dist = float(np.min(dists))

                   # ============================================================
                    # [修改点 3] TTC Penalty - 支持动态参数 (Curriculum)
                    # ============================================================
                    try:
                        ttc_dist_max = float(getattr(Config, "TTC_DIST_MAX", 25.0))
                        ttc_thr = float(getattr(Config, "TTC_THRESHOLD_S", 2.5))
                        
                        # [Curriculum Hook] 动态读取 TTC 惩罚权重 (0.0 -> 0.8)
                        ttc_scale = float(getattr(Config, "TTC_PENALTY_SCALE", 0.0))
                        
                        closing_eps = float(getattr(Config, "TTC_CLOSING_SPEED_EPS", 0.5))

                        if ttc_scale > 1e-6: # 只有开启了 TTC 惩罚才计算
                            rel_pos = valid_rels[:, :2]
                            rel_vel = valid_rels[:, 2:4]
                            dist = np.linalg.norm(rel_pos, axis=1)
                            mask = dist > 1e-6
                            if np.any(mask):
                                dist_m = dist[mask]
                                if ttc_dist_max > 1e-6:
                                    close_mask = dist_m < ttc_dist_max
                                else:
                                    close_mask = np.ones_like(dist_m, dtype=bool)
                                if np.any(close_mask):
                                    rp = rel_pos[mask][close_mask]
                                    rv = rel_vel[mask][close_mask]
                                    d = dist_m[close_mask]
                                    u = rp / d.reshape(-1, 1)
                                    closing = -np.sum(rv * u, axis=1) 
                                    closing_mask = closing > closing_eps
                                    if np.any(closing_mask):
                                        ttc = d[closing_mask] / closing[closing_mask]
                                        min_ttc = float(np.min(ttc))
                                        if ttc_thr > 1e-6 and min_ttc < ttc_thr:
                                            ttc_penalty = -ttc_scale * float(
                                                np.clip((ttc_thr - min_ttc) / ttc_thr, 0.0, 1.0)
                                            )
                                            rewards[agent_id] += ttc_penalty
                    except Exception:
                        pass

            if min_dist is not None and min_dist < safety_dist:
                linear_p = (safety_dist - min_dist) / max(safety_dist, 1e-6)
                safety_penalty -= safety_scale * 0.5 * float(np.clip(linear_p, 0.0, 1.0))

                if min_dist < critical_dist:
                    exponent = float(np.clip(critical_dist - min_dist, 0.0, 6.0))
                    exp_p = 0.1 * float(np.exp(exponent))
                    safety_penalty -= safety_scale * exp_p

                rewards[agent_id] += safety_penalty

                if valid_rels.size > 0:
                    rel_pos = valid_rels[:, :2]
                    rel_vel = valid_rels[:, 2:4]
                    norm = np.linalg.norm(rel_pos, axis=1, keepdims=True)
                    valid = norm[:, 0] > 1e-6
                    if np.any(valid):
                        u = rel_pos[valid] / norm[valid]
                        closing = -np.sum(rel_vel[valid] * u, axis=1)
                        if closing.size > 0:
                            max_closing = float(np.max(closing))
                            approach_scale = float(getattr(Config, "APPROACH_PENALTY_SCALE", 0.2))
                            approach_penalty = -approach_scale * float(np.clip(max_closing / 10.0, 0.0, 1.0))
                            rewards[agent_id] += approach_penalty

            # --------- Deadlock / long-idle (only when low risk) ---------
            idle_count_prev = int(self._idle_counts.get(agent_id, 0))
            idle_steps_thr = int(getattr(Config, "IDLE_LONG_STEPS", 60))
            idle_count = idle_count_prev
            if not done_i:
                # Only count idle when we can measure progress
                if delta_long is None:
                    idle_count = 0
                else:
                    prog_eps = float(getattr(Config, "IDLE_PROGRESS_EPS", 0.15))

                    # Risk gate (avoid punishing yielding at intersections)
                    safe_min_dist = float(getattr(Config, "IDLE_SAFE_MIN_DIST", 10.0))
                    safe_min_ttc = float(getattr(Config, "IDLE_SAFE_MIN_TTC", 3.0))
                    low_risk = True
                    if min_dist is not None and min_dist < safe_min_dist:
                        low_risk = False
                    if min_ttc is not None and min_ttc < safe_min_ttc:
                        low_risk = False

                    low_speed = speed_kmh < float(getattr(Config, "IDLE_SPEED_KMH", 1.0))
                    low_progress = float(delta_long) < prog_eps

                    if low_risk and low_speed and low_progress:
                        idle_count += 1
                    else:
                        idle_count = 0

                    if idle_steps_thr > 0 and idle_count >= idle_steps_thr:
                        idle_long_cost = float(getattr(Config, "IDLE_LONG_PENALTY", 2.0))
                        idle_long_penalty = -idle_long_cost
                        rewards[agent_id] += idle_long_penalty

            # Keep reporting the last idle_count on terminal step, but reset internal counter
            new_idle_counts[agent_id] = 0 if done_i else idle_count
            idle_long_event = (idle_steps_thr > 0) and (idle_count >= idle_steps_thr)

            # Attach reward breakdown for logging/diagnostics
            if agent_id not in infos:
                infos[agent_id] = {}
            infos[agent_id]["reward_breakdown"] = {
                "base": float(base_reward),
                "success_bonus": float(success_bonus),
                "speed_reward": float(speed_reward),
                "overspeed_penalty": float(overspeed_penalty),
                "idle_penalty": float(idle_penalty),
                "idle_long_penalty": float(idle_long_penalty),
                "progress_reward": float(progress_reward),
                "lane_center_penalty": float(lane_center_penalty),
                "heading_penalty": float(heading_penalty),
                "safety_penalty": float(safety_penalty),
                "approach_penalty": float(approach_penalty),
                "ttc_penalty": float(ttc_penalty),
                "smooth_penalty": float(smooth_penalty),
                "action_mag_penalty": float(mag_penalty),
                "crash_penalty": float(crash_penalty),
                "out_of_road_penalty": float(out_of_road_penalty),
                "total": float(rewards[agent_id]),
            }
            infos[agent_id]["speed_kmh"] = float(speed_kmh)
            if dyn_target_speed is not None:
                infos[agent_id]["dynamic_target_speed_kmh"] = float(dyn_target_speed)
            if min_dist is not None:
                infos[agent_id]["min_neighbor_dist"] = float(min_dist)
            if min_ttc is not None:
                infos[agent_id]["min_ttc_s"] = float(min_ttc)
            infos[agent_id]["idle_count"] = int(idle_count_prev if done_i else new_idle_counts.get(agent_id, 0))
            if abs_lat is not None:
                infos[agent_id]["abs_lane_lat"] = float(abs_lat)
            if heading_err is not None:
                infos[agent_id]["heading_err_rad"] = float(heading_err)

            # === [新增] 交互强度指标 (Interaction Intensity Metrics) ===
            # 用于论文分析：量化场景的交互复杂度
            has_neighbor = len(processed_obs[agent_id].get("neighbors_true", [])) > 0
            infos[agent_id]["has_neighbor"] = int(has_neighbor)
            infos[agent_id]["num_neighbors"] = len(processed_obs[agent_id].get("neighbors_true", []))
            
            # Near-Miss事件：高风险交互（用于安全性分析）
            near_miss = (
                (min_ttc is not None and min_ttc < 2.0) or 
                (min_dist is not None and min_dist < 5.0)
            )
            infos[agent_id]["near_miss"] = int(near_miss)
            
            # 停车状态：低效率指标（用于效率分析）
            is_stopped = speed_kmh < 1.0
            infos[agent_id]["is_stopped"] = int(is_stopped)

            # Unified event signals for downstream aggregation
            ttc_thr = float(getattr(Config, "TTC_THRESHOLD_S", 2.5))
            safety_dist = float(getattr(Config, "SAFETY_DIST", 8.0))
            infos[agent_id]["event"] = {
                "crash": bool(event_crash),
                "out_of_road": bool(event_out_of_road),
                "success": bool(event_success),
                "truncated": bool(truncated_i),
                "idle_long": bool(idle_long_event),
                "risk_ttc": bool(min_ttc is not None and min_ttc < ttc_thr),
                "risk_dist": bool(min_dist is not None and min_dist < safety_dist),
            }
            if done_i:
                if event_success:
                    term = "success"
                elif event_crash:
                    term = "crash"
                elif event_out_of_road:
                    term = "out_of_road"
                elif truncated_i:
                    term = "timeout"
                else:
                    term = "other"
                infos[agent_id]["terminal_reason"] = term

        self.prev_actions = new_prev_actions
        self._prev_longs = new_prev_longs
        self._idle_counts = new_idle_counts
            
        merged_dones = {k: dones.get(k, False) or truncated.get(k, False) for k in dones.keys()}
        merged_dones["__all__"] = merged_dones.get("__all__", False) or truncated.get("__all__", False)
        for k, info in infos.items():
            info["truncated"] = truncated.get(k, False)

        return processed_obs, rewards, merged_dones, infos

    def _get_neighbors(self, agent_id, all_positions):
        """DEPRECATED: kept for backward compatibility.

        Use _get_neighbors_within_radius_sorted() + top-K slicing instead.
        """
        return self._get_neighbors_within_radius_sorted(agent_id, all_positions)

    def _get_neighbors_within_radius_sorted(self, agent_id, all_positions):
        """Return neighbors within COMM_RADIUS, sorted by (distance, tie-break).

        Tie-break uses str(agent_id) to guarantee deterministic ordering.
        """
        if agent_id not in all_positions:
            return []
        my_pos = np.asarray(all_positions[agent_id], dtype=np.float32)
        comm_r = float(getattr(Config, "COMM_RADIUS", 80.0))
        comm_r = max(0.0, comm_r)

        candidates = []
        for other_id, other_pos in all_positions.items():
            if agent_id == other_id:
                continue
            try:
                dist = float(np.linalg.norm(my_pos - np.asarray(other_pos, dtype=np.float32)))
            except Exception:
                continue
            if dist < comm_r:
                candidates.append((dist, str(other_id), other_id))

        candidates.sort(key=lambda x: (x[0], x[1]))
        return [oid for _, __, oid in candidates]

    def _get_future_waypoints(self, agent_id):
        """
        Extract next N waypoints relative to ego vehicle's coordinate system.
        """
        if agent_id not in self.agents:
            return np.zeros((Config.PRED_WAYPOINTS_NUM, 2), dtype=np.float32)
            
        vehicle = self.agents[agent_id]

        # Use navigation reference lanes so GT follows planned route through intersections.
        # Fallback to vehicle.lane if navigation is not ready.
        navi = getattr(vehicle, "navigation", None)
        ref_lanes = None
        next_ref_lanes = None
        if navi is not None:
            ref_lanes = getattr(navi, "current_ref_lanes", None)
            next_ref_lanes = getattr(navi, "next_ref_lanes", None)

        lanes_seq = []
        if isinstance(ref_lanes, (list, tuple)) and len(ref_lanes) > 0:
            # Prefer the actual lane if it is one of the reference lanes
            if getattr(vehicle, "lane", None) in ref_lanes:
                lanes_seq.append(vehicle.lane)
            else:
                lanes_seq.append(ref_lanes[0])
        elif getattr(vehicle, "lane", None) is not None:
            lanes_seq.append(vehicle.lane)

        if isinstance(next_ref_lanes, (list, tuple)) and len(next_ref_lanes) > 0:
            # Only need the next segment for short-horizon waypoints.
            lanes_seq.append(next_ref_lanes[0])

        if len(lanes_seq) == 0:
            return np.zeros((Config.PRED_WAYPOINTS_NUM, 2), dtype=np.float32)

        base_lane = lanes_seq[0]
        try:
            long0, _ = base_lane.local_coordinates(vehicle.position)
        except Exception:
            long0 = 0.0
        long0 = float(np.clip(long0, 0.0, float(getattr(base_lane, "length", 0.0) or 0.0)))

        step_dist = float(getattr(Config, "WAYPOINT_STEP_DIST", 5.0))
        future_points = []
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        c, s = np.cos(heading), np.sin(heading)

        for i in range(1, int(getattr(Config, "PRED_WAYPOINTS_NUM", 5)) + 1):
            look_ahead = float(i) * step_dist

            lane = lanes_seq[0]
            lane_long = long0 + look_ahead
            lane_len = float(getattr(lane, "length", 0.0) or 0.0)

            if lane_long > lane_len and len(lanes_seq) > 1:
                # Spill over to next lane
                remain = lane_long - lane_len
                lane = lanes_seq[1]
                lane_len = float(getattr(lane, "length", 0.0) or 0.0)
                lane_long = float(np.clip(remain, 0.0, lane_len))
            else:
                lane_long = float(np.clip(lane_long, 0.0, lane_len))

            try:
                point = lane.position(lane_long, 0)
            except Exception:
                # As a final fallback, reuse current position
                point = np.asarray(vehicle.position, dtype=np.float32)

            rel_pos = point - vehicle.position
            local_x = float(rel_pos[0] * c + rel_pos[1] * s)
            local_y = float(-rel_pos[0] * s + rel_pos[1] * c)
            future_points.append([local_x, local_y])

        return np.array(future_points, dtype=np.float32)

    def update_config_params(self, params: dict):
        """
        允许外部进程在运行时动态修改 Config 参数 (用于 Curriculum Learning)
        """
        for k, v in params.items():
            # 尝试更新 Config 类属性
            if hasattr(Config, k):
                setattr(Config, k, v)
        # 也可以打印一下确认同步成功
        # print(f"[Env Worker] Updated Config: {params}")

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        # Spawn debug/safety check (right after env.reset)
        self._debug_spawn_capacity_after_reset(tag="reset")

        # Clear episode states
        self.prev_actions = {}
        self._idle_counts = {}
        self._prev_longs = {}
        self._prev_lane_ids = {}
        self._comm_burst_remaining = {}
        self._stale_neighbor_cache = {}
        
        processed_obs = {}
        agent_positions = {}
        agent_velocities = {}
        for agent_id in obs.keys():
            if agent_id in self.agents:
                agent_positions[agent_id] = self.agents[agent_id].position
                agent_velocities[agent_id] = self.agents[agent_id].velocity
            else:
                agent_positions[agent_id] = np.array([0.0, 0.0])
                agent_velocities[agent_id] = np.array([0.0, 0.0])
        
        for agent_id, raw_obs in obs.items():
            noisy_obs = raw_obs.copy()
            lidar_start_idx = -Config.LIDAR_NUM_LASERS
            noise = np.random.normal(0, Config.NOISE_STD, size=Config.LIDAR_NUM_LASERS)
            noisy_obs[lidar_start_idx:] += noise
            noisy_obs[lidar_start_idx:] = np.clip(noisy_obs[lidar_start_idx:], 0.0, 1.0)
            
            neighbors_true = self._get_neighbors_within_radius_sorted(agent_id, agent_positions)
            K = int(getattr(Config, "MAX_NEIGHBORS", 0) or 0)
            if K < 0:
                K = 0
            neighbors_true = neighbors_true[:K] if K > 0 else []

            masked_neighbors = []
            neighbor_rel_pos = []
            neighbor_rel_pos_true = []
            if agent_id in self.agents:
                ego_pos = self.agents[agent_id].position
                ego_vel = self.agents[agent_id].velocity
                ego_heading = float(self.agents[agent_id].heading_theta)
                pos_denom, vel_denom = self._get_rel_denoms()
                masked_neighbors, neighbor_rel_pos, neighbor_rel_pos_true = self._build_masked_neighbors_and_rel(
                    agent_id,
                    neighbors_true,
                    agent_positions,
                    agent_velocities,
                    ego_pos,
                    ego_vel,
                    ego_heading,
                    pos_denom,
                    vel_denom,
                )

                if self._debug_graph_checks_enabled():
                    eps = self._debug_graph_eps()
                    assert len(masked_neighbors) == len(neighbor_rel_pos)
                    assert len(neighbors_true) == len(neighbor_rel_pos_true)
                    assert len(neighbors_true) <= K
                    if neighbors_true:
                        n0 = neighbors_true[0]
                        recompute = self._compute_neighbor_rel(
                            ego_pos,
                            ego_vel,
                            ego_heading,
                            agent_positions[n0],
                            agent_velocities[n0],
                            pos_denom=pos_denom,
                            vel_denom=vel_denom,
                            normalize=False,
                        )
                        diff = np.max(np.abs(np.asarray(recompute, dtype=np.float32) - np.asarray(neighbor_rel_pos_true[0], dtype=np.float32)))
                        assert float(diff) <= float(eps)
                    if masked_neighbors:
                        n1 = masked_neighbors[0]
                        recompute = self._compute_neighbor_rel(
                            ego_pos,
                            ego_vel,
                            ego_heading,
                            agent_positions[n1],
                            agent_velocities[n1],
                            pos_denom=pos_denom,
                            vel_denom=vel_denom,
                            normalize=True,
                        )
                        diff = np.max(np.abs(np.asarray(recompute, dtype=np.float32) - np.asarray(neighbor_rel_pos[0], dtype=np.float32)))
                        assert float(diff) <= float(eps)
            else:
                for _ in masked_neighbors:
                    neighbor_rel_pos.append([0.0, 0.0, 0.0, 0.0])
                for _ in neighbors_true:
                    neighbor_rel_pos_true.append([0.0, 0.0, 0.0, 0.0])

            gt_waypoints = self._get_future_waypoints(agent_id)
            
            processed_obs[agent_id] = {
                "node_features": noisy_obs,
                "neighbors": masked_neighbors,
                "neighbors_true": neighbors_true,
                "neighbor_rel_pos": neighbor_rel_pos,
                "neighbor_rel_pos_true": neighbor_rel_pos_true,
                "gt_waypoints": gt_waypoints,
                "raw_position": agent_positions[agent_id],
                "agent_id": agent_id,
            }
            
        # Gymnasium reset returns (obs, info); SB3 SubprocVecEnv expects that. Keep signature aligned.
        return processed_obs, info
