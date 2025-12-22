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
        if config:
            meta_config.update(config)
            
        super().__init__(meta_config)
        # Track previous actions for smoothing/reward shaping
        self.prev_actions = {}
        # Track previous longitudinal position for progress shaping
        self._prev_longs = {}
        # Track previous lane id to avoid progress jumps on lane switch
        self._prev_lane_ids = {}
        # Track long-idle steps (deadlock) per agent
        self._idle_counts = {}

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

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
            neighbors = self._get_neighbors(agent_id, agent_positions)
            
            # --- Sim-to-Real: Packet Loss ---
            # Randomly mask neighbors
            masked_neighbors = []
            for n_id in neighbors:
                if np.random.random() > Config.MASK_RATIO:
                    masked_neighbors.append(n_id)
            
            # Calculate relative positions and velocities for GAT
            neighbor_rel_pos = []
            neighbor_rel_pos_true = []
            if agent_id in self.agents:
                ego_pos = self.agents[agent_id].position
                ego_vel = self.agents[agent_id].velocity
                ego_heading = self.agents[agent_id].heading_theta
                c, s = np.cos(ego_heading), np.sin(ego_heading)

                # True rel pos (no packet loss) for reward/safety/event stats
                for n_id in neighbors:
                    n_pos = agent_positions[n_id]
                    n_vel = agent_velocities[n_id]
                    rel_pos = n_pos - ego_pos
                    local_x = rel_pos[0] * c + rel_pos[1] * s
                    local_y = -rel_pos[0] * s + rel_pos[1] * c
                    rel_vel = n_vel - ego_vel
                    local_vx = rel_vel[0] * c + rel_vel[1] * s
                    local_vy = -rel_vel[0] * s + rel_vel[1] * c
                    neighbor_rel_pos_true.append([local_x, local_y, local_vx, local_vy])

                # Masked rel pos for observation
                for n_id in masked_neighbors:
                    n_pos = agent_positions[n_id]
                    n_vel = agent_velocities[n_id]
                    rel_pos = n_pos - ego_pos
                    local_x = rel_pos[0] * c + rel_pos[1] * s
                    local_y = -rel_pos[0] * s + rel_pos[1] * c
                    rel_vel = n_vel - ego_vel
                    local_vx = rel_vel[0] * c + rel_vel[1] * s
                    local_vy = -rel_vel[0] * s + rel_vel[1] * c
                    neighbor_rel_pos.append([local_x, local_y, local_vx, local_vy])
            else:
                # Fallback if agent not found (e.g. done state)
                for _ in masked_neighbors:
                    neighbor_rel_pos.append([0.0, 0.0, 0.0, 0.0])
                for _ in neighbors:
                    neighbor_rel_pos_true.append([0.0, 0.0, 0.0, 0.0])
            
            # --- Auxiliary Task: Ground Truth Waypoints ---
            gt_waypoints = self._get_future_waypoints(agent_id)
            
            processed_obs[agent_id] = {
                "node_features": noisy_obs,
                "neighbors": masked_neighbors, # List of agent_ids
                "neighbor_rel_pos": neighbor_rel_pos, # List of [x, y, vx, vy]
                "neighbor_rel_pos_true": neighbor_rel_pos_true, # reward/safety only (no packet loss)
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
        """
        Find neighbors within COMM_RADIUS.
        """
        my_pos = all_positions[agent_id]
        neighbors = []
        
        for other_id, other_pos in all_positions.items():
            if agent_id == other_id:
                continue
                
            dist = np.linalg.norm(my_pos - other_pos)
            if dist < Config.COMM_RADIUS:
                neighbors.append(other_id)
                
        # Sort by distance (optional, but good for consistent input if we truncate)
        # For now, we just return all valid neighbors. 
        # The GAT module will handle padding/truncation to MAX_NEIGHBORS.
        return neighbors

    def _get_future_waypoints(self, agent_id):
        """
        Extract next N waypoints relative to ego vehicle's coordinate system.
        """
        if agent_id not in self.agents:
            return np.zeros((Config.PRED_WAYPOINTS_NUM, 2), dtype=np.float32)
            
        vehicle = self.agents[agent_id]
        
        # Robust implementation using current lane
        future_points = []
        current_lane = vehicle.lane
        
        # Get current longitudinal position
        long, lat = current_lane.local_coordinates(vehicle.position)
        
        for i in range(1, Config.PRED_WAYPOINTS_NUM + 1):
            look_ahead_dist = i * 5.0 
            target_long = long + look_ahead_dist
            
            # Get point on lane
            point = current_lane.position(target_long, 0)
            
            # Transform to ego coordinates
            rel_pos = point - vehicle.position
            heading = vehicle.heading_theta
            c, s = np.cos(heading), np.sin(heading)
            
            # Rotate by -heading
            local_x = rel_pos[0] * c + rel_pos[1] * s
            local_y = -rel_pos[0] * s + rel_pos[1] * c
            
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
    # Clear episode states
        self.prev_actions = {}
        self._idle_counts = {}
        self._prev_longs = {}
        self._prev_lane_ids = {}
        
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
            
            neighbors = self._get_neighbors(agent_id, agent_positions)
            masked_neighbors = []
            for n_id in neighbors:
                if np.random.random() > Config.MASK_RATIO:
                    masked_neighbors.append(n_id)
            
            neighbor_rel_pos = []
            if agent_id in self.agents:
                ego_pos = self.agents[agent_id].position
                ego_vel = self.agents[agent_id].velocity
                ego_heading = self.agents[agent_id].heading_theta
                c, s = np.cos(ego_heading), np.sin(ego_heading)
                
                for n_id in masked_neighbors:
                    n_pos = agent_positions[n_id]
                    n_vel = agent_velocities[n_id]
                    
                    rel_pos = n_pos - ego_pos
                    local_x = rel_pos[0] * c + rel_pos[1] * s
                    local_y = -rel_pos[0] * s + rel_pos[1] * c
                    
                    rel_vel = n_vel - ego_vel
                    local_vx = rel_vel[0] * c + rel_vel[1] * s
                    local_vy = -rel_vel[0] * s + rel_vel[1] * c
                    
                    neighbor_rel_pos.append([local_x, local_y, local_vx, local_vy])
            else:
                for _ in masked_neighbors:
                    neighbor_rel_pos.append([0.0, 0.0, 0.0, 0.0])

            gt_waypoints = self._get_future_waypoints(agent_id)
            
            processed_obs[agent_id] = {
                "node_features": noisy_obs,
                "neighbors": masked_neighbors,
                "neighbor_rel_pos": neighbor_rel_pos,
                "gt_waypoints": gt_waypoints,
                "raw_position": agent_positions[agent_id]
            }
            
        # Gymnasium reset returns (obs, info); SB3 SubprocVecEnv expects that. Keep signature aligned.
        return processed_obs, info
