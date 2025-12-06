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
        
    def step(self, actions):
        """
        Step the environment and process observations.
        Returns a 4-tuple (obs, rewards, dones, infos) so it can be used inside SubprocVecEnv workers.
        """
        obs, rewards, dones, truncated, infos = super().step(actions)
        
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
            if agent_id in self.agents:
                ego_pos = self.agents[agent_id].position
                ego_vel = self.agents[agent_id].velocity
                ego_heading = self.agents[agent_id].heading_theta
                c, s = np.cos(ego_heading), np.sin(ego_heading)
                
                for n_id in masked_neighbors:
                    n_pos = agent_positions[n_id]
                    n_vel = agent_velocities[n_id]
                    
                    # Relative Position
                    rel_pos = n_pos - ego_pos
                    local_x = rel_pos[0] * c + rel_pos[1] * s
                    local_y = -rel_pos[0] * s + rel_pos[1] * c
                    
                    # Relative Velocity
                    rel_vel = n_vel - ego_vel
                    local_vx = rel_vel[0] * c + rel_vel[1] * s
                    local_vy = -rel_vel[0] * s + rel_vel[1] * c
                    
                    neighbor_rel_pos.append([local_x, local_y, local_vx, local_vy])
            else:
                # Fallback if agent not found (e.g. done state)
                for _ in masked_neighbors:
                    neighbor_rel_pos.append([0.0, 0.0, 0.0, 0.0])
            
            # --- Auxiliary Task: Ground Truth Waypoints ---
            gt_waypoints = self._get_future_waypoints(agent_id)
            
            processed_obs[agent_id] = {
                "node_features": noisy_obs,
                "neighbors": masked_neighbors, # List of agent_ids
                "neighbor_rel_pos": neighbor_rel_pos, # List of [x, y, vx, vy]
                "gt_waypoints": gt_waypoints,   # Shape (PRED_WAYPOINTS_NUM, 2)
                "raw_position": agent_positions[agent_id], # Useful for visualization/debugging
                "agent_id": agent_id # Explicitly track agent_id
            }

        # --- Reward Shaping ---
        for agent_id in rewards.keys():
            # 1. Stronger Collision Penalty
            if dones[agent_id] and infos[agent_id].get('crash', False):
                rewards[agent_id] -= 20.0 # Extra penalty on top of MetaDrive's default
            
            # 2. Safety Distance Penalty
            # Check distance to closest neighbor
            if agent_id in processed_obs and processed_obs[agent_id]['neighbor_rel_pos']:
                # Calculate distances from relative positions
                rels = np.array(processed_obs[agent_id]['neighbor_rel_pos']) # (N, 4)
                dists = np.linalg.norm(rels[:, :2], axis=1)
                min_dist = np.min(dists)
                
                if min_dist < Config.SAFETY_DIST:
                    # Penalty proportional to how close they are
                    # e.g. if dist is 0, penalty is -1.0
                    rewards[agent_id] -= (Config.SAFETY_DIST - min_dist) / Config.SAFETY_DIST
            
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

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        
        # We need to process the initial observation as well
        # Reuse the logic from step() or just return raw obs if the loop handles it.
        # But for consistency, let's wrap it.
        # However, reset() signature in Gym is (obs, info).
        
        # To avoid code duplication, we can just return the raw obs here 
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
            # Noise
            noisy_obs = raw_obs.copy()
            lidar_start_idx = -Config.LIDAR_NUM_LASERS
            noise = np.random.normal(0, Config.NOISE_STD, size=Config.LIDAR_NUM_LASERS)
            noisy_obs[lidar_start_idx:] += noise
            noisy_obs[lidar_start_idx:] = np.clip(noisy_obs[lidar_start_idx:], 0.0, 1.0)
            
            # Graph
            neighbors = self._get_neighbors(agent_id, agent_positions)
            # No packet loss on first frame? Or yes? Let's apply it.
            masked_neighbors = []
            for n_id in neighbors:
                if np.random.random() > Config.MASK_RATIO:
                    masked_neighbors.append(n_id)
            
            # Relative positions and velocities
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

            # Waypoints
            gt_waypoints = self._get_future_waypoints(agent_id)
            
            processed_obs[agent_id] = {
                "node_features": noisy_obs,
                "neighbors": masked_neighbors,
                "neighbor_rel_pos": neighbor_rel_pos,
                "gt_waypoints": gt_waypoints,
                "raw_position": agent_positions[agent_id]
            }
            
        return processed_obs
