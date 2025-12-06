import os
import sys
import time
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.models.policy import CooperativePolicy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MARL Model")
    parser.add_argument("--model_path", type=str, default="logs/marl_experiment/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--render", action="store_true", default=True, help="Enable rendering")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cpu/cuda)")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    
    # Override config for rendering
    eval_config = {
        "use_render": args.render,
        "show_interface": True,
        "show_logo": False,
        "show_fps": True,
        "window_size": (1200, 900),
        # "manual_control": True, # Uncomment if you want to control one agent manually to test others
    }
    
    device = torch.device(args.device)
    print(f"Evaluating on device: {device}")
    
    print(f"Initializing Environment...")
    env = GraphEnvWrapper(config=eval_config)
    
    # Get dims from a dummy reset
    obs_dict, _ = env.reset()
    if not obs_dict:
        print("Error: No agents found in environment.")
        return

    sample_agent = list(obs_dict.keys())[0]
    input_dim = obs_dict[sample_agent]['node_features'].shape[0]
    action_dim = env.action_space[sample_agent].shape[0]
    
    # Load Policy
    policy = CooperativePolicy(input_dim, action_dim).to(device)
    
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        # Load state dict
        state_dict = torch.load(args.model_path, map_location=device)
        policy.load_state_dict(state_dict)
    else:
        print(f"Warning: Model not found at {args.model_path}, using random weights.")
        
    policy.eval()
    
    for episode in range(args.episodes):
        print(f"Starting Episode {episode + 1}/{args.episodes}")
        obs_dict, _ = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            if args.render:
                env.render()
            
            # Prepare batch (Same logic as train.py inference)
            active_agents = list(obs_dict.keys())
            if not active_agents:
                break
                
            agent_to_idx = {a_id: i for i, a_id in enumerate(active_agents)}
            
            batch_node_features = []
            batch_neighbor_indices = []
            batch_neighbor_mask = []
            batch_neighbor_rel_pos = []
            
            for agent_id in active_agents:
                agent_obs = obs_dict[agent_id]
                batch_node_features.append(agent_obs['node_features'])
                
                n_indices = []
                n_mask = []
                n_rel_pos = []
                
                raw_neighbors = agent_obs['neighbors']
                raw_rel_pos = agent_obs['neighbor_rel_pos']
                
                for i, n_id in enumerate(raw_neighbors):
                    if n_id in agent_to_idx:
                        n_indices.append(agent_to_idx[n_id])
                        n_mask.append(1.0)
                        n_rel_pos.append(raw_rel_pos[i])
                
                while len(n_indices) < Config.MAX_NEIGHBORS:
                    n_indices.append(0) # Point to 0, masked out
                    n_mask.append(0.0)
                    n_rel_pos.append([0.0, 0.0, 0.0, 0.0])
                    
                batch_neighbor_indices.append(n_indices[:Config.MAX_NEIGHBORS])
                batch_neighbor_mask.append(n_mask[:Config.MAX_NEIGHBORS])
                batch_neighbor_rel_pos.append(n_rel_pos[:Config.MAX_NEIGHBORS])
            
            obs_tensor_batch = {
                "node_features": torch.tensor(np.array(batch_node_features), dtype=torch.float32).to(device),
                "neighbor_indices": torch.tensor(np.array(batch_neighbor_indices), dtype=torch.long).to(device),
                "neighbor_mask": torch.tensor(np.array(batch_neighbor_mask), dtype=torch.float32).to(device),
                "neighbor_rel_pos": torch.tensor(np.array(batch_neighbor_rel_pos), dtype=torch.float32).to(device)
            }
            
            with torch.no_grad():
                results = policy(obs_tensor_batch)
                action_mean = results["action_mean"]
                # Use deterministic action (mean) for evaluation
                actions = action_mean.cpu().numpy()
                
            action_dict = {
                agent_id: actions[i]
                for i, agent_id in enumerate(active_agents)
            }
            
            # Step
            next_obs_dict, rewards, dones, truncated, infos = env.step(action_dict)
            
            for r in rewards.values():
                total_reward += r
            
            # Update dones
            if not next_obs_dict:
                break
                
            obs_dict = next_obs_dict
            step += 1
            
            # Optional: Slow down slightly to make it watchable
            # time.sleep(0.02)
            
        print(f"Episode {episode + 1} finished. Steps: {step}, Total Reward: {total_reward:.2f}")
        
    env.close()

if __name__ == "__main__":
    evaluate()
