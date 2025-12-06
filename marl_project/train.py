import os
import sys
import shutil
import argparse
import json
import subprocess
import time
from collections import defaultdict, deque

# Add the project root directory to sys.path to allow imports from marl_project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from marl_project.config import Config
from marl_project.env_wrapper import GraphEnvWrapper
from marl_project.models.policy import CooperativePolicy

def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training for MARL")
    parser.add_argument("--lr", type=float, default=Config.LR, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=Config.GAMMA, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--ppo_epochs", type=int, default=Config.PPO_EPOCHS, help="PPO epochs")
    parser.add_argument("--aux_loss_coef", type=float, default=Config.AUX_LOSS_COEF, help="Aux loss coefficient")
    parser.add_argument("--map_type", type=str, default=Config.MAP_TYPE, help="Map type (S, C, X)")
    
    # New args
    parser.add_argument("--aux_decay", type=int, default=1, help="Decay aux loss coef (1=True, 0=False)")
    parser.add_argument("--clip_eps", type=float, default=Config.CLIP_EPSILON, help="PPO clip epsilon")
    parser.add_argument("--grad_norm", type=float, default=Config.MAX_GRAD_NORM, help="Max gradient norm")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cpu/cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_time", type=int, default=30, help="Max wall-clock seconds to train before stopping")
    
    return parser.parse_args()

def update_config(args):
    for key, value in vars(args).items():
        key_upper = key.upper()
        if hasattr(Config, key_upper):
            setattr(Config, key_upper, value)
            print(f"Config updated: {key_upper} = {value}")

def calculate_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    """
    Calculate GAE for a single agent's trajectory.
    """
    advantages = []
    gae = 0
    
    # Ensure inputs are tensors or numpy arrays
    # We assume they are tensors on the correct device
    
    for i in reversed(range(len(rewards))):
        mask = 1.0 - dones[i]
        delta = rewards[i] + gamma * next_value * mask - values[i]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[i]
        
    return torch.tensor(advantages, dtype=torch.float32, device=rewards.device)

def train():
    # --- Setup ---
    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = parse_args()
    update_config(args)
    
    log_dir = "logs/marl_experiment"
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config for reproducibility
    config_path = os.path.join(os.path.dirname(__file__), "config.py")
    if os.path.exists(config_path):
        shutil.copy(config_path, log_dir)
        
    # Dump final config to hparams.json
    hparams = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f, indent=4, default=str)

    writer = SummaryWriter(log_dir)

    # Device configuration
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- SubprocVecEnv with per-worker logs ---
    def make_env(rank):
        def _thunk():
            os.makedirs("logs", exist_ok=True)
            sys.stdout = open(os.path.join("logs", f"env_{rank}.log"), "w", buffering=1)
            sys.stderr = sys.stdout
            env_inst = GraphEnvWrapper()
            env_inst.seed(42 + rank)
            return env_inst
        return _thunk

    vec_env = SubprocVecEnv([make_env(i) for i in range(Config.NUM_ENVS)], start_method="spawn")
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Get dimensions from a dummy reset
    obs_list = vec_env.reset()
    obs_dict = obs_list[0] if isinstance(obs_list, (list, tuple)) else obs_list
    sample_agent = list(obs_dict.keys())[0]
    input_dim = obs_dict[sample_agent]['node_features'].shape[0]
    action_dim = vec_env.action_space[sample_agent].shape[0]

    policy = torch.compile(CooperativePolicy(input_dim, action_dim).to(device), mode="reduce-overhead")
    optimizer = optim.Adam(policy.parameters(), lr=Config.LR)
    
    print(f"Initialized Policy. Input: {input_dim}, Action: {action_dim}")
    
    # --- Training Parameters ---
    num_updates = 1000
    steps_per_update = Config.N_STEPS
    gamma = Config.GAMMA
    gae_lambda = 0.95
    clip_epsilon = args.clip_eps
    max_grad_norm = args.grad_norm
    
    # Checkpointing
    best_reward = -float('inf')
    saved_checkpoints = deque(maxlen=3)
    
    # --- Main Loop ---
    start_update = 1
    global_step = 0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            state_dict = torch.load(args.resume, map_location=device)
            policy.load_state_dict(state_dict)
            
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
    for update in range(start_update, num_updates + 1):
        if args.max_time and (time.time() - wall_start) >= args.max_time:
            print("Max wall-clock time reached, stopping.")
            break
        # --- 1. Rollout Phase ---
        buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "agent_ids": [],   # <--- ADDED THIS explicitly
            "old_values": [],      # <--- Stored detached tensors
            "old_logprobs": [],    # <--- Stored detached tensors
            "aux_preds": [],       # <--- Stored detached tensors
            "gt_waypoints": []
        }
        
        # Reward Tracking
        completed_episode_rewards = []
        episode_rewards = {}
        
        obs_list = vec_env.reset()
        episode_rewards = [{a_id: 0.0 for a_id in obs} for obs in obs_list]
        
        for step in range(steps_per_update):
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
                        if n_id in agent_to_idx:
                            n_indices.append(agent_to_idx[n_id])
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
            
            # Convert to Tensor and move to device
            obs_tensor_batch = {
                "node_features": torch.tensor(np.array(batch_node_features), dtype=torch.float32, pin_memory=True),
                "neighbor_indices": torch.tensor(np.array(batch_neighbor_indices), dtype=torch.long, pin_memory=True),
                "neighbor_mask": torch.tensor(np.array(batch_neighbor_mask), dtype=torch.float32, pin_memory=True),
                "neighbor_rel_pos": torch.tensor(np.array(batch_neighbor_rel_pos), dtype=torch.float32, pin_memory=True),
                "gt_waypoints": torch.tensor(np.array(batch_gt_waypoints), dtype=torch.float32, pin_memory=True)
            }
            
            # Inference (using forward_actor_critic for no_grad)
            results = policy.forward_actor_critic({k: v.to(device, non_blocking=True) for k, v in obs_tensor_batch.items()})
                
            # Sample Actions
            action_mean = results["action_mean"]
            action_std = results["action_std"]
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.sample()
            action_log_probs = dist.log_prob(actions).sum(dim=-1)
            values = results["value"].squeeze(-1)
            aux_preds = results["pred_waypoints"]
            
            # Execute Step
            env_actions = []
            cursor = 0
            for env_idx, obs_dict in enumerate(obs_list):
                count = env_agent_counts[env_idx]
                env_actions.append({aid: actions[cursor + j].cpu().numpy() for j, aid in enumerate(obs_dict.keys())})
                cursor += count

            next_obs_list, rewards_list, dones_list, infos_list = vec_env.step(env_actions)
            
            # Track Rewards
            for env_idx, obs_dict in enumerate(obs_list):
                rewards = rewards_list[env_idx]
                dones = dones_list[env_idx]
                for a_id in obs_dict.keys():
                    key = f"{env_idx}:{a_id}"
                    episode_rewards[env_idx][key] = episode_rewards[env_idx].get(key, 0.0) + rewards.get(a_id, 0.0)
                    if dones.get(a_id, False):
                        completed_episode_rewards.append(episode_rewards[env_idx][key])
                        episode_rewards[env_idx][key] = 0.0

            # Store in buffer
            buffer["obs"].append({k: v.cpu() for k, v in obs_tensor_batch.items()})
            buffer["actions"].append(actions.detach().cpu())
            buffer["agent_ids"].append(active_agents)
            
            # Store detached tensors for PPO update
            buffer["old_values"].append(values.detach().cpu())
            buffer["old_logprobs"].append(action_log_probs.detach().cpu())
            buffer["aux_preds"].append(aux_preds.detach().cpu())
            buffer["gt_waypoints"].append(obs_tensor_batch["gt_waypoints"].detach())
            
            # Rewards and Dones
            flat_rewards = []
            flat_dones = []
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
        flat_advantages = torch.zeros_like(flat_rewards).to(device)
        flat_returns = torch.zeros_like(flat_rewards).to(device)
        
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
        }
        
        # Aux Loss Scheduler
        if args.aux_decay:
            current_aux_coef = max(0.1, Config.AUX_LOSS_COEF * (1.0 - (update - 1) / num_updates))
        else:
            current_aux_coef = Config.AUX_LOSS_COEF
        
        total_ppo_loss = 0
        total_value_loss = 0
        total_aux_loss = 0
        total_loss = 0
        
        dataset = TensorDataset(
            flat_obs["node_features"].pin_memory(),
            flat_obs["neighbor_indices"].pin_memory(),
            flat_obs["neighbor_mask"].pin_memory(),
            flat_obs["neighbor_rel_pos"].pin_memory(),
            flat_actions.pin_memory(),
            flat_old_logprobs.pin_memory(),
            flat_old_values.pin_memory(),
            flat_returns.pin_memory(),
            flat_advantages.pin_memory()
        )
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

        for _ in range(Config.PPO_EPOCHS):
            for (b_nodes, b_nidx, b_nmask, b_nrel, b_actions, b_old_logprobs,
                 b_old_values, b_returns, b_advantages) in loader:
                obs_batch = {
                    "node_features": b_nodes.to(device, non_blocking=True),
                    "neighbor_indices": b_nidx.to(device, non_blocking=True),
                    "neighbor_mask": b_nmask.to(device, non_blocking=True),
                    "neighbor_rel_pos": b_nrel.to(device, non_blocking=True),
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
                v_clipped = b_old_values_gpu + torch.clamp(new_values - b_old_values_gpu, -clip_epsilon, clip_epsilon)
                v_loss_unclipped = (new_values - b_returns_gpu) ** 2
                v_loss_clipped = (v_clipped - b_returns_gpu) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = -dist_entropy.mean() * 0.01
                step_loss = ppo_loss + value_loss + entropy_loss + current_aux_coef * aux_loss

                optimizer.zero_grad()
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                total_ppo_loss += ppo_loss.item()
                total_value_loss += value_loss.item()
                total_aux_loss += aux_loss.item()
                total_loss += step_loss.item()
            
        # --- Logging ---
        avg_divisor = max(1, steps_per_update * Config.PPO_EPOCHS)
        avg_loss = total_loss / avg_divisor
        
        mean_step_reward = flat_rewards.mean().item()
        writer.add_scalar("Reward/Mean_Step", mean_step_reward, global_step)
        
        mean_ep_reward = 0.0
        if len(completed_episode_rewards) > 0:
            mean_ep_reward = np.mean(completed_episode_rewards)
            writer.add_scalar("Reward/Mean_Episode", mean_ep_reward, global_step)
            
        elapsed = max(1e-8, time.time() - wall_start)
        steps_per_sec = global_step / elapsed
        try:
            gpu_util = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).decode().strip().split("\n")[0]
        except Exception:
            gpu_util = "NA"
        print(f"Update {update}/{num_updates} | Loss: {avg_loss:.4f} | Aux: {total_aux_loss/avg_divisor:.4f} | EpReward: {mean_ep_reward:.2f} | SPS: {steps_per_sec:.2f} | GPU%: {gpu_util}")
        
        writer.add_scalar("Loss/Total", avg_loss, global_step)
        writer.add_scalar("Loss/Aux", total_aux_loss/avg_divisor, global_step)
        writer.add_scalar("Loss/PPO", total_ppo_loss/avg_divisor, global_step)
        
        # --- Save ---
        # Save Best Model
        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            torch.save(policy.state_dict(), f"{log_dir}/best_model.pth")
            print(f"New best model saved with reward {best_reward:.2f}")

        # Save Last 3 Checkpoints
        if update % 50 == 0:
            ckpt_path = f"{log_dir}/ckpt_{update:06d}.pth"
            torch.save(policy.state_dict(), ckpt_path)
            saved_checkpoints.append(ckpt_path)
            print(f"Model saved at update {update}")

    env.close()
    writer.close()

if __name__ == "__main__":
    train()
