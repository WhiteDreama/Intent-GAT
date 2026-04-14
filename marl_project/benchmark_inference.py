"""
Benchmark inference latency, payload, and parameter count for communication baselines.

Usage:
    python -m marl_project.benchmark_inference
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from marl_project.config import Config
from marl_project.mappo_modules import MAPPOPolicy
from marl_project.models.policy import CooperativePolicy


METHOD_CONFIGS = {
    "No-Comm": {"experiment_mode": "no_comm", "payload_bytes": 0.0},
    "Intent-GAT (ours)": {"experiment_mode": "ours", "payload_bytes": None},
    "Full-State Share": {"experiment_mode": "oracle", "payload_bytes": 428.0},
    "No-Aux": {"experiment_mode": "no_aux", "payload_bytes": None},
    "MAPPO-IPS": {"experiment_mode": "mappo_ips", "payload_bytes": None},
    "Where2Comm-style": {"experiment_mode": "where2comm", "payload_bytes": None},
}

CHECKPOINT_MAP = {
    "No-Comm": "logs/marl_experiment/baseline_no_comm/best_success_model.pth",
    "Intent-GAT (ours)": "logs/marl_experiment/baseline_intent_gat/best_success_model.pth",
    "Full-State Share": "logs/marl_experiment/baseline_raw_full/best_success_model.pth",
    "No-Aux": "logs/marl_experiment/baseline_no_aux/best_success_model.pth",
    "MAPPO-IPS": "logs/marl_experiment/baseline_mappo_ips/best_success_model.pth",
    "Where2Comm-style": "logs/marl_experiment/baseline_where2comm/best_success_model.pth",
}

NUM_AGENTS = 4
MAX_NEIGHBORS = 8
INPUT_DIM = 91
ACTION_DIM = 2
WARMUP_STEPS = 100
BENCH_STEPS = 1000


def build_dummy_batch(num_agents, device):
    batch = {
        "node_features": torch.randn(num_agents, INPUT_DIM, device=device),
        "neighbor_indices": torch.zeros(num_agents, MAX_NEIGHBORS, dtype=torch.long, device=device),
        "neighbor_mask": torch.ones(num_agents, MAX_NEIGHBORS, device=device),
        "neighbor_rel_pos": torch.randn(num_agents, MAX_NEIGHBORS, 4, device=device),
    }
    for i in range(num_agents):
        for j in range(MAX_NEIGHBORS):
            batch["neighbor_indices"][i, j] = (i + j + 1) % num_agents
    return batch


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_model(experiment_mode, device):
    Config.apply_experiment_mode(experiment_mode)
    if experiment_mode in {"mappo", "mappo_ips"}:
        return MAPPOPolicy(input_dim=INPUT_DIM, action_dim=ACTION_DIM, num_agents=NUM_AGENTS).to(device)
    return CooperativePolicy(input_dim=INPUT_DIM, action_dim=ACTION_DIM).to(device)


def _load_checkpoint_if_available(model, method_name, device):
    ckpt_path = CHECKPOINT_MAP.get(method_name)
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "actor_state_dict" in ckpt:
        model.load_state_dict(ckpt["actor_state_dict"], strict=False)
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)


def benchmark_method(method_name, config_dict, device):
    model = _build_model(config_dict["experiment_mode"], device)
    _load_checkpoint_if_available(model, method_name, device)
    model.eval()

    params = count_params(model)
    batch = build_dummy_batch(NUM_AGENTS, device)

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCH_STEPS):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) / BENCH_STEPS * 1000.0
    payload = config_dict["payload_bytes"]
    if payload is None:
        payload = float(getattr(model, "estimate_payload_bytes", lambda _batch: 0.0)(batch))

    return latency_ms, payload, params


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | N={NUM_AGENTS} agents | {BENCH_STEPS} steps")
    print("=" * 70)

    results = {}
    for name, cfg in METHOD_CONFIGS.items():
        lat, pay, par = benchmark_method(name, cfg, device)
        results[name] = (lat, pay, par)
        print(f"{name:25s}  Latency={lat:7.2f} ms/step  Payload={pay:7.1f} B  #Params={par:,}")

    print("\n" + "=" * 70)
    print("LaTeX rows for tab:inference_cost:")
    print("=" * 70)
    for name, (lat, pay, par) in results.items():
        par_k = par / 1000
        par_str = f"{par_k:.1f}k" if par_k < 1000 else f"{par/1e6:.2f}M"
        print(f"{name:25s} & {lat:.2f} & {pay:.1f} & {par_str} \\\\")

    out_dir = "logs/eval_compare"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "inference_cost.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("method,latency_ms_per_step,payload_bytes,num_params\n")
        for name, (lat, pay, par) in results.items():
            f.write(f"{name},{lat:.4f},{pay:.4f},{par}\n")
    print(f"\nSaved CSV to {csv_path}")


if __name__ == "__main__":
    main()
