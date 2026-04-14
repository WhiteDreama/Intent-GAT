[中文版](README.md)

# Cooperative MARL Driving on MetaDrive

This repository contains a cooperative multi-agent driving project built on top of MetaDrive. It focuses on cooperative decision-making under communication constraints, robustness evaluation, and mixed-traffic testing. The main research code lives in `marl_project/`, while the simulator source code is located in `metadrive/`.

## Preview

![Project Preview](fig/image.png)

## Highlights

- Graph-based multi-agent cooperative driving training and evaluation
- Multiple comparison modes: `ours`, `no_comm`, `no_aux`, `lidar_only`, `oracle`, `tarmac`, `mappo`, `mappo_ips`, `where2comm`
- Robustness tests for packet loss, observation noise, map variation, and mixed-traffic penetration sweeps
- Batch checkpoint evaluation, failure-case export, visualization, and inference-cost benchmarking

## Repository Structure

- `marl_project/train.py`: training entry point
- `marl_project/evaluate.py`: evaluation, robustness testing, and visualization
- `marl_project/benchmark_inference.py`: inference latency, payload, and parameter-count benchmark
- `marl_project/config.py`: main experiment configuration
- `marl_project/config_tarmac.py`: TarMAC-aligned configuration
- `marl_project/json/`: evaluation sweep configs
- `logs/marl_experiment/`: training logs and checkpoint outputs

## Environment Setup

Run the following commands from the repository root.

### Option 1: Conda environment

```bash
conda env create -f metadrive_env.yml
conda activate metadrive
pip install -e .
```

### Option 2: pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start for `marl_project`

### 1. Train a model

```bash
python marl_project/train.py --exp_name baseline_intent_gat --experiment_mode ours --device cuda:0
```

Common `--experiment_mode` values:

- `ours`
- `no_comm`
- `no_aux`
- `lidar_only`
- `oracle`
- `tarmac`
- `mappo`
- `mappo_ips`
- `where2comm`

### 2. Check training outputs

Training results are saved by default to `logs/marl_experiment/<exp_name>/`. Typical output files include:

- `best_model.pth`
- `best_success_model.pth`
- `ckpt_*.pth`
- `hparams.json`

In most cases, `best_success_model.pth` is the recommended checkpoint for evaluation.

### 3. Evaluate a single checkpoint

```bash
python marl_project/evaluate.py --model_path logs/marl_experiment/baseline_intent_gat/best_success_model.pth --model_type ours --episodes 20 --save_json logs/eval_ours.json
```

### 4. Run a mixed-traffic stress test

```bash
python marl_project/evaluate.py --model_path logs/marl_experiment/baseline_intent_gat/best_success_model.pth --model_type ours --mpr_sweep marl_project/json/eval_stress.json --episodes 20 --save_json logs/eval_stress.json
```

### 5. Run a communication robustness test

```bash
python marl_project/evaluate.py --model_path logs/marl_experiment/baseline_intent_gat/best_success_model.pth --model_type ours --mask 0.10 --episodes 20 --save_json logs/eval_mask_0.10.json
```

### 6. Visualize model behavior

```bash
python marl_project/evaluate.py --model_path logs/marl_experiment/baseline_intent_gat/best_success_model.pth --model_type ours --episodes 3 --render --top_down --pause_at_end
```

### 7. Benchmark inference cost

```bash
python -m marl_project.benchmark_inference
```

This script reports per-step inference latency, communication payload, and parameter count for different methods, and saves the results to `logs/eval_compare/inference_cost.csv`.

## Configuration Entry Points

If you want to quickly change experiment settings, start with `marl_project/config.py`. Commonly adjusted parameters include:

- `NUM_AGENTS`
- `MAP_MODE` / `MAP_BLOCK_NUM` / `MAP_TYPE`
- `LR` / `BATCH_SIZE` / `PPO_EPOCHS`
- `COMM_RADIUS` / `MASK_RATIO` / `NOISE_STD`
- `EXPERIMENT_MODE`

## Notes

- All commands assume you are running from the repository root
- `marl_project` is the main experiment directory, while `metadrive/` contains the simulator code
- For controlled comparisons, it is recommended to keep `config.py` fixed and only switch `--experiment_mode` and `--exp_name` from the command line
