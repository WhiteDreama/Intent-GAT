#!/usr/bin/env python3
"""
One-off full packet-loss sweep runner for the TarMAC baseline.

This mirrors the semantics-fixed evaluation protocol already used in the paper:
- episodes: 50
- start_seed: 10000
- num_agents: 6
- map_sequence: X
- comm_mode: iid
- stale_steps: 0
- stochastic evaluation

Outputs:
- per-rho summary.json / status.json / stdout.log / stderr.log
- aggregated packet_loss_summary.csv / packet_loss_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "marl_project" / "evaluate.py"
DEFAULT_PYTHON = sys.executable
DEFAULT_START_SEED = 10000
DEFAULT_NUM_AGENTS = 6
DEFAULT_MAP_SEQUENCE = "X"
RHO_VALUES = [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model_path: Path
    model_type: str


SPEC = MethodSpec(
    name="tarmac",
    model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_tarmac" / "best_model.pth",
    model_type="tarmac",
)


BASE_FIELD_ORDER = [
    "method",
    "rho",
    "rho_dir",
    "model_type",
    "model_path",
    "n_episodes",
    "start_seed",
    "num_agents",
    "n_finished",
    "n_success",
    "sr_mean",
    "sr_std",
    "sr_se",
    "cr_mean",
    "oor_mean",
    "steps_mean",
    "risk.avg_min_ttc_s",
    "risk.high_risk_ttc_rate",
    "risk.avg_min_dist_m",
    "risk.high_risk_dist_rate",
    "returncode",
    "runtime_sec",
    "summary_json",
    "status_json",
    "parse_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full TarMAC packet-loss sweep with per-rho logging.")
    parser.add_argument("--output_root", type=str, required=True, help="Directory for this TarMAC sweep run.")
    parser.add_argument("--python_exe", type=str, default=DEFAULT_PYTHON, help="Python executable for evaluate.py.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per rho.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip rho points with an existing summary.json.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json_safe(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, f"missing file: {path.name}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_rho_arg(rho: float) -> str:
    return f"{rho:.2f}"


def format_rho_dir(rho: float) -> str:
    return f"rho_{rho:.2f}".replace(".", "p")


def flatten_mapping(prefix: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    flat: Dict[str, Any] = {}
    for key, sub_value in value.items():
        full_key = f"{prefix}.{key}"
        if isinstance(sub_value, dict):
            flat.update(flatten_mapping(full_key, sub_value))
        else:
            flat[full_key] = sub_value
    return flat


def compute_sr_std(sr_mean: Optional[float]) -> Optional[float]:
    if sr_mean is None:
        return None
    p = min(1.0, max(0.0, sr_mean))
    return math.sqrt(p * (1.0 - p))


def compute_sr_se(sr_mean: Optional[float], n_finished: Optional[int]) -> Optional[float]:
    if sr_mean is None or n_finished is None or n_finished <= 0:
        return None
    p = min(1.0, max(0.0, sr_mean))
    return math.sqrt((p * (1.0 - p)) / float(n_finished))


def get_combo_dir(output_root: Path, rho: float) -> Path:
    return output_root / SPEC.name / format_rho_dir(rho)


def build_command(python_exe: str, episodes: int, rho: float, summary_json: Path) -> List[str]:
    return [
        python_exe,
        str(EVAL_SCRIPT),
        "--model_path",
        str(SPEC.model_path),
        "--model_type",
        SPEC.model_type,
        "--episodes",
        str(episodes),
        "--start_seed",
        str(DEFAULT_START_SEED),
        "--num_agents",
        str(DEFAULT_NUM_AGENTS),
        "--map_sequence",
        DEFAULT_MAP_SEQUENCE,
        "--mask",
        format_rho_arg(rho),
        "--comm_mode",
        "iid",
        "--stale_steps",
        "0",
        "--save_json",
        str(summary_json),
        "--stochastic",
    ]


def run_combo(python_exe: str, episodes: int, rho: float, combo_dir: Path) -> int:
    combo_dir.mkdir(parents=True, exist_ok=True)
    summary_json = combo_dir / "summary.json"
    status_json = combo_dir / "status.json"
    stdout_log = combo_dir / "stdout.log"
    stderr_log = combo_dir / "stderr.log"
    command_json = combo_dir / "command.json"

    command = build_command(python_exe=python_exe, episodes=episodes, rho=rho, summary_json=summary_json)
    dump_json(command_json, {"command": command, "cwd": str(REPO_ROOT)})

    started_at = now_iso()
    start_time = time.perf_counter()

    with stdout_log.open("w", encoding="utf-8") as out_f, stderr_log.open("w", encoding="utf-8") as err_f:
        process = subprocess.run(command, cwd=str(REPO_ROOT), stdout=out_f, stderr=err_f, text=True)
        returncode = int(process.returncode)

    runtime_sec = round(time.perf_counter() - start_time, 3)
    finished_at = now_iso()

    dump_json(
        status_json,
        {
            "method": SPEC.name,
            "rho": rho,
            "model_type": SPEC.model_type,
            "model_path": str(SPEC.model_path),
            "returncode": returncode,
            "runtime_sec": runtime_sec,
            "started_at": started_at,
            "finished_at": finished_at,
            "summary_json": str(summary_json),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "command": command,
        },
    )
    return returncode


def extract_row(output_root: Path, rho: float) -> Dict[str, Any]:
    combo_dir = get_combo_dir(output_root, rho)
    summary_json = combo_dir / "summary.json"
    status_json = combo_dir / "status.json"
    summary, summary_error = load_json_safe(summary_json)
    status, status_error = load_json_safe(status_json)

    row: Dict[str, Any] = {
        "method": "Tarmac",
        "rho": rho,
        "rho_dir": format_rho_dir(rho),
        "model_type": SPEC.model_type,
        "model_path": str(SPEC.model_path),
        "n_episodes": 50,
        "start_seed": DEFAULT_START_SEED,
        "num_agents": DEFAULT_NUM_AGENTS,
        "n_finished": None,
        "n_success": None,
        "sr_mean": None,
        "sr_std": None,
        "sr_se": None,
        "cr_mean": None,
        "oor_mean": None,
        "steps_mean": None,
        "returncode": None,
        "runtime_sec": None,
        "summary_json": str(summary_json),
        "status_json": str(status_json),
        "parse_error": summary_error or status_error,
    }

    if isinstance(status, dict):
        row["returncode"] = to_int(status.get("returncode"))
        row["runtime_sec"] = to_float(status.get("runtime_sec"))

    if not isinstance(summary, dict):
        return row

    metrics = summary.get("metrics", {})
    terminal_counts = summary.get("terminal_counts", {})
    risk = summary.get("risk", {})

    row["n_finished"] = to_int(summary.get("n_finished"))
    row["n_success"] = to_int(summary.get("n_success"))
    sr_mean = to_float(summary.get("success_rate"))
    row["sr_mean"] = sr_mean
    row["sr_std"] = compute_sr_std(sr_mean)
    row["sr_se"] = compute_sr_se(sr_mean, row["n_finished"])
    row["cr_mean"] = to_float(summary.get("collision_rate"))
    row["oor_mean"] = to_float(summary.get("out_of_road_rate"))
    row["steps_mean"] = to_float(metrics.get("completion_steps"))
    row.update(flatten_mapping("risk", risk))

    if row["cr_mean"] is None and row["n_finished"]:
        row["cr_mean"] = to_float(terminal_counts.get("crash_vehicle")) / float(row["n_finished"])
    if row["oor_mean"] is None and row["n_finished"]:
        row["oor_mean"] = to_float(terminal_counts.get("out_of_road")) / float(row["n_finished"])
    return row


def write_aggregate(output_root: Path) -> None:
    rows = [extract_row(output_root, rho) for rho in RHO_VALUES]
    csv_path = output_root / "packet_loss_summary.csv"
    json_path = output_root / "packet_loss_summary.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELD_ORDER, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    dump_json(json_path, rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not SPEC.model_path.exists():
        raise FileNotFoundError(f"TarMAC checkpoint not found: {SPEC.model_path}")

    manifest = {
        "created_at": now_iso(),
        "method": SPEC.name,
        "model_type": SPEC.model_type,
        "model_path": str(SPEC.model_path),
        "episodes": args.episodes,
        "start_seed": DEFAULT_START_SEED,
        "num_agents": DEFAULT_NUM_AGENTS,
        "map_sequence": DEFAULT_MAP_SEQUENCE,
        "comm_mode": "iid",
        "stale_steps": 0,
        "rhos": RHO_VALUES,
        "python_exe": args.python_exe,
    }
    dump_json(output_root / "run_manifest.json", manifest)

    for rho in RHO_VALUES:
        combo_dir = get_combo_dir(output_root, rho)
        summary_json = combo_dir / "summary.json"
        if args.skip_existing and summary_json.exists():
            print(f"[skip] rho={rho:.2f} existing summary found: {summary_json}")
            continue
        print(f"[run] rho={rho:.2f}")
        returncode = run_combo(args.python_exe, args.episodes, rho, combo_dir)
        print(f"[done] rho={rho:.2f} returncode={returncode}")

    write_aggregate(output_root)
    print(f"[aggregate] wrote {output_root / 'packet_loss_summary.csv'}")
    print(f"[aggregate] wrote {output_root / 'packet_loss_summary.json'}")


if __name__ == "__main__":
    main()
