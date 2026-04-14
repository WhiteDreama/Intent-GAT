#!/usr/bin/env python3
"""
Batch runner for packet loss robustness sweeps.

This script calls marl_project/evaluate.py for each method x rho combination,
stores raw outputs in dedicated directories, and aggregates summary.json files
into a single CSV/JSON table.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "marl_project" / "evaluate.py"

DEFAULT_START_SEED = 10000
DEFAULT_NUM_AGENTS = 6
DEFAULT_MAP_SEQUENCE = "X"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    model_path: Path
    model_type: str


METHOD_SPECS: Dict[str, MethodSpec] = {
    "ours": MethodSpec(
        name="ours",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_intent_gat" / "best_success_model.pth",
        model_type="ours",
    ),
    "dense_comm": MethodSpec(
        name="dense_comm",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_Dense_Comm" / "best_success_model.pth",
        model_type="oracle",
    ),
    "no_aux": MethodSpec(
        name="no_aux",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_no_aux" / "best_success_model.pth",
        model_type="no_aux",
    ),
    "lidar_only": MethodSpec(
        name="lidar_only",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_lidar_only" / "best_success_model.pth",
        model_type="lidar_only",
    ),
    "mappo_ips": MethodSpec(
        name="mappo_ips",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "MAPPO-IPS" / "best_success_model.pth",
        model_type="mappo_ips",
    ),
    "where2comm": MethodSpec(
        name="where2comm",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "where2Comm" / "best_success_model.pth",
        model_type="where2comm",
    ),
}


PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "methods": ["ours", "dense_comm"],
        "rhos": [0.00, 0.20],
        "episodes": 3,
    },
    "full": {
        "methods": ["ours", "dense_comm", "no_aux", "lidar_only", "mappo_ips", "where2comm"],
        "rhos": [0.00, 0.05, 0.10, 0.20, 0.50, 1.00],
        "episodes": 50,
    },
}


BASE_FIELD_ORDER: List[str] = [
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
    parser = argparse.ArgumentParser(description="Run packet loss robustness sweeps for multiple baselines.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default="smoke", help="Sweep preset to run.")
    parser.add_argument("--output_root", type=str, default=None, help="Root directory for sweep outputs.")
    parser.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable used to run evaluate.py.")
    parser.add_argument("--aggregate_only", action="store_true", help="Skip evaluation and only aggregate existing results.")
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true", help="Skip combos with an existing summary.json.")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false", help="Re-run combos even if summary.json exists.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop at the first failed evaluation run.")
    parser.set_defaults(skip_existing=True)
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


def safe_nested_get(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


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


def infer_rho_from_dir(combo_dir: Path) -> Optional[float]:
    name = combo_dir.name
    if not name.startswith("rho_"):
        return None
    raw = name[len("rho_"):].replace("p", ".")
    return to_float(raw)


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


def summary_fallback_rate(count_key: str, terminal_counts: Dict[str, Any], n_finished: Optional[int]) -> Optional[float]:
    if not isinstance(terminal_counts, dict) or not n_finished or n_finished <= 0:
        return None
    value = to_float(terminal_counts.get(count_key))
    if value is None:
        return None
    return value / float(n_finished)


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


def default_output_root(preset_name: str) -> Path:
    return REPO_ROOT / "logs" / f"packet_loss_sweep_{preset_name}"


def get_combo_dir(output_root: Path, method: str, rho: float) -> Path:
    return output_root / method / format_rho_dir(rho)


def build_command(python_exe: str, spec: MethodSpec, episodes: int, rho: float, summary_json: Path) -> List[str]:
    return [
        python_exe,
        "marl_project/evaluate.py",
        "--model_path",
        str(spec.model_path),
        "--model_type",
        spec.model_type,
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


def write_command_manifest(path: Path, command: Sequence[str], cwd: Path) -> None:
    dump_json(
        path,
        {
            "command": list(command),
            "cwd": str(cwd),
        },
    )


def run_combo(
    python_exe: str,
    spec: MethodSpec,
    episodes: int,
    rho: float,
    combo_dir: Path,
) -> int:
    combo_dir.mkdir(parents=True, exist_ok=True)
    summary_json = combo_dir / "summary.json"
    status_json = combo_dir / "status.json"
    stdout_log = combo_dir / "stdout.log"
    stderr_log = combo_dir / "stderr.log"
    command_json = combo_dir / "command.json"

    command = build_command(python_exe=python_exe, spec=spec, episodes=episodes, rho=rho, summary_json=summary_json)
    write_command_manifest(command_json, command, REPO_ROOT)

    started_at = now_iso()
    start_time = time.perf_counter()
    returncode = -1

    with stdout_log.open("w", encoding="utf-8") as out_f, stderr_log.open("w", encoding="utf-8") as err_f:
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        returncode = int(process.returncode)

    finished_at = now_iso()
    runtime_sec = round(time.perf_counter() - start_time, 3)

    dump_json(
        status_json,
        {
            "method": spec.name,
            "rho": rho,
            "model_type": spec.model_type,
            "model_path": str(spec.model_path),
            "returncode": returncode,
            "runtime_sec": runtime_sec,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": command,
            "summary_json": str(summary_json),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
        },
    )
    return returncode


def extract_row(
    method: str,
    combo_dir: Path,
    summary: Optional[Dict[str, Any]],
    summary_error: Optional[str],
    status: Optional[Dict[str, Any]],
    status_error: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "method": method,
        "rho": None,
        "rho_dir": combo_dir.name,
        "model_type": None,
        "model_path": None,
        "n_episodes": None,
        "start_seed": None,
        "num_agents": None,
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
        "summary_json": str(combo_dir / "summary.json"),
        "status_json": str(combo_dir / "status.json"),
        "parse_error": None,
    }

    rho_from_dir = infer_rho_from_dir(combo_dir)
    if rho_from_dir is not None:
        row["rho"] = rho_from_dir

    if isinstance(status, dict):
        row["rho"] = to_float(status.get("rho")) if status.get("rho") is not None else row["rho"]
        row["model_type"] = status.get("model_type")
        row["model_path"] = status.get("model_path")
        row["returncode"] = to_int(status.get("returncode"))
        row["runtime_sec"] = to_float(status.get("runtime_sec"))

    if isinstance(summary, dict):
        row["model_path"] = summary.get("model_path", row["model_path"])
        row["model_type"] = row["model_type"] or safe_nested_get(status or {}, "model_type")
        row["n_episodes"] = to_int(summary.get("episodes"))
        row["start_seed"] = to_int(summary.get("start_seed"))
        row["num_agents"] = to_int(summary.get("num_agents"))
        row["rho"] = to_float(safe_nested_get(summary, "robustness", "mask_ratio")) if safe_nested_get(summary, "robustness", "mask_ratio") is not None else row["rho"]

        terminal_counts = safe_nested_get(summary, "terminal_counts")
        if isinstance(terminal_counts, dict):
            n_finished = 0
            has_any_count = False
            for value in terminal_counts.values():
                count = to_int(value)
                if count is None:
                    continue
                n_finished += count
                has_any_count = True
            if has_any_count:
                row["n_finished"] = n_finished
            row["n_success"] = to_int(terminal_counts.get("success"))

        row["sr_mean"] = to_float(safe_nested_get(summary, "rates", "success"))
        row["cr_mean"] = to_float(safe_nested_get(summary, "rates", "crash"))
        row["oor_mean"] = to_float(safe_nested_get(summary, "rates", "out_of_road"))
        row["steps_mean"] = to_float(safe_nested_get(summary, "steps", "mean"))

        if row["sr_mean"] is None:
            row["sr_mean"] = summary_fallback_rate("success", terminal_counts or {}, row["n_finished"])
        if row["cr_mean"] is None:
            row["cr_mean"] = summary_fallback_rate("crash", terminal_counts or {}, row["n_finished"])
        if row["oor_mean"] is None:
            row["oor_mean"] = summary_fallback_rate("out_of_road", terminal_counts or {}, row["n_finished"])
        if row["n_success"] is None and row["sr_mean"] is not None and row["n_finished"] is not None:
            row["n_success"] = int(round(row["sr_mean"] * row["n_finished"]))

        row["sr_std"] = compute_sr_std(row["sr_mean"])
        row["sr_se"] = compute_sr_se(row["sr_mean"], row["n_finished"])

        row.update(flatten_mapping("risk", safe_nested_get(summary, "risk")))

    errors = [err for err in (summary_error, status_error) if err]
    if errors:
        row["parse_error"] = " | ".join(errors)
    return row


def discover_combo_dirs(output_root: Path) -> List[Path]:
    combo_dirs = set()
    if not output_root.exists():
        return []

    for path in output_root.rglob("summary.json"):
        combo_dirs.add(path.parent)
    for path in output_root.rglob("status.json"):
        combo_dirs.add(path.parent)

    for method_dir in output_root.iterdir():
        if not method_dir.is_dir():
            continue
        for rho_dir in method_dir.iterdir():
            if rho_dir.is_dir():
                combo_dirs.add(rho_dir)

    method_order = {name: idx for idx, name in enumerate(METHOD_SPECS.keys())}

    def sort_key(path: Path) -> Tuple[int, float, str]:
        method_name = path.parent.name
        rho_value = infer_rho_from_dir(path)
        return (
            method_order.get(method_name, len(method_order)),
            rho_value if rho_value is not None else float("inf"),
            str(path),
        )

    return sorted(combo_dirs, key=sort_key)


def build_field_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set(BASE_FIELD_ORDER)
    extras: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in seen:
                extras.append(key)
                seen.add(key)
    risk_extras = sorted([key for key in extras if key.startswith("risk.") and key not in BASE_FIELD_ORDER])
    other_extras = sorted([key for key in extras if not key.startswith("risk.") and key not in BASE_FIELD_ORDER])
    return BASE_FIELD_ORDER + risk_extras + other_extras


def aggregate_results(output_root: Path) -> Tuple[List[Dict[str, Any]], Path, Path]:
    combo_dirs = discover_combo_dirs(output_root)
    rows: List[Dict[str, Any]] = []

    for combo_dir in combo_dirs:
        method_name = combo_dir.parent.name
        summary_path = combo_dir / "summary.json"
        status_path = combo_dir / "status.json"
        summary, summary_error = load_json_safe(summary_path)
        status, status_error = load_json_safe(status_path)
        row = extract_row(
            method=method_name,
            combo_dir=combo_dir,
            summary=summary if isinstance(summary, dict) else None,
            summary_error=summary_error if not isinstance(summary, dict) else None,
            status=status if isinstance(status, dict) else None,
            status_error=status_error if not isinstance(status, dict) else None,
        )
        rows.append(row)

    csv_path = output_root / "packet_loss_summary.csv"
    json_path = output_root / "packet_loss_summary.json"
    output_root.mkdir(parents=True, exist_ok=True)

    fieldnames = build_field_order(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    dump_json(
        json_path,
        {
            "generated_at": now_iso(),
            "output_root": str(output_root),
            "rows": rows,
        },
    )
    return rows, csv_path, json_path


def main() -> int:
    args = parse_args()
    preset = PRESETS[args.preset]
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(args.preset).resolve()

    if not args.aggregate_only:
        failures = 0
        for method_name in preset["methods"]:
            spec = METHOD_SPECS[method_name]
            if not spec.model_path.exists():
                print(f"[sweep] Missing checkpoint for {method_name}: {spec.model_path}", file=sys.stderr)
                failures += 1
                if args.fail_fast:
                    break
                continue

            for rho in preset["rhos"]:
                combo_dir = get_combo_dir(output_root, method_name, rho)
                summary_json = combo_dir / "summary.json"
                if args.skip_existing and summary_json.exists():
                    print(f"[sweep] Skip existing: {method_name} {format_rho_dir(rho)}")
                    continue

                print(f"[sweep] Running {method_name} @ rho={rho:.2f}")
                returncode = run_combo(
                    python_exe=args.python_exe,
                    spec=spec,
                    episodes=int(preset["episodes"]),
                    rho=float(rho),
                    combo_dir=combo_dir,
                )
                if returncode != 0:
                    failures += 1
                    print(
                        f"[sweep] FAILED {method_name} @ rho={rho:.2f} (returncode={returncode})",
                        file=sys.stderr,
                    )
                    if args.fail_fast:
                        rows, csv_path, json_path = aggregate_results(output_root)
                        print(f"[sweep] Aggregated {len(rows)} rows to {csv_path}")
                        print(f"[sweep] JSON summary: {json_path}")
                        return returncode

        rows, csv_path, json_path = aggregate_results(output_root)
        print(f"[sweep] Aggregated {len(rows)} rows to {csv_path}")
        print(f"[sweep] JSON summary: {json_path}")
        return 1 if failures else 0

    rows, csv_path, json_path = aggregate_results(output_root)
    print(f"[sweep] Aggregated {len(rows)} rows to {csv_path}")
    print(f"[sweep] JSON summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
