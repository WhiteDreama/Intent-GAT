#!/usr/bin/env python3
"""
Run the Mixed-Traffic Load Robustness Experiment.

This evaluates three methods under matched mixed-traffic sweeps:
- Intent-GAT (ours)
- MAPPO-IPS
- TarMAC

For each method, the script runs the same configured traffic-density sweep on:
- X: unsignalized intersection
- r: roundabout

Outputs per method/scenario:
- summary.json (raw evaluate.py JSON payload)
- sweep_table.csv (raw evaluate.py CSV table)
- stdout.log / stderr.log / status.json / command.json

Top-level outputs:
- mixed_traffic_summary.csv
- mixed_traffic_summary.json
- run_manifest.json
- master_stdout.log / master_stderr.log
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "marl_project" / "evaluate.py"
DEFAULT_SWEEP_CONFIG = REPO_ROOT / "marl_project" / "json" / "mpr_config_mixed_traffic_load_a6.json"
DEFAULT_START_SEED = 10000


@dataclass(frozen=True)
class MethodSpec:
    name: str
    display_name: str
    model_path: Path
    model_type: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    display_name: str
    map_sequence: str


METHOD_SPECS: Dict[str, MethodSpec] = {
    "ours": MethodSpec(
        name="ours",
        display_name="Intent-GAT",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_intent_gat" / "best_success_model.pth",
        model_type="ours",
    ),
    "mappo_ips": MethodSpec(
        name="mappo_ips",
        display_name="MAPPO-IPS",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "MAPPO-IPS_v1" / "best_success_model.pth",
        model_type="mappo_ips",
    ),
    "tarmac": MethodSpec(
        name="tarmac",
        display_name="TarMAC",
        model_path=REPO_ROOT / "logs" / "marl_experiment" / "baseline_tarmac" / "best_success_model.pth",
        model_type="tarmac",
    ),
}


SCENARIO_SPECS: Dict[str, ScenarioSpec] = {
    "intersection": ScenarioSpec(name="intersection", display_name="Intersection", map_sequence="X"),
    "roundabout": ScenarioSpec(name="roundabout", display_name="Roundabout", map_sequence="r"),
}


FIELD_ORDER: List[str] = [
    "scenario",
    "scenario_label",
    "map_sequence",
    "method",
    "method_label",
    "model_type",
    "model_path",
    "sweep",
    "episodes",
    "start_seed",
    "num_agents",
    "traffic_density",
    "success",
    "crash",
    "out_of_road",
    "timeout",
    "steps_mean",
    "high_risk_ttc_rate",
    "avg_min_ttc_s",
    "observed_mpr_mean",
    "observed_rl_vehicles_mean",
    "observed_bg_vehicles_mean",
    "returncode",
    "runtime_sec",
    "summary_json",
    "table_csv",
    "status_json",
    "parse_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mixed-Traffic Load Robustness Experiment.")
    parser.add_argument("--output_root", type=str, default=None, help="Output root directory.")
    parser.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable used to run evaluate.py.")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per traffic-density condition.")
    parser.add_argument("--start_seed", type=int, default=DEFAULT_START_SEED, help="Evaluation start seed.")
    parser.add_argument("--sweep_config", type=str, default=str(DEFAULT_SWEEP_CONFIG), help="Path to the MPR sweep JSON config.")
    parser.add_argument("--aggregate_only", action="store_true", help="Skip evaluation and aggregate existing outputs only.")
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true", help="Skip completed method/scenario runs.")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false", help="Re-run method/scenario runs.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop at the first failed run.")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_SPECS.keys()), default=list(METHOD_SPECS.keys()))
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIO_SPECS.keys()), default=list(SCENARIO_SPECS.keys()))
    parser.add_argument("--plot_after", action="store_true", help="Plot the final aggregated figure after aggregation.")
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


class TeeLogger:
    def __init__(self, stdout_path: Path, stderr_path: Path) -> None:
        ensure_parent(stdout_path)
        ensure_parent(stderr_path)
        self._stdout_path = stdout_path
        self._stderr_path = stderr_path
        self._stdout = stdout_path.open("a", encoding="utf-8")
        self._stderr = stderr_path.open("a", encoding="utf-8")

    def info(self, message: str) -> None:
        line = message.rstrip()
        print(line)
        self._stdout.write(line + "\n")
        self._stdout.flush()

    def error(self, message: str) -> None:
        line = message.rstrip()
        print(line, file=sys.stderr)
        self._stderr.write(line + "\n")
        self._stderr.flush()

    def close(self) -> None:
        self._stdout.close()
        self._stderr.close()


def default_output_root() -> Path:
    return REPO_ROOT / "logs" / f"mixed_traffic_load_robustness_{timestamp_tag()}"


def build_command(
    python_exe: str,
    method: MethodSpec,
    scenario: ScenarioSpec,
    episodes: int,
    start_seed: int,
    sweep_config: Path,
    summary_json: Path,
    table_csv: Path,
) -> List[str]:
    return [
        python_exe,
        str(EVAL_SCRIPT),
        "--model_path",
        str(method.model_path),
        "--model_type",
        method.model_type,
        "--episodes",
        str(episodes),
        "--start_seed",
        str(start_seed),
        "--map_sequence",
        scenario.map_sequence,
        "--mpr_sweep",
        str(sweep_config),
        "--save_json",
        str(summary_json),
        "--save_table",
        str(table_csv),
        "--stochastic",
    ]


def get_run_dir(output_root: Path, scenario_name: str, method_name: str) -> Path:
    return output_root / scenario_name / method_name


def write_command_manifest(path: Path, command: Sequence[str], cwd: Path) -> None:
    dump_json(path, {"command": list(command), "cwd": str(cwd)})


def read_sweep_names(sweep_config: Path) -> List[str]:
    with sweep_config.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Sweep config must contain a JSON list: {sweep_config}")
    names: List[str] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sweep config entry {idx} must be a dict.")
        names.append(str(item.get("name", f"sweep_{idx:02d}")))
    return names


def run_method_scenario(
    python_exe: str,
    method: MethodSpec,
    scenario: ScenarioSpec,
    episodes: int,
    start_seed: int,
    sweep_config: Path,
    output_root: Path,
    logger: TeeLogger,
) -> int:
    run_dir = get_run_dir(output_root, scenario.name, method.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_json = run_dir / "summary.json"
    table_csv = run_dir / "sweep_table.csv"
    status_json = run_dir / "status.json"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    command_json = run_dir / "command.json"

    command = build_command(
        python_exe=python_exe,
        method=method,
        scenario=scenario,
        episodes=episodes,
        start_seed=start_seed,
        sweep_config=sweep_config,
        summary_json=summary_json,
        table_csv=table_csv,
    )
    write_command_manifest(command_json, command, REPO_ROOT)

    logger.info(
        f"[Run] scenario={scenario.name} ({scenario.map_sequence}) | method={method.name} | "
        f"episodes={episodes} | summary={summary_json}"
    )
    started_at = now_iso()
    start_time = time.perf_counter()
    with stdout_log.open("w", encoding="utf-8") as out_f, stderr_log.open("w", encoding="utf-8") as err_f:
        proc = subprocess.run(command, cwd=str(REPO_ROOT), stdout=out_f, stderr=err_f, text=True)
    runtime_sec = round(time.perf_counter() - start_time, 3)
    returncode = int(proc.returncode)
    finished_at = now_iso()

    dump_json(
        status_json,
        {
            "scenario": scenario.name,
            "scenario_label": scenario.display_name,
            "map_sequence": scenario.map_sequence,
            "method": method.name,
            "method_label": method.display_name,
            "model_type": method.model_type,
            "model_path": str(method.model_path),
            "returncode": returncode,
            "runtime_sec": runtime_sec,
            "started_at": started_at,
            "finished_at": finished_at,
            "summary_json": str(summary_json),
            "table_csv": str(table_csv),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "command": command,
        },
    )
    if returncode == 0:
        logger.info(f"[Done] scenario={scenario.name} | method={method.name} | runtime={runtime_sec:.1f}s")
    else:
        logger.error(f"[Fail] scenario={scenario.name} | method={method.name} | returncode={returncode}")
    return returncode


def extract_rows_for_run(
    output_root: Path,
    scenario: ScenarioSpec,
    method: MethodSpec,
) -> List[Dict[str, Any]]:
    run_dir = get_run_dir(output_root, scenario.name, method.name)
    summary_json = run_dir / "summary.json"
    table_csv = run_dir / "sweep_table.csv"
    status_json = run_dir / "status.json"
    summary_payload, summary_error = load_json_safe(summary_json)
    status_payload, status_error = load_json_safe(status_json)

    returncode = None
    runtime_sec = None
    if isinstance(status_payload, dict):
        returncode = status_payload.get("returncode")
        runtime_sec = status_payload.get("runtime_sec")

    rows: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]]
    if isinstance(summary_payload, dict) and isinstance(summary_payload.get("runs"), list):
        runs = [item for item in summary_payload["runs"] if isinstance(item, dict)]
    elif isinstance(summary_payload, dict):
        runs = [summary_payload]
    else:
        runs = []

    if not runs:
        rows.append(
            {
                "scenario": scenario.name,
                "scenario_label": scenario.display_name,
                "map_sequence": scenario.map_sequence,
                "method": method.name,
                "method_label": method.display_name,
                "model_type": method.model_type,
                "model_path": str(method.model_path),
                "sweep": None,
                "episodes": None,
                "start_seed": None,
                "num_agents": None,
                "traffic_density": None,
                "success": None,
                "crash": None,
                "out_of_road": None,
                "timeout": None,
                "steps_mean": None,
                "high_risk_ttc_rate": None,
                "avg_min_ttc_s": None,
                "observed_mpr_mean": None,
                "observed_rl_vehicles_mean": None,
                "observed_bg_vehicles_mean": None,
                "returncode": returncode,
                "runtime_sec": runtime_sec,
                "summary_json": str(summary_json),
                "table_csv": str(table_csv),
                "status_json": str(status_json),
                "parse_error": summary_error or status_error or "missing runs payload",
            }
        )
        return rows

    for run in sorted(runs, key=lambda item: (to_float(item.get("traffic_density")) or -1.0, str(item.get("sweep", "")))):
        rates = run.get("rates", {}) if isinstance(run.get("rates"), dict) else {}
        risk = run.get("risk", {}) if isinstance(run.get("risk"), dict) else {}
        observed = run.get("observed", {}) if isinstance(run.get("observed"), dict) else {}
        steps = run.get("steps", {}) if isinstance(run.get("steps"), dict) else {}
        rows.append(
            {
                "scenario": scenario.name,
                "scenario_label": scenario.display_name,
                "map_sequence": scenario.map_sequence,
                "method": method.name,
                "method_label": method.display_name,
                "model_type": method.model_type,
                "model_path": str(method.model_path),
                "sweep": run.get("sweep"),
                "episodes": run.get("episodes"),
                "start_seed": run.get("start_seed"),
                "num_agents": run.get("num_agents"),
                "traffic_density": run.get("traffic_density"),
                "success": rates.get("success"),
                "crash": rates.get("crash"),
                "out_of_road": rates.get("out_of_road"),
                "timeout": rates.get("timeout"),
                "steps_mean": steps.get("mean"),
                "high_risk_ttc_rate": risk.get("high_risk_ttc_rate"),
                "avg_min_ttc_s": risk.get("avg_min_ttc_s"),
                "observed_mpr_mean": observed.get("mpr_mean"),
                "observed_rl_vehicles_mean": observed.get("rl_vehicles_mean"),
                "observed_bg_vehicles_mean": observed.get("bg_vehicles_mean"),
                "returncode": returncode,
                "runtime_sec": runtime_sec,
                "summary_json": str(summary_json),
                "table_csv": str(table_csv),
                "status_json": str(status_json),
                "parse_error": summary_error or status_error,
            }
        )
    return rows


def write_aggregate(output_root: Path, methods: Iterable[str], scenarios: Iterable[str], logger: TeeLogger) -> Tuple[Path, Path]:
    rows: List[Dict[str, Any]] = []
    for scenario_name in scenarios:
        scenario = SCENARIO_SPECS[scenario_name]
        for method_name in methods:
            method = METHOD_SPECS[method_name]
            rows.extend(extract_rows_for_run(output_root, scenario, method))

    rows.sort(
        key=lambda row: (
            row.get("scenario") or "",
            row.get("method") or "",
            to_float(row.get("traffic_density")) if row.get("traffic_density") is not None else -1.0,
            str(row.get("sweep") or ""),
        )
    )

    csv_path = output_root / "mixed_traffic_summary.csv"
    json_path = output_root / "mixed_traffic_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    dump_json(json_path, rows)
    logger.info(f"[Aggregate] CSV saved to {csv_path}")
    logger.info(f"[Aggregate] JSON saved to {json_path}")
    return csv_path, json_path


def maybe_run_plot(output_root: Path, csv_path: Path, logger: TeeLogger) -> None:
    plot_script = REPO_ROOT / "scripts" / "plot_mixed_traffic_load_experiment.py"
    if not plot_script.exists():
        logger.error(f"[Plot] Missing plotting script: {plot_script}")
        return
    command = [
        sys.executable,
        str(plot_script),
        "--input_csv",
        str(csv_path),
        "--output_dir",
        str(output_root),
    ]
    logger.info(f"[Plot] Running: {' '.join(command)}")
    proc = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.stderr:
        logger.error(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"Plot script failed with return code {proc.returncode}")


def validate_prerequisites(methods: Iterable[str], sweep_config: Path) -> None:
    if not sweep_config.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_config}")
    _ = read_sweep_names(sweep_config)
    for method_name in methods:
        spec = METHOD_SPECS[method_name]
        if not spec.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found for {method_name}: {spec.model_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_config = Path(args.sweep_config)
    logger = TeeLogger(output_root / "master_stdout.log", output_root / "master_stderr.log")

    try:
        validate_prerequisites(args.methods, sweep_config)
        run_manifest = {
            "experiment_name": "Mixed-Traffic Load Robustness Experiment",
            "created_at": now_iso(),
            "output_root": str(output_root),
            "python_exe": args.python_exe,
            "episodes": args.episodes,
            "start_seed": args.start_seed,
            "sweep_config": str(sweep_config),
            "sweep_names": read_sweep_names(sweep_config),
            "methods": [
                {
                    "name": METHOD_SPECS[name].name,
                    "display_name": METHOD_SPECS[name].display_name,
                    "model_path": str(METHOD_SPECS[name].model_path),
                    "model_type": METHOD_SPECS[name].model_type,
                }
                for name in args.methods
            ],
            "scenarios": [
                {
                    "name": SCENARIO_SPECS[name].name,
                    "display_name": SCENARIO_SPECS[name].display_name,
                    "map_sequence": SCENARIO_SPECS[name].map_sequence,
                }
                for name in args.scenarios
            ],
        }
        dump_json(output_root / "run_manifest.json", run_manifest)
        logger.info(f"[Start] Mixed-Traffic Load Robustness Experiment -> {output_root}")
        logger.info(f"[Config] sweep={sweep_config} | episodes={args.episodes} | start_seed={args.start_seed}")

        if not args.aggregate_only:
            for scenario_name in args.scenarios:
                scenario = SCENARIO_SPECS[scenario_name]
                for method_name in args.methods:
                    method = METHOD_SPECS[method_name]
                    run_dir = get_run_dir(output_root, scenario.name, method.name)
                    summary_json = run_dir / "summary.json"
                    if args.skip_existing and summary_json.exists():
                        logger.info(f"[Skip] Existing summary found for scenario={scenario.name}, method={method.name}")
                        continue
                    returncode = run_method_scenario(
                        python_exe=args.python_exe,
                        method=method,
                        scenario=scenario,
                        episodes=args.episodes,
                        start_seed=args.start_seed,
                        sweep_config=sweep_config,
                        output_root=output_root,
                        logger=logger,
                    )
                    if returncode != 0 and args.fail_fast:
                        raise RuntimeError(f"Evaluation failed for scenario={scenario.name}, method={method.name}")

        csv_path, _ = write_aggregate(output_root, args.methods, args.scenarios, logger)
        if args.plot_after:
            maybe_run_plot(output_root, csv_path, logger)
        dump_json(
            output_root / "run_status.json",
            {
                "status": "completed",
                "finished_at": now_iso(),
                "aggregate_csv": str(csv_path),
            },
        )
        logger.info("[Finish] Experiment pipeline completed.")
    except Exception as exc:
        dump_json(
            output_root / "run_status.json",
            {
                "status": "failed",
                "finished_at": now_iso(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        logger.error(f"[Error] {type(exc).__name__}: {exc}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
