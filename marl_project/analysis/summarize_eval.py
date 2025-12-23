"""Aggregate evaluation outputs into a single comparison table.

Supports inputs produced by marl_project/evaluate.py:
- Single-model JSON (a dict with keys like model_path/rates/return/...)
- Batch JSON (a dict with key "models": [ ... ])

Examples:
  python marl_project/summarize_eval.py --inputs logs/eval_*.json --out_csv logs/eval_all.csv
  python marl_project/summarize_eval.py --inputs "logs/**/eval_batch*.json" --out_csv logs/eval_all.csv
"""

import os
import sys
import json
import glob
import argparse
import csv
import re
from typing import Any, Dict, List, Optional, Tuple


def _ensure_repo_root_on_path() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    if root_dir not in sys.path:
        sys.path.append(root_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize evaluation JSON outputs into a single CSV table")
    p.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more glob patterns or file paths to eval JSON files",
    )
    p.add_argument("--out_csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--print", action="store_true", help="Print a compact table to stdout")
    return p.parse_args()


def _expand_inputs(patterns: List[str]) -> List[str]:
    paths: List[str] = []
    for pat in patterns:
        if os.path.exists(pat) and os.path.isfile(pat):
            paths.append(pat)
            continue
        matches = glob.glob(pat, recursive=True)
        for m in matches:
            if os.path.isfile(m) and m.lower().endswith(".json"):
                paths.append(m)
    # de-dup while preserving order
    seen = set()
    out: List[str] = []
    for p in paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(p)
    return out


def _sort_key_for_ckpt(path: str) -> Tuple[int, str]:
    bn = os.path.basename(path)
    m = re.search(r"ckpt_(\d+)\.pth$", bn)
    if m:
        try:
            return int(m.group(1)), path
        except Exception:
            pass
    return 10**18, path


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_models(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "models" in payload and isinstance(payload["models"], list):
        return [m for m in payload["models"] if isinstance(m, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{100.0 * x:5.2f}%"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        ("ckpt", 10),
        ("success", 9),
        ("crash", 9),
        ("out", 9),
        ("timeout", 9),
        ("ret", 10),
        ("steps", 8),
        ("neigh", 7),
    ]
    line = " ".join([h[0].ljust(h[1]) for h in headers])
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        ckpt = str(r.get("ckpt", "-"))[:10].ljust(10)
        success = _format_pct(_to_float(r.get("success")))
        crash = _format_pct(_to_float(r.get("crash")))
        out = _format_pct(_to_float(r.get("out_of_road")))
        timeout = _format_pct(_to_float(r.get("timeout")))
        ret = _to_float(r.get("return_mean"))
        steps = _to_float(r.get("steps_mean"))
        neigh = _to_float(r.get("neighbors_mean"))
        ret_s = (f"{ret:9.2f}" if ret is not None else "-".rjust(10))
        steps_s = (f"{steps:7.1f}" if steps is not None else "-".rjust(8))
        neigh_s = (f"{neigh:6.2f}" if neigh is not None else "-".rjust(7))
        print(f"{ckpt} {success} {crash} {out} {timeout} {ret_s} {steps_s} {neigh_s}")


def main() -> None:
    args = parse_args()
    _ensure_repo_root_on_path()

    inputs = _expand_inputs(args.inputs)
    if not inputs:
        raise FileNotFoundError("No input JSON files matched --inputs")

    rows: List[Dict[str, Any]] = []
    for path in inputs:
        payload = _load_json(path)
        models = _extract_models(payload)
        for m in models:
            model_path = m.get("model_path")
            bn = os.path.basename(model_path) if isinstance(model_path, str) else "-"
            ckpt_num, _ = _sort_key_for_ckpt(bn)

            rates = m.get("rates", {}) if isinstance(m.get("rates"), dict) else {}

            rows.append(
                {
                    "source_json": path,
                    "model_path": model_path,
                    "ckpt": (ckpt_num if ckpt_num < 10**18 else bn.replace(".pth", "")),
                    "episodes": m.get("episodes"),
                    "start_seed": m.get("start_seed"),
                    "eval_full_reward": m.get("eval_full_reward"),
                    "deterministic": m.get("deterministic"),
                    "success": rates.get("success"),
                    "crash": rates.get("crash"),
                    "out_of_road": rates.get("out_of_road"),
                    "timeout": rates.get("timeout"),
                    "return_mean": _safe_get(m, ["return", "mean"]),
                    "return_std": _safe_get(m, ["return", "std"]),
                    "steps_mean": _safe_get(m, ["steps", "mean"]),
                    "neighbors_mean": _safe_get(m, ["graph", "neighbors_mean"]),
                    "avg_min_ttc_s": _safe_get(m, ["risk", "avg_min_ttc_s"]),
                    "high_risk_ttc_rate": _safe_get(m, ["risk", "high_risk_ttc_rate"]),
                    "avg_min_dist_m": _safe_get(m, ["risk", "avg_min_dist_m"]),
                    "high_risk_dist_rate": _safe_get(m, ["risk", "high_risk_dist_rate"]),
                    "ade": _safe_get(m, ["aux", "ade"]),
                    "fde": _safe_get(m, ["aux", "fde"]),
                }
            )

    rows = sorted(rows, key=lambda r: (r.get("ckpt", 10**18), str(r.get("model_path", ""))))

    out_csv = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to: {out_csv}")

    if args.print:
        _print_table(rows)


if __name__ == "__main__":
    main()
