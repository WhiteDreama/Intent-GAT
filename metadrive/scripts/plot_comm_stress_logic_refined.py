#!/usr/bin/env python3
"""
Refined communication-stress figures with logically consistent metric views.

Outputs:
- fig_comm_stress_metrics_refined.{png,pdf}
- fig_comm_stress_degradation_refined.{png,pdf}

The degradation figure uses consistent "worse is positive" semantics:
- SR row: baseline - current (success-rate loss)
- CR row: current - baseline (collision-rate increase)
- HR-TTC row: current - baseline (risk increase)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


METHOD_DISPLAY = {
    "ours": "Intent-GAT",
    "dense_comm": "Dense Comm",
    "mappo_ips": "MAPPO-IPS",
    "tarmac": "TarMAC",
    "lidar_only": "LiDAR-Only",
}

METHOD_ORDER = ["lidar_only", "tarmac", "mappo_ips", "dense_comm", "ours"]
METHOD_STYLE = {
    "ours": {"color": "#1f4e79", "linestyle": "-", "marker": "o", "zorder": 5},
    "dense_comm": {"color": "#8c2d19", "linestyle": "--", "marker": "s", "zorder": 4},
    "mappo_ips": {"color": "#7f6000", "linestyle": "-.", "marker": "^", "zorder": 3},
    "tarmac": {"color": "#4f7f3b", "linestyle": (0, (1.2, 1.8)), "marker": "D", "zorder": 2},
    "lidar_only": {"color": "#6a51a3", "linestyle": ":", "marker": "P", "zorder": 1},
}

STRESS_ORDER = ["staleness", "burst"]
STRESS_TITLE = {
    "staleness": "(a) Staleness Stress",
    "burst": "(b) Burst Loss Stress",
}
STRESS_XLABEL = {
    "staleness": "Delay steps $\\Delta$",
    "burst": "Burst length $L$",
}

RAW_METRICS = [
    ("SR", "Success rate (%)"),
    ("CR", "Collision rate (%)"),
    ("HR_TTC", "HR-TTC (%)"),
]

DELTA_METRICS = [
    ("SR", "Success-rate loss (pp)"),
    ("CR", "Collision-rate increase (pp)"),
    ("HR_TTC", "HR-TTC increase (pp)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot refined communication-stress figures.")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV, typically comm_stress_three_metrics.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save refined figures")
    return parser.parse_args()


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.2,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            row = {str(k).strip(): v for k, v in row.items()}
            method = row.get("method")
            stress_type = row.get("stress_type")
            setting_value = to_float(row.get("setting_value"))
            if method not in METHOD_DISPLAY or stress_type not in STRESS_ORDER or setting_value is None:
                continue
            sr = to_float(row.get("SR"))
            cr = to_float(row.get("CR"))
            hr_ttc = to_float(row.get("HR_TTC"))
            oor = to_float(row.get("OOR"))
            steps = to_float(row.get("Steps"))
            if sr is None or cr is None or hr_ttc is None:
                continue
            rows.append(
                {
                    "method": method,
                    "stress_type": stress_type,
                    "setting_value": setting_value,
                    "SR": sr,
                    "CR": cr,
                    "HR_TTC": hr_ttc,
                    "OOR": oor,
                    "Steps": steps,
                }
            )
    if not rows:
        raise ValueError(f"No valid rows found in {path}")
    return rows


def values_for_metric(rows: List[Dict[str, Any]], metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(metric)
        if value is not None:
            values.append(100.0 * float(value))
    return values


def compute_delta_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baseline: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["stress_type"]))
        if key not in baseline or float(row["setting_value"]) < float(baseline[key]["setting_value"]):
            baseline[key] = row

    delta_rows: List[Dict[str, Any]] = []
    for row in rows:
        base = baseline[(str(row["method"]), str(row["stress_type"]))]
        delta_rows.append(
            {
                "method": row["method"],
                "stress_type": row["stress_type"],
                "setting_value": row["setting_value"],
                "SR": (float(base["SR"]) - float(row["SR"])) * 100.0,
                "CR": (float(row["CR"]) - float(base["CR"])) * 100.0,
                "HR_TTC": (float(row["HR_TTC"]) - float(base["HR_TTC"])) * 100.0,
            }
        )
    return delta_rows


def compute_limits(values: List[float], symmetric_zero: bool = False) -> Tuple[float, float]:
    if not values:
        return (0.0, 1.0)
    v_min = min(values)
    v_max = max(values)
    if symmetric_zero:
        bound = max(abs(v_min), abs(v_max))
        bound = max(bound, 1.0)
        step = 1.0 if bound <= 5 else 2.0
        upper = np.ceil((bound + 0.4) / step) * step
        return (-upper, upper)
    span = max(v_max - v_min, 1.0)
    pad = max(0.8, 0.08 * span)
    lower = np.floor((v_min - pad) / 1.0) * 1.0
    upper = np.ceil((v_max + pad) / 1.0) * 1.0
    return (lower, upper)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.55, alpha=0.85)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_metric_grid(
    rows: List[Dict[str, Any]],
    metric_specs: List[Tuple[str, str]],
    output_pdf: Path,
    output_png: Path,
    delta_mode: bool,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.15, 6.3), sharex="col")
    line_width = 1.8
    marker_size = 4.0

    def metric_to_plot_value(metric_name: str, value: Any) -> float:
        numeric_value = float(value)
        if delta_mode:
            return numeric_value
        return 100.0 * numeric_value

    metric_limits: Dict[str, Tuple[float, float]] = {}
    for metric, _label in metric_specs:
        vals = [metric_to_plot_value(metric, row[metric]) for row in rows if row.get(metric) is not None]
        metric_limits[metric] = compute_limits(vals, symmetric_zero=delta_mode)

    for col_idx, stress_type in enumerate(STRESS_ORDER):
        stress_rows = [row for row in rows if row["stress_type"] == stress_type]
        x_values = sorted({float(row["setting_value"]) for row in stress_rows})

        for row_idx, (metric, ylabel) in enumerate(metric_specs):
            ax = axes[row_idx, col_idx]
            for method in METHOD_ORDER:
                method_rows = [row for row in stress_rows if row["method"] == method and row.get(metric) is not None]
                method_rows.sort(key=lambda r: float(r["setting_value"]))
                if not method_rows:
                    continue
                x = np.asarray([float(r["setting_value"]) for r in method_rows], dtype=np.float64)
                y = np.asarray([metric_to_plot_value(metric, r[metric]) for r in method_rows], dtype=np.float64)
                style = METHOD_STYLE[method]
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=line_width,
                    markersize=marker_size,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    label=METHOD_DISPLAY[method],
                    zorder=style["zorder"],
                )

            if row_idx == 0:
                ax.set_title(STRESS_TITLE[stress_type], pad=5.0)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 2:
                ax.set_xlabel(STRESS_XLABEL[stress_type])

            ax.set_xticks(x_values)
            ax.set_xticklabels([str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.2f}" for x in x_values])
            ax.set_ylim(*metric_limits[metric])
            style_axes(ax)
            if delta_mode:
                ax.axhline(0.0, color="#8c8c8c", linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955], pad=0.7, w_pad=1.2, h_pad=1.0)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    rows = load_rows(Path(args.input_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    draw_metric_grid(
        rows=rows,
        metric_specs=RAW_METRICS,
        output_pdf=output_dir / "fig_comm_stress_metrics_refined.pdf",
        output_png=output_dir / "fig_comm_stress_metrics_refined.png",
        delta_mode=False,
    )

    delta_rows = compute_delta_rows(rows)
    draw_metric_grid(
        rows=delta_rows,
        metric_specs=DELTA_METRICS,
        output_pdf=output_dir / "fig_comm_stress_degradation_refined.pdf",
        output_png=output_dir / "fig_comm_stress_degradation_refined.png",
        delta_mode=True,
    )


if __name__ == "__main__":
    main()
