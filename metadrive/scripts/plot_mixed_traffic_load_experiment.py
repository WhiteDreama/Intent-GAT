#!/usr/bin/env python3
"""
Plot the Mixed-Traffic Load Robustness Experiment.

Produces:
- mixed_traffic_load_robustness.pdf
- mixed_traffic_load_robustness.png
- mixed_traffic_high_load_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


METHOD_DISPLAY = {
    "ours": "Intent-GAT",
    "mappo_ips": "MAPPO-IPS",
    "tarmac": "TarMAC",
}

SCENARIO_ORDER = ["intersection", "roundabout"]
SCENARIO_TITLE = {
    "intersection": "(a) Unsignalized Intersection",
    "roundabout": "(b) Roundabout",
}

METHOD_ORDER = ["tarmac", "mappo_ips", "ours"]
METHOD_STYLE = {
    "ours": {"color": "#1f4e79", "linestyle": "-", "marker": "o", "zorder": 3},
    "mappo_ips": {"color": "#8c2d19", "linestyle": "--", "marker": "s", "zorder": 2},
    "tarmac": {"color": "#6b6b6b", "linestyle": "-.", "marker": "^", "zorder": 1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the Mixed-Traffic Load Robustness Experiment.")
    parser.add_argument("--input_csv", type=str, required=True, help="Aggregated CSV produced by run_mixed_traffic_load_experiment.py")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for figures and supporting CSV.")
    return parser.parse_args()


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.2,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
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
        rows = []
        for row in reader:
            row = {str(k).strip(): v for k, v in row.items()}
            if row.get("method") not in METHOD_DISPLAY:
                continue
            rho = to_float(row.get("traffic_density"))
            sr = to_float(row.get("success"))
            if rho is None or sr is None:
                continue
            row["traffic_density"] = rho
            row["success"] = sr
            row["crash"] = to_float(row.get("crash"))
            row["high_risk_ttc_rate"] = to_float(row.get("high_risk_ttc_rate"))
            row["observed_bg_vehicles_mean"] = to_float(row.get("observed_bg_vehicles_mean"))
            row["observed_mpr_mean"] = to_float(row.get("observed_mpr_mean"))
            rows.append(row)
    return rows


def write_high_load_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    high_rows: List[Dict[str, Any]] = []
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("scenario")), str(row.get("method")))
        grouped.setdefault(key, []).append(row)

    for (scenario, method), items in grouped.items():
        best = max(items, key=lambda item: float(item["traffic_density"]))
        high_rows.append(
            {
                "scenario": scenario,
                "method": METHOD_DISPLAY[method],
                "traffic_density": best["traffic_density"],
                "success": best["success"],
                "crash": best.get("crash"),
                "high_risk_ttc_rate": best.get("high_risk_ttc_rate"),
                "observed_bg_vehicles_mean": best.get("observed_bg_vehicles_mean"),
                "observed_mpr_mean": best.get("observed_mpr_mean"),
            }
        )

    high_rows.sort(key=lambda row: (row["scenario"], row["method"]))
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "method",
                "traffic_density",
                "success",
                "crash",
                "high_risk_ttc_rate",
                "observed_bg_vehicles_mean",
                "observed_mpr_mean",
            ],
        )
        writer.writeheader()
        for row in high_rows:
            writer.writerow(row)


def plot(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "mixed_traffic_load_robustness.pdf"
    png_path = output_dir / "mixed_traffic_load_robustness.png"
    summary_path = output_dir / "mixed_traffic_high_load_summary.csv"

    write_high_load_summary(rows, summary_path)

    present_scenarios = [scenario for scenario in SCENARIO_ORDER if any(row.get("scenario") == scenario for row in rows)]
    if not present_scenarios:
        raise ValueError("No supported scenarios found in the input CSV.")
    unique_densities = sorted(
        {
            float(row["traffic_density"])
            for row in rows
            if row.get("traffic_density") is not None
        }
    )
    if not unique_densities:
        raise ValueError("No valid traffic_density values found in the input CSV.")

    if len(present_scenarios) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(3.45, 2.65), sharey=True)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, len(present_scenarios), figsize=(6.8, 2.65), sharey=True)
    line_width = 1.8
    marker_size = 4.0

    for ax, scenario in zip(axes, present_scenarios):
        scenario_rows = [row for row in rows if row.get("scenario") == scenario]
        for method in METHOD_ORDER:
            method_rows = [row for row in scenario_rows if row.get("method") == method]
            method_rows.sort(key=lambda row: float(row["traffic_density"]))
            if not method_rows:
                continue
            x = np.asarray([float(row["traffic_density"]) for row in method_rows], dtype=np.float64)
            y = 100.0 * np.asarray([float(row["success"]) for row in method_rows], dtype=np.float64)
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

        ax.set_title(SCENARIO_TITLE[scenario], pad=4.0)
        x_min = min(unique_densities)
        x_max = max(unique_densities)
        pad = 0.01 if x_max > x_min else 0.02
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_xticks(unique_densities)
        ax.set_xticklabels([f"{x:.2f}" for x in unique_densities], rotation=0)
        ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.55, alpha=0.85)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Configured traffic density")

    axes[0].set_ylabel("Success rate (%)")

    all_y = [100.0 * float(row["success"]) for row in rows if row.get("success") is not None]
    if all_y:
        y_min = max(0.0, np.floor((min(all_y) - 3.0) / 5.0) * 5.0)
        y_max = min(100.0, np.ceil((max(all_y) + 3.0) / 5.0) * 5.0)
        if y_max - y_min < 15.0:
            mid = 0.5 * (y_min + y_max)
            y_min = max(0.0, mid - 8.0)
            y_max = min(100.0, mid + 8.0)
        axes[0].set_ylim(y_min, y_max)

    legend_ax = axes[-1]
    handles, labels = legend_ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="lower left",
        frameon=False,
        handlelength=2.2,
        borderpad=0.2,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.6, w_pad=1.0)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    rows = load_rows(Path(args.input_csv))
    if not rows:
        raise ValueError(f"No valid rows found in {args.input_csv}")
    plot(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
