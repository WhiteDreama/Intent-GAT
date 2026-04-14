#!/usr/bin/env python3
"""
Regenerate a clean communication-stress figure from a completed CSV.

Input columns:
- method
- stress_type
- setting_value
- SR
- CR
- HR_TTC

Primary output:
- two-panel SR figure for staleness and burst communication stress
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
    "dense_comm": "Dense Comm",
    "mappo_ips": "MAPPO-IPS",
    "tarmac": "TarMAC",
    "lidar_only": "LiDAR-Only",
}

METHOD_ORDER = ["lidar_only", "tarmac", "mappo_ips", "dense_comm", "ours"]
METHOD_STYLE = {
    "ours": {"color": "#1f4e79", "linestyle": "-", "marker": "o", "zorder": 5},
    "dense_comm": {"color": "#7f6000", "linestyle": "-.", "marker": "^", "zorder": 4},
    "mappo_ips": {"color": "#8c2d19", "linestyle": "--", "marker": "s", "zorder": 3},
    "tarmac": {"color": "#6b6b6b", "linestyle": (0, (4.0, 1.4)), "marker": "D", "zorder": 2},
    "lidar_only": {"color": "#4f7f3b", "linestyle": ":", "marker": "P", "zorder": 1},
}

STRESS_ORDER = ["staleness", "burst"]
STRESS_TITLE = {
    "staleness": "(a) Staleness Stress",
    "burst": "(b) Burst Loss Stress",
}
STRESS_XLABEL = {
    "staleness": "Stale steps",
    "burst": "Burst length",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate communication-stress figures from CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV with communication-stress results.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save regenerated figures.")
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
            "legend.fontsize": 7.7,
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
            method = str(row.get("method", "")).strip()
            stress_type = str(row.get("stress_type", "")).strip()
            if method not in METHOD_DISPLAY or stress_type not in STRESS_ORDER:
                continue
            setting_value = to_float(row.get("setting_value"))
            sr = to_float(row.get("SR"))
            cr = to_float(row.get("CR"))
            hr_ttc = to_float(row.get("HR_TTC"))
            if setting_value is None or sr is None:
                continue
            rows.append(
                {
                    "method": method,
                    "stress_type": stress_type,
                    "setting_value": setting_value,
                    "SR": sr,
                    "CR": cr,
                    "HR_TTC": hr_ttc,
                }
            )
    return rows


def plot_sr(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "fig_comm_stress_sr.png"
    pdf_path = output_dir / "fig_comm_stress_sr.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), sharey=True)
    line_width = 1.8
    marker_size = 4.0

    all_y = [100.0 * float(row["SR"]) for row in rows if row.get("SR") is not None]
    y_min = max(0.0, np.floor((min(all_y) - 3.0) / 5.0) * 5.0)
    y_max = min(100.0, np.ceil((max(all_y) + 3.0) / 5.0) * 5.0)

    for ax, stress_type in zip(axes, STRESS_ORDER):
        stress_rows = [row for row in rows if row.get("stress_type") == stress_type]
        x_values = sorted({float(row["setting_value"]) for row in stress_rows})

        for method in METHOD_ORDER:
            method_rows = [row for row in stress_rows if row.get("method") == method]
            method_rows.sort(key=lambda row: float(row["setting_value"]))
            if not method_rows:
                continue

            x = np.asarray([float(row["setting_value"]) for row in method_rows], dtype=np.float64)
            y = 100.0 * np.asarray([float(row["SR"]) for row in method_rows], dtype=np.float64)
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

        ax.set_title(STRESS_TITLE[stress_type], pad=4.0)
        ax.set_xlabel(STRESS_XLABEL[stress_type])
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:.2f}" for x in x_values])
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.55, alpha=0.85)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Success rate (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(
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
    plot_sr(rows, Path(args.output_dir))


if __name__ == "__main__":
    main()
