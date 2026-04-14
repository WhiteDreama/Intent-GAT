#!/usr/bin/env python3
"""
Publication-ready packet-loss robustness figure for the T-ITS manuscript.

The figure is intentionally restrained:
- one standalone panel
- no interpolation between unobserved points
- subtle confidence bands
- direct labels instead of a large legend box
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

# Data source: keep the actual packet-loss points unchanged.
CSV_PATH = REPO_ROOT / "logs" / "packet_loss_sweep_full_semantics_fixed" / "packet_loss_summary_v2.csv"

# Export names requested by the user.
OUTPUT_PDF = REPO_ROOT / "figures" / "fig_packet_loss_robustness.pdf"
OUTPUT_PNG = REPO_ROOT / "figures" / "fig_packet_loss_robustness.png"

RHO_ORDER = [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]
RHO_TO_POS = {rho: idx for idx, rho in enumerate(RHO_ORDER)}
SELECTED_METHODS = ["ours", "mappo_ips", "dense_comm", "lidar_only", "where2comm"]

METHOD_DISPLAY = {
    "ours": "Intent-GAT (Ours)",
    "mappo_ips": "MAPPO-IPS",
    "dense_comm": "Dense Comm",
    "lidar_only": "LiDAR-Only",
    "where2comm": "Where2Comm-style",
}

# Plot order emphasizes the scientific story:
# weaker / secondary baselines first, strongest method last on top.
PLOT_ORDER = ["where2comm", "lidar_only", "dense_comm", "mappo_ips", "ours"]

# IEEE-style line choices:
# - line width kept moderate for print clarity
# - marker sizes kept moderate to avoid clutter
# - both style and marker differ so the figure remains readable in grayscale
METHOD_STYLE = {
    "ours": {"color": "#1f4e79", "linestyle": "-", "marker": "o", "zorder": 5},
    "mappo_ips": {"color": "#8c2d19", "linestyle": "--", "marker": "s", "zorder": 4},
    "dense_comm": {"color": "#7f6000", "linestyle": "-.", "marker": "^", "zorder": 3},
    "lidar_only": {"color": "#4f7f3b", "linestyle": ":", "marker": "D", "zorder": 2},
    "where2comm": {"color": "#6a51a3", "linestyle": (0, (3.2, 1.2, 1.0, 1.2)), "marker": "P", "zorder": 1},
}


def configure_ieee_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.3,
            "ytick.major.size": 3.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def to_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_data(csv_path: Path) -> Dict[str, List[Dict[str, float]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    grouped: Dict[str, List[Dict[str, float]]] = {method: [] for method in SELECTED_METHODS}
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method not in grouped:
            continue

        rho = to_float(row.get("rho"))
        sr_mean = to_float(row.get("sr_mean"))
        sr_se = to_float(row.get("sr_se"))
        if rho is None or sr_mean is None:
            continue

        grouped[method].append(
            {
                "rho": float(rho),
                "sr_mean": float(sr_mean),
                "sr_se": 0.0 if sr_se is None else float(sr_se),
            }
        )

    for method in grouped:
        grouped[method].sort(key=lambda item: item["rho"])
    return grouped


def compute_label_positions(label_items: List[Dict[str, float]], lower: float, upper: float, min_gap: float) -> Dict[str, float]:
    ordered = sorted(label_items, key=lambda item: item["y_actual"], reverse=True)
    adjusted: Dict[str, float] = {}
    current_top = upper
    for item in ordered:
        y = min(item["y_actual"], current_top)
        adjusted[item["method"]] = y
        current_top = y - min_gap

    if ordered:
        min_y = min(adjusted.values())
        if min_y < lower:
            shift = lower - min_y
            for key in adjusted:
                adjusted[key] += shift

    return adjusted


def plot_figure(grouped: Dict[str, List[Dict[str, float]]]) -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.55, 2.78))

    # Style choices:
    # - line width: 2.05 keeps the curves clear in print without looking heavy
    # - marker size: 4.0 is readable in one-column export without clutter
    # - confidence alpha: 0.10 keeps intervals visible but secondary
    line_width = 2.05
    marker_size = 4.0
    band_alpha = 0.10

    label_items: List[Dict[str, float]] = []

    for method in PLOT_ORDER:
        rows = grouped.get(method, [])
        if not rows:
            continue

        row_by_rho = {float(item["rho"]): item for item in rows}
        method_rows = [row_by_rho[rho] for rho in RHO_ORDER if rho in row_by_rho]
        if not method_rows:
            continue

        # Use equal spacing for the tested packet-loss settings so adjacent labels
        # such as 0.00 / 0.05 / 0.10 do not visually crowd each other.
        x = np.asarray([RHO_TO_POS[item["rho"]] for item in method_rows], dtype=np.float64)
        y = 100.0 * np.asarray([item["sr_mean"] for item in method_rows], dtype=np.float64)
        e = 100.0 * np.asarray([item["sr_se"] for item in method_rows], dtype=np.float64)
        style = METHOD_STYLE[method]

        ax.fill_between(
            x,
            y - e,
            y + e,
            color=style["color"],
            alpha=band_alpha,
            linewidth=0.0,
            zorder=style["zorder"] - 0.2,
        )
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=line_width,
            markersize=marker_size,
            markerfacecolor="white",
            markeredgewidth=0.95,
            zorder=style["zorder"],
        )

        label_items.append(
            {
                "method": method,
                "x_last": float(x[-1]),
                "y_actual": float(y[-1]),
                "color": style["color"],
            }
        )

    ax.set_xlabel("Packet loss rate")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(np.arange(len(RHO_ORDER), dtype=np.float64))
    ax.set_xticklabels([f"{rho:.2f}" for rho in RHO_ORDER])
    ax.set_xlim(-0.20, len(RHO_ORDER) - 1 + 0.92)

    # Tight y-range around the actual data to avoid empty space while staying honest.
    ax.set_ylim(42.0, 82.0)
    ax.set_yticks([45, 55, 65, 75])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.55, alpha=0.85)
    ax.grid(False, axis="x")

    # Direct labeling is cleaner than a boxed legend for this five-curve figure.
    label_y = compute_label_positions(label_items, lower=43.5, upper=79.5, min_gap=2.5)
    x_label = len(RHO_ORDER) - 1 + 0.38
    for item in sorted(label_items, key=lambda entry: label_y[entry["method"]], reverse=True):
        method = item["method"]
        y_actual = item["y_actual"]
        y_text = label_y[method]
        color = item["color"]

        if abs(y_text - y_actual) > 0.25:
            ax.plot(
                [item["x_last"], x_label - 0.012],
                [y_actual, y_text],
                color=color,
                linewidth=0.75,
                alpha=0.75,
                solid_capstyle="round",
                zorder=6,
            )

        ax.text(
            x_label,
            y_text,
            METHOD_DISPLAY[method],
            color=color,
            fontsize=7.8,
            va="center",
            ha="left",
            fontweight="semibold" if method == "ours" else "normal",
            clip_on=False,
        )

    # Export DPI: 600 for publication-ready raster output.
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    configure_ieee_style()
    grouped = load_data(CSV_PATH)
    plot_figure(grouped)


if __name__ == "__main__":
    main()
