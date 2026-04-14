#!/usr/bin/env python3
"""
Generate supporting mixed-traffic outputs from a completed summary CSV:
- CR vs configured traffic density figure
- high-load SR / CR / timeout summary table
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
    parser = argparse.ArgumentParser(description="Plot CR curves and export a high-load summary table.")
    parser.add_argument("--input_csv", type=str, required=True, help="Completed mixed-traffic summary CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for figure/table files.")
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
        rows: List[Dict[str, Any]] = []
        for row in reader:
            row = {str(k).strip(): v for k, v in row.items()}
            method = row.get("method")
            if method not in METHOD_DISPLAY:
                continue
            scenario = row.get("scenario")
            density = to_float(row.get("traffic_density"))
            success = to_float(row.get("success"))
            crash = to_float(row.get("crash"))
            timeout = to_float(row.get("timeout"))
            if scenario not in SCENARIO_ORDER or density is None:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": row.get("scenario_label") or scenario.title(),
                    "method": method,
                    "method_label": METHOD_DISPLAY[method],
                    "traffic_density": density,
                    "success": success,
                    "crash": crash,
                    "timeout": timeout,
                }
            )
    return rows


def write_high_load_summary_csv(rows: List[Dict[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["scenario"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for (scenario, method), items in grouped.items():
        high = max(items, key=lambda item: float(item["traffic_density"]))
        summary_rows.append(
            {
                "scenario": scenario,
                "method": METHOD_DISPLAY[method],
                "traffic_density": high["traffic_density"],
                "sr": high.get("success"),
                "cr": high.get("crash"),
                "timeout": high.get("timeout"),
            }
        )

    summary_rows.sort(key=lambda row: (row["scenario"], row["method"]))
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario", "method", "traffic_density", "sr", "cr", "timeout"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_rows


def write_high_load_summary_tex(summary_rows: List[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Scenario & Method & Density & SR & CR & Timeout \\",
        r"\midrule",
    ]
    for row in summary_rows:
        scenario_label = "Intersection" if row["scenario"] == "intersection" else "Roundabout"
        lines.append(
            f"{scenario_label} & {row['method']} & "
            f"{float(row['traffic_density']):.2f} & "
            f"{100.0 * float(row['sr']):.1f} & "
            f"{100.0 * float(row['cr']):.1f} & "
            f"{100.0 * float(row['timeout']):.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cr(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "mixed_traffic_cr_vs_density.pdf"
    png_path = output_dir / "mixed_traffic_cr_vs_density.png"

    present_scenarios = [scenario for scenario in SCENARIO_ORDER if any(row.get("scenario") == scenario for row in rows)]
    if len(present_scenarios) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(3.45, 2.65), sharey=True)
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, len(present_scenarios), figsize=(6.8, 2.65), sharey=True)

    densities = sorted({float(row["traffic_density"]) for row in rows})
    line_width = 1.8
    marker_size = 4.0

    for ax, scenario in zip(axes, present_scenarios):
        scenario_rows = [row for row in rows if row.get("scenario") == scenario]
        for method in METHOD_ORDER:
            method_rows = [row for row in scenario_rows if row.get("method") == method and row.get("crash") is not None]
            method_rows.sort(key=lambda row: float(row["traffic_density"]))
            if not method_rows:
                continue
            x = np.asarray([float(row["traffic_density"]) for row in method_rows], dtype=np.float64)
            y = 100.0 * np.asarray([float(row["crash"]) for row in method_rows], dtype=np.float64)
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
        x_min = min(densities)
        x_max = max(densities)
        pad = 0.01 if x_max > x_min else 0.02
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_xticks(densities)
        ax.set_xticklabels([f"{x:.2f}" for x in densities])
        ax.grid(True, axis="y", color="#e7e7e7", linewidth=0.55, alpha=0.85)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Configured traffic density")

    axes[0].set_ylabel("Collision rate (%)")

    all_y = [100.0 * float(row["crash"]) for row in rows if row.get("crash") is not None]
    if all_y:
        y_min = max(0.0, np.floor((min(all_y) - 2.0) / 5.0) * 5.0)
        y_max = min(100.0, np.ceil((max(all_y) + 2.0) / 5.0) * 5.0)
        axes[0].set_ylim(y_min, y_max)

    legend_ax = axes[-1]
    handles, labels = legend_ax.get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        loc="upper left",
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "mixed_traffic_high_load_sr_cr_timeout.csv"
    summary_tex = output_dir / "mixed_traffic_high_load_sr_cr_timeout.tex"

    summary_rows = write_high_load_summary_csv(rows, summary_csv)
    write_high_load_summary_tex(summary_rows, summary_tex)
    plot_cr(rows, output_dir)


if __name__ == "__main__":
    main()
