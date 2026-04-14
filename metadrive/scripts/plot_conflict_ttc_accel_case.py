#!/usr/bin/env python3
"""
Plot Figure 6: representative conflict episode TTC + ego acceleration.

This script reads the exported windowed CSV from export_conflict_case_csv.py and
generates a publication-ready two-panel figure using matplotlib only.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

# Suggested paper caption:
#
# """Micro-kinematic response in a representative interaction episode. The top
# panel shows the time-to-collision (TTC) to a shared reference
# interaction-critical vehicle tracked across methods, and the bottom panel
# shows the ego vehicle's longitudinal acceleration. Intent-GAT reacts earlier
# to the reference interaction while maintaining a moderate, non-extreme
# braking profile. This figure is intended as a qualitative mechanism
# illustration rather than a standalone safety ranking."""
#
# Short alternative:
#
# """Representative interaction episode with a shared reference vehicle.
# Intent-GAT responds earlier to the developing interaction while exhibiting a
# moderate braking profile."""
#
# The caption is intentionally conservative: this case is used to illustrate
# earlier and more proactive engagement with the shared reference actor, not to
# claim a strict safety advantage over all baselines.

# =========================
# Top-level Config
# =========================
# This figure is tuned for the selected representative case seed10002_ep002.
# LiDAR-Only is excluded by default because the fixed critical actor coverage
# and TTC validity are too weak in this case for a fair three-way comparison.
csv_path = (
    REPO_ROOT
    / "logs"
    / "representative_conflict_case_seed10002_agent5"
    / "conflict_case_seed10002_ep002_windowed.csv"
)
target_episode_id = 2
selected_methods = ["ours", "no_aux", "dense_comm"]
ttc_threshold = 2.5
xlim = (-3.0, 3.0)
smoothing_window = 5
# TTC is clipped only at plot time so the exported CSV remains physically
# faithful while the displayed range focuses on the conflict region.
ttc_display_clip = 6.0

METHOD_ORDER = ["ours", "no_aux", "dense_comm", "lidar_only"]
METHOD_DISPLAY = {
    "ours": "Intent-GAT (Ours)",
    "no_aux": "No-Aux",
    "dense_comm": "Dense Comm",
    "lidar_only": "LiDAR-Only",
}
METHOD_STYLE = {
    "ours": {"color": "#000000", "linestyle": "-", "marker": "o"},
    "no_aux": {"color": "#444444", "linestyle": "--", "marker": "s"},
    "dense_comm": {"color": "#6a6a6a", "linestyle": "-.", "marker": "^"},
    "lidar_only": {"color": "#8c8c8c", "linestyle": ":", "marker": "D"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot representative conflict-case TTC/acceleration figure.")
    parser.add_argument("--csv_path", type=str, default=str(csv_path))
    parser.add_argument("--target_episode_id", type=int, default=target_episode_id)
    parser.add_argument("--methods", type=str, default=",".join(selected_methods))
    parser.add_argument("--ttc_threshold", type=float, default=ttc_threshold)
    parser.add_argument("--xlim_left", type=float, default=xlim[0])
    parser.add_argument("--xlim_right", type=float, default=xlim[1])
    parser.add_argument("--smoothing_window", type=int, default=smoothing_window)
    parser.add_argument("--ttc_display_clip", type=float, default=ttc_display_clip)
    parser.add_argument("--output_stem", type=str, default="fig6_ttc_accel_case_seed10002_ep002")
    return parser.parse_args()


def configure_ieee_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "lines.linewidth": 2.1,
            "lines.markersize": 2.9,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#666666",
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def to_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_and_filter_data(
    csv_path_value: Path,
    target_episode_id_value: Optional[int],
    selected_methods_value: Sequence[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[int]]:
    with csv_path_value.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path_value}")

    available_episode_ids = sorted({to_int(row.get("episode_id")) for row in rows if to_int(row.get("episode_id")) is not None})
    chosen_episode = available_episode_ids[0] if available_episode_ids else None
    if target_episode_id_value is not None:
        chosen_episode = int(target_episode_id_value)

    filtered = [
        row for row in rows
        if (chosen_episode is None or to_int(row.get("episode_id")) == chosen_episode)
        and str(row.get("method")) in set(selected_methods_value)
    ]
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for method in METHOD_ORDER:
        method_rows = [row for row in filtered if str(row.get("method")) == method]
        method_rows.sort(key=lambda r: (to_float(r.get("time_s")) or 0.0, to_int(r.get("step")) or 0))
        if method_rows:
            by_method[method] = method_rows
    return by_method, chosen_episode


def smooth_acceleration(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    if int(window) <= 1:
        return list(values)
    half = int(window) // 2
    smoothed: List[Optional[float]] = []
    for idx in range(len(values)):
        chunk = [v for v in values[max(0, idx - half): min(len(values), idx + half + 1)] if v is not None]
        smoothed.append(None if not chunk else float(np.mean(chunk)))
    return smoothed


def detect_reaction_onset_time(
    times: Sequence[Optional[float]],
    smoothed_accel: Sequence[Optional[float]],
    threshold: float = -1.0,
    consecutive_steps: int = 2,
) -> Optional[float]:
    run = 0
    start_time: Optional[float] = None
    for time_value, accel_value in zip(times, smoothed_accel):
        if time_value is None:
            run = 0
            start_time = None
            continue
        if float(time_value) >= 0.0 and accel_value is not None and float(accel_value) < float(threshold):
            if run == 0:
                start_time = float(time_value)
            run += 1
            if run >= int(consecutive_steps):
                return start_time
        else:
            run = 0
            start_time = None
    return None


def plot_case(
    by_method: Dict[str, List[Dict[str, Any]]],
    ttc_threshold_value: float,
    xlim_value: Tuple[float, float],
    smoothing_window_value: int,
    ttc_display_clip_value: float,
    output_stem_value: str,
) -> Tuple[Path, Path]:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figures_dir / f"{output_stem_value}.pdf"
    png_path = figures_dir / f"{output_stem_value}.png"
    generic_pdf_path = figures_dir / "fig6_ttc_accel_case.pdf"
    generic_png_path = figures_dir / "fig6_ttc_accel_case.png"

    fig, (ax_ttc, ax_accel) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(3.45, 4.15),
        constrained_layout=True,
    )

    plotted_methods: List[str] = []
    tmins: List[float] = []
    tmaxs: List[float] = []
    valid_ttc_counts: Dict[str, int] = {}
    diagnostics: Dict[str, Dict[str, Optional[float]]] = {}

    for method in METHOD_ORDER:
        rows = by_method.get(method)
        if not rows:
            print(f"[Figure6] Warning: missing method '{method}' in CSV, skipping.")
            continue
        style = METHOD_STYLE[method]
        display = METHOD_DISPLAY[method]
        times = [to_float(row.get("time_s")) for row in rows]
        ttc_vals = [to_float(row.get("ttc")) for row in rows]
        accel_vals = [to_float(row.get("ego_accel")) for row in rows]
        # Acceleration is smoothed only for display because the raw finite
        # difference is noisy; TTC is intentionally left unsmoothed so the
        # threshold crossing and missing segments remain faithful.
        # TTC is intentionally not smoothed so threshold crossings and missing
        # segments remain honest; acceleration is smoothed only at plot time to
        # suppress finite-difference noise from the exported raw trace.
        accel_vals = smooth_acceleration(accel_vals, smoothing_window_value)
        ttc_plot = [min(ttc_display_clip_value, v) if v is not None else np.nan for v in ttc_vals]
        accel_plot = [v if v is not None else np.nan for v in accel_vals]
        time_plot = [v if v is not None else np.nan for v in times]

        ax_ttc.plot(
            time_plot,
            ttc_plot,
            label=display,
            **style,
            markevery=3,
            linewidth=2.1,
            markersize=2.9,
        )
        ax_accel.plot(
            time_plot,
            accel_plot,
            **style,
            markevery=3,
            linewidth=2.1,
            markersize=2.9,
        )
        plotted_methods.append(method)
        finite_times = [t for t in times if t is not None]
        if finite_times:
            tmins.append(min(finite_times))
            tmaxs.append(max(finite_times))
        valid_ttc_counts[method] = sum(
            1 for t, v in zip(times, ttc_vals)
            if t is not None and xlim_value[0] <= t <= xlim_value[1] and v is not None and np.isfinite(v)
        )
        ttc_post_window = [
            float(v) for t, v in zip(times, ttc_vals)
            if t is not None and v is not None and 0.0 <= float(t) <= 2.5 and np.isfinite(v)
        ]
        accel_post_window = [
            float(v) for t, v in zip(times, accel_vals)
            if t is not None and v is not None and 0.0 <= float(t) <= 2.5 and np.isfinite(v)
        ]
        diagnostics[method] = {
            "min_ttc_post_window": min(ttc_post_window) if ttc_post_window else None,
            "reaction_onset_time": detect_reaction_onset_time(times, accel_vals),
            "smoothed_min_accel": min(accel_post_window) if accel_post_window else None,
        }

    ax_ttc.axhline(ttc_threshold_value, color="#777777", linestyle="--", linewidth=1.0)
    ax_ttc.axvline(0.0, color="#8a8a8a", linestyle="--", linewidth=0.9)
    ax_accel.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax_accel.axvline(0.0, color="#8a8a8a", linestyle="--", linewidth=0.9)

    ax_ttc.set_ylabel("TTC (s)")
    ax_ttc.set_title("TTC to interaction-critical vehicle")
    ax_accel.set_ylabel("Acceleration (m/s$^2$)")
    ax_accel.set_xlabel("Time relative to conflict onset (s)")
    ax_accel.set_title("Ego longitudinal acceleration")

    for ax in (ax_ttc, ax_accel):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", color="#d9d9d9", linewidth=0.45, alpha=0.30)
        ax.set_facecolor("white")

    ax_ttc.set_xlim(*xlim_value)
    ax_ttc.set_ylim(0.0, ttc_display_clip_value)
    ax_accel.set_xlim(*xlim_value)
    ax_ttc.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        ncol=1,
        borderpad=0.25,
        handlelength=1.8,
        labelspacing=0.25,
        columnspacing=0.8,
    )

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    fig.savefig(generic_pdf_path)
    fig.savefig(generic_png_path, dpi=300)
    plt.close(fig)

    display_tmins = []
    display_tmaxs = []
    for rows in by_method.values():
        times = [to_float(row.get("time_s")) for row in rows]
        finite_in_window = [t for t in times if t is not None and xlim_value[0] <= t <= xlim_value[1]]
        if finite_in_window:
            display_tmins.append(min(finite_in_window))
            display_tmaxs.append(max(finite_in_window))
    time_range = (min(display_tmins), max(display_tmaxs)) if display_tmins and display_tmaxs else (None, None)
    print(f"[Figure6] plotted methods: {', '.join(plotted_methods)}")
    print(f"[Figure6] time range found: {time_range[0]} to {time_range[1]} s")
    for method in plotted_methods:
        print(f"[Figure6] valid TTC points in display window | {method}: {valid_ttc_counts.get(method, 0)}")
        print(
            "[Figure6] diagnostics | "
            f"{method}: min TTC[0,2.5]={diagnostics.get(method, {}).get('min_ttc_post_window')}, "
            f"reaction onset={diagnostics.get(method, {}).get('reaction_onset_time')}, "
            f"smoothed min accel={diagnostics.get(method, {}).get('smoothed_min_accel')}"
        )
    print(f"[Figure6] saved PDF: {pdf_path}")
    print(f"[Figure6] saved PNG: {png_path}")
    print(f"[Figure6] saved generic PDF: {generic_pdf_path}")
    print(f"[Figure6] saved generic PNG: {generic_png_path}")
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    configure_ieee_style()
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    by_method, chosen_episode = load_and_filter_data(
        csv_path_value=Path(args.csv_path),
        target_episode_id_value=args.target_episode_id,
        selected_methods_value=methods,
    )
    print(f"[Figure6] target episode id: {chosen_episode}")
    plot_case(
        by_method=by_method,
        ttc_threshold_value=float(args.ttc_threshold),
        xlim_value=(float(args.xlim_left), float(args.xlim_right)),
        smoothing_window_value=max(1, int(args.smoothing_window)),
        ttc_display_clip_value=float(args.ttc_display_clip),
        output_stem_value=str(args.output_stem),
    )


if __name__ == "__main__":
    main()
