#!/usr/bin/env python3
"""
Plot packet loss robustness curves in an IEEE T-ITS friendly style.

Preferred data source:
1. packet_loss_summary.csv
2. packet_loss_summary.json
3. recursively matched packet-loss summary CSV/JSON files
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = REPO_ROOT / "logs" / "packet_loss_sweep_full"
DEFAULT_RHO_ORDER = [0.00, 0.05, 0.10, 0.20, 0.50, 1.00]

METHOD_DISPLAY = {
    "ours": "Intent-GAT (Ours)",
    "dense_comm": "Dense Comm",
    "no_aux": "No-Aux",
    "lidar_only": "LiDAR-Only",
    "mappo_ips": "MAPPO-IPS",
    "where2comm": "Where2Comm-style",
}

METHOD_ORDER = ["ours", "dense_comm", "no_aux", "lidar_only", "mappo_ips", "where2comm"]

METHOD_STYLE = {
    "ours": {
        "color": "#000000",
        "linestyle": "-",
        "marker": "o",
    },
    "dense_comm": {
        "color": "#3f3f3f",
        "linestyle": "--",
        "marker": "s",
    },
    "no_aux": {
        "color": "#6a6a6a",
        "linestyle": "-.",
        "marker": "^",
    },
    "lidar_only": {
        "color": "#8a8a8a",
        "linestyle": ":",
        "marker": "D",
    },
    "mappo_ips": {
        "color": "#585858",
        "linestyle": (0, (5, 1.5)),
        "marker": "v",
    },
    "where2comm": {
        "color": "#7b7b7b",
        "linestyle": (0, (3, 1, 1, 1)),
        "marker": "P",
    },
}

CAPTION_TEXT = (
    "Closed-loop packet-loss robustness under varying packet loss rate $\\rho$. "
    "Success rate is reported over 50 evaluation episodes, and error bars denote the standard error "
    "estimated from finished agent outcomes. Intent-GAT maintains a stable success-rate curve across "
    "the sweep and consistently outperforms Dense Comm and LiDAR-Only. In this experiment, No-Aux "
    "also remains robust and stays close to, or slightly above, Intent-GAT on success rate, indicating "
    "that packet-loss sensitivity is modest for both intent-aware variants under the current protocol."
)

ROBUSTNESS_TEXT = (
    "Figure X reports closed-loop robustness under increasing packet loss. Intent-GAT exhibits a "
    "largely flat success-rate curve from $\\rho=0$ to $\\rho=1$, indicating that its policy quality "
    "degrades only mildly under severe communication loss. Dense Comm remains consistently below "
    "Intent-GAT across the entire sweep, suggesting weaker robustness to degraded message delivery. "
    "LiDAR-Only provides a lower communication-free reference. Notably, No-Aux also stays stable in "
    "this setting and does not show a clear monotonic disadvantage relative to Intent-GAT on success "
    "rate alone; therefore, the benefit of semantic anchoring may need to be interpreted jointly with "
    "other safety or interaction metrics rather than success rate in isolation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot packet loss robustness curves for publication.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_SWEEP_DIR),
        help="Directory containing packet loss sweep outputs.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Optional explicit CSV/JSON summary file. If omitted, the script auto-discovers one.",
    )
    parser.add_argument(
        "--singlecol_width",
        type=float,
        default=3.5,
        help="Single-column figure width in inches.",
    )
    parser.add_argument(
        "--doublecol_width",
        type=float,
        default=7.16,
        help="Double-column figure width in inches.",
    )
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
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.8,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def discover_input_file(input_dir: Path) -> Path:
    exact_csv = input_dir / "packet_loss_summary.csv"
    exact_json = input_dir / "packet_loss_summary.json"
    if exact_csv.exists():
        return exact_csv
    if exact_json.exists():
        return exact_json

    csv_candidates = sorted(input_dir.rglob("*packet*loss*summary*.csv"))
    if csv_candidates:
        return csv_candidates[0]
    json_candidates = sorted(input_dir.rglob("*packet*loss*summary*.json"))
    if json_candidates:
        return json_candidates[0]
    raise FileNotFoundError(f"No packet-loss summary CSV/JSON found under {input_dir}")


def load_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return payload["rows"]
        if isinstance(payload, list):
            return payload
        raise ValueError(f"Unsupported JSON structure in {path}")
    raise ValueError(f"Unsupported input file type: {path.suffix}")


def safe_get(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in ("", None):
            return row[key]
    return None


def to_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def adapt_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = safe_get(row, "method", "model_id")
    if method not in METHOD_DISPLAY:
        return None

    rho = to_float(safe_get(row, "rho", "mask_ratio", "robustness.mask_ratio"))
    sr_mean = to_float(safe_get(row, "sr_mean", "success_rate_mean", "rates.success", "success"))
    sr_se = to_float(safe_get(row, "sr_se", "success_rate_se"))
    sr_std = to_float(safe_get(row, "sr_std", "success_rate_std"))
    n_episodes = safe_get(row, "n_episodes", "episodes")
    n_success = safe_get(row, "n_success", "success_count")
    risk_ttc = to_float(safe_get(row, "risk.avg_min_ttc_s", "avg_min_ttc_s_mean"))
    risk_ttc_rate = to_float(safe_get(row, "risk.high_risk_ttc_rate", "high_risk_ttc_rate_mean"))
    risk_dist = to_float(safe_get(row, "risk.avg_min_dist_m", "avg_min_dist_m_mean"))
    risk_dist_rate = to_float(safe_get(row, "risk.high_risk_dist_rate", "high_risk_dist_rate_mean"))

    if rho is None or sr_mean is None:
        return None

    return {
        "method": method,
        "display_name": METHOD_DISPLAY[method],
        "rho": rho,
        "sr_mean": sr_mean,
        "sr_se": sr_se,
        "sr_std": sr_std,
        "n_episodes": n_episodes,
        "n_success": n_success,
        "risk.avg_min_ttc_s": risk_ttc,
        "risk.high_risk_ttc_rate": risk_ttc_rate,
        "risk.avg_min_dist_m": risk_dist,
        "risk.high_risk_dist_rate": risk_dist_rate,
    }


def choose_error_field(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    if any(to_float(r.get("sr_se")) is not None for r in rows):
        return "sr_se"
    if any(to_float(r.get("sr_std")) is not None for r in rows):
        return "sr_std"
    return None


def build_series(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[float, Dict[str, Any]]]:
    series: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for row in rows:
        method = str(row["method"])
        rho = float(row["rho"])
        series.setdefault(method, {})[rho] = row
    return series


def compute_ylim(series: Dict[str, Dict[float, Dict[str, Any]]], error_field: Optional[str]) -> Tuple[float, float]:
    mins: List[float] = []
    maxs: List[float] = []
    for method_map in series.values():
        for point in method_map.values():
            mean_pct = 100.0 * float(point["sr_mean"])
            err_val = to_float(point.get(error_field)) if error_field else None
            err_pct = 100.0 * err_val if err_val is not None else 0.0
            mins.append(mean_pct - err_pct)
            maxs.append(mean_pct + err_pct)

    if not mins or not maxs:
        return 0.0, 100.0

    lower = max(0.0, 5.0 * math.floor((min(mins) - 2.0) / 5.0))
    upper = min(100.0, 5.0 * math.ceil((max(maxs) + 2.0) / 5.0))
    if upper - lower < 15.0:
        center = 0.5 * (upper + lower)
        lower = max(0.0, 5.0 * math.floor((center - 8.0) / 5.0))
        upper = min(100.0, 5.0 * math.ceil((center + 8.0) / 5.0))
    return lower, upper


def plot_packet_loss_curve(
    rows: Sequence[Dict[str, Any]],
    width_in: float,
    height_in: float,
    output_pdf: Path,
    output_png: Path,
    legend_ncol: int,
    top_margin: float,
) -> None:
    configure_ieee_style()
    series = build_series(rows)
    error_field = choose_error_field(rows)
    y_min, y_max = compute_ylim(series, error_field)

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(top=top_margin, left=0.13, right=0.98, bottom=0.18)

    x_values = DEFAULT_RHO_ORDER
    x_labels = [f"{x:.2f}" for x in x_values]

    for method in METHOD_ORDER:
        method_points = series.get(method, {})
        xs: List[float] = []
        ys: List[float] = []
        yerrs: List[float] = []
        for rho in x_values:
            point = method_points.get(rho)
            if point is None:
                continue
            xs.append(rho)
            ys.append(100.0 * float(point["sr_mean"]))
            err_val = to_float(point.get(error_field)) if error_field else None
            yerrs.append(100.0 * err_val if err_val is not None else 0.0)

        if not xs:
            continue

        style = METHOD_STYLE[method]
        ax.errorbar(
            xs,
            ys,
            yerr=yerrs if error_field else None,
            label=METHOD_DISPLAY[method],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            marker=style["marker"],
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            capsize=2.2 if error_field else 0.0,
            elinewidth=0.85 if error_field else 0.0,
            capthick=0.85 if error_field else 0.0,
            zorder=3,
        )

    ax.set_xlabel(r"Packet Loss Rate $\rho$")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, color="#c7c7c7", alpha=0.8)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.35, color="#dfdfdf", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=legend_ncol,
        columnspacing=1.2,
        handlelength=2.3,
        borderpad=0.4,
    )

    ensure_parent(output_pdf)
    fig.savefig(output_pdf, format="pdf")
    fig.savefig(output_png, format="png", dpi=600)
    plt.close(fig)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    input_file = Path(args.input_file).resolve() if args.input_file else discover_input_file(input_dir)

    rows_raw = load_rows(input_file)
    adapted_rows = [row for row in (adapt_row(r) for r in rows_raw) if row is not None]
    adapted_rows = [row for row in adapted_rows if float(row["rho"]) in DEFAULT_RHO_ORDER]

    if not adapted_rows:
        raise RuntimeError(f"No valid packet-loss rows found in {input_file}")

    plot_packet_loss_curve(
        rows=adapted_rows,
        width_in=float(args.singlecol_width),
        height_in=2.65,
        output_pdf=input_dir / "fig_packet_loss_singlecol.pdf",
        output_png=input_dir / "fig_packet_loss_singlecol.png",
        legend_ncol=2,
        top_margin=0.80,
    )

    plot_packet_loss_curve(
        rows=adapted_rows,
        width_in=float(args.doublecol_width),
        height_in=3.05,
        output_pdf=input_dir / "fig_packet_loss_doublecol.pdf",
        output_png=input_dir / "fig_packet_loss_doublecol.png",
        legend_ncol=4,
        top_margin=0.83,
    )

    write_text(input_dir / "fig_packet_loss_caption.txt", CAPTION_TEXT)
    write_text(input_dir / "fig_packet_loss_results_paragraph.txt", ROBUSTNESS_TEXT)

    print(f"[packet-loss-plot] source_file={input_file}")
    if rows_raw:
        if isinstance(rows_raw[0], dict):
            print(f"[packet-loss-plot] source_columns={list(rows_raw[0].keys())}")
    print("[packet-loss-plot] outputs:")
    print(f"  - {input_dir / 'fig_packet_loss_singlecol.pdf'}")
    print(f"  - {input_dir / 'fig_packet_loss_singlecol.png'}")
    print(f"  - {input_dir / 'fig_packet_loss_doublecol.pdf'}")
    print(f"  - {input_dir / 'fig_packet_loss_doublecol.png'}")
    print(f"  - {input_dir / 'fig_packet_loss_caption.txt'}")
    print(f"  - {input_dir / 'fig_packet_loss_results_paragraph.txt'}")


if __name__ == "__main__":
    main()
