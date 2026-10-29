from __future__ import annotations

import argparse
import os
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "followbench_mpl"))

import matplotlib.pyplot as plt
import numpy as np


from followbench.visualization.aggregate import display_metric
from followbench.visualization.constants import COLORS, DEFAULT_PLOT_METRICS, DEFAULT_RESULTS_DIR, LINE_STYLES, METHOD_LABELS, PLOTTED_METRICS
from followbench.visualization.data_loader import add_filter_args, load_records


def plot_sweep(records: list[dict], x_key: str, metrics: list[str], out_path: str) -> None:
    if not records:
        raise SystemExit("[VIS] no records matched filters")
    planners = sorted({r.get("planner") for r in records if r.get("planner")})
    positions = sorted({r.get("follow_position") for r in records if r.get("follow_position")})
    xs = sorted({float(r.get(x_key)) for r in records if r.get(x_key) is not None})
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(4.4 * len(metrics), 6.0), 3.8), squeeze=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for planner in planners:
            for pos in positions:
                y = []
                x_plot = []
                for x in xs:
                    vals = [
                        display_metric(r, metric)
                        for r in records
                        if r.get("planner") == planner
                        and r.get("follow_position") == pos
                        and r.get(x_key) is not None
                        and abs(float(r.get(x_key)) - x) < 1e-6
                    ]
                    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
                    if len(arr):
                        x_plot.append(x)
                        y.append(float(np.mean(arr)))
                if y:
                    ax.plot(
                        x_plot,
                        y,
                        marker="o",
                        color=COLORS.get(planner, "#777777"),
                        linestyle=LINE_STYLES.get(pos, "-"),
                        label=f"{METHOD_LABELS.get(planner, planner)}-{pos}",
                    )
        if PLOTTED_METRICS.get(metric, {}).get("kind") == "percent":
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(metric)
        ax.set_xlabel(x_key)
        ax.set_ylabel(PLOTTED_METRICS.get(metric, {}).get("unit", ""))
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=max(1, min(len(labels), 6)), bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sweep curves over desired_distance or num_pedestrians.")
    add_filter_args(parser)
    parser.add_argument("--x", choices=["desired_distance", "num_pedestrians"], required=True)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_PLOT_METRICS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    records = load_records(
        log_root=args.log_root,
        scenario=args.scenario,
        scenario_name=args.scenario_name,
        planners=args.planners,
        humans=args.humans,
        distance=args.distance,
        follow_position=args.follow_position,
    )
    out = args.output or os.path.join(DEFAULT_RESULTS_DIR, "sweep", f"sweep_{args.scenario or 'all'}_{args.x}.png")
    plot_sweep(records, args.x, args.metrics, out)
    print(f"[VIS] records={len(records)} output={out}")


if __name__ == "__main__":
    main()
