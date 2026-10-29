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
from followbench.visualization.constants import COLORS, DEFAULT_PLOT_METRICS, DEFAULT_RESULTS_DIR, METHOD_LABELS, PLOTTED_METRICS
from followbench.visualization.data_loader import add_filter_args, load_records


def plot_compare(records: list[dict], metrics: list[str], out_path: str) -> None:
    if not records:
        raise SystemExit("[VIS] no records matched filters")
    planners = sorted({r.get("planner") for r in records if r.get("planner")})
    rows = 1
    cols = len(metrics)
    fig, axes = plt.subplots(rows, cols, figsize=(max(4.0 * cols, 6.0), 3.6), squeeze=False)
    axes = axes.flatten()
    legend_handles = []
    legend_labels = []

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        colors = []
        for planner in planners:
            vals = [display_metric(r, metric) for r in records if r.get("planner") == planner]
            arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
            if len(arr) == 0:
                continue
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
            labels.append(METHOD_LABELS.get(planner, planner))
            colors.append(COLORS.get(planner, "#777777"))
        x = np.arange(len(means))
        kind = PLOTTED_METRICS.get(metric, {}).get("kind")
        if kind == "percent":
            ax.bar(x, means, color=colors, edgecolor="black", linewidth=1.0)
            ax.set_ylim(0, 105)
        else:
            ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=1.0)
            ax.set_ylim(bottom=0)
        for patch, label in zip(ax.patches, labels):
            if label not in legend_labels:
                legend_handles.append(patch)
                legend_labels.append(label)
        ax.set_title(metric)
        ax.set_ylabel(PLOTTED_METRICS.get(metric, {}).get("unit", ""))
        ax.set_xticks([])
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center", ncol=max(1, len(legend_labels)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot planner comparison bars for one FollowBench2.0 config.")
    add_filter_args(parser)
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
    out = args.output or os.path.join(
        DEFAULT_RESULTS_DIR,
        "compare",
        f"compare_{args.scenario or 'all'}_H{args.humans if args.humans is not None else 'all'}_{args.follow_position or 'all'}_D{args.distance if args.distance is not None else 'all'}.png",
    )
    plot_compare(records, args.metrics, out)
    print(f"[VIS] records={len(records)} output={out}")


if __name__ == "__main__":
    main()
