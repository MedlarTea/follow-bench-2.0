from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np


from followbench.visualization.aggregate import display_metric
from followbench.visualization.constants import DEFAULT_RESULTS_DIR, DEFAULT_TABLE_METRICS
from followbench.visualization.data_loader import add_filter_args, load_records


def export_wide_table(records: list[dict[str, Any]], group_keys: list[str], metrics: list[str], out_path: str) -> None:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(k) for k in group_keys)].append(record)

    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple("" if v is None else str(v) for v in kv[0])):
        row = {k: v for k, v in zip(group_keys, key)}
        row["count"] = len(items)
        for metric in metrics:
            values = [display_metric(item, metric) for item in items]
            arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(arr)) if len(arr) else None
            row[f"{metric}_std"] = float(np.std(arr)) if len(arr) else None
        rows.append(row)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = group_keys + ["count"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a wide summary table for FollowBench2.0 evaluation results.")
    add_filter_args(parser)
    parser.add_argument("--group-by", nargs="+", default=["scenario_type", "scenario_name", "planner", "num_pedestrians", "follow_position", "desired_distance"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_TABLE_METRICS)
    parser.add_argument("--output", default=os.path.join(DEFAULT_RESULTS_DIR, "tables", "summary_wide.csv"))
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
    export_wide_table(records, args.group_by, args.metrics, args.output)
    print(f"[VIS] records={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
