from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Any, Iterable

import numpy as np


from followbench.visualization.constants import DEFAULT_PLOT_METRICS, DEFAULT_RESULTS_DIR
from followbench.visualization.data_loader import add_filter_args, load_records


def display_metric(record: dict[str, Any], metric: str) -> float | None:
    total_time = float(record.get("total_time") or 0.0)
    if metric == "ASR":
        return float(bool(record.get("obstacle_avoidance_success"))) * 100.0
    if metric == "TVR":
        return float(record.get("target_visibility_ratio", 0.0)) * 100.0
    if metric == "SR":
        return float(bool(record.get("task_success"))) * 100.0
    if metric == "TinTPerson":
        return _ratio_time(record.get("time_in_target_personal_zone"), total_time)
    if metric == "TinPrivate":
        return _ratio_time(record.get("time_in_human_private_zone"), total_time)
    if metric == "PL":
        return _num(record.get("path_length"))
    if metric == "AvgVel":
        return _num(record.get("avg_velocity"))
    if metric == "AvgAcc":
        return _num(record.get("avg_acceleration"))
    if metric == "Jerk":
        return _num(record.get("avg_jerk"))
    if metric == "CostMean":
        return _num(record.get("planner_cost_ms_mean"))
    if metric == "CostP95":
        return _num(record.get("planner_cost_ms_p95"))
    if metric == "PolicyMean":
        return _num(_coalesce(record.get("policy_total_ms_mean"), record.get("planner_cost_ms_mean")))
    if metric == "PolicyP95":
        return _num(_coalesce(record.get("policy_total_ms_p95"), record.get("planner_cost_ms_p95")))
    if metric == "PlanMean":
        return _num(record.get("planner_core_ms_mean"))
    if metric == "PlanP95":
        return _num(record.get("planner_core_ms_p95"))
    if metric == "PercMean":
        return _num(record.get("perception_total_ms_mean"))
    if metric == "PercP95":
        return _num(record.get("perception_total_ms_p95"))
    if metric == "DetMean":
        return _num(record.get("perception_detection_ms_mean"))
    if metric == "TrackMean":
        return _num(record.get("perception_tracking_ms_mean"))
    if metric == "ReIDMean":
        return _num(record.get("perception_reid_ms_mean"))
    if metric == "MapMean":
        return _num(record.get("perception_mapping_ms_mean"))
    if metric == "FSMMean":
        return _num(record.get("perception_fsm_ms_mean"))
    return _num(record.get(metric))


def aggregate(records: Iterable[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(k) for k in group_keys)].append(record)

    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: tuple("" if v is None else str(v) for v in kv[0])):
        base = {k: v for k, v in zip(group_keys, key)}
        for metric in metrics:
            vals = [display_metric(item, metric) for item in items]
            arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
            row = dict(base)
            row.update(
                {
                    "metric": metric,
                    "count": int(len(arr)),
                    "mean": float(np.mean(arr)) if len(arr) else None,
                    "std": float(np.std(arr)) if len(arr) else None,
                    "ci95": float(1.96 * np.std(arr) / math.sqrt(len(arr))) if len(arr) else None,
                }
            )
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    preferred = ["scenario_type", "scenario_name", "planner", "num_pedestrians", "follow_position", "desired_distance", "metric", "count", "mean", "std", "ci95"]
    ordered = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _ratio_time(value, total_time: float) -> float | None:
    if value is None or total_time <= 0.0:
        return None
    return float(value) / total_time * 100.0


def _num(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FollowBench2.0 eval_result.json files.")
    add_filter_args(parser)
    parser.add_argument("--group-by", nargs="+", default=["scenario_type", "planner", "num_pedestrians", "follow_position", "desired_distance"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_PLOT_METRICS)
    parser.add_argument("--output", default=os.path.join(DEFAULT_RESULTS_DIR, "tables", "aggregate.csv"))
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
    rows = aggregate(records, args.group_by, args.metrics)
    write_csv(rows, args.output)
    print(f"[VIS] records={len(records)} rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
