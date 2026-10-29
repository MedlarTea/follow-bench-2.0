from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Iterable, Optional


from followbench.visualization.constants import DEFAULT_LOG_ROOT


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_run_dirs(log_root: str = DEFAULT_LOG_ROOT) -> Iterable[str]:
    pattern = os.path.join(os.path.abspath(log_root), "**", "eval_result.json")
    for eval_path in glob.iglob(pattern, recursive=True):
        run_dir = os.path.dirname(eval_path)
        if os.path.exists(os.path.join(run_dir, "episode_meta.json")):
            yield run_dir


def load_records(
    log_root: str = DEFAULT_LOG_ROOT,
    scenario: Optional[str] = None,
    scenario_name: Optional[str] = None,
    planners: Optional[list[str]] = None,
    humans: Optional[int] = None,
    distance: Optional[float] = None,
    follow_position: Optional[str] = None,
) -> list[dict[str, Any]]:
    records = []
    planner_set = set(planners or [])
    for run_dir in iter_run_dirs(log_root):
        meta = load_json(os.path.join(run_dir, "episode_meta.json"))
        result = load_json(os.path.join(run_dir, "eval_result.json"))
        record = {"run_dir": run_dir}
        record.update(meta)
        record.update(result)

        if scenario is not None and record.get("scenario_type") != scenario:
            continue
        if scenario_name is not None and record.get("scenario_name") != scenario_name:
            continue
        if planner_set and record.get("planner") not in planner_set:
            continue
        if humans is not None and int(record.get("num_pedestrians", -1)) != int(humans):
            continue
        if distance is not None and abs(float(record.get("desired_distance", -999.0)) - float(distance)) > 1e-6:
            continue
        if follow_position is not None and record.get("follow_position") != follow_position:
            continue
        records.append(record)
    return records


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--scenario", default=None, help="scenario_type filter, e.g. corner/park/passage")
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--planners", nargs="*", default=None)
    parser.add_argument("--humans", type=int, default=None, help="Non-target pedestrian count.")
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--follow-position", default=None)
