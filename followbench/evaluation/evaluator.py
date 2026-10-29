#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


from followbench.evaluation.metrics import write_eval_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one FollowBench 2.0 run directory.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = write_eval_result(args.run_dir)
    print(f"[EVAL] wrote {os.path.join(args.run_dir, 'eval_result.json')}")
    for key in (
        "obstacle_avoidance_success",
        "search_success",
        "target_visibility_ratio",
        "path_length",
        "avg_velocity",
        "avg_acceleration",
        "avg_jerk",
        "policy_total_ms_mean",
        "planner_core_ms_mean",
        "perception_total_ms_mean",
        "termination_reason",
    ):
        print(f"[EVAL] {key}: {result.get(key)}")


if __name__ == "__main__":
    main()
