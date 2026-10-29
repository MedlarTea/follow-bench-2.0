#!/usr/bin/env python3
"""
Analyze robot-target distance from a npc_gt_stream.jsonl file.
Usage: python tools/analysis/analyze_distance.py <path/to/npc_gt_stream.jsonl>
       python tools/analysis/analyze_distance.py --scene corner
           (latest run under $FB_OUTPUT_ROOT/runs/<scene>/, default scene: plaza)
"""
import argparse
import json
import math
import sys
import os
import glob


def load_latest_run(runs_dir: str) -> str:
    pattern = os.path.join(runs_dir, "*", "npc_gt_stream.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No npc_gt_stream.jsonl found under {runs_dir}")
    return files[-1]


def analyze(path: str) -> dict:
    dists = []
    times = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            robot = rec.get("robot")
            if robot is None:
                continue
            target_id = rec["target_track_id"]
            target = next((n for n in rec["npcs"] if n["track_id"] == target_id), None)
            if target is None:
                continue
            d = math.hypot(robot["x"] - target["x"], robot["y"] - target["y"])
            dists.append(d)
            times.append(rec["time_s"])

    if not dists:
        return {"error": "no data with robot field"}

    over5 = sum(1 for d in dists if d > 5.0)
    return {
        "ticks": len(dists),
        "duration_s": round(times[-1] - times[0], 1) if len(times) > 1 else 0,
        "mean_dist": round(sum(dists) / len(dists), 3),
        "max_dist": round(max(dists), 3),
        "min_dist": round(min(dists), 3),
        "pct_over5m": round(over5 / len(dists) * 100, 1),
        "ticks_over5m": over5,
        "pass": max(dists) <= 5.0,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robot-target distance stats from npc_gt_stream.jsonl.")
    ap.add_argument("path", nargs="?", help="npc_gt_stream.jsonl (default: latest run of --scene)")
    ap.add_argument("--scene", default="plaza", help="scene whose latest run is used when no path is given")
    args = ap.parse_args()
    if args.path:
        path = args.path
    else:
        from followbench.paths import scene_runs_dir
        path = load_latest_run(str(scene_runs_dir(args.scene)))

    print(f"Analyzing: {path}")
    result = analyze(path)
    for k, v in result.items():
        print(f"  {k}: {v}")
