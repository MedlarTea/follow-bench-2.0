from __future__ import annotations

import argparse
import json
import os
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "followbench_mpl"))

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def xy(record: dict | None):
    if not record:
        return None
    if record.get("x") is None or record.get("y") is None:
        return None
    return float(record["x"]), float(record["y"])


def xy_series(steps: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    indices = []
    points = []
    for idx, step in enumerate(steps):
        point = xy(step.get(key))
        if point is not None:
            indices.append(idx)
            points.append(point)
    return np.array(indices, dtype=int), np.array(points, dtype=float)


def plot_episode(run_dir: str, output_dir: str | None = None) -> list[str]:
    steps = load_jsonl(os.path.join(run_dir, "episode_step.jsonl"))
    collisions = load_jsonl(os.path.join(run_dir, "collision_events.jsonl")) if os.path.exists(os.path.join(run_dir, "collision_events.jsonl")) else []
    active = [s for s in steps if s.get("episode_active") and not s.get("paused")]
    if not active:
        raise SystemExit("[VIS] run has no active evaluation steps")
    out_dir = output_dir or os.path.join(run_dir, "plots")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    outputs = []
    outputs.append(_plot_trajectory(active, os.path.join(out_dir, "trajectory.png")))
    outputs.append(_plot_distance_visibility(active, os.path.join(out_dir, "distance_visibility.png")))
    outputs.append(_plot_motion(active, os.path.join(out_dir, "motion_profile.png")))
    outputs.append(_plot_clearance(active, collisions, os.path.join(out_dir, "clearance_collision.png")))
    outputs.append(_plot_planner_cost(active, os.path.join(out_dir, "planner_cost.png")))
    if any(_timing_value(s, ("perception_total_ms",)) is not None for s in active):
        outputs.append(_plot_perception_timing(active, os.path.join(out_dir, "perception_timing.png")))
    return outputs


def _plot_trajectory(steps: list[dict], out_path: str) -> str:
    robot_idx, r = xy_series(steps, "robot")
    _, t = xy_series(steps, "target")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    if len(r):
        vis = np.array([bool((steps[i].get("visibility") or {}).get("target_visible", False)) for i in robot_idx])
        ax.plot(r[:, 0], r[:, 1], color="#4c78a8", label="robot")
        if len(vis) == len(r):
            ax.scatter(r[~vis, 0], r[~vis, 1], s=12, color="#e45756", label="robot target invisible")
    if len(t):
        ax.plot(t[:, 0], t[:, 1], color="#54a24b", label="target")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Trajectory")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_distance_visibility(steps: list[dict], out_path: str) -> str:
    time = np.array([float(s["time_s"]) for s in steps])
    center_dist = []
    target_surface = []
    visible = []
    for s in steps:
        rxy = xy(s.get("robot"))
        txy = xy(s.get("target"))
        if rxy is None or txy is None:
            center_dist.append(np.nan)
        else:
            center_dist.append(float(np.linalg.norm(np.array(rxy) - np.array(txy))))
        target_surface.append((s.get("clearance") or {}).get("robot_target_surface_dist", np.nan))
        visible.append(bool((s.get("visibility") or {}).get("target_visible", False)))
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(time, center_dist, label="robot-target center dist")
    ax.plot(time, target_surface, label="robot-target surface dist")
    _shade_invisible(ax, time, visible)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")
    ax.set_title("Distance and Visibility")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_motion(steps: list[dict], out_path: str) -> str:
    time = np.array([float(s["time_s"]) for s in steps])
    robot_xy = np.array([xy(s.get("robot")) or (np.nan, np.nan) for s in steps], dtype=float)
    dt = np.diff(time)
    delta = np.diff(robot_xy, axis=0)
    speed = np.linalg.norm(delta, axis=1) / np.maximum(dt, 1e-6)
    speed[~np.isfinite(speed)] = np.nan
    acc = np.diff(speed) / np.maximum(dt[1:], 1e-6) if len(speed) > 1 else np.array([])
    v_cmd = np.array([(s.get("command") or {}).get("v_mps", np.nan) for s in steps])
    w_cmd = np.array([(s.get("command") or {}).get("w_radps", np.nan) for s in steps])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(time[1:], speed, label="speed")
    axes[0].plot(time, v_cmd, label="v_cmd", alpha=0.8)
    axes[1].plot(time[2:], acc, label="acceleration")
    axes[2].plot(time, w_cmd, label="w_cmd", color="#f58518")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_clearance(steps: list[dict], collisions: list[dict], out_path: str) -> str:
    time = np.array([float(s["time_s"]) for s in steps])
    min_human = np.array([(s.get("clearance") or {}).get("min_human_surface_dist", np.nan) for s in steps], dtype=float)
    target = np.array([(s.get("clearance") or {}).get("robot_target_surface_dist", np.nan) for s in steps], dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    ax.plot(time, min_human, label="min human surface dist")
    ax.plot(time, target, label="target surface dist")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="collision boundary")
    for event in collisions:
        if event.get("time_s") is not None:
            ax.axvline(float(event["time_s"]), color="#e45756", alpha=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("clearance [m]")
    ax.set_title("Clearance and Collisions")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_planner_cost(steps: list[dict], out_path: str) -> str:
    time = np.array([float(s["time_s"]) for s in steps])
    policy = _timing_series(steps, ("policy_total_ms",), fallback_key="planner_cost_ms")
    planner = _timing_series(steps, ("planner_core_ms",))
    perception = _timing_series(steps, ("perception_total_ms",))
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(time, policy, label="policy total")
    if np.isfinite(planner).any():
        ax.plot(time, planner, label="planner core")
    if np.isfinite(perception).any():
        ax.plot(time, perception, label="perception total")
    finite = policy[np.isfinite(policy)]
    if len(finite):
        ax.axhline(float(np.mean(finite)), color="#54a24b", linestyle="--", label="mean")
        ax.axhline(float(np.percentile(finite, 95)), color="#f58518", linestyle="--", label="p95")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ms")
    ax.set_title("Timing Profile")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_perception_timing(steps: list[dict], out_path: str) -> str:
    time = np.array([float(s["time_s"]) for s in steps])
    series = {
        "detection": _timing_series(steps, ("perception", "detection_ms")),
        "tracking": _timing_series(steps, ("perception", "tracking_ms")),
        "reid": _timing_series(steps, ("perception", "reid_ms")),
        "mapping": _timing_series(steps, ("perception", "mapping_ms")),
        "fsm": _timing_series(steps, ("perception", "fsm_ms")),
    }
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for label, vals in series.items():
        if np.isfinite(vals).any():
            ax.plot(time, vals, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("ms")
    ax.set_title("Perception Timing")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _timing_series(steps: list[dict], path: tuple[str, ...], fallback_key: str | None = None) -> np.ndarray:
    vals = []
    for step in steps:
        value = _timing_value(step, path)
        if value is None and fallback_key is not None:
            value = step.get(fallback_key)
        vals.append(np.nan if value is None else float(value))
    return np.array(vals, dtype=float)


def _timing_value(step: dict, path: tuple[str, ...]):
    current = step.get("timing") or {}
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _shade_invisible(ax, time: np.ndarray, visible: list[bool]) -> None:
    if len(time) == 0:
        return
    start = None
    for i, is_visible in enumerate(visible):
        if not is_visible and start is None:
            start = time[i]
        if (is_visible or i == len(visible) - 1) and start is not None:
            end = time[i] if is_visible else time[i]
            ax.axvspan(start, end, color="#e45756", alpha=0.12)
            start = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnostic figures for one FollowBench2.0 episode.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    outputs = plot_episode(args.run_dir, args.output_dir)
    for path in outputs:
        print(f"[VIS] wrote {path}")


if __name__ == "__main__":
    main()
