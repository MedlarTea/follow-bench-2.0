from __future__ import annotations

import json
import math
import os
from typing import Iterable

import numpy as np

from .schema_docs import write_schema_docs
from .schemas import (
    DEFAULT_HUMAN_COLLISION_MARGIN,
    DEFAULT_HUMAN_RADIUS,
    DEFAULT_ROBOT_RADIUS,
    DEFAULT_SIM_DT,
    DEFAULT_TARGET_LOSS_DIST_M,
    DEFAULT_TARGET_LOSS_N_LOST_STEPS,
    DEFAULT_TARGET_LOSS_WARMUP_SEC,
    DEFAULT_TARGET_RADIUS,
    DEFAULT_ZONE_THRESHOLDS,
)

# Termination reasons meaning the target finished its trajectory (end of the fixed horizon).
ROUTE_COMPLETED_TERMINATION_REASONS = {
    "target_round_trip_done",
    "target_completed_route",
    "target_completed_a_to_b",
}
LONG_TERM_LOSS_TERMINATION_REASONS = {"target_too_far"}
COLLISION_TERMINATION_REASONS = {
    "human_collision",
    "pedestrian_collision",
    "target_collision",
    "obstacle_collision",
}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _xy(obj: dict | None) -> np.ndarray | None:
    if not obj:
        return None
    if obj.get("x") is None or obj.get("y") is None:
        return None
    return np.array([float(obj["x"]), float(obj["y"])], dtype=float)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _visible(step: dict) -> bool:
    return bool((step.get("visibility") or {}).get("target_visible", False))


def _active_steps(steps: Iterable[dict]) -> list[dict]:
    return [s for s in steps if bool(s.get("episode_active", False)) and not bool(s.get("paused", False))]


def evaluate_run(run_dir: str) -> dict:
    meta = _load_json(os.path.join(run_dir, "episode_meta.json"))
    all_steps = _load_jsonl(os.path.join(run_dir, "episode_step.jsonl"))
    collision_events = _load_jsonl(os.path.join(run_dir, "collision_events.jsonl"))
    active_collision_events = [e for e in collision_events if e.get("episode_active", True)]
    steps = _active_steps(all_steps)

    dt = float(meta.get("dt", DEFAULT_SIM_DT))
    max_search_steps = int(meta.get("max_search_steps", round(float(meta.get("max_search_time_s", 5.0)) / max(dt, 1e-6))))
    zone_thresholds = list(meta.get("zone_thresholds", DEFAULT_ZONE_THRESHOLDS))
    zone_min = float(zone_thresholds[0])
    zone_max = float(zone_thresholds[1])
    robot_radius = float((meta.get("robot") or {}).get("radius", DEFAULT_ROBOT_RADIUS))
    target_radius = float((meta.get("target") or {}).get("radius", DEFAULT_TARGET_RADIUS))
    human_radius = float((meta.get("human") or {}).get("radius", DEFAULT_HUMAN_RADIUS))
    collision_margin = float((meta.get("collision") or {}).get("human_collision_margin", DEFAULT_HUMAN_COLLISION_MARGIN))
    loss_dist_m, n_lost_steps, loss_warmup_sec = _target_loss_params(meta, dt)

    max_steps = len(steps)
    termination_reason = meta.get("termination_reason")
    target_route_completed = termination_reason in ROUTE_COMPLETED_TERMINATION_REASONS
    if max_steps == 0:
        result = {
            "max_steps": 0,
            "max_search_steps": max_search_steps,
            "obstacle_avoidance_success": True,
            "target_identity_success": termination_reason != "target_identity_lost",
            "target_identity_switch_count": 0,
            "max_identity_mismatch_duration_s": 0.0,
            "identity_failure_track_id": None,
            "target_visibility_ratio": 0.0,
            "search_success": True,
            "search_path_length": 0.0,
            "avg_robot_target_dist_no_radius": 0.0,
            "path_length": 0.0,
            "avg_velocity": 0.0,
            "avg_acceleration": 0.0,
            "avg_jerk": 0.0,
            "time_in_target_personal_zone": 0.0,
            "time_in_human_private_zone": 0.0,
            "time_in_target_search": 0.0,
            "total_time": 0.0,
            "termination_reason": termination_reason,
            "task_success": False,
            "collision_free": len(active_collision_events) == 0 and termination_reason not in COLLISION_TERMINATION_REASONS,
            "long_term_target_loss": termination_reason in LONG_TERM_LOSS_TERMINATION_REASONS,
            "final_robot_target_dist": None,
            "final_within_loss_dist": False,
            "max_consecutive_target_loss_steps": 0,
            "target_loss_steps": 0,
            "target_loss_ratio": 0.0,
            "target_loss_dist_m": loss_dist_m,
            "target_loss_n_lost_steps": n_lost_steps,
            "target_route_completed": bool(target_route_completed),
            "active_total_time": 0.0,
            "collision_count": len(active_collision_events),
            "human_collision_count": sum(1 for e in active_collision_events if e.get("collision_type") == "human"),
            "planner_cost_ms_mean": None,
            "planner_cost_ms_p95": None,
            **_empty_timing_metrics(),
        }
        return result

    robot_xy = [_xy(s.get("robot")) for s in steps]
    target_xy = [_xy(s.get("target")) for s in steps]
    visible = np.array([_visible(s) for s in steps], dtype=bool)

    robot_target_dists = []
    target_surface_dists = []
    human_surface_mins = []
    for step, rxy, txy in zip(steps, robot_xy, target_xy):
        if rxy is not None and txy is not None:
            d = _dist(rxy, txy)
            robot_target_dists.append(d)
            target_surface_dists.append(d - robot_radius - target_radius)
        humans = step.get("humans") or []
        human_surfaces = []
        if rxy is not None:
            for human in humans:
                hxy = _xy(human)
                if hxy is not None:
                    human_surfaces.append(_dist(rxy, hxy) - robot_radius - human_radius)
        human_surface_mins.append(min(human_surfaces) if human_surfaces else float("inf"))

    deltas = []
    velocities = []
    for i in range(1, max_steps):
        if robot_xy[i - 1] is None or robot_xy[i] is None:
            delta_vec = np.zeros(2, dtype=float)
        else:
            delta_vec = robot_xy[i] - robot_xy[i - 1]
        deltas.append(float(np.linalg.norm(delta_vec)))
        velocities.append(delta_vec / max(dt, 1e-6))
    path_length = float(np.sum(deltas)) if deltas else 0.0
    velocities_arr = np.array(velocities, dtype=float).reshape((-1, 2)) if velocities else np.zeros((0, 2), dtype=float)
    speeds = np.linalg.norm(velocities_arr, axis=1) if len(velocities_arr) else np.zeros((0,), dtype=float)
    accelerations = np.diff(velocities_arr, axis=0) / max(dt, 1e-6) if len(velocities_arr) > 1 else np.zeros((0, 2), dtype=float)
    acc_norms = np.linalg.norm(accelerations, axis=1) if len(accelerations) else np.zeros((0,), dtype=float)
    jerks = np.diff(accelerations, axis=0) / max(dt, 1e-6) if len(accelerations) > 1 else np.zeros((0, 2), dtype=float)
    jerk_norms = np.linalg.norm(jerks, axis=1) if len(jerks) else np.zeros((0,), dtype=float)

    search_path_length = 0.0
    for i, delta in enumerate(deltas, start=1):
        if not visible[i]:
            search_path_length += float(delta)

    consecutive = 0
    max_consecutive = 0
    for is_visible in visible:
        if is_visible:
            consecutive = 0
        else:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)

    human_collision_count = sum(1 for e in active_collision_events if e.get("collision_type") == "human")
    geometry_collision = any(d <= collision_margin for d in human_surface_mins)
    geometry_collision = geometry_collision or any(d <= collision_margin for d in target_surface_dists)
    step_collision = any(bool(s.get("collision", False)) for s in steps)
    obstacle_avoidance_success = (human_collision_count == 0) and (not geometry_collision) and (not step_collision)
    # Any collision (pedestrian, target, static obstacle; sensor, map or geometry) fails the trial.
    collision_free = (
        bool(obstacle_avoidance_success)
        and len(active_collision_events) == 0
        and termination_reason not in COLLISION_TERMINATION_REASONS
    )

    loss_metrics = _target_loss_metrics(
        robot_xy=robot_xy,
        target_xy=target_xy,
        dt=dt,
        loss_dist_m=loss_dist_m,
        n_lost_steps=n_lost_steps,
        warmup_sec=loss_warmup_sec,
    )
    long_term_target_loss = bool(
        loss_metrics["max_consecutive_target_loss_steps"] > n_lost_steps
        or termination_reason in LONG_TERM_LOSS_TERMINATION_REASONS
    )
    final_dist = loss_metrics["final_robot_target_dist"]
    final_within_loss_dist = final_dist is not None and final_dist <= loss_dist_m
    task_success = collision_free and (not long_term_target_loss) and final_within_loss_dist
    identity_metrics = _target_identity_metrics(steps, termination_reason)

    target_surface_arr = np.array(target_surface_dists, dtype=float)
    human_surface_arr = np.array(human_surface_mins, dtype=float)
    result = {
        "max_steps": max_steps,
        "max_search_steps": max_search_steps,
        "obstacle_avoidance_success": bool(obstacle_avoidance_success),
        **identity_metrics,
        "target_visibility_ratio": float(np.sum(visible) / max(max_steps, 1)),
        "search_success": bool(max_consecutive <= max_search_steps),
        "search_path_length": float(search_path_length),
        "avg_robot_target_dist_no_radius": float(np.mean(robot_target_dists)) if robot_target_dists else 0.0,
        "path_length": path_length,
        "avg_velocity": float(np.mean(speeds)) if len(speeds) else 0.0,
        "avg_acceleration": float(np.mean(acc_norms)) if len(acc_norms) else 0.0,
        "avg_jerk": float(np.mean(jerk_norms)) if len(jerk_norms) else 0.0,
        "time_in_target_personal_zone": float(np.sum((target_surface_arr > zone_min) & (target_surface_arr < zone_max)) * dt) if len(target_surface_arr) else 0.0,
        "time_in_human_private_zone": float(np.sum(human_surface_arr < zone_min) * dt) if len(human_surface_arr) else 0.0,
        "time_in_target_search": float(np.sum(~visible) * dt),
        "total_time": float(max_steps * dt),
        "termination_reason": termination_reason,
        "task_success": bool(task_success),
        "collision_free": bool(collision_free),
        "long_term_target_loss": bool(long_term_target_loss),
        "final_robot_target_dist": final_dist,
        "final_within_loss_dist": bool(final_within_loss_dist),
        "max_consecutive_target_loss_steps": int(loss_metrics["max_consecutive_target_loss_steps"]),
        "target_loss_steps": int(loss_metrics["target_loss_steps"]),
        "target_loss_ratio": float(loss_metrics["target_loss_steps"] / max(max_steps, 1)),
        "target_loss_dist_m": loss_dist_m,
        "target_loss_n_lost_steps": n_lost_steps,
        "target_route_completed": bool(target_route_completed),
        "active_total_time": float(max_steps * dt),
        "collision_count": int(len(active_collision_events)),
        "human_collision_count": int(human_collision_count),
        "geometry_collision_detected": bool(geometry_collision),
        "min_human_surface_dist": float(np.min(human_surface_arr)) if len(human_surface_arr) else None,
        "planner_cost_ms_mean": _mean_field(steps, "planner_cost_ms"),
        "planner_cost_ms_p95": _percentile_field(steps, "planner_cost_ms", 95),
    }
    result.update(_timing_metrics(steps))
    return result


def _target_loss_params(meta: dict, dt: float) -> tuple[float, int, float]:
    cfg = meta.get("target_loss") or {}
    loss_dist_m = float(cfg.get("dist_m", DEFAULT_TARGET_LOSS_DIST_M))
    n_lost_steps = int(cfg.get("n_lost_steps", DEFAULT_TARGET_LOSS_N_LOST_STEPS))
    if n_lost_steps <= 0:
        fail_sec = cfg.get("fail_sec")
        if fail_sec is not None:
            n_lost_steps = max(int(math.ceil(float(fail_sec) / max(dt, 1e-6))), 0)
        else:
            n_lost_steps = DEFAULT_TARGET_LOSS_N_LOST_STEPS
    warmup_sec = float(cfg.get("warmup_sec", DEFAULT_TARGET_LOSS_WARMUP_SEC))
    return loss_dist_m, n_lost_steps, warmup_sec


def _target_loss_metrics(
    *,
    robot_xy: list[np.ndarray | None],
    target_xy: list[np.ndarray | None],
    dt: float,
    loss_dist_m: float,
    n_lost_steps: int,
    warmup_sec: float,
) -> dict:
    """Target loss: robot-target center distance > loss_dist_m.

    Long-term loss is a run of more than n_lost_steps consecutive lost steps.
    Steps with missing poses reset the run, matching the runtime monitor.
    """
    consecutive = 0
    max_consecutive = 0
    lost_steps = 0
    final_dist = None
    for i, (rxy, txy) in enumerate(zip(robot_xy, target_xy)):
        if rxy is None or txy is None:
            consecutive = 0
            continue
        d = _dist(rxy, txy)
        final_dist = d
        if i * dt < warmup_sec or d <= loss_dist_m:
            consecutive = 0
            continue
        lost_steps += 1
        consecutive += 1
        max_consecutive = max(max_consecutive, consecutive)
    return {
        "max_consecutive_target_loss_steps": max_consecutive,
        "target_loss_steps": lost_steps,
        "final_robot_target_dist": None if final_dist is None else float(final_dist),
    }


def _target_identity_metrics(steps: list[dict], termination_reason: str | None) -> dict:
    payloads = [s.get("target_identity") for s in steps if isinstance(s.get("target_identity"), dict)]
    if not payloads:
        return {
            "target_identity_success": termination_reason != "target_identity_lost",
            "target_identity_switch_count": 0,
            "max_identity_mismatch_duration_s": 0.0,
            "identity_failure_track_id": None,
        }
    max_mismatch = 0.0
    switch_count = 0
    failure_track_id = None
    for payload in payloads:
        try:
            max_mismatch = max(max_mismatch, float(payload.get("max_mismatch_duration_s") or 0.0))
        except (TypeError, ValueError):
            pass
        try:
            switch_count = max(switch_count, int(payload.get("switch_count") or 0))
        except (TypeError, ValueError):
            pass
        if payload.get("termination_triggered"):
            failure_track_id = payload.get("failure_track_id")
    return {
        "target_identity_success": termination_reason != "target_identity_lost",
        "target_identity_switch_count": int(switch_count),
        "max_identity_mismatch_duration_s": float(max_mismatch),
        "identity_failure_track_id": failure_track_id,
    }


def _mean_field(records: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in records if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _percentile_field(records: list[dict], key: str, pct: float) -> float | None:
    vals = [float(r[key]) for r in records if r.get(key) is not None]
    return float(np.percentile(vals, pct)) if vals else None


def write_eval_result(run_dir: str) -> dict:
    result = evaluate_run(run_dir)
    out_path = os.path.join(run_dir, "eval_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_schema_docs(run_dir)
    return result


def _empty_timing_metrics() -> dict:
    keys = [
        "policy_total_ms",
        "planner_core_ms",
        "perception_total_ms",
        "wrapper_overhead_ms",
        "perception_detection_ms",
        "perception_tracking_ms",
        "perception_reid_ms",
        "perception_mapping_ms",
        "perception_fsm_ms",
    ]
    out = {}
    for key in keys:
        out[f"{key}_mean"] = None
        out[f"{key}_p95"] = None
    return out


def _timing_metrics(steps: list[dict]) -> dict:
    paths = {
        "policy_total_ms": ("policy_total_ms",),
        "planner_core_ms": ("planner_core_ms",),
        "perception_total_ms": ("perception_total_ms",),
        "wrapper_overhead_ms": ("wrapper_overhead_ms",),
        "perception_detection_ms": ("perception", "detection_ms"),
        "perception_tracking_ms": ("perception", "tracking_ms"),
        "perception_reid_ms": ("perception", "reid_ms"),
        "perception_mapping_ms": ("perception", "mapping_ms"),
        "perception_fsm_ms": ("perception", "fsm_ms"),
    }
    out = {}
    for name, path in paths.items():
        vals = [_timing_value(step, path) for step in steps]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        out[f"{name}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{name}_p95"] = float(np.percentile(vals, 95)) if vals else None
    return out


def _timing_value(step: dict, path: tuple[str, ...]) -> float | None:
    timing = step.get("timing") or {}
    current = timing
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if current is None and path == ("policy_total_ms",):
        current = step.get("planner_cost_ms")
    if current is None:
        return None
    try:
        return float(current)
    except (TypeError, ValueError):
        return None
