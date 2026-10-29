from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Iterable, Optional

import numpy as np

from .metrics import write_eval_result
from .schemas import (
    DEFAULT_HUMAN_COLLISION_MARGIN,
    DEFAULT_HUMAN_RADIUS,
    DEFAULT_LOG_ROOT,
    DEFAULT_MAX_SEARCH_TIME_S,
    DEFAULT_ROBOT_LENGTH,
    DEFAULT_ROBOT_RADIUS,
    DEFAULT_ROBOT_WIDTH,
    DEFAULT_SIM_DT,
    DEFAULT_TARGET_LOSS_DIST_M,
    DEFAULT_TARGET_LOSS_N_LOST_STEPS,
    DEFAULT_TARGET_LOSS_WARMUP_SEC,
    DEFAULT_TARGET_RADIUS,
    DEFAULT_ZONE_THRESHOLDS,
    actor_footprint,
    format_distance,
    next_trial_id,
    num_pedestrians_from_total,
    planner_mode_from_debug,
    robot_to_dict,
    scenario_name_from_args,
    scenario_variant_from_args,
    state_to_dict,
    trial_dir_name,
)


class EvaluationLogger:
    def __init__(
        self,
        log_root: str,
        scenario_type: str,
        planner: str,
        num_walkers_total: int,
        follow_position: str,
        desired_distance: float,
        trial_id: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        self.log_root = os.path.abspath(log_root or DEFAULT_LOG_ROOT)
        self.scenario_type = str(scenario_type or "unknown")
        self.planner = str(planner or "unknown")
        self.num_walkers_total = int(num_walkers_total)
        self.num_pedestrians = num_pedestrians_from_total(self.num_walkers_total)
        self.follow_position = str(follow_position or "back")
        self.desired_distance = float(desired_distance)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_dir = os.path.join(
            self.log_root,
            self.scenario_type,
            self.planner,
            f"H{self.num_pedestrians}_{self.follow_position}_D{format_distance(self.desired_distance)}",
        )
        resolved_trial_id = int(trial_id) if trial_id is not None else next_trial_id(self.config_dir)
        self.trial_id = resolved_trial_id
        self.run_dir = os.path.join(self.config_dir, trial_dir_name(self.trial_id, self.timestamp))
        os.makedirs(self.run_dir, exist_ok=True)

        self.meta_path = os.path.join(self.run_dir, "episode_meta.json")
        self.step_path = os.path.join(self.run_dir, "episode_step.jsonl")
        self.collision_path = os.path.join(self.run_dir, "collision_events.jsonl")
        self.summary_path = os.path.join(self.run_dir, "eval_summary.txt")
        self._step_f = open(self.step_path, "w", encoding="utf-8")
        self._collision_f = open(self.collision_path, "w", encoding="utf-8")
        self._meta: dict[str, Any] = {}
        self._closed = False

    @classmethod
    def from_args(cls, args, scenario_type: str) -> "EvaluationLogger":
        return cls(
            log_root=getattr(args, "eval_log_root", DEFAULT_LOG_ROOT),
            scenario_type=getattr(args, "eval_scenario_type", None) or scenario_type,
            planner=getattr(args, "planner", "unknown"),
            num_walkers_total=int(getattr(args, "num_walkers", 0)),
            follow_position=getattr(args, "follow_position", "back"),
            desired_distance=float(getattr(args, "desired_distance", 1.5)),
            trial_id=getattr(args, "eval_trial_id", None),
        )

    def write_meta(self, args, robot_actor=None, target_actor=None, extras: Optional[dict] = None) -> None:
        dt = float(getattr(args, "dt", DEFAULT_SIM_DT))
        max_search_time_s = float(getattr(args, "max_search_time_s", DEFAULT_MAX_SEARCH_TIME_S))
        robot_meta = actor_footprint(
            robot_actor,
            fallback_radius=DEFAULT_ROBOT_RADIUS,
            fallback_length=DEFAULT_ROBOT_LENGTH,
            fallback_width=DEFAULT_ROBOT_WIDTH,
        )
        # Target collision radius is fixed (same as other humans): the walker
        # bbox varies with animation pose at spawn time, which made the
        # collision metric non-deterministic across same-seed runs. The bbox
        # footprint is kept for reference only.
        target_bbox = actor_footprint(target_actor, fallback_radius=DEFAULT_TARGET_RADIUS)
        target_meta = {
            "radius": DEFAULT_TARGET_RADIUS,
            "bbox_radius": target_bbox.get("radius"),
            "length": target_bbox.get("length"),
            "width": target_bbox.get("width"),
        }
        human_meta = {"radius": DEFAULT_HUMAN_RADIUS}
        self._meta = {
            "benchmark_version": "followbench2.0",
            "scenario_type": self.scenario_type,
            "scenario_name": scenario_name_from_args(args, self.scenario_type),
            "scenario_variant": scenario_variant_from_args(args, self.scenario_type),
            "map_name": getattr(args, "map_name", None),
            "planner": self.planner,
            "use_perception": bool(getattr(args, "use_perception", False)),
            "follow_position": self.follow_position,
            "desired_distance": self.desired_distance,
            "num_walkers_total": self.num_walkers_total,
            "num_pedestrians": self.num_pedestrians,
            "target_track_id": getattr(args, "target_track_id", None),
            "trial_id": self.trial_id,
            "timestamp": self.timestamp,
            "dt": dt,
            "duration_sec": float(getattr(args, "duration_sec", 0.0)),
            "max_search_time_s": max_search_time_s,
            "max_search_steps": int(round(max_search_time_s / max(dt, 1e-6))),
            "zone_thresholds": list(DEFAULT_ZONE_THRESHOLDS),
            "robot": robot_meta,
            "target": target_meta,
            "human": human_meta,
            "collision": {
                "terminate_on_human_collision": bool(getattr(args, "terminate_on_human_collision", True)),
                "human_collision_margin": float(getattr(args, "human_collision_margin", DEFAULT_HUMAN_COLLISION_MARGIN)),
                "use_collision_sensor": not bool(getattr(args, "disable_collision_sensor", False)),
                "use_geometry_fallback": True,
            },
            "target_loss": {
                "dist_m": float(getattr(args, "robot_target_far_fail_dist_m", DEFAULT_TARGET_LOSS_DIST_M)),
                "n_lost_steps": int(getattr(args, "robot_target_far_fail_steps", DEFAULT_TARGET_LOSS_N_LOST_STEPS)),
                "fail_sec": getattr(args, "robot_target_far_fail_sec", None),
                "warmup_sec": float(getattr(args, "robot_target_far_warmup_sec", DEFAULT_TARGET_LOSS_WARMUP_SEC)),
            },
            "termination_reason": None,
            "episode_success": None,
            "failure_reason": None,
            "termination": {
                "target_lost_fail_sec": getattr(args, "target_lost_fail_sec", None),
                "search_recovery_timeout_sec": getattr(args, "search_recovery_timeout_sec", None),
                "target_lost_warmup_sec": getattr(args, "target_lost_warmup_sec", None),
                "terminate_on_target_identity_loss": bool(getattr(args, "terminate_on_target_identity_loss", True)),
                "target_identity_associate_gate_m": getattr(args, "target_identity_associate_gate_m", None),
                "target_identity_wrong_dist_m": getattr(args, "target_identity_wrong_dist_m", None),
                "target_identity_startup_grace_sec": getattr(args, "target_identity_startup_grace_sec", None),
                "target_identity_confirm_sec": getattr(args, "target_identity_confirm_sec", None),
                "target_identity_fail_sec": getattr(args, "target_identity_fail_sec", None),
                "robot_stuck_window_sec": getattr(args, "robot_stuck_window_sec", None),
                "robot_stuck_grace_sec": getattr(args, "robot_stuck_grace_sec", None),
                "robot_stuck_min_progress_m": getattr(args, "robot_stuck_min_progress_m", None),
                "robot_stuck_blocked_human_surface_m": getattr(args, "robot_stuck_blocked_human_surface_m", None),
                "robot_stuck_blocked_target_surface_m": getattr(args, "robot_stuck_blocked_target_surface_m", None),
                "robot_stuck_target_wait_speed_mps": getattr(args, "robot_stuck_target_wait_speed_mps", None),
                "robot_stuck_target_wait_progress_m": getattr(args, "robot_stuck_target_wait_progress_m", None),
                "robot_stuck_target_escape_distance_m": getattr(args, "robot_stuck_target_escape_distance_m", None),
                "terminate_on_robot_stuck": bool(getattr(args, "terminate_on_robot_stuck", False)),
            },
            "scenario_extras": dict(extras) if extras else {},
        }
        self._write_meta()

    def update_extras(self, extras: dict) -> None:
        if not self._meta:
            return
        current = self._meta.setdefault("scenario_extras", {})
        current.update(extras)
        self._write_meta()

    def _write_meta(self) -> None:
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    def write_step(
        self,
        *,
        tick: int,
        time_s: float,
        episode_active: bool,
        paused: bool,
        target_track_id: str,
        robot_state,
        target_state,
        npc_states: Iterable,
        visible: bool,
        pixel_count: int,
        visibility_threshold: int,
        command: Optional[dict] = None,
        planner_cost_ms: Optional[float] = None,
        timing: Optional[dict] = None,
        debug_info: Optional[dict] = None,
        collision: bool = False,
        collision_source: Optional[str] = None,
        termination_triggered: bool = False,
        termination_reason: Optional[str] = None,
        clearance: Optional[dict] = None,
        target_identity: Optional[dict] = None,
    ) -> None:
        humans = [state_to_dict(s) for s in npc_states or [] if getattr(s, "track_id", None) != target_track_id]
        step_timing = timing or build_step_timing(policy_total_ms=planner_cost_ms, debug_info=debug_info)
        policy_total_ms = step_timing.get("policy_total_ms")
        record = {
            "tick": int(tick),
            "time_s": float(time_s),
            "episode_active": bool(episode_active),
            "paused": bool(paused),
            "planner_mode": planner_mode_from_debug(debug_info),
            "target_track_id": str(target_track_id),
            "robot": robot_to_dict(robot_state),
            "target": state_to_dict(target_state),
            "humans": [h for h in humans if h is not None],
            "visibility": {
                "target_visible": bool(visible),
                "target_pixel_count": int(pixel_count),
                "threshold": int(visibility_threshold),
            },
            "command": command,
            "planner_cost_ms": None if policy_total_ms is None else float(policy_total_ms),
            "timing": step_timing,
            "perception": build_step_perception_summary(debug_info),
            "planner": build_step_planner_summary(debug_info),
            "clearance": clearance,
            "target_identity": target_identity,
            "collision": bool(collision),
            "collision_source": collision_source,
            "termination_triggered": bool(termination_triggered),
            "termination_reason": termination_reason,
        }
        self._step_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_collision_event(self, event: dict) -> None:
        self._collision_f.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._collision_f.flush()

    def write_collision_events(self, events: Iterable[dict]) -> None:
        for event in events:
            self.write_collision_event(event)

    def finalize(
        self,
        termination_reason: Optional[str] = None,
        episode_success: Optional[bool] = None,
    ) -> Optional[dict]:
        if self._closed:
            return None
        if not self._meta:
            self.close_files()
            return None
        if termination_reason is not None:
            self._meta["termination_reason"] = termination_reason
        if episode_success is not None:
            self._meta["episode_success"] = bool(episode_success)
            self._meta["failure_reason"] = None if episode_success else termination_reason
        if termination_reason is not None or episode_success is not None:
            self._write_meta()
        self.close_files()
        result = write_eval_result(self.run_dir)
        self._write_summary(result)
        return result

    def close_files(self) -> None:
        if self._closed:
            return
        self._step_f.close()
        self._collision_f.close()
        self._closed = True

    def _write_summary(self, result: dict) -> None:
        lines = [
            f"run_dir: {self.run_dir}",
            f"obstacle_avoidance_success: {result.get('obstacle_avoidance_success')}",
            f"search_success: {result.get('search_success')}",
            f"target_visibility_ratio: {result.get('target_visibility_ratio')}",
            f"path_length: {result.get('path_length')}",
            f"avg_velocity: {result.get('avg_velocity')}",
            f"avg_acceleration: {result.get('avg_acceleration')}",
            f"avg_jerk: {result.get('avg_jerk')}",
            f"policy_total_ms_mean: {result.get('policy_total_ms_mean')}",
            f"policy_total_ms_p95: {result.get('policy_total_ms_p95')}",
            f"planner_core_ms_mean: {result.get('planner_core_ms_mean')}",
            f"planner_core_ms_p95: {result.get('planner_core_ms_p95')}",
            f"perception_total_ms_mean: {result.get('perception_total_ms_mean')}",
            f"perception_total_ms_p95: {result.get('perception_total_ms_p95')}",
            f"termination_reason: {result.get('termination_reason')}",
            f"collision_free: {result.get('collision_free')}",
            f"long_term_target_loss: {result.get('long_term_target_loss')}",
            f"max_consecutive_target_loss_steps: {result.get('max_consecutive_target_loss_steps')} (N_lost={result.get('target_loss_n_lost_steps')})",
            f"final_robot_target_dist: {result.get('final_robot_target_dist')}",
            f"task_success: {result.get('task_success')}",
        ]
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def compute_clearance(self, robot_state, target_state, npc_states: Iterable, target_track_id: str) -> dict:
        robot_radius = float((self._meta.get("robot") or {}).get("radius", DEFAULT_ROBOT_RADIUS))
        target_radius = float((self._meta.get("target") or {}).get("radius", DEFAULT_TARGET_RADIUS))
        human_radius = float((self._meta.get("human") or {}).get("radius", DEFAULT_HUMAN_RADIUS))
        return compute_clearance(
            robot_state,
            target_state,
            npc_states,
            target_track_id,
            robot_radius=robot_radius,
            target_radius=target_radius,
            human_radius=human_radius,
        )


def build_step_timing(policy_total_ms: Optional[float], debug_info: Optional[dict]) -> dict:
    debug_timing = (debug_info or {}).get("timing") or {}
    perception = debug_timing.get("perception")
    if perception is None and isinstance((debug_info or {}).get("perception"), dict):
        perception = ((debug_info or {}).get("perception") or {}).get("timing")
    perception = _normalize_perception_timing(perception or debug_timing)

    policy_total = _first_float(
        debug_timing,
        "policy_total_ms",
        "policy_ms",
        fallback=policy_total_ms,
    )
    perception_total = _first_float(
        debug_timing,
        "perception_total_ms",
        "percep_ms",
        fallback=perception.get("total_ms"),
    )
    if perception.get("total_ms") is None and perception_total is not None:
        perception["total_ms"] = perception_total
    planner_core = _first_float(debug_timing, "planner_core_ms", "planner_ms")

    if planner_core is None:
        if perception_total is None:
            planner_core = policy_total
        elif policy_total is not None:
            planner_core = max(0.0, float(policy_total) - float(perception_total))

    wrapper_overhead = _first_float(debug_timing, "wrapper_overhead_ms", "overhead_ms")
    if wrapper_overhead is None and policy_total is not None:
        known = 0.0
        if perception_total is not None:
            known += float(perception_total)
        if planner_core is not None:
            known += float(planner_core)
        wrapper_overhead = max(0.0, float(policy_total) - known)

    if all(v is None for k, v in perception.items() if k != "raw"):
        perception_out = None
    else:
        perception_out = perception

    return {
        "policy_total_ms": policy_total,
        "perception_total_ms": perception_total,
        "planner_core_ms": planner_core,
        "wrapper_overhead_ms": wrapper_overhead,
        "perception": perception_out,
    }


def build_step_perception_summary(debug_info: Optional[dict]) -> Optional[dict]:
    if not debug_info:
        return None
    perception_debug = debug_info.get("perception")
    if not isinstance(perception_debug, dict):
        perception_debug = {}
    target_pos = debug_info.get("target_pos")
    if not isinstance(target_pos, dict):
        target_pos = {}
    planner_target = debug_info.get("planner_target")
    if not isinstance(planner_target, dict):
        planner_target = {}

    bboxes_by_view = debug_info.get("track_bboxes_by_view")
    if not isinstance(bboxes_by_view, dict):
        bboxes_by_view = perception_debug.get("track_bboxes_by_view")
    if not isinstance(bboxes_by_view, dict):
        bboxes_by_view = {}
    bbox_counts_by_view = {}
    bbox_target_views = []
    bbox_total = 0
    for view, bboxes in bboxes_by_view.items():
        if not isinstance(bboxes, list):
            continue
        count = len(bboxes)
        bbox_counts_by_view[str(view)] = int(count)
        bbox_total += count
        if any(isinstance(b, dict) and bool(b.get("is_target", False)) for b in bboxes):
            bbox_target_views.append(str(view))

    tracked_peds = debug_info.get("tracked_peds")
    if not isinstance(tracked_peds, list):
        tracked_peds = perception_debug.get("tracked_peds")
    tracked_count = len(tracked_peds) if isinstance(tracked_peds, list) else None

    summary = {
        "state": _first_present(debug_info, "perception_state", fallback=perception_debug.get("state")),
        "has_target": _first_present(debug_info, "perception_has_target"),
        "target_lost": _first_present(debug_info, "target_lost"),
        "reid_match": _first_present(debug_info, "reid_match"),
        "active_track_id": _first_present(debug_info, "active_track_id", fallback=perception_debug.get("target_id")),
        "target_track_id": _first_present(debug_info, "target_track_id", fallback=perception_debug.get("target_id")),
        "target_view": _first_present(debug_info, "target_view", fallback=perception_debug.get("target_view")),
        "target_bbox": _first_present(debug_info, "target_bbox", fallback=perception_debug.get("target_bbox")),
        "target_xy_tracker": target_pos.get("tracker", perception_debug.get("target_xy")),
        "target_xy_gt": target_pos.get("gt"),
        "target_err_m": target_pos.get("err_m"),
        "tracked_count": tracked_count,
        "bbox_total": int(bbox_total),
        "bbox_counts_by_view": bbox_counts_by_view,
        "bbox_target_views": bbox_target_views,
        "track_bboxes_age": _first_present(debug_info, "track_bboxes_age"),
        "planner_target": {
            "state": planner_target.get("planner_target_state"),
            "xy": planner_target.get("perception_target_filtered_xy"),
            "raw_xy": planner_target.get("perception_target_raw_xy"),
            "velocity": planner_target.get("perception_target_velocity"),
            "heading_rad": planner_target.get("perception_target_heading_rad"),
            "heading_confident": planner_target.get("perception_target_heading_confident"),
            "speed": planner_target.get("perception_target_speed"),
            "yaw_source": planner_target.get("target_yaw_source"),
            "route_heading_available": planner_target.get("route_heading_available"),
            "route_segment_id": planner_target.get("route_segment_id"),
            "route_distance_m": planner_target.get("route_distance_m"),
            "side_goal_yaw_rad": planner_target.get("side_goal_yaw_rad"),
        },
    }
    return _json_safe(summary)


def build_step_planner_summary(debug_info: Optional[dict]) -> Optional[dict]:
    if not debug_info:
        return None
    summary = {
        "mode": planner_mode_from_debug(debug_info),
        "search_mode": debug_info.get("search_mode"),
        "search_detail_mode": debug_info.get("search_detail_mode"),
        "identity_recovery_active": debug_info.get("identity_recovery_active"),
        "identity_recovery_reason": debug_info.get("identity_recovery_reason"),
        "search_hidden_steps": debug_info.get("search_hidden_steps"),
        "search_occluder_id": debug_info.get("search_occluder_id"),
        "search_cost": debug_info.get("search_cost"),
        "search_anchor": debug_info.get("search_anchor"),
        "search_goal": debug_info.get("search_goal"),
        "goal_point": debug_info.get("goal_point"),
        "target_protected": debug_info.get("target_protected"),
        "raw_target_visible": debug_info.get("raw_target_visible"),
        "gt_target_always_visible": debug_info.get("gt_target_always_visible"),
        "enable_search": debug_info.get("enable_search"),
        "predictor_name": debug_info.get("predictor_name"),
        "obstacle_count": _list_len(debug_info.get("obstacles")),
        "traj_point_count": _list_len(debug_info.get("traj_points")),
        "search_sample_count": _list_len(debug_info.get("search_samples")),
        "predicted_target_traj_count": _list_len(debug_info.get("predicted_target_traj")),
        "map_debug_mode": debug_info.get("map_debug_mode"),
    }
    if all(value is None for value in summary.values()):
        return None
    return _json_safe(summary)


def _normalize_perception_timing(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    mapping_ms = _first_float(raw, "mapping_ms", "map_ms")
    if mapping_ms is None:
        parts = [_first_float(raw, key) for key in ("depth_ms", "project_ms", "merge_ms")]
        finite_parts = [p for p in parts if p is not None]
        mapping_ms = float(sum(finite_parts)) if finite_parts else _first_float(raw, "project_ms")
    return {
        "detection_ms": _first_float(raw, "detection_ms", "detect_ms", "yolo_ms"),
        "tracking_ms": _first_float(raw, "tracking_ms", "track_ms"),
        "reid_ms": _first_float(raw, "reid_ms", "reid_extract_ms", "id_ms"),
        "mapping_ms": mapping_ms,
        "fsm_ms": _first_float(raw, "fsm_ms"),
        "other_ms": _first_float(raw, "other_ms"),
        "total_ms": _first_float(raw, "total_ms", "perception_total_ms", "percep_ms"),
        "raw": _json_safe(raw) if raw else None,
    }


def _first_float(source: Optional[dict], *keys: str, fallback: Any = None) -> Optional[float]:
    source = source or {}
    for key in keys:
        value = source.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    if fallback is not None:
        try:
            return float(fallback)
        except (TypeError, ValueError):
            return None
    return None


def _first_present(source: Optional[dict], *keys: str, fallback: Any = None) -> Any:
    source = source or {}
    for key in keys:
        if key in source and source.get(key) is not None:
            return source.get(key)
    return fallback


def _list_len(value: Any) -> Optional[int]:
    if isinstance(value, (list, tuple)):
        return int(len(value))
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def compute_clearance(robot_state, target_state, npc_states: Iterable, target_track_id: str, robot_radius: float = DEFAULT_ROBOT_RADIUS, target_radius: float = DEFAULT_TARGET_RADIUS, human_radius: float = DEFAULT_HUMAN_RADIUS) -> dict:
    rxy = np.array([float(robot_state.x), float(robot_state.y)], dtype=float)
    out = {
        "robot_target_surface_dist": None,
        "min_human_surface_dist": None,
        "closest_human_track_id": None,
    }
    if target_state is not None:
        txy = np.array([float(target_state.x), float(target_state.y)], dtype=float)
        out["robot_target_surface_dist"] = float(np.linalg.norm(rxy - txy) - robot_radius - target_radius)
    best_dist = float("inf")
    best_id = None
    for state in npc_states or []:
        if getattr(state, "track_id", None) == target_track_id:
            continue
        hxy = np.array([float(state.x), float(state.y)], dtype=float)
        d = float(np.linalg.norm(rxy - hxy) - robot_radius - human_radius)
        if d < best_dist:
            best_dist = d
            best_id = str(getattr(state, "track_id", ""))
    if best_id is not None:
        out["min_human_surface_dist"] = best_dist
        out["closest_human_track_id"] = best_id
    return out


def make_geometry_collision_event(tick: int, time_s: float, clearance: dict, target_track_id: str, margin: float = DEFAULT_HUMAN_COLLISION_MARGIN) -> Optional[dict]:
    target_dist = clearance.get("robot_target_surface_dist")
    if target_dist is not None and float(target_dist) <= float(margin):
        return {
            "tick": int(tick),
            "time_s": float(time_s),
            "episode_active": True,
            "source": "geometry",
            "collision_type": "human",
            "other_track_id": target_track_id,
            "is_target": True,
            "min_human_surface_dist": None,
            "robot_target_surface_dist": float(target_dist),
        }
    min_dist = clearance.get("min_human_surface_dist")
    if min_dist is None or float(min_dist) > float(margin):
        return None
    other_track_id = clearance.get("closest_human_track_id")
    return {
        "tick": int(tick),
        "time_s": float(time_s),
        "episode_active": True,
        "source": "geometry",
        "collision_type": "human",
        "other_track_id": other_track_id,
        "is_target": bool(other_track_id == target_track_id),
        "min_human_surface_dist": float(min_dist),
    }


def enrich_sensor_collision_events(events: Iterable[dict], *, tick: int, time_s: float, episode_active: bool, actor_to_track, target_track_id: str) -> list[dict]:
    out = []
    for event in events:
        enriched = dict(event)
        other_id = enriched.get("other_actor_id", -1)
        other_track_id = actor_to_track(other_id)
        enriched.update(
            {
                "tick": int(tick),
                "time_s": float(time_s),
                "episode_active": bool(episode_active),
                "other_track_id": other_track_id,
                "is_target": bool(other_track_id == target_track_id),
            }
        )
        out.append(enriched)
    return out
