#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np

SCENE = "passage"

from followbench.paths import module_path, scene_asset, scene_runs_dir  # noqa: E402

from followbench.sim.core_types import FollowObservation, RobotState  # noqa: E402


from followbench.visualization.lidar_vis import debug_lidar_world_points, lidar_to_bev as _lidar_to_bev  # noqa: E402

from followbench.scenes.passage.npc_runtime import DIFFICULTY_PRESETS, PassageNpcRuntime  # noqa: E402
from followbench.sim.robot_runtime import ScoutRobotRuntime  # noqa: E402
from followbench.sim.humanoid_runtime import G1RobotRuntime  # noqa: E402
from followbench.sim.visibility_instance import InstanceVisibilityChecker  # noqa: E402
from followbench.sim.visibility_instance import _target_bbox_roi as _target_bbox_roi_xyxy  # noqa: E402
from followbench.scenes.passage.collision_monitor import PassageCollisionMonitor, PassageMapObstacleMonitor  # noqa: E402
import followbench.sim.weather as weather_mod  # noqa: E402
from followbench.evaluation.collision import actor_track_lookup  # noqa: E402
from followbench.evaluation.logging import (  # noqa: E402
    EvaluationLogger,
    enrich_sensor_collision_events,
    make_geometry_collision_event,
)
from followbench.evaluation.schemas import add_evaluation_args  # noqa: E402
from followbench.evaluation.sim_clock import SimulationPacer  # noqa: E402
from followbench.sim.carla_world import connect_world  # noqa: E402

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# ── Manual control constants ──────────────────────────────────────────────────
MANUAL_WALKER_SPEED_MPS = 1.8
MANUAL_ROBOT_V_MPS      = 2.0
MANUAL_ROBOT_W_RADPS    = 2.2
MANUAL_SPRINT_MULT      = 1.8
MANUAL_CAM_DIST         = 4.5
MANUAL_CAM_HEIGHT       = 2.0
MANUAL_MOUSE_SENS       = 0.20


class TargetIdentityMonitor:
    """Evaluation-only watchdog for perception target identity switches."""

    def __init__(
        self,
        *,
        enabled: bool,
        associate_gate_m: float,
        wrong_dist_m: float,
        startup_grace_sec: float,
        confirm_sec: float,
        fail_sec: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.associate_gate_m = max(float(associate_gate_m), 0.0)
        self.wrong_dist_m = max(float(wrong_dist_m), 0.0)
        self.startup_grace_sec = max(float(startup_grace_sec), 0.0)
        self.confirm_sec = max(float(confirm_sec), 0.0)
        self.fail_sec = max(float(fail_sec), self.confirm_sec)
        self.mismatch_duration_s = 0.0
        self.max_mismatch_duration_s = 0.0
        self.switch_count = 0
        self._was_mismatching = False
        self.failure_track_id: Optional[str] = None

    def reset(self) -> None:
        self.mismatch_duration_s = 0.0
        self.max_mismatch_duration_s = 0.0
        self.switch_count = 0
        self._was_mismatching = False
        self.failure_track_id = None

    def update(
        self,
        *,
        dt: float,
        eval_time_s: float,
        target_track_id: str,
        target_state,
        npc_states,
        policy_debug_info: Optional[dict],
    ) -> dict:
        payload = {
            "enabled": bool(self.enabled),
            "status": "disabled",
            "identity_state": "disabled",
            "perception_has_target": None,
            "perception_track_id": None,
            "tracker_xy": None,
            "gt_target_track_id": str(target_track_id),
            "gt_target_xy": None,
            "err_to_gt_target_m": None,
            "associated_gt_track_id": None,
            "associated_gt_dist_m": None,
            "mismatch_active": False,
            "mismatch_reason": None,
            "recovery_active": False,
            "recovery_reason": None,
            "mismatch_duration_s": float(self.mismatch_duration_s),
            "max_mismatch_duration_s": float(self.max_mismatch_duration_s),
            "switch_count": int(self.switch_count),
            "failure_track_id": self.failure_track_id,
            "termination_triggered": False,
            "termination_reason": None,
        }
        if not self.enabled:
            return payload

        payload["status"] = "no_perception"
        debug = policy_debug_info if isinstance(policy_debug_info, dict) else {}
        perception_has_target = bool(debug.get("perception_has_target", False))
        target_lost = bool(debug.get("target_lost", False))
        payload["perception_has_target"] = perception_has_target
        payload["perception_track_id"] = debug.get("active_track_id", debug.get("target_track_id"))
        recovery_reason = self._recovery_reason(debug)
        recovery_active = recovery_reason is not None
        payload["recovery_active"] = recovery_active
        payload["recovery_reason"] = recovery_reason

        gt_xy = None
        if target_state is not None:
            gt_xy = (float(target_state.x), float(target_state.y))
            payload["gt_target_xy"] = [gt_xy[0], gt_xy[1]]

        target_pos = debug.get("target_pos") if isinstance(debug.get("target_pos"), dict) else {}
        tracker_xy_raw = target_pos.get("tracker")
        tracker_xy = self._coerce_xy(tracker_xy_raw)
        if tracker_xy is not None:
            payload["tracker_xy"] = [tracker_xy[0], tracker_xy[1]]

        if target_lost or not perception_has_target or tracker_xy is None:
            payload["status"] = "recovering" if recovery_active else ("lost" if target_lost else "no_target")
            payload["identity_state"] = payload["status"]
            payload["mismatch_duration_s"] = float(self.mismatch_duration_s)
            payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
            payload["switch_count"] = int(self.switch_count)
            return payload

        associated_track_id, associated_dist = self._nearest_gt_track(tracker_xy, npc_states)
        payload["associated_gt_track_id"] = associated_track_id
        payload["associated_gt_dist_m"] = associated_dist
        if gt_xy is not None:
            payload["err_to_gt_target_m"] = float(np.hypot(tracker_xy[0] - gt_xy[0], tracker_xy[1] - gt_xy[1]))

        associated_to_target = (
            associated_track_id == str(target_track_id)
            and associated_dist is not None
            and associated_dist <= self.associate_gate_m
        )
        if associated_to_target:
            self.mismatch_duration_s = 0.0
            self._was_mismatching = False
            payload["status"] = "ok"
            payload["identity_state"] = "ok"
            payload["mismatch_duration_s"] = 0.0
            payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
            payload["switch_count"] = int(self.switch_count)
            return payload

        if recovery_active:
            self._was_mismatching = False
            payload["status"] = "recovering"
            payload["identity_state"] = "recovering"
            payload["mismatch_duration_s"] = float(self.mismatch_duration_s)
            payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
            payload["switch_count"] = int(self.switch_count)
            return payload

        err_to_target = payload["err_to_gt_target_m"]
        strict_wrong_dist_m = min(self.wrong_dist_m, 3.0)
        strict_associate_gate_m = min(self.associate_gate_m, 0.35)
        base_mismatch = (
            eval_time_s >= self.startup_grace_sec
            and associated_track_id is not None
            and associated_track_id != str(target_track_id)
            and associated_dist is not None
            and associated_dist <= self.associate_gate_m
            and err_to_target is not None
            and err_to_target >= self.wrong_dist_m
        )
        clear_wrong_association = (
            eval_time_s >= self.startup_grace_sec
            and associated_track_id is not None
            and associated_track_id != str(target_track_id)
            and associated_dist is not None
            and associated_dist <= strict_associate_gate_m
            and err_to_target is not None
            and err_to_target >= strict_wrong_dist_m
        )
        mismatch = base_mismatch or clear_wrong_association

        if mismatch:
            payload["mismatch_reason"] = (
                "clear_wrong_association"
                if clear_wrong_association and not base_mismatch
                else "wrong_target_distance"
            )
            if not self._was_mismatching:
                self.switch_count += 1
            self._was_mismatching = True
            self.mismatch_duration_s += max(float(dt), 0.0)
            self.max_mismatch_duration_s = max(self.max_mismatch_duration_s, self.mismatch_duration_s)
            self.failure_track_id = str(associated_track_id)
            payload["failure_track_id"] = self.failure_track_id
            payload["mismatch_active"] = True
            if self.mismatch_duration_s >= self.fail_sec:
                payload["status"] = "failed"
                payload["identity_state"] = "failed"
                payload["termination_triggered"] = True
                payload["termination_reason"] = "target_identity_lost"
            elif self.mismatch_duration_s >= self.confirm_sec:
                payload["status"] = "suspect"
                payload["identity_state"] = "suspect"
            else:
                payload["status"] = "warming"
                payload["identity_state"] = "warming"
        else:
            payload["status"] = "ambiguous"
            payload["identity_state"] = "ambiguous"
            self._was_mismatching = False

        payload["mismatch_duration_s"] = float(self.mismatch_duration_s)
        payload["max_mismatch_duration_s"] = float(self.max_mismatch_duration_s)
        payload["switch_count"] = int(self.switch_count)
        return payload

    @staticmethod
    def _recovery_reason(debug: dict) -> Optional[str]:
        if bool(debug.get("identity_recovery_active", False)):
            return "identity_recovery_active"
        for key in ("recovery_state", "search_detail_mode", "search_mode"):
            value = debug.get(key)
            if value is None:
                continue
            if str(value).lower() == "reacquire_transition":
                return f"{key}=reacquire_transition"
        return None

    @staticmethod
    def _coerce_xy(value) -> Optional[tuple[float, float]]:
        if value is None:
            return None
        try:
            if len(value) < 2:
                return None
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _nearest_gt_track(tracker_xy: tuple[float, float], npc_states) -> tuple[Optional[str], Optional[float]]:
        best_track = None
        best_dist = None
        tx, ty = tracker_xy
        for state in npc_states or []:
            track_id = getattr(state, "track_id", None)
            if track_id is None:
                continue
            dist = float(np.hypot(float(state.x) - tx, float(state.y) - ty))
            if best_dist is None or dist < best_dist:
                best_track = str(track_id)
                best_dist = dist
        return best_track, best_dist


class TargetDistanceMonitor:
    def __init__(
        self,
        *,
        warmup_sec: float,
        fail_dist_m: float,
        fail_sec: float,
        fail_steps: int = 0,
    ) -> None:
        self.warmup_sec = max(float(warmup_sec), 0.0)
        self.fail_dist_m = max(float(fail_dist_m), 0.0)
        self.fail_sec = max(float(fail_sec), 0.0)
        self.fail_steps = max(int(fail_steps), 0)
        self.duration_s = 0.0
        self.max_duration_s = 0.0
        self.steps = 0
        self.max_steps = 0

    def reset(self) -> None:
        self.duration_s = 0.0
        self.max_duration_s = 0.0
        self.steps = 0
        self.max_steps = 0

    def update(self, *, dt: float, eval_time_s: float, following: bool, distance_m: Optional[float]) -> dict:
        payload = {
            "enabled": True,
            "distance_m": distance_m,
            "warmup_sec": float(self.warmup_sec),
            "fail_dist_m": float(self.fail_dist_m),
            "fail_sec": float(self.fail_sec),
            "fail_steps": int(self.fail_steps),
            "duration_s": float(self.duration_s),
            "max_duration_s": float(self.max_duration_s),
            "steps": int(self.steps),
            "max_steps": int(self.max_steps),
            "active": False,
            "termination_triggered": False,
            "termination_reason": None,
        }
        if not following or distance_m is None or eval_time_s < self.warmup_sec or distance_m <= self.fail_dist_m:
            self.duration_s = 0.0
            self.steps = 0
            payload["duration_s"] = 0.0
            payload["steps"] = 0
            return payload
        self.duration_s += max(float(dt), 0.0)
        self.steps += 1
        self.max_duration_s = max(self.max_duration_s, self.duration_s)
        self.max_steps = max(self.max_steps, self.steps)
        payload["duration_s"] = float(self.duration_s)
        payload["max_duration_s"] = float(self.max_duration_s)
        payload["steps"] = int(self.steps)
        payload["max_steps"] = int(self.max_steps)
        payload["active"] = True
        # Step threshold takes priority when configured (>0); otherwise fall back to seconds.
        triggered_by_steps = self.fail_steps > 0 and self.steps > self.fail_steps
        triggered_by_sec = self.fail_steps <= 0 and self.duration_s > self.fail_sec
        if triggered_by_steps or triggered_by_sec:
            payload["termination_triggered"] = True
            payload["termination_reason"] = "target_too_far"
        return payload


class RobotStuckMonitor:
    def __init__(
        self,
        *,
        window_sec: float,
        grace_sec: float,
        min_progress_m: float,
        target_wait_speed_mps: float,
        target_escape_distance_m: float,
        static_obstacle_clearance_m: float,
    ) -> None:
        self.window_sec = max(float(window_sec), 0.5)
        self.grace_sec = max(float(grace_sec), 0.0)
        self.min_progress_m = max(float(min_progress_m), 0.0)
        self.target_wait_speed_mps = max(float(target_wait_speed_mps), 0.0)
        self.target_escape_distance_m = max(float(target_escape_distance_m), 0.0)
        self.static_obstacle_clearance_m = max(float(static_obstacle_clearance_m), 0.0)
        self.history: List[Tuple[float, float, float]] = []

    def reset(self) -> None:
        self.history.clear()

    def update(
        self,
        *,
        eval_time_s: float,
        following: bool,
        robot_state: Optional[RobotState],
        target_state,
        target_surface_dist_m: Optional[float],
        obstacle_clearance_m: Optional[float],
    ) -> dict:
        payload = {
            "enabled": True,
            "window_sec": float(self.window_sec),
            "grace_sec": float(self.grace_sec),
            "min_progress_m": float(self.min_progress_m),
            "target_escape_distance_m": float(self.target_escape_distance_m),
            "static_obstacle_clearance_m": float(self.static_obstacle_clearance_m),
            "progress_m": None,
            "target_surface_dist_m": target_surface_dist_m,
            "target_speed_mps": None if target_state is None else float(getattr(target_state, "speed", 0.0)),
            "obstacle_clearance_m": obstacle_clearance_m,
            "active": False,
            "termination_triggered": False,
            "termination_reason": None,
        }
        if not following or robot_state is None:
            self.history.clear()
            return payload
        self.history.append((float(eval_time_s), float(robot_state.x), float(robot_state.y)))
        while len(self.history) > 2 and eval_time_s - self.history[0][0] > self.window_sec:
            self.history.pop(0)
        if eval_time_s < self.grace_sec or len(self.history) < 2:
            return payload
        history_span = self.history[-1][0] - self.history[0][0]
        if history_span < min(self.window_sec * 0.8, self.window_sec - 1e-6):
            return payload
        x0, y0 = self.history[0][1], self.history[0][2]
        x1, y1 = self.history[-1][1], self.history[-1][2]
        progress_m = float(np.hypot(x1 - x0, y1 - y0))
        payload["progress_m"] = progress_m
        if progress_m >= self.min_progress_m:
            return payload
        target_speed = payload["target_speed_mps"]
        target_waiting = (
            target_surface_dist_m is not None
            and target_surface_dist_m <= self.target_escape_distance_m
            and target_speed is not None
            and target_speed <= self.target_wait_speed_mps
        )
        if target_waiting:
            return payload
        near_static_obstacle = (
            obstacle_clearance_m is not None and obstacle_clearance_m <= self.static_obstacle_clearance_m
        )
        far_from_target = (
            target_surface_dist_m is not None and target_surface_dist_m >= self.target_escape_distance_m
        )
        if not (near_static_obstacle or far_from_target):
            return payload
        payload["active"] = True
        payload["termination_triggered"] = True
        payload["termination_reason"] = (
            "robot_stuck_static_obstacle" if near_static_obstacle else "robot_stuck_far_from_target"
        )
        return payload


def _collision_termination_reason(event: dict) -> Optional[str]:
    if bool(event.get("is_target", False)):
        return "target_collision"
    collision_type = str(event.get("collision_type", "")).lower()
    if collision_type == "human":
        return "pedestrian_collision"
    if collision_type == "obstacle":
        return "obstacle_collision"
    return None


def _select_collision_termination(events: List[dict]) -> tuple[Optional[str], Optional[dict]]:
    priority = {
        "target_collision": 0,
        "pedestrian_collision": 1,
        "obstacle_collision": 2,
    }
    best_reason = None
    best_event = None
    for event in events:
        reason = _collision_termination_reason(event)
        if reason is None:
            continue
        if best_reason is None or priority[reason] < priority[best_reason]:
            best_reason = reason
            best_event = event
    return best_reason, best_event


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Passage episode manager")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--load-map", action="store_true")
    p.add_argument("--map-name", type=str, default="Town10HD_Opt")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--duration-sec", type=float, default=300.0)
    p.add_argument("--num-walkers", type=int, default=None,
                   help="Total walkers incl. target. Default: difficulty preset (6/10/14/18).")
    p.add_argument(
        "--npc-rng-seed",
        type=int,
        default=None,
        help="NPC random seed. Omit for a fresh seed each run; set an integer to reproduce.",
    )
    p.add_argument("--npc-min-speed", type=float, default=None,
                   help="Override the difficulty-preset NPC min speed (m/s).")
    p.add_argument("--npc-max-speed", type=float, default=None,
                   help="Override the difficulty-preset NPC max speed (m/s).")
    p.add_argument("--target-track-id", type=str, default="N01")
    p.add_argument("--visibility-threshold", type=int, default=400)
    p.add_argument("--grid-npz", type=str,
                   default=scene_asset(SCENE, "gridmap_roi.npz"))
    p.add_argument("--routes-json", type=str,
                   default=scene_asset(SCENE, "routes.json"))
    p.add_argument("--flow-points-json", type=str,
                   default=scene_asset(SCENE, "gridmap_roi_flow_points.json"))
    p.add_argument("--robot-spawn-x", type=float, default=0.0)
    p.add_argument("--robot-spawn-y", type=float, default=0.0)
    p.add_argument("--robot-spawn-z", type=float, default=0.5)
    p.add_argument("--robot-spawn-yaw-deg", type=float, default=0.0)
    p.add_argument("--robot-spawn-mode",
                   choices=["fixed", "roi_random", "near_target", "behind_target"],
                   default="near_target")
    p.add_argument("--robot-type", choices=["scout", "g1"], default="scout")
    p.add_argument("--robot-spawn-min-dist", type=float, default=0.8)
    p.add_argument("--robot-spawn-max-dist", type=float, default=2.0)
    p.add_argument("--sensor-image-w", type=int, default=800)
    p.add_argument("--sensor-image-h", type=int, default=600)
    p.add_argument("--sensor-fov-deg", type=float, default=90.0)
    p.add_argument("--sensor-lidar-range", type=float, default=30.0)
    p.add_argument("--sensor-lidar-height-agl", type=float, default=0.50,
                   help="Single-line LiDAR scan plane height above ground (m).")
    p.add_argument("--robot-rescue-lift-z", type=float, default=0.25)
    p.add_argument("--planner",
                   choices=["pid", "dwa_traj", "dwa_traj_depth_tpt", "sfm",
                            "rda", "rda_lidar", "gt_planner", "rda_traj", "rda_search",
                            "rda_depth_tpt", "bso_hfc",
                            "trackvla", "oa_vat", "custom"],
                   default="pid")
    from followbench.interfaces.policy_factory import add_policy_factory_args  # noqa: PLC0415
    add_policy_factory_args(p)
    p.add_argument("--follow-position", choices=["back", "left_side", "right_side"], default="back")
    p.add_argument("--desired-distance", type=float, default=1.5)
    p.add_argument(
        "--rda-search-use-camera-visibility",
        action="store_true",
        help="Let rda_search enter search modes from camera visibility. Default keeps GT-target runs in follow mode.",
    )
    p.add_argument(
        "--rda-search-disable-target-protection",
        action="store_true",
        help="Disable the side-follow target protected zone injected for rda_search.",
    )
    p.add_argument(
        "--rda-traj-disable-lidar-obstacles",
        action="store_true",
        help="Disable lidar/NPC obstacle injection for rda_traj (legacy behavior).",
    )
    p.add_argument(
        "--use-perception",
        action="store_true",
        help="Wrap the chosen --planner with a YOLO+depth+ReID perception "
             "frontend (multi-view). Target and NPC positions come from the "
             "tracker instead of GT. Compatible with every planner except "
             "rda_depth_tpt / trackvla / oa_vat (each carries its own perception).",
    )
    p.add_argument(
        "--reid-mode",
        choices=["basic", "kpr"],
        default="basic",
        help="Appearance ReID backbone (only when --use-perception): basic or kpr.",
    )
    p.add_argument(
        "--reid-kpr-config",
        type=str,
        default="kpr_occ_duke_test",
        help="KPR config name (only used when --reid-mode=kpr).",
    )
    p.add_argument(
        "--reid-device",
        type=str,
        default="auto",
        help="Device for the ReID backbone (auto/cuda/cpu).",
    )
    p.add_argument(
        "--lost-policy",
        choices=["gt_fallback", "brake", "last_known"],
        default="last_known",
        help="Behaviour when the perception frontend reports target lost. rda_search only accepts last_known.",
    )
    p.add_argument(
        "--kinematics-mode",
        choices=["ema", "kf"],
        default="kf",
        help="Per-track velocity estimator used by the depth tracker (only when --use-perception).",
    )
    p.add_argument(
        "--kf-pos-sigma",
        type=float,
        default=0.20,
        help="(--kinematics-mode=kf) Measurement noise std-dev in metres.",
    )
    p.add_argument(
        "--kf-vel-sigma-q",
        type=float,
        default=0.05,
        help="(--kinematics-mode=kf) Process noise std-dev on velocity (m/s).",
    )
    p.add_argument(
        "--trackvla-instruction",
        type=str,
        default="Follow the person",
        help="(planner=trackvla only) Natural-language instruction fed to the VLM each tick.",
    )
    p.add_argument(
        "--trackvla-ckpt",
        type=str,
        default=None,
        help="(planner=trackvla only) Override the default .pt checkpoint path under data/trackvla/ckpts/.",
    )
    p.add_argument("--trackvla-auto-caption", action="store_true", default=True,
                   help="(planner=trackvla only) Use Qwen3-VL to caption the target's "
                        "clothing on the first visible frame, then set the instruction to "
                        "'Follow the person wearing ...'. Disable with --trackvla-no-auto-caption.")
    p.add_argument("--trackvla-no-auto-caption", dest="trackvla_auto_caption",
                   action="store_false",
                   help="(planner=trackvla only) Force plain --trackvla-instruction.")
    p.add_argument("--trackvla-caption-ckpt", type=str, default=None,
                   help="(planner=trackvla only) Qwen3-VL ckpt dir; "
                        "default data/trackvla/LLM_hf/qwen3-vl-4b-instruct.")
    p.add_argument("--trackvla-caption-device", type=str, default=None,
                   help="(planner=trackvla only) Captioner device; default cuda if available.")
    p.add_argument("--oa-vat-text-query", type=str, default="person",
                   help="(planner=oa_vat) YOLOe text prompt for first-frame and re-acq detection")
    p.add_argument("--oa-vat-kf-mode", type=str, default="confaware",
                   choices=["none", "standard", "confaware"],
                   help="(planner=oa_vat) Kalman filter mode for occlusion gap-fill")
    p.add_argument("--oa-vat-online-enhance", action="store_true", default=True,
                   help="(planner=oa_vat) EMA-update reference DINOv3 feature every N frames")
    p.add_argument("--oa-vat-no-online-enhance", dest="oa_vat_online_enhance",
                   action="store_false", help="(planner=oa_vat) disable online enhance")
    p.add_argument("--oa-vat-dinov3", type=str, default="dinov3_vitb16",
                   choices=["dinov3_vitb16", "dinov3_vits16"],
                   help="(planner=oa_vat) DINOv3 backbone")
    p.add_argument("--oa-vat-match-thresh", type=float, default=0.4,
                   help="(planner=oa_vat) DINOv3 cosine threshold for first-frame match")
    p.add_argument("--oa-vat-reacq-thresh", type=float, default=0.5,
                   help="(planner=oa_vat) DINOv3 cosine threshold for re-acquisition")
    p.add_argument("--oa-vat-yolo-conf", type=float, default=0.1,
                   help="(planner=oa_vat) YOLOe confidence threshold")
    p.add_argument("--oa-vat-target-area-ratio", type=float, default=0.12,
                   help="(planner=oa_vat) Desired bbox area / frame area for distance control")
    p.add_argument("--teleop", action="store_true",
                   help="Drive the robot with WASD and log the episode as planner=teleop.")
    p.add_argument("--operator-id", type=str, default="anonymous",
                   help="Anonymous operator identifier stored in episode metadata.")
    p.add_argument("--num-episodes", type=int, default=1,
                   help="Number of teleoperation episodes to run in one CARLA/UI session.")
    p.add_argument("--sim-speed", type=float, default=1.0,
                   help="Teleop simulation speed relative to wall clock (1.0=realtime).")
    p.add_argument("--no-ui", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--record-planning-video",
        action="store_true",
        help="Record rgb_planning.mp4 (front RGB on top, planning BEV + timeline below) "
             "into the evaluation trial dir. Requires --enable-evaluation.",
    )
    p.add_argument("--auto-spawn-sec", type=float, default=-1.0)
    p.add_argument("--auto-follow-sec", type=float, default=-1.0)
    p.add_argument("--output-dir", type=str, default=scene_runs_dir(SCENE))
    p.add_argument("--draw-debug", action="store_true")
    p.add_argument("--draw-laser", action="store_true")
    p.add_argument("--async-mode", action="store_true")
    p.add_argument("--async-after-spawn", action="store_true")
    p.add_argument("--difficulty",
                   choices=["easy", "hard", "boss", "legend"],
                   default="easy",
                   help="Crowd difficulty preset. Cycle at runtime with D key.")
    p.add_argument("--weather",
                   choices=["random"] + weather_mod.CHOICES,
                   default="random",
                   help="Weather preset. 'random' re-rolls each time difficulty is selected.")
    add_evaluation_args(p)
    p.set_defaults(
        target_identity_startup_grace_sec=1.5,
        target_identity_confirm_sec=1.5,
        target_identity_fail_sec=2.5,
        target_identity_wrong_dist_m=4.0,
        terminate_on_target_identity_loss=False,
    )
    p.add_argument("--map-obstacle-overlap-confirm-steps", type=int, default=2)
    p.add_argument("--map-obstacle-clearance-probe-m", type=float, default=2.0)
    p.add_argument("--robot-target-far-fail-dist-m", type=float, default=3.0)
    p.add_argument("--robot-target-far-fail-sec", type=float, default=2.0)
    p.add_argument(
        "--robot-target-far-fail-steps",
        type=int,
        default=100,
        help="Long-term target loss threshold N_lost: the episode fails once the robot-target distance exceeds "
             "--robot-target-far-fail-dist-m for more than N_lost consecutive steps (target_too_far). "
             "Set <=0 to fall back to --robot-target-far-fail-sec.",
    )
    p.add_argument("--robot-target-far-warmup-sec", type=float, default=0.0)
    stuck_group = p.add_mutually_exclusive_group()
    stuck_group.add_argument(
        "--terminate-on-robot-stuck",
        dest="terminate_on_robot_stuck",
        action="store_true",
        default=False,
    )
    stuck_group.add_argument(
        "--no-terminate-on-robot-stuck",
        dest="terminate_on_robot_stuck",
        action="store_false",
    )
    p.add_argument("--robot-stuck-window-sec", type=float, default=3.0)
    p.add_argument("--robot-stuck-grace-sec", type=float, default=3.0)
    p.add_argument("--robot-stuck-min-progress-m", type=float, default=0.25)
    p.add_argument("--robot-stuck-target-wait-speed-mps", type=float, default=0.15)
    p.add_argument("--robot-stuck-target-escape-distance-m", type=float, default=6.0)
    p.add_argument("--robot-stuck-static-obstacle-clearance-m", type=float, default=0.35)
    return p.parse_args()


_FOLLOW_OFFSET_RAD = {
    "back":       math.pi,
    "front":      0.0,
    # CARLA target-body sides: visual-left is yaw - 90 deg, visual-right is yaw + 90 deg.
    "left_side":  -math.pi / 2.0,
    "right_side": math.pi / 2.0,
}


def follow_position_spawn_xy(target_x, target_y, target_yaw_rad,
                              follow_position, desired_distance,
                              roi_map, forbidden_xys=None,
                              min_clearance_m: float = 0.7,
                              search_max_radius: float = 2.0):
    """Pick a spawn (x, y) at the requested follow offset, avoiding existing
    walkers. ``forbidden_xys`` is a list of (x, y) world coords the candidate
    must stay at least ``min_clearance_m`` away from."""
    forbidden = list(forbidden_xys) if forbidden_xys else []
    clearance_sq = float(min_clearance_m) * float(min_clearance_m)

    def _ok(x: float, y: float) -> bool:
        if not roi_map.is_world_walkable(x, y):
            return False
        for fx, fy in forbidden:
            dx = x - float(fx)
            dy = y - float(fy)
            if dx * dx + dy * dy < clearance_sq:
                return False
        return True

    offset = _FOLLOW_OFFSET_RAD.get(follow_position, math.pi)
    base_angle = target_yaw_rad + offset
    ideal_x = float(target_x + math.cos(base_angle) * desired_distance)
    ideal_y = float(target_y + math.sin(base_angle) * desired_distance)
    if _ok(ideal_x, ideal_y):
        return ideal_x, ideal_y, True
    for radius in np.arange(0.2, search_max_radius + 1e-3, 0.2):
        for ang in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
            xx = float(ideal_x + math.cos(ang) * radius)
            yy = float(ideal_y + math.sin(ang) * radius)
            if _ok(xx, yy):
                return xx, yy, True
    return ideal_x, ideal_y, False


def load_passage_route_segments(
    routes_json: str,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    try:
        with open(routes_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[EP] warning: failed to load route segments from {routes_json}: {exc}")
        return []

    routes = payload.get("routes", {})
    if not isinstance(routes, dict):
        return []

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for route_id in sorted(
        routes,
        key=lambda k: (0, int(k)) if str(k).isdigit() else (1, str(k)),
    ):
        raw_points = routes.get(route_id)
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            continue
        points: List[Tuple[float, float]] = []
        for item in raw_points:
            try:
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError, IndexError):
                points = []
                break
        for a, b in zip(points, points[1:]):
            if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-6:
                segments.append((a, b))
    return segments


def resolve_robot_spawn_tf(args, npc: PassageNpcRuntime, target_track_id: str) -> carla.Transform:
    if args.robot_spawn_mode == "fixed":
        return carla.Transform(
            carla.Location(x=args.robot_spawn_x, y=args.robot_spawn_y, z=args.robot_spawn_z),
            carla.Rotation(yaw=args.robot_spawn_yaw_deg),
        )
    t = npc.get_target_by_track_id(target_track_id)
    if t is None:
        loc = npc.roi_map.sample_world_location(npc.rng, z=args.robot_spawn_z, jitter=True)
        return carla.Transform(carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z)),
                               carla.Rotation(yaw=args.robot_spawn_yaw_deg))
    # Treat the target as another forbidden xy: keeps the chosen spawn position
    # at least min_clearance_m (default 0.7m) from the target center so the
    # robot never lands inside the target capsule (was the dominant cause of
    # geometry/target_collision @ first active tick).
    other_xys = [(s.x, s.y) for s in npc.get_states() if s.track_id != target_track_id]
    other_xys.append((float(t.x), float(t.y)))
    target_yaw_rad = yaw_rad_from_velocity(float(t.vx), float(t.vy), float(t.yaw_deg))
    if args.robot_spawn_mode == "roi_random":
        loc = npc.roi_map.sample_world_location(npc.rng, z=args.robot_spawn_z, jitter=True)
        return carla.Transform(carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z)),
                               carla.Rotation(yaw=float(np.degrees(target_yaw_rad))))

    z = float(max(args.robot_spawn_z, t.z + 0.25))
    x, y, _ok = follow_position_spawn_xy(
        float(t.x), float(t.y), target_yaw_rad,
        str(args.follow_position), float(args.desired_distance),
        npc.roi_map, forbidden_xys=other_xys,
    )
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=float(np.degrees(target_yaw_rad))),
    )


def yaw_rad_from_velocity(vx: float, vy: float, fallback_deg: float) -> float:
    if abs(vx) + abs(vy) > 1e-4:
        return float(np.arctan2(vy, vx))
    return float(np.deg2rad(fallback_deg))


def _compute_target_bbox_for_trackvla(
    args,
    target_actor,
    robot,
    rgb_intrinsics,
):
    """Project the target actor's 3D bounding box into front-camera pixel coords.

    Returns (u0, v0, u1, v1) or None. Only computed when the active planner is
    ``trackvla`` AND auto-captioning is enabled — keeps other runs CARLA-free.
    """
    if getattr(args, "planner", None) != "trackvla":
        return None
    if not getattr(args, "trackvla_auto_caption", False):
        return None
    if target_actor is None or rgb_intrinsics is None:
        return None
    rgb_sensor = robot.sensors.get("rgb") if hasattr(robot, "sensors") else None
    if rgb_sensor is None:
        return None
    try:
        return _target_bbox_roi_xyxy(
            target_actor,
            rgb_sensor,
            int(rgb_intrinsics.get("width", 0)),
            int(rgb_intrinsics.get("height", 0)),
            float(args.sensor_fov_deg),
        )
    except Exception:
        return None


def step_world(world: carla.World, synchronous_mode: bool) -> None:
    if synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()


def apply_world_mode(world: carla.World, base_settings: carla.WorldSettings,
                     sync_mode: bool, dt: float) -> None:
    s = world.get_settings()
    s.no_rendering_mode = base_settings.no_rendering_mode
    s.substepping = base_settings.substepping
    s.max_substep_delta_time = base_settings.max_substep_delta_time
    s.max_substeps = base_settings.max_substeps
    s.synchronous_mode = bool(sync_mode)
    s.fixed_delta_seconds = float(dt) if sync_mode else None
    world.apply_settings(s)


def draw_status(world: carla.World, text: str) -> None:
    world.debug.draw_string(
        carla.Location(x=0.0, y=0.0, z=3.0), text,
        draw_shadow=False, color=carla.Color(255, 255, 255), life_time=0.1)


def _apply_manual_target_control(target_actor, key_state, cam_yaw: float = 0.0) -> None:
    if target_actor is None or key_state is None:
        return
    move_forward = float(key_state[pygame.K_w]) - float(key_state[pygame.K_s])
    move_right   = float(key_state[pygame.K_d]) - float(key_state[pygame.K_a])
    sprint = bool(key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT])
    speed  = MANUAL_WALKER_SPEED_MPS * (MANUAL_SPRINT_MULT if sprint else 1.0)
    if abs(move_forward) + abs(move_right) < 1e-6:
        target_actor.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=0.0, jump=False))
        return
    yaw_rad = math.radians(cam_yaw)
    dx = move_forward * math.cos(yaw_rad) - move_right * math.sin(yaw_rad)
    dy = move_forward * math.sin(yaw_rad) + move_right * math.cos(yaw_rad)
    n  = math.hypot(dx, dy)
    if n < 1e-6:
        return
    target_actor.apply_control(carla.WalkerControl(
        direction=carla.Vector3D(dx / n, dy / n, 0.0), speed=float(speed), jump=False))


def _apply_manual_robot_control(
    robot: ScoutRobotRuntime,
    key_state,
) -> Tuple[float, float]:
    if key_state is None:
        robot.hold_still()
        return 0.0, 0.0
    forward = float(key_state[pygame.K_w]) - float(key_state[pygame.K_s])
    turn    = float(key_state[pygame.K_d]) - float(key_state[pygame.K_a])
    sprint  = bool(key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT])
    scale   = MANUAL_SPRINT_MULT if sprint else 1.0
    v_cmd = MANUAL_ROBOT_V_MPS * scale * forward
    w_cmd = MANUAL_ROBOT_W_RADPS * scale * turn
    robot.apply_velocity_command(v_cmd, w_cmd)
    return v_cmd, w_cmd


def _update_manual_spectator(spectator, actor, cam_yaw: float, cam_pitch: float) -> None:
    loc       = actor.get_location()
    yaw_rad   = math.radians(cam_yaw)
    pitch_rad = math.radians(cam_pitch)
    ox = -MANUAL_CAM_DIST * math.cos(yaw_rad) * math.cos(pitch_rad)
    oy = -MANUAL_CAM_DIST * math.sin(yaw_rad) * math.cos(pitch_rad)
    oz =  MANUAL_CAM_HEIGHT - MANUAL_CAM_DIST * math.sin(pitch_rad)
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x + ox, y=loc.y + oy, z=loc.z + max(oz, 0.5)),
        carla.Rotation(pitch=cam_pitch, yaw=cam_yaw, roll=0.0)))


def _to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3][:, :, ::-1]


def _depth_to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    d = (arr[:, :, 2].astype(np.float32)
         + arr[:, :, 1].astype(np.float32) * 256.0
         + arr[:, :, 0].astype(np.float32) * 65536.0) / (256.0**3 - 1.0) * 1000.0
    gray = np.clip(np.log1p(d) / np.log1p(100.0) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _depth_to_meters(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return (arr[:, :, 2].astype(np.float32)
            + arr[:, :, 1].astype(np.float32) * 256.0
            + arr[:, :, 0].astype(np.float32) * 65536.0) / (256.0**3 - 1.0) * 1000.0


def _instance_to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    inst_id = (arr[:, :, 2].astype(np.uint32) << 16
               | arr[:, :, 1].astype(np.uint32) << 8
               | arr[:, :, 0].astype(np.uint32))
    out = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    out[:, :, 0] = ((inst_id * 37) % 255).astype(np.uint8)
    out[:, :, 1] = ((inst_id * 67) % 255).astype(np.uint8)
    out[:, :, 2] = ((inst_id * 97) % 255).astype(np.uint8)
    out[inst_id == 0] = 0
    return out


def _lidar_to_points(lidar_data) -> Optional[np.ndarray]:
    if lidar_data is None:
        return None
    return np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()


def _draw_bboxes_on(img: Optional[np.ndarray], bboxes, stale: int) -> Optional[np.ndarray]:
    if img is None or not bboxes:
        return img
    img = img.copy()
    for b in bboxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        tid = int(b["track_id"])
        is_tgt = bool(b.get("is_target", False))
        if is_tgt:
            color = (0, 220, 0) if stale == 0 else (0, 140, 0)
        else:
            color = (255, 215, 0) if stale == 0 else (160, 130, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ID {tid}{' (TGT)' if is_tgt else ''}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 8, y1 + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main() -> None:
    args = build_parser()
    # Resolve walker count / speed range from the difficulty preset unless the
    # user overrides them explicitly on the CLI.
    _preset = DIFFICULTY_PRESETS[args.difficulty]
    speed_overridden = args.npc_min_speed is not None or args.npc_max_speed is not None
    if args.num_walkers is None:
        args.num_walkers = int(_preset["num_walkers"])
    if args.npc_min_speed is None:
        args.npc_min_speed = float(_preset["min_speed"])
    if args.npc_max_speed is None:
        args.npc_max_speed = float(_preset["max_speed"])
    if args.teleop and args.no_ui:
        raise SystemExit("--teleop requires the pygame UI; remove --no-ui.")
    if args.teleop and not HAS_PYGAME:
        raise SystemExit("--teleop requires pygame, but pygame could not be imported.")
    if args.teleop and args.use_perception:
        raise SystemExit("--teleop is incompatible with --use-perception.")
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be at least 1.")
    if args.num_episodes > 1 and not args.teleop:
        raise SystemExit("--num-episodes > 1 is supported only with --teleop.")
    if args.num_episodes > 1 and args.eval_trial_id is not None:
        raise SystemExit("--eval-trial-id cannot be fixed when --num-episodes > 1.")
    if args.sim_speed <= 0.0:
        raise SystemExit("--sim-speed must be greater than zero.")
    if args.teleop and args.async_mode:
        raise SystemExit("--teleop time scaling requires synchronous mode.")
    eval_args = argparse.Namespace(**vars(args))
    if args.teleop:
        eval_args.planner = "teleop"

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_stamp)
    os.makedirs(out_dir, exist_ok=True)
    gt_path   = os.path.join(out_dir, "npc_gt_stream.jsonl")
    vis_path  = os.path.join(out_dir, "visibility_stream.jsonl")
    calib_path = os.path.join(out_dir, "robot_sensor_calibration.json")

    client = carla.Client(args.host, args.port)
    # 20 s default times out under N-way parallel boot (eval_multi_gpu.sh)
    # because every CARLA simultaneously reads Town10HD_Opt from the same
    # SSD. 60 s leaves ample slack on a single-instance run too.
    client.set_timeout(float(os.environ.get("CARLA_CLIENT_TIMEOUT_S", "300")))
    world = connect_world(client, args.map_name, args.load_map)

    orig_settings  = world.get_settings()
    requested_async = bool(args.async_mode)
    use_sync_mode  = (not requested_async) or bool(args.async_after_spawn)
    apply_world_mode(world, orig_settings, sync_mode=use_sync_mode, dt=args.dt)
    print(f"[EP] world_mode={'SYNC' if use_sync_mode else 'ASYNC'} dt={args.dt}")

    npc = PassageNpcRuntime(
        world=world,
        routes_json=args.routes_json,
        grid_npz=args.grid_npz,
        flow_points_json=args.flow_points_json,
        target_track_id=args.target_track_id,
        npc_min_speed=args.npc_min_speed,
        npc_max_speed=args.npc_max_speed,
        spawn_z=args.robot_spawn_z,
        seed=args.npc_rng_seed,
    )
    robot_cls = G1RobotRuntime if args.robot_type == "g1" else ScoutRobotRuntime
    robot = robot_cls(world=world, dt=args.dt)
    robot.rescue_lift_z = float(args.robot_rescue_lift_z)
    policy_kwargs = {
        "desired_distance": float(args.desired_distance),
        "follow_position": str(args.follow_position),
    }
    if args.use_perception:
        perception_kwargs = dict(
            reid_mode=args.reid_mode,
            reid_kpr_config=args.reid_kpr_config,
            reid_device=args.reid_device,
            lost_policy=args.lost_policy,
            kinematics_mode=args.kinematics_mode,
            kf_pos_sigma=args.kf_pos_sigma,
            kf_vel_sigma_q=args.kf_vel_sigma_q,
            planner_target_state="stable_route_heading",
            route_segments=load_passage_route_segments(args.routes_json),
        )
    else:
        perception_kwargs = None

    if args.planner == "custom":
        from followbench.interfaces.policy_factory import build_custom_policy  # noqa: PLC0415
        policy = build_custom_policy(args, scene="passage")
    elif args.planner == "dwa_traj":
        if perception_kwargs is not None:
            from baselines.adapters.dwa_traj_perception_adapter import DwaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            policy = DwaTrajPerceptionFollowerPolicy(
                dt=args.dt,
                **policy_kwargs,
                **perception_kwargs,
                route_heading_max_distance_m=1.0,
                enable_spacing_reverse=True,
                visibility_aware_reverse=True,
                perception_goal_behind_deadband=0.45,
                goal_behind_confirm_steps=3,
                startup_guard_steps=max(1, int(round(0.75 / max(float(args.dt), 1e-3)))),
                spacing_reverse_min_speed=0.08,
                spacing_reverse_max_speed=0.22,
                spacing_reverse_max_yawrate=0.35,
                severe_goal_behind_m=0.80,
            )
        else:
            from baselines.adapters.dwa_traj_adapter import DwaTrajFollowerPolicy  # noqa: PLC0415
            policy = DwaTrajFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "dwa_traj_depth_tpt":
        from baselines.adapters.dwa_traj_depth_tpt_adapter import DwaTrajDepthTptFollowerPolicy  # noqa: PLC0415
        policy = DwaTrajDepthTptFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "sfm":
        if perception_kwargs is not None:
            from baselines.adapters.sfm_perception_adapter import SfmPerceptionFollowerPolicy  # noqa: PLC0415
            policy = SfmPerceptionFollowerPolicy(
                dt=args.dt,
                **policy_kwargs,
                **perception_kwargs,
                route_heading_max_distance_m=1.0,
                startup_goal_behind_guard=True,
                startup_guard_steps=max(1, int(round(0.75 / max(float(args.dt), 1e-3)))),
                goal_behind_deadband=0.45,
                side_follow_goal_smoothing_alpha=(
                    0.5 if args.follow_position in {"left_side", "right_side"} else None
                ),
                side_follow_startup_guard_steps=(
                    max(1, int(round(0.75 / max(float(args.dt), 1e-3))))
                    if args.follow_position in {"left_side", "right_side"}
                    else 0
                ),
                side_follow_goal_bearing_limit_deg=(
                    50.0 if args.follow_position in {"left_side", "right_side"} else None
                ),
                side_follow_goal_jump_limit_m=(
                    0.20 if args.follow_position in {"left_side", "right_side"} else None
                ),
            )
        else:
            from baselines.adapters.sfm_adapter import SfmFollowerPolicy  # noqa: PLC0415
            policy = SfmFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda":
        if perception_kwargs is not None:
            from baselines.adapters.rda_perception_adapter import RdaPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.rda_adapter import RdaFollowerPolicy  # noqa: PLC0415
            policy = RdaFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda_lidar":
        if perception_kwargs is not None:
            from baselines.adapters.rda_lidar_perception_adapter import RdaLidarPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaLidarPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.rda_lidar_adapter import RdaLidarFollowerPolicy  # noqa: PLC0415
            policy = RdaLidarFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "gt_planner":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner gt_planner "
                "(gt_planner already consumes full GT + lidar)."
            )
        from baselines.adapters.gt_planner_adapter import GtPlannerFollowerPolicy  # noqa: PLC0415
        policy = GtPlannerFollowerPolicy(dt=args.dt, **policy_kwargs, grid_npz=args.grid_npz)
    elif args.planner == "rda_traj":
        traj_kwargs = dict(
            use_lidar_obstacles=not bool(args.rda_traj_disable_lidar_obstacles),
        )
        if perception_kwargs is not None:
            from baselines.adapters.rda_traj_perception_adapter import RdaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaTrajPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs, **traj_kwargs)
        else:
            from baselines.adapters.rda_traj_adapter import RdaTrajFollowerPolicy  # noqa: PLC0415
            policy = RdaTrajFollowerPolicy(dt=args.dt, **policy_kwargs, **traj_kwargs)
    elif args.planner == "rda_search":
        search_kwargs = dict(
            gt_target_always_visible=not bool(args.rda_search_use_camera_visibility),
            protect_target_for_side_follow=not bool(args.rda_search_disable_target_protection),
        )
        if perception_kwargs is not None:
            from baselines.adapters.rda_search_perception_adapter import RdaSearchPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaSearchPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs, **search_kwargs)
        else:
            from baselines.adapters.rda_search_adapter import RdaSearchFollowerPolicy  # noqa: PLC0415
            policy = RdaSearchFollowerPolicy(dt=args.dt, **policy_kwargs, **search_kwargs)
    elif args.planner == "rda_depth_tpt":
        from baselines.adapters.rda_depth_tpt_adapter import RdaDepthTptFollowerPolicy  # noqa: PLC0415
        policy = RdaDepthTptFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "bso_hfc":
        if perception_kwargs is not None:
            from baselines.adapters.bso_hfc_perception_adapter import BsoHfcPerceptionFollowerPolicy  # noqa: PLC0415
            policy = BsoHfcPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.bso_hfc_adapter import BsoHfcFollowerPolicy  # noqa: PLC0415
            policy = BsoHfcFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "trackvla":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner trackvla "
                "(trackvla is itself an end-to-end vision-language-action policy)."
            )
        from baselines.adapters.trackvla_adapter import TrackVlaFollowerPolicy  # noqa: PLC0415
        policy = TrackVlaFollowerPolicy(
            dt=args.dt,
            instruction=args.trackvla_instruction,
            ckpt_path=args.trackvla_ckpt,
            render_overlay=bool(args.debug),
            auto_caption=bool(args.trackvla_auto_caption),
            caption_ckpt_path=args.trackvla_caption_ckpt,
            caption_device=args.trackvla_caption_device,
        )
    elif args.planner == "oa_vat":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner oa_vat "
                "(oa_vat carries its own YOLOe + ORTrack + DINOv3 perception)."
            )
        from baselines.adapters.oa_vat_adapter import OaVatFollowerPolicy  # noqa: PLC0415
        policy = OaVatFollowerPolicy(
            dt=args.dt,
            text_query=args.oa_vat_text_query,
            kf_mode=args.oa_vat_kf_mode,
            online_enhance=args.oa_vat_online_enhance,
            dinov3_model=args.oa_vat_dinov3,
            match_thresh=args.oa_vat_match_thresh,
            reacq_match_thresh=args.oa_vat_reacq_thresh,
            yolo_conf_thresh=args.oa_vat_yolo_conf,
            target_area_ratio=args.oa_vat_target_area_ratio,
        )
    else:  # pid (default)
        if perception_kwargs is not None:
            from baselines.adapters.pid_perception_adapter import PidPerceptionFollowerPolicy  # noqa: PLC0415
            policy = PidPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.pid_adapter import PIDFollowerPolicy  # noqa: PLC0415
            policy = PIDFollowerPolicy(dt=args.dt, **policy_kwargs)

    vis_checker = InstanceVisibilityChecker(threshold=args.visibility_threshold)
    eval_logger = EvaluationLogger.from_args(
        eval_args, scenario_type="passage"
    ) if args.enable_evaluation else None
    collision_monitor = None
    map_obstacle_monitor = None
    identity_monitor = TargetIdentityMonitor(
        enabled=bool(eval_logger is not None and args.use_perception),
        associate_gate_m=float(args.target_identity_associate_gate_m),
        wrong_dist_m=float(args.target_identity_wrong_dist_m),
        startup_grace_sec=float(args.target_identity_startup_grace_sec),
        confirm_sec=float(args.target_identity_confirm_sec),
        fail_sec=float(args.target_identity_fail_sec),
    )
    target_distance_monitor = TargetDistanceMonitor(
        warmup_sec=float(args.robot_target_far_warmup_sec),
        fail_dist_m=float(args.robot_target_far_fail_dist_m),
        fail_sec=float(args.robot_target_far_fail_sec),
        fail_steps=int(getattr(args, "robot_target_far_fail_steps", 0)),
    )
    robot_stuck_monitor = RobotStuckMonitor(
        window_sec=float(args.robot_stuck_window_sec),
        grace_sec=float(args.robot_stuck_grace_sec),
        min_progress_m=float(args.robot_stuck_min_progress_m),
        target_wait_speed_mps=float(args.robot_stuck_target_wait_speed_mps),
        target_escape_distance_m=float(args.robot_stuck_target_escape_distance_m),
        static_obstacle_clearance_m=float(args.robot_stuck_static_obstacle_clearance_m),
    )
    termination_reason = None
    episode_success: Optional[bool] = None
    if eval_logger is not None:
        print(f"[EVAL] run_dir={eval_logger.run_dir}")
    planning_video: Dict[str, Optional[object]] = {"rec": None}
    if args.npc_rng_seed is not None:
        print(f"[EP] npc_rng_seed={args.npc_rng_seed}")
    dbg_vis = None
    if args.debug:
        from followbench.visualization.debug_visualizer import DebugVisualizer
        dbg_vis = DebugVisualizer(half_size=30.0)

    _LIVE_VIEWER_PATH = module_path("visualization", "live_viewer.py")
    _debug_proc: Optional[subprocess.Popen] = None

    spawned   = False
    following = False
    paused    = False
    paused_at = 0.0
    active_target_track_id = args.target_track_id
    running   = True
    tick      = 0
    completed_episodes = 0
    # Defaults were resolved from the difficulty preset in main().
    desired_num_walkers = int(args.num_walkers)
    desired_min_speed   = float(args.npc_min_speed)
    desired_max_speed   = float(args.npc_max_speed)

    ui_enabled = (not args.no_ui) and HAS_PYGAME
    if ui_enabled:
        pygame.init()
        panel_w, panel_h, hud_h = 400, 260, 170
        pygame.display.set_mode((panel_w * 3, hud_h + panel_h * 2))
        title = "Passage Teleoperation" if args.teleop else "Passage"
        pygame.display.set_caption(
            f"{title} | G spawn | F start/stop | WASD drive | R reset | Q quit")
        ui_font = pygame.font.SysFont("monospace", 18)
        latest_panels = {k: np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                         for k in ("rgb_left", "rgb", "rgb_right", "depth", "instance", "lidar")}
    elif not args.no_ui:
        print("[WARN] pygame not available, key UI disabled.")

    def on_signal(_sig, _frm):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_signal)

    gt_f  = open(gt_path,  "w", encoding="utf-8")
    vis_f = open(vis_path, "w", encoding="utf-8")

    print("[EP] ready. Press G to spawn, then F to follow.")
    print(f"[EP] routes={args.routes_json}")

    def do_spawn() -> None:
        nonlocal spawned, use_sync_mode, collision_monitor, map_obstacle_monitor, termination_reason, episode_success
        if spawned:
            return
        identity_monitor.reset()
        target_distance_monitor.reset()
        robot_stuck_monitor.reset()
        termination_reason = None
        episode_success = None
        for a in world.get_actors():
            try:
                is_scout = "vehicle.scout" in a.type_id
                is_g1 = "walker.pedestrian.g1" in a.type_id
                is_known = a.attributes.get("role_name", "") in {
                    "followbench_robot", "sfm_follow_robot", "scout_follow"}
                if is_scout or is_g1 or is_known:
                    a.destroy()
            except Exception:
                pass
        npc.spawn(desired_num_walkers,
                  min_speed=desired_min_speed, max_speed=desired_max_speed)
        if not npc.agents:
            raise RuntimeError("No NPC spawned.")
        step_world(world, use_sync_mode)
        tgt_init = npc.get_target_by_track_id(active_target_track_id)
        tgt_actor = npc.get_actor_by_track_id(active_target_track_id)
        if tgt_actor is None or tgt_init is None:
            tgt_init = npc.get_states()[0]
            print(f"[EP][WARN] target {active_target_track_id} missing, fallback to {tgt_init.track_id}")
            tgt_actor = npc.get_actor_by_track_id(tgt_init.track_id)

        robot_tf = resolve_robot_spawn_tf(args, npc=npc,
                                          target_track_id=active_target_track_id)
        print(f"[EP] robot_spawn at=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f})")
        _spawn_forbidden_xys = [(s.x, s.y) for s in npc.get_states()]
        try:
            robot.spawn(robot_tf, roi_map=npc.roi_map,
                        forbidden_xys=_spawn_forbidden_xys, min_clearance_m=0.7)
        except RuntimeError:
            # Initial location may be invalid for vehicle spawn (e.g. flow-point area is
            # pedestrian-only). Fall back to random ROI samples.
            spawned_robot = False
            for _ in range(60):
                sample_loc = npc.roi_map.sample_world_location(
                    npc.rng, z=robot_tf.location.z, jitter=True)
                retry_tf = carla.Transform(sample_loc, robot_tf.rotation)
                try:
                    robot.spawn(retry_tf, roi_map=npc.roi_map,
                                forbidden_xys=_spawn_forbidden_xys, min_clearance_m=0.7)
                    spawned_robot = True
                    print(f"[EP] robot_spawn fallback at="
                          f"({retry_tf.location.x:.2f},{retry_tf.location.y:.2f})")
                    break
                except RuntimeError:
                    continue
            if not spawned_robot:
                raise RuntimeError("[EP] robot spawn failed after all retries")
        step_world(world, use_sync_mode)

        # Hard correction if robot ended up too far from target OR uncomfortably
        # close (robot+target capsule radii ~ 0.7m: anything below that means
        # geometry overlap on the very first follow tick → target_collision).
        # Also re-check after a few ticks so the target has time to settle.
        if tgt_init is not None and robot.actor is not None and tgt_actor is not None:
            _far_thresh = float(args.desired_distance) + 1.5
            _too_close_thresh = 0.7
            for _attempt in range(2):
                rloc = robot.actor.get_location()
                ttf = tgt_actor.get_transform()
                tloc = ttf.location
                dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
                if not (dist0 > _far_thresh or dist0 < _too_close_thresh):
                    break
                tvel = tgt_actor.get_velocity()
                target_yaw_rad = yaw_rad_from_velocity(
                    float(tvel.x), float(tvel.y), float(ttf.rotation.yaw)
                )
                _correction_other_xys = [(s.x, s.y) for s in npc.get_states() if s.track_id != active_target_track_id]
                # Mirror resolve_robot_spawn_tf: forbid placing on top of target.
                _correction_other_xys.append((float(tloc.x), float(tloc.y)))
                new_x, new_y, _ok = follow_position_spawn_xy(
                    float(tloc.x), float(tloc.y), target_yaw_rad,
                    str(args.follow_position), float(args.desired_distance),
                    npc.roi_map, forbidden_xys=_correction_other_xys,
                )
                why = "too_far" if dist0 > _far_thresh else "too_close"
                print(f"[EP] spawn correction (attempt {_attempt+1}, {why}): "
                      f"dist0={dist0:.2f}m -> ({new_x:.2f},{new_y:.2f}) ok={_ok}")
                fix_tf = carla.Transform(
                    carla.Location(x=new_x, y=new_y, z=robot_tf.location.z),
                    carla.Rotation(yaw=float(np.degrees(target_yaw_rad))),
                )
                robot.force_place(fix_tf)
                step_world(world, use_sync_mode)

        robot.spawn_sensors(
            image_w=args.sensor_image_w, image_h=args.sensor_image_h,
            cam_fov=args.sensor_fov_deg, lidar_range=args.sensor_lidar_range,
            lidar_scan_height_agl=args.sensor_lidar_height_agl,
            enable_side_views=True,
        )
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(robot.get_calibration(), f, indent=2)
        if eval_logger is not None:
            eval_logger.write_meta(
                eval_args,
                robot_actor=robot.actor,
                target_actor=tgt_actor,
                extras={
                    "difficulty": current_difficulty,
                    "weather": current_weather,
                    "npc_rng_seed": args.npc_rng_seed,
                    "num_walkers": len(npc.agents),
                    "max_concurrent": int(npc._max_concurrent),
                    "npc_speed_range": [float(npc._min_speed), float(npc._max_speed)],
                    "lookalike_track_ids": list(npc.lookalike_track_ids),
                    "difficulty_history": list(difficulty_history),
                    "weather_history": list(weather_history),
                    "termination_policy": {
                        "target_identity_startup_grace_sec": float(args.target_identity_startup_grace_sec),
                        "target_identity_confirm_sec": float(args.target_identity_confirm_sec),
                        "target_identity_fail_sec": float(args.target_identity_fail_sec),
                        "robot_target_far_fail_dist_m": float(args.robot_target_far_fail_dist_m),
                        "robot_target_far_fail_sec": float(args.robot_target_far_fail_sec),
                        "robot_target_far_warmup_sec": float(args.robot_target_far_warmup_sec),
                        "robot_stuck_window_sec": float(args.robot_stuck_window_sec),
                        "robot_stuck_grace_sec": float(args.robot_stuck_grace_sec),
                        "robot_stuck_min_progress_m": float(args.robot_stuck_min_progress_m),
                        "robot_stuck_target_wait_speed_mps": float(args.robot_stuck_target_wait_speed_mps),
                        "robot_stuck_target_escape_distance_m": float(args.robot_stuck_target_escape_distance_m),
                        "robot_stuck_static_obstacle_clearance_m": float(args.robot_stuck_static_obstacle_clearance_m),
                        "map_obstacle_overlap_confirm_steps": int(args.map_obstacle_overlap_confirm_steps),
                    },
                    "robot_type": args.robot_type,
                    "sim_speed": args.sim_speed,
                    **({
                        "control_mode": "human_keyboard",
                        "operator_id": args.operator_id,
                        "session_episode": completed_episodes + 1,
                        "session_num_episodes": args.num_episodes,
                    } if args.teleop else {}),
                },
            )
            collision_monitor = PassageCollisionMonitor(
                world,
                robot.actor,
                enabled=not bool(args.disable_collision_sensor),
            )
            if robot.actor is not None:
                extent = robot.actor.bounding_box.extent
                map_obstacle_monitor = PassageMapObstacleMonitor(
                    grid_npz=args.grid_npz,
                    robot_radius_m=max(float(extent.x), float(extent.y)),
                    overlap_confirm_steps=int(args.map_obstacle_overlap_confirm_steps),
                    clearance_probe_m=float(args.map_obstacle_clearance_probe_m),
                )
        npc.set_motion_active(False)
        spawned = True
        print(f"[EP] spawned walkers={desired_num_walkers}")
        if dbg_vis is not None:
            rs = robot.get_state()
            if rs is not None:
                dbg_vis.start(rs.x, rs.y, args.grid_npz)
        try:
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=-42.0, y=155.0, z=12.0),
                carla.Rotation(pitch=-50.0, yaw=90.0, roll=0.0)))
        except Exception:
            pass
        if requested_async and args.async_after_spawn and use_sync_mode:
            apply_world_mode(world, orig_settings, sync_mode=False, dt=args.dt)
            use_sync_mode = False
            print("[EP] world_mode switched to ASYNC after spawn.")

    manual_mode        = "none"
    _cam_yaw           = 0.0
    _cam_pitch         = -20.0
    _DIFFICULTIES      = ["easy", "hard", "boss", "legend"]
    current_difficulty = args.difficulty
    npc.set_difficulty(current_difficulty)

    # Weather is drawn from a stream derived from the episode seed (paired across
    # policies) with difficulty-dependent weights (boss/legend biased toward
    # dusk/rain/fog).  Without a seed, a fresh unseeded stream is used.
    _weather_rng = weather_mod.seeded_rng(args.npc_rng_seed)

    def _select_and_apply_weather() -> str:
        if args.weather == "random":
            name = weather_mod.sample_for_difficulty(current_difficulty, _weather_rng)
        else:
            name = args.weather
        weather_mod.apply(world, name)
        print(f"[EP] weather → {name.upper()} ({weather_mod.WEATHER_LABELS_CN[name]})")
        return name

    current_weather: str = _select_and_apply_weather()
    weather_history: List[Dict] = [{"tick": 0, "weather": current_weather}]
    difficulty_history: List[Dict] = [{"tick": 0, "difficulty": current_difficulty}]

    def do_reset(*, announce: bool = True) -> None:
        nonlocal spawned, following, tick, manual_mode, collision_monitor, map_obstacle_monitor
        nonlocal termination_reason, episode_success, episode_start_t, episode_start_tick
        if spawned:
            npc.destroy()
            robot.destroy()
        if collision_monitor is not None:
            collision_monitor.destroy()
            collision_monitor = None
        map_obstacle_monitor = None
        identity_monitor.reset()
        target_distance_monitor.reset()
        robot_stuck_monitor.reset()
        termination_reason = None
        episode_success = None
        if dbg_vis is not None:
            dbg_vis.stop()
        spawned   = False
        following = False
        episode_start_t = None
        episode_start_tick = 0
        tick      = 0
        manual_mode = "none"
        if ui_enabled:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        vis_checker.reset()
        if announce:
            print("[EP] reset. Press G to spawn.")

    def finalize_current_eval(reason: Optional[str], success: Optional[bool]) -> None:
        if planning_video["rec"] is not None:
            planning_video["rec"].close(reason, success)
            planning_video["rec"] = None
        if eval_logger is None:
            return
        result = eval_logger.finalize(reason, episode_success=success)
        if result is not None:
            print(f"[EVAL] result: {os.path.join(eval_logger.run_dir, 'eval_result.json')}")
            print(
                f"[EVAL] obstacle_avoidance_success={result.get('obstacle_avoidance_success')} "
                f"search_success={result.get('search_success')} "
                f"TVR={result.get('target_visibility_ratio')}"
            )

    def finish_episode(reason: str, success: bool) -> None:
        nonlocal completed_episodes, eval_logger, running
        nonlocal auto_spawn_done, auto_follow_done
        nonlocal current_weather, weather_history
        finalize_current_eval(reason, success)
        completed_episodes += 1
        print(f"[EP] completed episode {completed_episodes}/{args.num_episodes} reason={reason}")
        if completed_episodes >= args.num_episodes:
            running = False
            return
        do_reset(announce=False)
        current_weather = _select_and_apply_weather()
        weather_history = [{"tick": 0, "weather": current_weather}]
        auto_spawn_done = False
        auto_follow_done = False
        eval_logger = EvaluationLogger.from_args(
            eval_args, scenario_type="passage"
        ) if args.enable_evaluation else None
        if eval_logger is not None:
            print(f"[EVAL] next run_dir={eval_logger.run_dir}")
        print(
            f"[EP] ready for episode {completed_episodes + 1}/{args.num_episodes}. "
            "Press G to spawn, then F to start."
        )

    def do_toggle_manual_target() -> None:
        nonlocal manual_mode, _cam_yaw, _cam_pitch
        if not spawned:
            return
        if manual_mode == "target":
            manual_mode = "none"
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            print("[EP] manual target OFF")
        else:
            manual_mode = "target"
            ta = npc.get_actor_by_track_id(active_target_track_id)
            if ta is not None:
                _cam_yaw = float(ta.get_transform().rotation.yaw)
            _cam_pitch = -20.0
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()
            print(f"[EP] manual target ON ({active_target_track_id}) | WASD  Mouse  H release")

    def do_toggle_manual_robot() -> None:
        nonlocal manual_mode, _cam_yaw, _cam_pitch
        if not spawned:
            return
        if manual_mode == "robot":
            manual_mode = "none"
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            print("[EP] manual robot OFF")
        else:
            manual_mode = "robot"
            if robot.actor is not None:
                _cam_yaw = float(robot.actor.get_transform().rotation.yaw)
            _cam_pitch = -20.0
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
            pygame.mouse.get_rel()
            print("[EP] manual robot ON | WASD  Mouse  J release")

    def do_toggle_follow() -> None:
        nonlocal following, episode_start_t, episode_start_tick, start_t
        if not spawned:
            return
        following = not following
        npc.set_motion_active(following)
        policy.reset()
        vis_checker.reset()
        if following and episode_start_t is None:
            episode_start_t = sim_clock()
            episode_start_tick = tick
            if args.teleop:
                start_t = episode_start_t
            print(f"[EP] episode START  target={active_target_track_id}")
        if args.teleop:
            if following and manual_mode != "robot":
                do_toggle_manual_robot()
            elif not following and manual_mode == "robot":
                do_toggle_manual_robot()
                robot.hold_still()
        print(f"[EP] following={'ON' if following else 'OFF'} target={active_target_track_id}")

    # Sim clock: seconds of simulated time (world steps * dt). Used instead of
    # wall clock so auto-spawn/follow timing and duration are reproducible.
    sim_steps = 0

    def sim_clock() -> float:
        return sim_steps * float(args.dt)

    start_t          = sim_clock()
    episode_start_t: Optional[float] = None
    episode_start_tick = 0
    auto_spawn_done  = False
    auto_follow_done = False
    rgb_intrinsics_cached       = None
    rgb_extr_cached             = None
    rgb_intrinsics_left_cached  = None
    rgb_extr_left_cached        = None
    rgb_intrinsics_right_cached = None
    rgb_extr_right_cached       = None
    lidar_extr_cached     = None
    last_rgb_np         = None
    last_lidar_np       = None
    last_depth_np       = None
    last_rgb_left_np    = None
    last_rgb_right_np   = None
    last_depth_left_np  = None
    last_depth_right_np = None

    sim_pacer = SimulationPacer(args.dt, args.sim_speed, enabled=args.teleop)

    def step_main_loop(frame_started_at: float) -> None:
        nonlocal sim_steps
        step_world(world, use_sync_mode)
        sim_steps += 1
        sim_pacer.end_frame(frame_started_at)

    try:
        while running:
            frame_started_at = sim_pacer.begin_frame()
            now_t   = sim_clock()
            elapsed = now_t - start_t
            episode_end_reason = None
            episode_end_success = False
            duration_active = (not args.teleop) or episode_start_t is not None
            duration_elapsed = (
                max(tick - episode_start_tick, 0) * args.dt
                if args.teleop else elapsed
            )
            if duration_active and duration_elapsed >= args.duration_sec:
                if args.teleop:
                    finish_episode(termination_reason or "duration_limit", False)
                    if running:
                        step_main_loop(frame_started_at)
                        continue
                break

            if args.auto_spawn_sec >= 0.0 and not auto_spawn_done and elapsed >= args.auto_spawn_sec:
                do_spawn(); auto_spawn_done = True
            if args.auto_follow_sec >= 0.0 and not auto_follow_done and elapsed >= args.auto_follow_sec:
                if not spawned:
                    do_spawn(); auto_spawn_done = True
                if not following:
                    do_toggle_follow()
                auto_follow_done = True

            if ui_enabled:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        running = False
                    elif e.type == pygame.MOUSEMOTION and manual_mode != "none":
                        _cam_yaw   = (_cam_yaw + e.rel[0] * MANUAL_MOUSE_SENS) % 360.0
                        _cam_pitch = float(np.clip(
                            _cam_pitch - e.rel[1] * MANUAL_MOUSE_SENS * 0.4, -60.0, 10.0))
                    elif e.type == pygame.KEYDOWN:
                        k = e.key
                        if k == pygame.K_q:
                            running = False
                        elif k == pygame.K_ESCAPE and manual_mode != "none":
                            if args.teleop:
                                print("[EP] teleop is locked; press F to pause")
                            else:
                                manual_mode = "none"
                                pygame.mouse.set_visible(True)
                                pygame.event.set_grab(False)
                        elif k == pygame.K_SPACE and spawned:
                            paused = not paused
                            paused_at = now_t
                        elif k == pygame.K_h and spawned and not args.teleop:
                            do_toggle_manual_target()
                        elif k == pygame.K_j and spawned and not args.teleop:
                            do_toggle_manual_robot()
                        elif k == pygame.K_g and not spawned:
                            do_spawn()
                        elif k == pygame.K_r:
                            do_reset()
                        elif k == pygame.K_f and spawned:
                            do_toggle_follow()
                        elif k == pygame.K_v:
                            if _debug_proc is None or _debug_proc.poll() is not None:
                                _debug_proc = subprocess.Popen([
                                    sys.executable, _LIVE_VIEWER_PATH,
                                    "--npz", args.grid_npz, "--half", "20",
                                ])
                                print("[EP] NPC live viewer: ON  (V to close)")
                            else:
                                _debug_proc.terminate()
                                _debug_proc = None
                                print("[EP] NPC live viewer: OFF")
                        elif k == pygame.K_t and spawned:
                            states = npc.get_states()
                            if states:
                                ids = [s.track_id for s in states]
                                if active_target_track_id not in ids:
                                    active_target_track_id = ids[0]
                                else:
                                    idx = ids.index(active_target_track_id)
                                    active_target_track_id = ids[(idx + 1) % len(ids)]
                                print(f"[EP] target → {active_target_track_id}")
                        # Teleop fixes the difficulty at process launch.  D is
                        # always steer-right there, including while paused.
                        elif k == pygame.K_d and manual_mode == "none" and not args.teleop:
                            idx = _DIFFICULTIES.index(current_difficulty)
                            current_difficulty = _DIFFICULTIES[(idx + 1) % len(_DIFFICULTIES)]
                            npc.set_difficulty(current_difficulty)
                            if not speed_overridden:
                                desired_min_speed = float(npc._min_speed)
                                desired_max_speed = float(npc._max_speed)
                            current_weather = _select_and_apply_weather()
                            difficulty_history.append({"tick": tick, "difficulty": current_difficulty})
                            weather_history.append({"tick": tick, "weather": current_weather})
                            if eval_logger is not None:
                                eval_logger.update_extras({
                                    "difficulty": current_difficulty,
                                    "weather": current_weather,
                                    "difficulty_history": list(difficulty_history),
                                    "weather_history": list(weather_history),
                                })
                            print(f"[EP] difficulty → {current_difficulty.upper()}  weather → {current_weather}")
                        elif k == pygame.K_UP and not args.teleop:
                            desired_num_walkers = min(40, desired_num_walkers + 2)
                            print(f"[EP] walkers → {desired_num_walkers}")
                        elif k == pygame.K_DOWN and not args.teleop:
                            desired_num_walkers = max(2, desired_num_walkers - 2)
                            print(f"[EP] walkers → {desired_num_walkers}")
                        elif k == pygame.K_LEFTBRACKET and not args.teleop:
                            desired_min_speed = max(0.3, desired_min_speed - 0.1)
                            print(f"[EP] min_speed → {desired_min_speed:.2f}")
                        elif k == pygame.K_RIGHTBRACKET and not args.teleop:
                            desired_min_speed = min(desired_max_speed, desired_min_speed + 0.1)
                            print(f"[EP] min_speed → {desired_min_speed:.2f}")
                        elif k == pygame.K_MINUS and not args.teleop:
                            desired_max_speed = max(desired_min_speed, desired_max_speed - 0.1)
                            print(f"[EP] max_speed → {desired_max_speed:.2f}")
                        elif k == pygame.K_EQUALS and not args.teleop:
                            desired_max_speed = min(2.5, desired_max_speed + 0.1)
                            print(f"[EP] max_speed → {desired_max_speed:.2f}")

            if not spawned and ui_enabled:
                surf = pygame.display.get_surface()
                if surf is not None:
                    surf.fill((18, 22, 28))
                    for i, text in enumerate([
                        (f"State: spawned={spawned} follow={following} "
                         f"| walkers={desired_num_walkers} "
                         f"speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"),
                        (f"Difficulty: {current_difficulty.upper()}  "
                         f"({'LOCKED BY LAUNCH' if args.teleop else 'D to cycle: easy→hard→boss→legend'})"),
                        "Keys: G spawn  F follow  T target  H manual-NPC  J robot  R reset  SPACE pause  Q quit",
                        "Tune: Up/Down walkers | [ ] min_speed | - / = max_speed  (reset+G to apply)",
                    ]):
                        surf.blit(ui_font.render(text, True, (230, 235, 240)), (16, 16 + i * 34))
                    pygame.display.flip()

            if paused:
                if spawned:
                    npc.step(dt=args.dt, robot_actor=robot.actor,
                             target_track_id=active_target_track_id)
                    robot.hold_still()
                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                        overlay.fill((0, 0, 0, 140))
                        surf.blit(overlay, (0, 0))
                        t = ui_font.render("PAUSED — SPACE to resume", True, (255, 220, 0))
                        surf.blit(t, ((surf.get_width() - t.get_width()) // 2,
                                      (surf.get_height() - t.get_height()) // 2))
                        pygame.display.flip()
                step_main_loop(frame_started_at)
                continue

            if spawned:
                npc.step(dt=args.dt, robot_actor=robot.actor,
                         target_track_id=active_target_track_id)
                npc_states    = npc.get_states()
                target_state  = next((s for s in npc_states
                                      if s.track_id == active_target_track_id), None)
                target_actor  = npc.get_actor_by_track_id(active_target_track_id)
                robot_state   = robot.get_state()
                v_cmd = None
                w_cmd = None
                planner_cost_ms = None
                policy_debug_info = {}

                if manual_mode == "target" and ui_enabled:
                    key_state = pygame.key.get_pressed()
                    _apply_manual_target_control(target_actor, key_state, cam_yaw=_cam_yaw)
                    if target_actor is not None:
                        _update_manual_spectator(world.get_spectator(), target_actor,
                                                 _cam_yaw, _cam_pitch)
                elif manual_mode == "robot" and ui_enabled:
                    key_state = pygame.key.get_pressed()
                    v_cmd, w_cmd = _apply_manual_robot_control(robot, key_state)
                    if args.teleop:
                        policy_debug_info = {"mode": "teleop"}
                    if robot.actor is not None:
                        _update_manual_spectator(world.get_spectator(), robot.actor,
                                                 _cam_yaw, _cam_pitch)

                visible, pix_count = False, 0
                inst_img       = robot.get_sensor_data("instance")
                inst_img_left  = robot.get_sensor_data("instance_left")
                inst_img_right = robot.get_sensor_data("instance_right")
                rgb_img        = robot.get_sensor_data("rgb")
                depth_img      = robot.get_sensor_data("depth")
                lidar_img      = robot.get_sensor_data("lidar")
                rgb_img_left   = robot.get_sensor_data("rgb_left")
                depth_img_left = robot.get_sensor_data("depth_left")
                rgb_img_right  = robot.get_sensor_data("rgb_right")
                depth_img_right = robot.get_sensor_data("depth_right")
                if target_actor is not None and "instance" in robot.sensors:
                    visible, pix_count = vis_checker.evaluate_multi(
                        views=[
                            (inst_img, robot.sensors.get("instance")),
                            (inst_img_left, robot.sensors.get("instance_left")),
                            (inst_img_right, robot.sensors.get("instance_right")),
                        ],
                        target_actor=target_actor,
                        image_w=args.sensor_image_w,
                        image_h=args.sensor_image_h,
                        fov_deg=args.sensor_fov_deg,
                    )
                # Pull bbox overlay info from the planner (perception adapters expose it).
                _dbg_fn = getattr(policy, "get_debug_info", None)
                _dbg_info = _dbg_fn() if callable(_dbg_fn) else {}
                bboxes_by_view = _dbg_info.get("track_bboxes_by_view") or {}
                _bbox_stale = int(_dbg_info.get("track_bboxes_age", 0))
                if ui_enabled:
                    r = _to_rgb(rgb_img)
                    if r is not None:
                        r = _draw_bboxes_on(r, bboxes_by_view.get("front"), _bbox_stale)
                        latest_panels["rgb"] = cv2.resize(r, (400, 260))
                    rl = _to_rgb(rgb_img_left)
                    if rl is not None:
                        rl = _draw_bboxes_on(rl, bboxes_by_view.get("left"), _bbox_stale)
                        latest_panels["rgb_left"] = cv2.resize(rl, (400, 260))
                    rr = _to_rgb(rgb_img_right)
                    if rr is not None:
                        rr = _draw_bboxes_on(rr, bboxes_by_view.get("right"), _bbox_stale)
                        latest_panels["rgb_right"] = cv2.resize(rr, (400, 260))
                    d = _depth_to_rgb(depth_img)
                    if d is not None:
                        latest_panels["depth"] = cv2.resize(d, (400, 260))
                    ins = _instance_to_rgb(inst_img)
                    if ins is not None:
                        latest_panels["instance"] = cv2.resize(ins, (400, 260))
                    lbev = _lidar_to_bev(lidar_img)
                    if lbev is not None:
                        latest_panels["lidar"] = lbev

                if manual_mode == "robot":
                    pass  # manual WASD applied above; keep planner/hold_still from overriding it
                elif target_state is not None and following:
                    target_yaw_rad = yaw_rad_from_velocity(
                        target_state.vx, target_state.vy, fallback_deg=target_state.yaw_deg)
                    if rgb_intrinsics_cached is None:
                        calib = robot.get_calibration()
                        rgb_intrinsics_cached       = calib.get("rgb", {}).get("intrinsics")
                        rgb_extr_cached             = calib.get("rgb", {}).get("extrinsics_robot_to_sensor")
                        lidar_extr_cached           = calib.get("lidar", {}).get("extrinsics_robot_to_sensor")
                        rgb_intrinsics_left_cached  = calib.get("rgb_left", {}).get("intrinsics")
                        rgb_extr_left_cached        = calib.get("rgb_left", {}).get("extrinsics_robot_to_sensor")
                        rgb_intrinsics_right_cached = calib.get("rgb_right", {}).get("intrinsics")
                        rgb_extr_right_cached       = calib.get("rgb_right", {}).get("extrinsics_robot_to_sensor")
                    rgb_np = _to_rgb(rgb_img)
                    if rgb_np is not None:
                        last_rgb_np = rgb_np
                    lidar_np = _lidar_to_points(lidar_img)
                    if lidar_np is not None:
                        last_lidar_np = lidar_np
                    depth_np = _depth_to_meters(depth_img)
                    if depth_np is not None:
                        last_depth_np = depth_np
                    rgb_left_np = _to_rgb(rgb_img_left)
                    if rgb_left_np is not None:
                        last_rgb_left_np = rgb_left_np
                    rgb_right_np = _to_rgb(rgb_img_right)
                    if rgb_right_np is not None:
                        last_rgb_right_np = rgb_right_np
                    depth_left_np = _depth_to_meters(depth_img_left)
                    if depth_left_np is not None:
                        last_depth_left_np = depth_left_np
                    depth_right_np = _depth_to_meters(depth_img_right)
                    if depth_right_np is not None:
                        last_depth_right_np = depth_right_np
                    obs = FollowObservation(
                        tick=tick, dt=args.dt,
                        robot=robot_state, target=target_state, npcs=npc_states,
                        target_visible=visible, target_pixel_count=pix_count,
                        extras={
                            "target_yaw_rad": target_yaw_rad,
                            "target_bbox_xyxy": _compute_target_bbox_for_trackvla(
                                args, target_actor, robot, rgb_intrinsics_cached,
                            ),
                        },
                        rgb_image=last_rgb_np, depth_image=last_depth_np,
                        lidar_points=last_lidar_np,
                        rgb_intrinsics=rgb_intrinsics_cached,
                        rgb_extrinsics_robot_to_sensor=rgb_extr_cached,
                        lidar_extrinsics_robot_to_sensor=lidar_extr_cached,
                        rgb_image_left=last_rgb_left_np,
                        rgb_image_right=last_rgb_right_np,
                        depth_image_left=last_depth_left_np,
                        depth_image_right=last_depth_right_np,
                        rgb_intrinsics_left=rgb_intrinsics_left_cached,
                        rgb_intrinsics_right=rgb_intrinsics_right_cached,
                        rgb_extrinsics_left_robot_to_sensor=rgb_extr_left_cached,
                        rgb_extrinsics_right_robot_to_sensor=rgb_extr_right_cached,
                    )
                    _t_policy = time.perf_counter()
                    action = policy.act(obs)
                    planner_cost_ms = (time.perf_counter() - _t_policy) * 1000.0
                    _dbg_fn_eval = getattr(policy, "get_debug_info", None)
                    policy_debug_info = _dbg_fn_eval() if callable(_dbg_fn_eval) else {}
                    v_cmd = float(action.v_mps)
                    w_cmd = float(action.w_radps)
                    robot.apply_velocity_command(v_cmd, w_cmd)
                    # if args.draw_laser:
                    #     world.debug.draw_line(
                    #         carla.Location(x=robot_state.x, y=robot_state.y, z=robot_state.z + 1.2),
                    #         carla.Location(x=target_state.x, y=target_state.y, z=target_state.z + 1.4),
                    #         thickness=0.02, color=carla.Color(50, 10, 10),
                    #         life_time=max(args.dt * 2.0, 0.12))
                else:
                    robot.hold_still()

                eval_clearance = None
                eval_collision = False
                eval_collision_source = None
                target_identity_status = None
                target_task_done = bool(following and spawned and npc.episode_done)
                eval_termination_triggered = False
                eval_step_termination_reason = None
                if eval_logger is not None:
                    eval_time_s = now_t - start_t
                    eval_clearance = eval_logger.compute_clearance(
                        robot_state,
                        target_state,
                        npc_states,
                        active_target_track_id,
                    )
                    map_obstacle_status = (
                        map_obstacle_monitor.update(x=float(robot_state.x), y=float(robot_state.y))
                        if map_obstacle_monitor is not None
                        else {
                            "grid_overlap": False,
                            "grid_overlap_confirm_count": 0,
                            "grid_overlap_confirm_steps": 0,
                            "grid_clearance_m": None,
                            "termination_triggered": False,
                            "termination_reason": None,
                        }
                    )
                    actor_to_track = actor_track_lookup(npc_states)
                    sensor_events = (
                        enrich_sensor_collision_events(
                            collision_monitor.drain(),
                            tick=tick,
                            time_s=eval_time_s,
                            episode_active=following,
                            actor_to_track=actor_to_track,
                            target_track_id=active_target_track_id,
                        )
                        if collision_monitor is not None
                        else []
                    )
                    geometry_event = (
                        make_geometry_collision_event(
                            tick,
                            eval_time_s,
                            eval_clearance,
                            active_target_track_id,
                            margin=float(args.human_collision_margin),
                        )
                        if following
                        else None
                    )
                    eval_collision_events = list(sensor_events)
                    if following and bool(map_obstacle_status.get("termination_triggered", False)) and map_obstacle_monitor is not None:
                        eval_collision_events.append(
                            map_obstacle_monitor.make_event(
                                tick=tick,
                                time_s=eval_time_s,
                                episode_active=following,
                                status=map_obstacle_status,
                            )
                        )
                    if geometry_event is not None:
                        eval_collision_events.append(geometry_event)
                    collision_reason = None
                    collision_event = None
                    if eval_collision_events:
                        eval_logger.write_collision_events(eval_collision_events)
                        eval_collision = True
                        collision_reason, collision_event = _select_collision_termination(eval_collision_events)
                        eval_collision_source = (
                            None if collision_event is None else collision_event.get("source")
                        )
                    if following and args.use_perception:
                        target_identity_status = identity_monitor.update(
                            dt=float(args.dt),
                            eval_time_s=eval_time_s,
                            target_track_id=active_target_track_id,
                            target_state=target_state,
                            npc_states=npc_states,
                            policy_debug_info=policy_debug_info,
                        )
                        identity_loss_triggered = bool(target_identity_status.get("termination_triggered"))
                        if identity_loss_triggered and not bool(args.terminate_on_target_identity_loss):
                            target_identity_status["termination_triggered"] = False
                            target_identity_status["termination_reason"] = None
                        if identity_loss_triggered and bool(args.terminate_on_target_identity_loss):
                            target_identity_status["termination_triggered"] = True
                    target_center_dist_m = None
                    if target_state is not None:
                        target_center_dist_m = float(np.hypot(robot_state.x - target_state.x, robot_state.y - target_state.y))
                    target_far_status = target_distance_monitor.update(
                        dt=float(args.dt),
                        eval_time_s=eval_time_s,
                        following=following,
                        distance_m=target_center_dist_m,
                    )
                    target_surface_dist_m = None
                    if eval_clearance is not None:
                        target_surface_dist_m = eval_clearance.get("robot_target_surface_dist")
                        if target_surface_dist_m is not None:
                            target_surface_dist_m = float(target_surface_dist_m)
                    robot_stuck_status = robot_stuck_monitor.update(
                        eval_time_s=eval_time_s,
                        following=following,
                        robot_state=robot_state,
                        target_state=target_state,
                        target_surface_dist_m=target_surface_dist_m,
                        obstacle_clearance_m=map_obstacle_status.get("grid_clearance_m"),
                    )
                    failure_reason = None
                    if following and collision_reason is not None:
                        should_terminate_collision = (
                            collision_reason == "obstacle_collision"
                            or bool(args.terminate_on_human_collision)
                        )
                        if should_terminate_collision:
                            failure_reason = collision_reason
                    if failure_reason is None and target_identity_status is not None:
                        if bool(target_identity_status.get("termination_triggered", False)):
                            failure_reason = str(target_identity_status.get("termination_reason"))
                    if failure_reason is None and bool(args.terminate_on_robot_stuck):
                        if bool(robot_stuck_status.get("termination_triggered", False)):
                            failure_reason = str(robot_stuck_status.get("termination_reason"))
                    if failure_reason is None and bool(target_far_status.get("termination_triggered", False)):
                        failure_reason = str(target_far_status.get("termination_reason"))
                    if failure_reason is not None:
                        termination_reason = failure_reason
                        episode_success = False
                        eval_step_termination_reason = failure_reason
                        eval_termination_triggered = True
                    elif target_task_done:
                        eval_step_termination_reason = "target_completed_a_to_b"
                        eval_termination_triggered = True
                    if not isinstance(policy_debug_info, dict):
                        policy_debug_info = {}
                    policy_debug_info["map_obstacle"] = map_obstacle_status
                    policy_debug_info["target_far"] = target_far_status
                    policy_debug_info["robot_stuck"] = robot_stuck_status
                    eval_logger.write_step(
                        tick=tick,
                        time_s=eval_time_s,
                        episode_active=following,
                        paused=paused,
                        target_track_id=active_target_track_id,
                        robot_state=robot_state,
                        target_state=target_state,
                        npc_states=npc_states,
                        visible=visible,
                        pixel_count=pix_count,
                        visibility_threshold=args.visibility_threshold,
                        command=None if v_cmd is None else {"v_mps": v_cmd, "w_radps": w_cmd},
                        planner_cost_ms=planner_cost_ms,
                        debug_info=policy_debug_info,
                        collision=eval_collision,
                        collision_source=eval_collision_source,
                        termination_triggered=eval_termination_triggered,
                        termination_reason=eval_step_termination_reason,
                        clearance=eval_clearance,
                        target_identity=target_identity_status,
                    )
                    if args.record_planning_video and following and robot_state is not None:
                        if planning_video["rec"] is None:
                            from followbench.scenes.passage.planning_video import PlanningVideoRecorder  # noqa: PLC0415
                            planning_video["rec"] = PlanningVideoRecorder(
                                run_dir=eval_logger.run_dir,
                                fps=1.0 / float(args.dt),
                                grid_npz=args.grid_npz,
                                dt=float(args.dt),
                                target_track_id=str(active_target_track_id),
                                desired_distance=float(args.desired_distance),
                                fail_dist_m=float(args.robot_target_far_fail_dist_m),
                                title=(f"{args.planner} reid={getattr(args, 'reid_mode', '?')} "
                                       f"diff={current_difficulty} weather={current_weather} "
                                       f"seed={args.npc_rng_seed}"),
                            )
                        try:
                            _vis_bboxes = (policy_debug_info or {}).get("track_bboxes_by_view") or {}
                            planning_video["rec"].write(
                                tick=tick,
                                time_s=float(tick) * float(args.dt),
                                rgb_front=_draw_bboxes_on(last_rgb_np, _vis_bboxes.get("front"), 0),
                                robot_state=robot_state,
                                target_state=target_state,
                                npc_states=npc_states,
                                debug=policy_debug_info,
                                v_cmd=v_cmd,
                                w_cmd=w_cmd,
                                visible=visible,
                                extra_hud=(f"term={eval_step_termination_reason}"
                                           if eval_step_termination_reason else ""),
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(f"[VIS][WARN] planning video frame failed: {exc}")
                    if eval_termination_triggered:
                        if eval_step_termination_reason == "target_completed_a_to_b":
                            termination_reason = "target_completed_a_to_b"
                            episode_success = True
                            print("[EVAL] terminating episode: target_completed_a_to_b")
                        else:
                            print(f"[EVAL] terminating episode: {eval_step_termination_reason}")
                        npc.set_motion_active(False)
                        robot.hold_still()
                        episode_end_reason = termination_reason
                        episode_end_success = bool(episode_success)

                gt_f.write(json.dumps({
                    "tick": tick, "time_s": now_t - start_t,
                    "target_track_id": active_target_track_id,
                    "robot": {"x": robot_state.x, "y": robot_state.y,
                              "z": robot_state.z, "yaw_rad": robot_state.yaw_rad,
                              "speed": robot_state.speed},
                    "npcs": [s.__dict__ for s in npc_states],
                }, ensure_ascii=False) + "\n")
                vis_f.write(json.dumps({
                    "tick": tick, "time_s": now_t - start_t,
                    "target_track_id": active_target_track_id,
                    "target_visible": bool(visible),
                    "target_pixel_count": int(pix_count),
                    "threshold": int(args.visibility_threshold),
                }, ensure_ascii=False) + "\n")

                # if args.draw_debug and target_state is not None:
                #     world.debug.draw_string(
                #         carla.Location(x=target_state.x, y=target_state.y, z=target_state.z + 2.6),
                #         f"TGT {active_target_track_id} vis={int(visible)} pix={pix_count}",
                #         draw_shadow=False, color=carla.Color(255, 255, 0),
                #         life_time=max(args.dt * 2.0, 0.12))

                # Write live viewer state every 3 ticks
                if _debug_proc is not None and _debug_proc.poll() is None and tick % 3 == 0:
                    npc.write_debug_state(
                        tick=tick, robot_state=robot_state,
                        target_track_id=active_target_track_id,
                        sim_t=now_t - start_t,
                    )

                if dbg_vis is not None and robot_state is not None:
                    _dbg_fn = getattr(policy, "get_debug_info", None)
                    _dbg_info = _dbg_fn() if callable(_dbg_fn) else {}
                    _lidar_world = debug_lidar_world_points(
                        robot, robot_state, lidar_img, last_lidar_np, lidar_extr_cached,
                    )
                    dbg_vis.update(
                        tick=tick,
                        robot_state=robot_state,
                        npc_states=npc_states,
                        active_target_id=active_target_track_id,
                        obstacles=_dbg_info.get("obstacles"),
                        traj_points=_dbg_info.get("traj_points"),
                        global_path=_dbg_info.get("global_path"),
                        goal_point=_dbg_info.get("goal_point"),
                        search_samples=_dbg_info.get("search_samples"),
                        search_goal=_dbg_info.get("search_goal"),
                        predicted_target_traj=(
                            _dbg_info.get("predicted_target_traj")
                            or _dbg_info.get("target_predicted_traj")
                        ),
                        map_occupied_cells=_dbg_info.get("map_occupied_cells"),
                        map_observed_free_cells=_dbg_info.get("map_observed_free_cells"),
                        map_outline=_dbg_info.get("map_outline"),
                        map_occupancy_rgba=_dbg_info.get("map_occupancy_rgba"),
                        map_esdf_rgba=_dbg_info.get("map_esdf_rgba"),
                        map_hybrid_rgba=_dbg_info.get("map_hybrid_rgba"),
                        map_debug_extent=_dbg_info.get("map_debug_extent"),
                        map_debug_mode=_dbg_info.get("map_debug_mode"),
                        lidar_range_max=_dbg_info.get("lidar_range_max"),
                        lidar_points_world=_lidar_world,
                    )

                draw_status(world,
                    f"spawned={spawned} follow={following} "
                    f"target={active_target_track_id} pix={pix_count}")
                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        lines = [
                            (f"State: spawned={spawned} follow={following} "
                             f"target={active_target_track_id} "
                             f"| walkers={desired_num_walkers} "
                             f"speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"),
                            (f"Difficulty: {current_difficulty.upper()}  "
                             f"({'LOCKED BY LAUNCH' if args.teleop else 'D to cycle: easy→hard→boss→legend'})  "
                             f"passing={npc._passing_count}/{npc._max_concurrent}"),
                            (
                                "Keys: WASD drive  Shift sprint  Mouse look  F pause  R reset  SPACE pause  Q quit"
                                if args.teleop else
                                "Keys: G spawn  F follow  T target  H manual-NPC  J robot  R reset  SPACE pause  Q quit"
                            ),
                            "Tune: Up/Down walkers | [ ] min_speed | - / = max_speed  (reset+G to apply)",
                            (f"Visibility(instance): pix={pix_count} "
                             f"thr={args.visibility_threshold} => visible={int(visible)}"),
                        ]
                        for i, text in enumerate(lines):
                            surf.blit(ui_font.render(text, True, (230, 235, 240)), (16, 16 + i * 34))
                        for title, cx, cy, key in [
                            ("RGB Left",  0, 0, "rgb_left"),
                            ("RGB Front", 1, 0, "rgb"),
                            ("RGB Right", 2, 0, "rgb_right"),
                            ("Depth",     0, 1, "depth"),
                            ("Instance",  1, 1, "instance"),
                            ("LiDAR BEV", 2, 1, "lidar"),
                        ]:
                            x, y = cx * 400, 170 + cy * 260
                            surf.blit(pygame.surfarray.make_surface(
                                latest_panels[key].transpose(1, 0, 2)), (x, y))
                            pygame.draw.rect(surf, (60, 60, 60), pygame.Rect(x, y, 400, 260), 1)
                            surf.blit(ui_font.render(title, True, (255, 220, 0)), (x + 6, y + 6))
                        pygame.display.flip()

            if episode_end_reason is not None:
                if args.teleop:
                    finish_episode(episode_end_reason, episode_end_success)
                else:
                    break
                if running:
                    step_main_loop(frame_started_at)
                    continue

            step_main_loop(frame_started_at)
            tick += 1

            if spawned and npc.episode_done and running:
                ep_dur = (sim_clock() - episode_start_t) if episode_start_t is not None else 0.0
                if termination_reason is None:
                    termination_reason = "target_completed_a_to_b"
                if episode_success is None:
                    episode_success = True
                npc.set_motion_active(False)
                robot.hold_still()
                print(f"[EP] episode DONE  reason={termination_reason}  "
                      f"duration={ep_dur:.2f}s  ticks={tick}")
                if args.teleop:
                    finish_episode(termination_reason, bool(episode_success))
                else:
                    running = False

    finally:
        if collision_monitor is not None:
            collision_monitor.destroy()
        gt_f.close()
        vis_f.close()
        npc.destroy()
        robot.destroy()
        world.apply_settings(orig_settings)
        if ui_enabled:
            pygame.quit()
        if _debug_proc is not None:
            try:
                _debug_proc.terminate()
            except Exception:
                pass
        if dbg_vis is not None:
            dbg_vis.stop()
        if eval_logger is not None:
            finalize_current_eval(termination_reason, episode_success)
        print("[EP] cleanup done.")
        print(f"[EP] GT stream:     {gt_path}")
        print(f"[EP] vis stream:    {vis_path}")
        print(f"[EP] calibration:   {calib_path}")


if __name__ == "__main__":
    main()
