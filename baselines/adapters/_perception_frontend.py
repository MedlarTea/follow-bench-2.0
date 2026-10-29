"""Perception frontend — reusable wrapper turning a GT-reading inner planner
into one that runs entirely off the appearance-ReID perception pipeline.

Responsibilities (per tick):

1. Build an optional GT hint for the perception FSM (only while the FSM is
   still in ``initial`` — the very first lock-on).
2. Run the unified ``PerceptionPipeline`` with multi-view RGB+depth and get
   the current target world XY + the full ``all_tracks`` list.
3. Synthesize a new ``obs.target`` (``NpcState``) and ``obs.npcs`` list from
   the tracker output and emit a ``FollowObservation`` with those fields
   replaced.  The inner planner sees a normal ``FollowObservation`` — it
   doesn't know perception happened.
4. When the target is lost, apply one of three **lost policies**:
      ``gt_fallback`` — pass the raw ``obs`` through (GT-driven, debug only).
      ``last_known``  — reuse the last successful tracker XY as a stationary
                        target so the inner MPC coasts to it and waits for
                        ReID to re-acquire.
      ``brake``       — tell the wrapper to emit zero action this tick.

The class returns a ``PerceptionFrontendStep`` with ``modified_obs``,
``target_locked``, ``brake``, the raw ``PerceptionResult``, and a debug dict
ready for ``get_debug_info()`` consumption.

Consumers decide how to use the result:
  - Simple wrappers (pid / sfm / rda_lidar / ...) do
      ``action = FollowAction(0,0) if step.brake else inner.act(step.modified_obs)``.
  - The ``rda_search`` wrapper additionally overrides
    ``modified_obs.target_visible / target_pixel_count`` from ``target_locked``
    so the search planner can enter its own search mode when perception
    reports lost.
"""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


from followbench.sim.core_types import FollowObservation, NpcState

try:
    from followbench.perception.perception_pipeline import (
        PerceptionConfig,
        PerceptionPipeline,
        PerceptionResult,
    )
    _HAS_PERCEPTION = True
    _PERCEPTION_IMPORT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_PERCEPTION = False
    _PERCEPTION_IMPORT_ERR = str(_e)


_VALID_LOST_POLICIES = frozenset({"gt_fallback", "brake", "last_known"})
_VALID_PLANNER_TARGET_STATES = frozenset({"detected_motion", "smoothed_motion_heading", "stable_route_heading"})


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class PerceptionFrontendConfig:
    """Knobs for the perception frontend. Defaults match the previous
    ``rda_perception`` adapter exactly."""

    dt: float = 0.1
    yolo_model: str = "yolo11s.pt"
    tracker_cfg: str = "botsort.yaml"
    tracker_device: str = "cuda"
    tracker_yolo_stride: int = 1
    tracker_max_range_m: float = 15.0

    reid_mode: str = "basic"              # 'basic' | 'kpr'
    reid_kpr_config: str = "kpr_occ_duke_test"
    reid_device: str = "auto"

    # Target-lost behaviour (only consulted when the FSM reports no target).
    #   gt_fallback — debug-only: pass GT target through the inner planner.
    #   last_known  — coast to the last successful tracker xy, speed=0.
    #   brake       — the wrapper emits (v=0, w=0) and skips the inner planner.
    lost_policy: str = "last_known"

    # Velocity estimator for the depth tracker.
    #   'ema' — finite-difference + EMA (fast, no extra deps, noisier velocity).
    #   'kf'  — constant-velocity Kalman filter (filterpy required, smoother
    #           velocity — recommended when downstream planners rely on vx/vy
    #           for trajectory prediction, e.g. rda_traj + --use-perception).
    kinematics_mode: str = "kf"
    kf_pos_sigma: float = 0.20     # KF: measurement noise σ (m)
    kf_vel_sigma_q: float = 0.05   # KF: process noise σ on velocity (m/s)

    # Log tag used by the per-tick status line: e.g. ``[PID_PRC]``.
    log_prefix: str = "PRC"

    # How the synthetic target state passed to the inner planner is built.
    #   detected_motion
    #       Default/current behaviour: copy the perception tracker's detected
    #       position, velocity, heading, and speed directly into ``obs.target``.
    #       Keep this default so existing perception planners do not change.
    #   smoothed_motion_heading
    #       DWA side-follow mode: still use only camera-detected target
    #       positions, but estimate planner-facing velocity/heading from a
    #       short multi-frame position history.  This avoids moving the
    #       left/right-side goal around with single-frame depth-tracker jitter.
    #   stable_route_heading
    #       Corner side-follow mode: use camera-detected target positions plus
    #       route-segment tangents as the main heading, with only a small
    #       motion-history correction.  This avoids using GT actor yaw while
    #       keeping left/right goals stable at low speed.
    planner_target_state: str = "detected_motion"
    route_segments: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    route_motion_correction_max_deg: float = 30.0
    route_yaw_rate_limit_degps: float = 45.0
    route_heading_max_distance_m: Optional[float] = None
    planner_follow_position: Optional[str] = None
    startup_use_gt_yaw: bool = True
    startup_gt_yaw_frames: int = 5


@dataclass
class PerceptionFrontendStep:
    """Result of a single frontend tick, ready for the wrapper to dispatch."""

    modified_obs: FollowObservation           # target / npcs replaced (or unchanged)
    target_locked: bool                       # FSM currently has a lock
    brake: bool                               # wrapper should emit zero action
    perception_result: PerceptionResult
    # Debug payload for ``get_debug_info()``. Already shaped for downstream
    # visualizers (track_bboxes_by_view, tracked_peds, target_pos, timing).
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _PlannerTargetState:
    x: float
    y: float
    vx: float
    vy: float
    yaw_rad: float
    speed: float
    heading_confident: bool


class _SmoothedMotionHeadingFilter:
    """Planner-facing target state from camera-detected positions only.

    The depth tracker already estimates velocity, but side-follow planners use
    heading as geometry: ``left_side/right_side`` is a 90-degree offset from the
    target heading.  Estimating that heading from a short position history makes
    the side-follow goal stable without using the target person's ground truth.
    """

    def __init__(
        self,
        dt: float,
        history_size: int = 10,
        xy_alpha: float = 0.45,
        speed_alpha: float = 0.35,
        min_samples: int = 4,
        min_heading_speed: float = 0.20,
        min_heading_displacement: float = 0.30,
        yaw_rate_limit: float = 0.9,
    ) -> None:
        self.dt = max(float(dt), 1e-3)
        self.history_size = max(int(history_size), 2)
        self.xy_alpha = float(xy_alpha)
        self.speed_alpha = float(speed_alpha)
        self.min_samples = max(int(min_samples), 2)
        self.min_heading_speed = float(min_heading_speed)
        self.min_heading_displacement = float(min_heading_displacement)
        self.yaw_rate_limit = float(yaw_rate_limit)
        self._xy: Tuple[float, float] | None = None
        self._history: Deque[Tuple[float, float]] = deque(maxlen=self.history_size)
        self._yaw_rad: float | None = None
        self._speed: float = 0.0

    def reset(self) -> None:
        self._xy = None
        self._history.clear()
        self._yaw_rad = None
        self._speed = 0.0

    def update(self, raw_xy: Tuple[float, float]) -> _PlannerTargetState:
        rx, ry = float(raw_xy[0]), float(raw_xy[1])
        if self._xy is None:
            sx, sy = rx, ry
        else:
            px, py = self._xy
            a = self.xy_alpha
            sx = a * rx + (1.0 - a) * px
            sy = a * ry + (1.0 - a) * py
        self._xy = (sx, sy)
        self._history.append((sx, sy))

        vx, vy = self._fit_velocity()
        measured_speed = math.hypot(vx, vy)
        self._speed = self.speed_alpha * measured_speed + (1.0 - self.speed_alpha) * self._speed

        heading_confident = False
        displacement = self._history_displacement()
        if (
            len(self._history) >= self.min_samples
            and displacement >= self.min_heading_displacement
            and measured_speed >= self.min_heading_speed
        ):
            desired_yaw = math.atan2(vy, vx)
            self._yaw_rad = (
                desired_yaw
                if self._yaw_rad is None
                else _step_angle(self._yaw_rad, desired_yaw, self.yaw_rate_limit * self.dt)
            )
            heading_confident = True
        elif self._yaw_rad is None and measured_speed > 1e-3:
            # Seed the angle from perception motion only; keep it marked
            # unconfident until enough history supports side-follow geometry.
            self._yaw_rad = math.atan2(vy, vx)

        yaw = float(self._yaw_rad) if self._yaw_rad is not None else 0.0
        speed = float(self._speed if heading_confident else min(self._speed, self.min_heading_speed))
        return _PlannerTargetState(
            x=sx,
            y=sy,
            vx=speed * math.cos(yaw),
            vy=speed * math.sin(yaw),
            yaw_rad=yaw,
            speed=speed,
            heading_confident=heading_confident,
        )

    def _fit_velocity(self) -> Tuple[float, float]:
        n = len(self._history)
        if n < 2:
            return 0.0, 0.0
        xs = [p[0] for p in self._history]
        ys = [p[1] for p in self._history]
        ts = [i * self.dt for i in range(n)]
        t_mean = sum(ts) / n
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        denom = sum((t - t_mean) ** 2 for t in ts)
        if denom <= 1e-9:
            return 0.0, 0.0
        vx = sum((t - t_mean) * (x - x_mean) for t, x in zip(ts, xs)) / denom
        vy = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys)) / denom
        return float(vx), float(vy)

    def _history_displacement(self) -> float:
        if len(self._history) < 2:
            return 0.0
        x0, y0 = self._history[0]
        x1, y1 = self._history[-1]
        return float(math.hypot(x1 - x0, y1 - y0))


class _StableRouteHeadingFilter:
    """Route-aware heading from perceived target positions.

    The route tangent is the main heading source.  Motion history only chooses
    route direction and applies a bounded correction, so startup and local
    avoidance do not rotate the side-follow goal behind the target.
    """

    def __init__(
        self,
        dt: float,
        route_segments: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]],
        history_size: int = 10,
        min_samples: int = 5,
        min_motion_displacement: float = 0.30,
        min_motion_speed: float = 0.20,
        max_motion_correction_deg: float = 30.0,
        yaw_rate_limit_degps: float = 45.0,
        direction_switch_confirm_frames: int = 5,
        max_route_distance_m: Optional[float] = None,
    ) -> None:
        self.dt = max(float(dt), 1e-3)
        self.route_segments = _normalize_route_segments(route_segments)
        self.history_size = max(int(history_size), 2)
        self.min_samples = max(int(min_samples), 2)
        self.min_motion_displacement = float(min_motion_displacement)
        self.min_motion_speed = float(min_motion_speed)
        self.max_motion_correction = math.radians(float(max_motion_correction_deg))
        self.yaw_rate_limit = math.radians(float(yaw_rate_limit_degps))
        self.direction_switch_confirm_frames = max(int(direction_switch_confirm_frames), 1)
        self.max_route_distance_m = None if max_route_distance_m is None else max(float(max_route_distance_m), 0.0)
        self._history: Deque[Tuple[float, float]] = deque(maxlen=self.history_size)
        self._yaw_rad: float | None = None
        self._last_confident_yaw_rad: float | None = None
        self._direction_sign: float = 1.0
        self._pending_direction_sign: float | None = None
        self._pending_direction_count: int = 0
        self._speed: float = 0.0

    def reset(self) -> None:
        self._history.clear()
        self._yaw_rad = None
        self._last_confident_yaw_rad = None
        self._direction_sign = 1.0
        self._pending_direction_sign = None
        self._pending_direction_count = 0
        self._speed = 0.0

    def update(
        self,
        raw_xy: Tuple[float, float],
        detected_vx: float,
        detected_vy: float,
        detected_yaw_rad: float,
        detected_speed: float,
        robot_xy: Optional[Tuple[float, float]] = None,
        follow_position: Optional[str] = None,
    ) -> Tuple[_PlannerTargetState, Dict[str, Any]]:
        rx, ry = float(raw_xy[0]), float(raw_xy[1])
        self._history.append((rx, ry))
        motion = self._motion_estimate()
        motion_yaw = motion["yaw"]
        motion_speed = motion["speed"]
        motion_confident = bool(motion["confident"])

        nearest_route = self._nearest_route(rx, ry)
        route_rejected_by_distance = False
        if (
            nearest_route is not None
            and self.max_route_distance_m is not None
            and float(nearest_route["distance"]) > self.max_route_distance_m
        ):
            route = None
            route_rejected_by_distance = True
        else:
            route = nearest_route
        route_available = route is not None
        yaw_source = "last_confident"
        heading_confident = False
        correction = 0.0
        route_yaw = None
        segment_id = None
        route_distance = None

        if route is not None:
            segment_id = int(route["segment_id"])
            route_distance = float(route["distance"])
            tangent_yaw = float(route["yaw"])
            sign = self._direction_sign
            if motion_confident and motion_yaw is not None:
                along = math.cos(_wrap_pi(float(motion_yaw) - tangent_yaw))
                desired_sign = 1.0 if along >= 0.0 else -1.0
                sign = self._confirmed_direction(desired_sign)
            elif robot_xy is not None and follow_position in {"left_side", "right_side"}:
                sign = self._direction_from_robot_side(
                    tangent_yaw=tangent_yaw,
                    target_xy=(rx, ry),
                    robot_xy=robot_xy,
                    follow_position=str(follow_position),
                )
                self._direction_sign = sign
            route_yaw = _wrap_pi(tangent_yaw if sign >= 0.0 else tangent_yaw + math.pi)
            yaw_raw = route_yaw
            if motion_confident and motion_yaw is not None:
                correction = max(
                    -self.max_motion_correction,
                    min(self.max_motion_correction, _wrap_pi(float(motion_yaw) - route_yaw)),
                )
                yaw_raw = _wrap_pi(route_yaw + correction)
            yaw_source = "stable_route_heading"
            heading_confident = True
        else:
            if nearest_route is not None:
                segment_id = int(nearest_route["segment_id"])
                route_distance = float(nearest_route["distance"])
            if motion_confident and motion_yaw is not None:
                yaw_raw = float(motion_yaw)
                yaw_source = "motion_fallback"
                heading_confident = True
            elif self._last_confident_yaw_rad is not None:
                yaw_raw = float(self._last_confident_yaw_rad)
            else:
                yaw_raw = float(detected_yaw_rad)
                yaw_source = "detected_fallback"

        if self._yaw_rad is None:
            yaw = float(yaw_raw)
        elif heading_confident:
            yaw = _step_angle(self._yaw_rad, float(yaw_raw), self.yaw_rate_limit * self.dt)
        else:
            yaw = float(self._yaw_rad)
        self._yaw_rad = yaw
        if heading_confident:
            self._last_confident_yaw_rad = yaw

        speed_meas = float(motion_speed if motion_confident else max(float(detected_speed), math.hypot(float(detected_vx), float(detected_vy))))
        self._speed = 0.35 * max(speed_meas, 0.0) + 0.65 * self._speed
        speed = float(max(self._speed, 0.0))
        state = _PlannerTargetState(
            x=rx,
            y=ry,
            vx=speed * math.cos(yaw),
            vy=speed * math.sin(yaw),
            yaw_rad=yaw,
            speed=speed,
            heading_confident=heading_confident,
        )
        debug = {
            "route_heading_available": bool(route_available),
            "route_segment_id": segment_id,
            "route_distance_m": route_distance,
            "route_heading_rejected_by_distance": bool(route_rejected_by_distance),
            "route_heading_max_distance_m": self.max_route_distance_m,
            "route_yaw_rad": route_yaw,
            "motion_yaw_rad": motion_yaw,
            "motion_speed_mps": float(motion_speed),
            "motion_heading_confident": motion_confident,
            "side_goal_yaw_rad": float(yaw),
            "yaw_correction_deg": float(math.degrees(correction)),
            "yaw_source": yaw_source,
            "route_direction_sign": float(self._direction_sign),
        }
        return state, debug

    def _motion_estimate(self) -> Dict[str, Any]:
        n = len(self._history)
        if n < 2:
            return {"yaw": None, "speed": 0.0, "confident": False}
        x0, y0 = self._history[0]
        x1, y1 = self._history[-1]
        displacement = float(math.hypot(x1 - x0, y1 - y0))
        if n < self.min_samples or displacement < self.min_motion_displacement:
            return {"yaw": None, "speed": displacement / max((n - 1) * self.dt, 1e-3), "confident": False}
        xs = [p[0] for p in self._history]
        ys = [p[1] for p in self._history]
        ts = [i * self.dt for i in range(n)]
        t_mean = sum(ts) / n
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        denom = sum((t - t_mean) ** 2 for t in ts)
        if denom <= 1e-9:
            return {"yaw": None, "speed": 0.0, "confident": False}
        vx = sum((t - t_mean) * (x - x_mean) for t, x in zip(ts, xs)) / denom
        vy = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys)) / denom
        speed = float(math.hypot(vx, vy))
        confident = bool(speed >= self.min_motion_speed)
        yaw = float(math.atan2(vy, vx)) if speed > 1e-6 else None
        return {"yaw": yaw, "speed": speed, "confident": confident}

    def _nearest_route(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        if not self.route_segments:
            return None
        p = (float(x), float(y))
        best = None
        best_dist = float("inf")
        for idx, (a, b) in enumerate(self.route_segments):
            ax, ay = a
            bx, by = b
            dx = bx - ax
            dy = by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 1e-9:
                continue
            t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / seg_len2))
            px = ax + t * dx
            py = ay + t * dy
            dist = float(math.hypot(p[0] - px, p[1] - py))
            if dist < best_dist:
                best_dist = dist
                best = {
                    "segment_id": idx,
                    "distance": dist,
                    "yaw": float(math.atan2(dy, dx)),
                }
        return best

    def _confirmed_direction(self, desired_sign: float) -> float:
        desired_sign = 1.0 if float(desired_sign) >= 0.0 else -1.0
        if desired_sign == self._direction_sign:
            self._pending_direction_sign = None
            self._pending_direction_count = 0
            return self._direction_sign
        if self._pending_direction_sign != desired_sign:
            self._pending_direction_sign = desired_sign
            self._pending_direction_count = 1
        else:
            self._pending_direction_count += 1
        if self._pending_direction_count >= self.direction_switch_confirm_frames:
            self._direction_sign = desired_sign
            self._pending_direction_sign = None
            self._pending_direction_count = 0
        return self._direction_sign

    @staticmethod
    def _direction_from_robot_side(
        tangent_yaw: float,
        target_xy: Tuple[float, float],
        robot_xy: Tuple[float, float],
        follow_position: str,
    ) -> float:
        offset = math.pi * 0.5 if follow_position == "right_side" else -math.pi * 0.5
        tx, ty = float(target_xy[0]), float(target_xy[1])
        rx, ry = float(robot_xy[0]), float(robot_xy[1])
        best_sign = 1.0
        best_score = float("inf")
        for sign in (1.0, -1.0):
            yaw = _wrap_pi(float(tangent_yaw) if sign >= 0.0 else float(tangent_yaw) + math.pi)
            side_yaw = _wrap_pi(yaw + offset)
            side_x = tx + math.cos(side_yaw)
            side_y = ty + math.sin(side_yaw)
            score = float(math.hypot(rx - side_x, ry - side_y))
            if score < best_score:
                best_score = score
                best_sign = sign
        return best_sign


# ── The class ─────────────────────────────────────────────────────────────────

class PerceptionFrontend:
    """Stateful helper: owns the perception pipeline + the last-known cache.

    Usage inside a wrapper:
        self._fe = PerceptionFrontend(PerceptionFrontendConfig(
            dt=dt, reid_mode=..., log_prefix="PID_PRC"))
        ...
        step = self._fe.step(obs)
        if step.brake:
            action = FollowAction(v_mps=0.0, w_radps=0.0)
        else:
            action = self._inner.act(step.modified_obs)
    """

    def __init__(self, cfg: PerceptionFrontendConfig) -> None:
        if not _HAS_PERCEPTION:
            raise ImportError(
                "PerceptionFrontend requires perception_pipeline.py. Make "
                "sure followbench.perception is importable and its "
                "dependencies (ultralytics, torch, ...) are installed.\n"
                f"Original error: {_PERCEPTION_IMPORT_ERR}"
            )

        lost_policy = str(cfg.lost_policy).lower()
        if lost_policy not in _VALID_LOST_POLICIES:
            raise ValueError(
                f"lost_policy must be one of {sorted(_VALID_LOST_POLICIES)}; "
                f"got {cfg.lost_policy!r}"
            )
        planner_target_state = str(cfg.planner_target_state).lower()
        if planner_target_state not in _VALID_PLANNER_TARGET_STATES:
            raise ValueError(
                "planner_target_state must be one of "
                f"{sorted(_VALID_PLANNER_TARGET_STATES)}; got {cfg.planner_target_state!r}"
            )
        self._cfg = cfg
        self._lost_policy = lost_policy
        self._planner_target_state = planner_target_state

        self._perception = PerceptionPipeline(PerceptionConfig(
            yolo_model=cfg.yolo_model,
            tracker_cfg=cfg.tracker_cfg,
            tracker_device=cfg.tracker_device,
            tracker_yolo_stride=cfg.tracker_yolo_stride,
            tracker_max_range_m=cfg.tracker_max_range_m,
            reid_mode=cfg.reid_mode,
            reid_kpr_config=cfg.reid_kpr_config,
            reid_device=cfg.reid_device,
            dt=cfg.dt,
            kinematics_mode=cfg.kinematics_mode,
            kf_pos_sigma=cfg.kf_pos_sigma,
            kf_vel_sigma_q=cfg.kf_vel_sigma_q,
        ))

        # Last known target pose (tracker-derived). Used by ``last_known``
        # policy to build a stationary synthetic target while ReID recovers.
        self._last_known_xy: Optional[Tuple[float, float]] = None
        self._last_known_yaw_rad: float = 0.0
        self._last_known_speed: float = 0.0
        self._startup_yaw_ready_count = 0
        self._startup_yaw_switched = False
        self._target_state_filter = _SmoothedMotionHeadingFilter(dt=cfg.dt)
        self._route_heading_filter = _StableRouteHeadingFilter(
            dt=cfg.dt,
            route_segments=cfg.route_segments,
            max_motion_correction_deg=cfg.route_motion_correction_max_deg,
            yaw_rate_limit_degps=cfg.route_yaw_rate_limit_degps,
            max_route_distance_m=cfg.route_heading_max_distance_m,
        )
        self._last_planner_target_debug: Dict[str, Any] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._perception.reset()
        self._last_known_xy = None
        self._last_known_yaw_rad = 0.0
        self._last_known_speed = 0.0
        self._startup_yaw_ready_count = 0
        self._startup_yaw_switched = False
        self._target_state_filter.reset()
        self._route_heading_filter.reset()
        self._last_planner_target_debug = {}

    @property
    def state_name(self) -> str:
        return self._perception.state_name

    @property
    def lost_policy(self) -> str:
        return self._lost_policy

    @property
    def last_known_xy(self) -> Optional[Tuple[float, float]]:
        return self._last_known_xy

    # ── Per-tick step ─────────────────────────────────────────────────────────

    def step(self, obs: FollowObservation) -> PerceptionFrontendStep:
        # ── (1) Only seed the FSM with a GT hint while it's still booting.
        gt = obs.target
        gt_hint: Optional[Tuple[float, float]] = None
        if gt is not None and self._perception.state_name == "initial":
            gt_hint = (float(gt.x), float(gt.y))

        # ── (2) Run the unified perception pipeline (multi-view). Missing
        #        side-cam fields in ``obs`` are silently degraded inside.
        t0 = time.perf_counter()
        result: PerceptionResult = self._perception.update(
            rgb_image=obs.rgb_image,
            depth_image=obs.depth_image,
            robot_x=float(obs.robot.x),
            robot_y=float(obs.robot.y),
            robot_z=float(obs.robot.z),
            robot_yaw=float(obs.robot.yaw_rad),
            rgb_intrinsics=obs.rgb_intrinsics,
            rgb_extrinsics=obs.rgb_extrinsics_robot_to_sensor,
            gt_target_xy=gt_hint,
            rgb_image_left=obs.rgb_image_left,
            depth_image_left=obs.depth_image_left,
            rgb_intrinsics_left=obs.rgb_intrinsics_left,
            rgb_extrinsics_left=obs.rgb_extrinsics_left_robot_to_sensor,
            rgb_image_right=obs.rgb_image_right,
            depth_image_right=obs.depth_image_right,
            rgb_intrinsics_right=obs.rgb_intrinsics_right,
            rgb_extrinsics_right=obs.rgb_extrinsics_right_robot_to_sensor,
        )
        t_percep_ms = (time.perf_counter() - t0) * 1000.0

        # ── (3) Error term for logs only — never enters the control loop.
        gt_xy: Optional[Tuple[float, float]] = (
            (float(gt.x), float(gt.y)) if gt is not None else None
        )
        gt_yaw_rad: Optional[float] = float(math.radians(gt.yaw_deg)) if gt is not None else None
        tracker_xy = result.target_xy_world
        err_m: Optional[float] = (
            math.hypot(tracker_xy[0] - gt_xy[0], tracker_xy[1] - gt_xy[1])
            if (tracker_xy is not None and gt_xy is not None) else None
        )

        # ── (4) Wrap every non-target track as an NpcState for the MPC.
        #        Prefix "D" = depth-tracker (distinguishes from real GT NPCs).
        synthetic_npcs: List[NpcState] = [
            NpcState(
                track_id=f"D{tp.track_id:03d}",
                actor_id=-1,
                x=float(tp.x), y=float(tp.y), z=0.0,
                vx=float(tp.vx), vy=float(tp.vy), vz=0.0,
                yaw_deg=float(math.degrees(tp.yaw_rad)),
                speed=float(tp.speed),
            )
            for tp in result.all_tracks
            if tp.track_id != result.target_id
        ]

        # ── (5) Main branch: perception has a world XY for the target.
        if tracker_xy is not None:
            target_tp = next(
                (tp for tp in result.all_tracks if tp.track_id == result.target_id),
                None,
            )
            tvx = float(target_tp.vx) if target_tp is not None else 0.0
            tvy = float(target_tp.vy) if target_tp is not None else 0.0
            tyaw_rad = float(target_tp.yaw_rad) if target_tp is not None else 0.0
            tspeed = float(target_tp.speed) if target_tp is not None else 0.0
            planner_target = self._build_planner_target_state(
                tracker_xy=(float(tracker_xy[0]), float(tracker_xy[1])),
                detected_vx=tvx,
                detected_vy=tvy,
                detected_yaw_rad=tyaw_rad,
                detected_speed=tspeed,
                robot_xy=(float(obs.robot.x), float(obs.robot.y)),
                gt_yaw_rad=gt_yaw_rad,
            )
            self._last_known_xy = (planner_target.x, planner_target.y)
            self._last_known_yaw_rad = planner_target.yaw_rad
            self._last_known_speed = planner_target.speed

            # z is carried from GT for control completeness — BEV tracking is
            # 2-D but the MPC expects a 3-D NpcState.
            synthetic_target = NpcState(
                track_id="T_PRC",
                actor_id=-1,
                x=planner_target.x,
                y=planner_target.y,
                z=float(gt.z) if gt is not None else 0.0,
                vx=planner_target.vx, vy=planner_target.vy, vz=0.0,
                yaw_deg=float(math.degrees(planner_target.yaw_rad)),
                speed=planner_target.speed,
            )
            modified_obs = dataclasses.replace(
                obs,
                target=synthetic_target,
                npcs=synthetic_npcs,
                extras=self._planner_extras(obs, planner_target, result),
            )

            # Per-tick status line — same format as the previous adapter's.
            gt_str = (
                f"  gt=({gt_xy[0]:.2f},{gt_xy[1]:.2f})  err={err_m:.2f}m"
                if gt_xy is not None else "  gt=N/A"
            )
            print(
                f"[{self._cfg.log_prefix}] tick={obs.tick}  state={result.state_name}"
                f"  target_id={result.target_id}"
                f"  pos=({tracker_xy[0]:.2f},{tracker_xy[1]:.2f})"
                f"{gt_str}"
                f"  npcs={len(synthetic_npcs)}  total_tracks={len(result.all_tracks)}"
                f"  t_track={result.timing.get('track_ms', 0.0):.1f}ms"
                f"  t_reid={result.timing.get('reid_extract_ms', 0.0):.1f}ms"
                f"  t_fsm={result.timing.get('fsm_ms', 0.0):.1f}ms"
                f"  t_percep={t_percep_ms:.1f}ms",
                flush=True,
            )

            debug = self._build_debug(result, tracker_xy, gt_xy, err_m, t_percep_ms)
            return PerceptionFrontendStep(
                modified_obs=modified_obs,
                target_locked=True,
                brake=False,
                perception_result=result,
                debug=debug,
            )

        # ── (6) Lost branch: perception says no target.
        gt_str = (f"  gt=({gt_xy[0]:.2f},{gt_xy[1]:.2f})"
                  if gt_xy is not None else "  gt=N/A")
        lk_str = (f"  last_known=({self._last_known_xy[0]:.2f},{self._last_known_xy[1]:.2f})"
                  if self._last_known_xy is not None else "  last_known=N/A")
        print(
            f"[{self._cfg.log_prefix}] tick={obs.tick}  state={result.state_name}"
            f"  target lost — policy={self._lost_policy}{gt_str}{lk_str}"
            f"  total_tracks={len(result.all_tracks)}"
            f"  t_percep={t_percep_ms:.1f}ms",
            flush=True,
        )

        self._last_planner_target_debug = {
            "planner_target_state": self._planner_target_state,
            "perception_target_raw_xy": None,
            "perception_target_filtered_xy": None,
            "perception_target_velocity": [0.0, 0.0],
            "perception_target_heading_rad": None,
            "perception_target_heading_confident": False,
            "perception_target_speed": 0.0,
        }
        debug = self._build_debug(result, None, gt_xy, None, t_percep_ms)

        if self._lost_policy == "gt_fallback":
            # Pass raw GT obs through the inner planner (debug-only path).
            return PerceptionFrontendStep(
                modified_obs=obs,
                target_locked=False,
                brake=False,
                perception_result=result,
                debug=debug,
            )

        if self._lost_policy == "last_known" and self._last_known_xy is not None:
            lk_x, lk_y = self._last_known_xy
            synthetic_target = NpcState(
                track_id="T_PRC_LAST",
                actor_id=-1,
                x=float(lk_x),
                y=float(lk_y),
                z=float(gt.z) if gt is not None else 0.0,
                vx=0.0, vy=0.0, vz=0.0,
                yaw_deg=float(math.degrees(self._last_known_yaw_rad)),
                speed=0.0,
            )
            self._last_planner_target_debug = {
                "planner_target_state": self._planner_target_state,
                "perception_target_raw_xy": None,
                "perception_target_filtered_xy": [float(lk_x), float(lk_y)],
                "perception_target_velocity": [0.0, 0.0],
                "perception_target_heading_rad": float(self._last_known_yaw_rad),
                "perception_target_heading_confident": False,
                "perception_target_speed": 0.0,
            }
            debug["planner_target"] = dict(self._last_planner_target_debug)
            last_known_state = _PlannerTargetState(
                x=float(lk_x),
                y=float(lk_y),
                vx=0.0,
                vy=0.0,
                yaw_rad=float(self._last_known_yaw_rad),
                speed=0.0,
                heading_confident=False,
            )
            modified_obs = dataclasses.replace(
                obs,
                target=synthetic_target,
                npcs=synthetic_npcs,
                extras=self._planner_extras(obs, last_known_state, result),
            )
            return PerceptionFrontendStep(
                modified_obs=modified_obs,
                target_locked=False,
                brake=False,
                perception_result=result,
                debug=debug,
            )

        # brake — or ``last_known`` with no cached pose yet.
        return PerceptionFrontendStep(
            modified_obs=obs,
            target_locked=False,
            brake=True,
            perception_result=result,
            debug=debug,
        )

    def _build_planner_target_state(
        self,
        tracker_xy: Tuple[float, float],
        detected_vx: float,
        detected_vy: float,
        detected_yaw_rad: float,
        detected_speed: float,
        robot_xy: Optional[Tuple[float, float]] = None,
        gt_yaw_rad: Optional[float] = None,
    ) -> _PlannerTargetState:
        startup_frames = max(1, int(self._cfg.startup_gt_yaw_frames))

        if self._planner_target_state == "smoothed_motion_heading":
            state = self._target_state_filter.update(tracker_xy)
            if self._cfg.startup_use_gt_yaw and gt_yaw_rad is not None and not self._startup_yaw_switched:
                if bool(state.heading_confident):
                    self._startup_yaw_ready_count += 1
                else:
                    self._startup_yaw_ready_count = 0
                if self._startup_yaw_ready_count >= startup_frames:
                    self._startup_yaw_switched = True
            use_gt_yaw = bool(self._cfg.startup_use_gt_yaw and gt_yaw_rad is not None and not self._startup_yaw_switched)
            yaw_source = "gt_startup" if use_gt_yaw else "perception_smoothed_motion_heading"
            yaw_rad = float(gt_yaw_rad if use_gt_yaw else state.yaw_rad) if gt_yaw_rad is not None else float(state.yaw_rad)
            speed = float(state.speed)
            if use_gt_yaw:
                state = _PlannerTargetState(
                    x=float(state.x),
                    y=float(state.y),
                    vx=speed * math.cos(yaw_rad),
                    vy=speed * math.sin(yaw_rad),
                    yaw_rad=yaw_rad,
                    speed=speed,
                    heading_confident=True,
                )
            self._last_planner_target_debug = {
                "planner_target_state": self._planner_target_state,
                "perception_target_raw_xy": [float(tracker_xy[0]), float(tracker_xy[1])],
                "perception_target_filtered_xy": [float(state.x), float(state.y)],
                "perception_target_velocity": [float(state.vx), float(state.vy)],
                "perception_target_heading_rad": float(state.yaw_rad),
                "perception_target_heading_confident": bool(state.heading_confident),
                "perception_target_speed": float(state.speed),
                "target_yaw_source": yaw_source,
                "startup_yaw_ready_count": int(self._startup_yaw_ready_count),
                "startup_yaw_switched": bool(self._startup_yaw_switched),
            }
            return state

        if self._planner_target_state == "stable_route_heading":
            state, route_debug = self._route_heading_filter.update(
                raw_xy=tracker_xy,
                detected_vx=detected_vx,
                detected_vy=detected_vy,
                detected_yaw_rad=detected_yaw_rad,
                detected_speed=detected_speed,
                robot_xy=robot_xy,
                follow_position=self._cfg.planner_follow_position,
            )
            if self._cfg.startup_use_gt_yaw and gt_yaw_rad is not None and not self._startup_yaw_switched:
                route_ready = bool(route_debug.get("route_heading_available")) and bool(route_debug.get("motion_heading_confident"))
                if route_ready:
                    self._startup_yaw_ready_count += 1
                else:
                    self._startup_yaw_ready_count = 0
                if self._startup_yaw_ready_count >= startup_frames:
                    self._startup_yaw_switched = True
            use_gt_yaw = bool(self._cfg.startup_use_gt_yaw and gt_yaw_rad is not None and not self._startup_yaw_switched)
            yaw_source = "gt_startup" if use_gt_yaw else str(route_debug.get("yaw_source", "perception_stable_route_heading"))
            yaw_rad = float(gt_yaw_rad if use_gt_yaw else state.yaw_rad) if gt_yaw_rad is not None else float(state.yaw_rad)
            speed = float(state.speed)
            if use_gt_yaw:
                state = _PlannerTargetState(
                    x=float(state.x),
                    y=float(state.y),
                    vx=speed * math.cos(yaw_rad),
                    vy=speed * math.sin(yaw_rad),
                    yaw_rad=yaw_rad,
                    speed=speed,
                    heading_confident=True,
                )
            self._last_planner_target_debug = {
                "planner_target_state": self._planner_target_state,
                "perception_target_raw_xy": [float(tracker_xy[0]), float(tracker_xy[1])],
                "perception_target_filtered_xy": [float(state.x), float(state.y)],
                "perception_target_velocity": [float(state.vx), float(state.vy)],
                "perception_target_heading_rad": float(state.yaw_rad),
                "perception_target_heading_confident": bool(state.heading_confident),
                "perception_target_speed": float(state.speed),
                "target_yaw_source": yaw_source,
                "startup_yaw_ready_count": int(self._startup_yaw_ready_count),
                "startup_yaw_switched": bool(self._startup_yaw_switched),
                **route_debug,
            }
            return state

        state = _PlannerTargetState(
            x=float(tracker_xy[0]),
            y=float(tracker_xy[1]),
            vx=float(detected_vx),
            vy=float(detected_vy),
            yaw_rad=float(detected_yaw_rad),
            speed=float(detected_speed),
            heading_confident=True,
        )
        if self._cfg.startup_use_gt_yaw and gt_yaw_rad is not None and not self._startup_yaw_switched:
            self._startup_yaw_ready_count += 1
            if self._startup_yaw_ready_count >= startup_frames:
                self._startup_yaw_switched = True
            yaw_rad = float(gt_yaw_rad)
            speed = float(state.speed)
            state = _PlannerTargetState(
                x=float(state.x),
                y=float(state.y),
                vx=speed * math.cos(yaw_rad),
                vy=speed * math.sin(yaw_rad),
                yaw_rad=yaw_rad,
                speed=speed,
                heading_confident=True,
            )
            yaw_source = "gt_startup"
        else:
            yaw_source = "perception_detected_motion"
        self._last_planner_target_debug = {
            "planner_target_state": self._planner_target_state,
            "perception_target_raw_xy": [float(tracker_xy[0]), float(tracker_xy[1])],
            "perception_target_filtered_xy": [float(state.x), float(state.y)],
            "perception_target_velocity": [float(state.vx), float(state.vy)],
            "perception_target_heading_rad": float(state.yaw_rad),
            "perception_target_heading_confident": True,
            "perception_target_speed": float(state.speed),
            "target_yaw_source": yaw_source,
            "startup_yaw_ready_count": int(self._startup_yaw_ready_count),
            "startup_yaw_switched": bool(self._startup_yaw_switched),
        }
        return state

    def _planner_extras(
        self,
        obs: FollowObservation,
        target_state: _PlannerTargetState,
        result: Optional[PerceptionResult] = None,
    ) -> Any:
        if self._planner_target_state not in {"smoothed_motion_heading", "stable_route_heading"}:
            return obs.extras
        extras = dict(obs.extras or {})
        # DWA falls back to ``obs.extras`` for target yaw when target velocity
        # is small.  Overwrite those yaw hints so no GT actor yaw enters this
        # planner-facing perception path.
        for key in ("target_yaw_rad", "target_actor_yaw_rad", "target_velocity_yaw_rad"):
            extras[key] = float(target_state.yaw_rad)
        extras["target_side_goal_yaw_rad"] = float(target_state.yaw_rad)
        extras["target_yaw_source"] = str(self._last_planner_target_debug.get("target_yaw_source", f"perception_{self._planner_target_state}"))
        if result is not None:
            extras["perception_has_target"] = result.target_xy_world is not None
            extras["perception_target_view"] = result.target_view
            extras["perception_target_bbox"] = None if result.target_bbox is None else [float(v) for v in result.target_bbox]
            extras["perception_target_id"] = int(result.target_id)
        return extras

    # ── Debug payload ─────────────────────────────────────────────────────────

    def _build_debug(
        self,
        result: PerceptionResult,
        tracker_xy: Optional[Tuple[float, float]],
        gt_xy: Optional[Tuple[float, float]],
        err_m: Optional[float],
        t_percep_ms: float,
    ) -> Dict[str, Any]:
        percep_debug = self._perception.get_debug_info()
        raw_timing = dict(result.timing or {})
        perception_timing = {
            "detection_ms": _first_timing(raw_timing, "detection_ms", "detect_ms", "yolo_ms"),
            "tracking_ms": _first_timing(raw_timing, "tracking_ms", "track_ms"),
            "reid_ms": _first_timing(raw_timing, "reid_ms", "reid_extract_ms", "id_ms"),
            "mapping_ms": _first_timing(raw_timing, "mapping_ms", "map_ms", "project_ms"),
            "fsm_ms": _first_timing(raw_timing, "fsm_ms"),
            "other_ms": _first_timing(raw_timing, "other_ms"),
            "total_ms": float(t_percep_ms),
            "raw": raw_timing,
        }
        return {
            "tracked_peds":         percep_debug.get("tracked_peds", []),
            "track_bboxes_by_view": percep_debug.get("track_bboxes_by_view", {}),
            "track_bboxes_age":     0,
            "target_track_id":      result.target_id,
            "target_lost":          tracker_xy is None,
            "perception_has_target": tracker_xy is not None,
            "perception_state":     result.state_name,
            "reid_match":           tracker_xy is not None and result.target_id is not None,
            "active_track_id":      result.target_id,
            "target_pos": {
                "tracker": tracker_xy,
                "gt":      gt_xy,
                "err_m":   err_m,
            },
            "planner_target":        dict(self._last_planner_target_debug),
            "perception":           percep_debug,
            "timing": {
                "perception_total_ms": float(t_percep_ms),
                "perception": perception_timing,
            },
        }


def _first_timing(source: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = source.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _normalize_route_segments(route_segments: Any) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for item in route_segments or []:
        try:
            a, b = item
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.hypot(bx - ax, by - ay) <= 1e-6:
            continue
        out.append(((ax, ay), (bx, by)))
    return out


def _step_angle(current: float, desired: float, max_step: float) -> float:
    err = _wrap_pi(float(desired) - float(current))
    step = max(-abs(float(max_step)), min(abs(float(max_step)), err))
    return _wrap_pi(float(current) + step)


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


__all__ = [
    "PerceptionFrontend",
    "PerceptionFrontendConfig",
    "PerceptionFrontendStep",
]
