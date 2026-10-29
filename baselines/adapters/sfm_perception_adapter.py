"""SFM (Social Force Model) follower with a perception frontend."""
from __future__ import annotations

import dataclasses
import math
import os
import sys
import time

import numpy as np


from baselines.adapters.sfm_adapter import SfmFollowerPolicy
from baselines.adapters._perception_frontend import (
    PerceptionFrontend,
    PerceptionFrontendConfig,
)
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter


class SfmPerceptionFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        yolo_model: str = "yolo11s.pt",
        tracker_device: str = "cuda",
        tracker_yolo_stride: int = 1,
        reid_mode: str = "basic",
        reid_kpr_config: str = "kpr_occ_duke_test",
        reid_device: str = "auto",
        lost_policy: str = "last_known",
        kinematics_mode: str = "kf",
        kf_pos_sigma: float = 0.20,
        kf_vel_sigma_q: float = 0.05,
        planner_target_state: str = "stable_route_heading",
        route_segments=None,
        route_heading_max_distance_m=None,
        side_follow_goal_smoothing_alpha: float | None = None,
        side_follow_startup_guard_steps: int = 0,
        side_follow_goal_bearing_limit_deg: float | None = None,
        side_follow_goal_jump_limit_m: float | None = None,
        **inner_kwargs,
    ) -> None:
        self._inner = SfmFollowerPolicy(dt=dt, **inner_kwargs)
        self._follow_position = str(inner_kwargs.get("follow_position", "back"))
        self._dt = float(dt)
        alpha = None if side_follow_goal_smoothing_alpha is None else float(side_follow_goal_smoothing_alpha)
        self._side_follow_goal_smoothing_alpha = None if alpha is None else float(np.clip(alpha, 0.0, 1.0))
        self._side_follow_startup_guard_steps = max(int(side_follow_startup_guard_steps), 0)
        self._side_follow_goal_bearing_limit_deg = (
            None if side_follow_goal_bearing_limit_deg is None else max(float(side_follow_goal_bearing_limit_deg), 0.0)
        )
        self._side_follow_goal_jump_limit_m = (
            None if side_follow_goal_jump_limit_m is None else max(float(side_follow_goal_jump_limit_m), 0.0)
        )
        self._frontend = PerceptionFrontend(PerceptionFrontendConfig(
            dt=dt, yolo_model=yolo_model, tracker_device=tracker_device,
            tracker_yolo_stride=tracker_yolo_stride, reid_mode=reid_mode,
            reid_kpr_config=reid_kpr_config, reid_device=reid_device,
            lost_policy=lost_policy, log_prefix="SFM_PRC",
            kinematics_mode=kinematics_mode,
            kf_pos_sigma=kf_pos_sigma,
            kf_vel_sigma_q=kf_vel_sigma_q,
            planner_target_state=planner_target_state,
            route_segments=route_segments,
            route_heading_max_distance_m=route_heading_max_distance_m,
            planner_follow_position=inner_kwargs.get("follow_position"),
        ))
        self._last_debug: dict = {}
        self._last_smoothed_target_xy: np.ndarray | None = None
        self._last_smoothed_target_yaw_rad: float | None = None
        self._last_stable_goal_xy: np.ndarray | None = None
        self._side_follow_tick = 0

    def reset(self) -> None:
        self._inner.reset()
        self._frontend.reset()
        self._last_debug = {}
        self._last_smoothed_target_xy = None
        self._last_smoothed_target_yaw_rad = None
        self._last_stable_goal_xy = None
        self._side_follow_tick = 0

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        step = self._frontend.step(obs)
        modified_obs = step.modified_obs
        stabilizer_debug = {"enabled": False, "active": False}
        if not step.brake:
            modified_obs, stabilizer_debug = self._stabilize_side_follow_input(step.modified_obs)
        t_planner = time.perf_counter()
        if step.brake:
            action = FollowAction(v_mps=0.0, w_radps=0.0)
        else:
            action = self._inner.act(modified_obs)
        planner_core_ms = 0.0 if step.brake else (time.perf_counter() - t_planner) * 1000.0
        inner_debug = getattr(self._inner, "get_debug_info", lambda: {})() or {}
        timing = dict(inner_debug.get("timing") or {})
        timing.update(step.debug.get("timing") or {})
        timing["planner_core_ms"] = planner_core_ms
        self._last_debug = {
            **inner_debug,
            **step.debug,
            "timing": timing,
            "side_follow_input_stabilizer": stabilizer_debug,
        }
        return action

    def _stabilize_side_follow_input(self, obs: FollowObservation) -> tuple[FollowObservation, dict]:
        debug = {"enabled": False, "active": False}
        if self._follow_position not in {"left_side", "right_side"}:
            return obs, debug
        if (
            self._side_follow_goal_smoothing_alpha is None
            and self._side_follow_startup_guard_steps <= 0
            and self._side_follow_goal_bearing_limit_deg is None
            and self._side_follow_goal_jump_limit_m is None
        ):
            return obs, debug
        target = obs.target
        if target is None:
            return obs, debug

        self._side_follow_tick += 1
        debug["enabled"] = True

        raw_target_xy = np.array([float(target.x), float(target.y)], dtype=float)
        raw_target_yaw = float(math.radians(float(target.yaw_deg)))
        raw_speed = float(target.speed)
        if not math.isfinite(raw_speed) or raw_speed <= 1e-6:
            raw_speed = float(np.hypot(float(target.vx), float(target.vy)))

        alpha = self._side_follow_goal_smoothing_alpha
        if alpha is None or self._last_smoothed_target_xy is None:
            smoothed_target_xy = raw_target_xy.copy()
        else:
            smoothed_target_xy = alpha * raw_target_xy + (1.0 - alpha) * self._last_smoothed_target_xy
        if alpha is None or self._last_smoothed_target_yaw_rad is None:
            smoothed_target_yaw = raw_target_yaw
        else:
            smoothed_target_yaw = _blend_angle(self._last_smoothed_target_yaw_rad, raw_target_yaw, alpha)

        raw_goal_xy = _side_goal_xy(
            target_xy=raw_target_xy,
            target_yaw_rad=raw_target_yaw,
            desired_distance=float(self._inner.desired_distance),
            follow_position=self._follow_position,
        )
        stable_goal_xy = _side_goal_xy(
            target_xy=smoothed_target_xy,
            target_yaw_rad=smoothed_target_yaw,
            desired_distance=float(self._inner.desired_distance),
            follow_position=self._follow_position,
        )

        startup_guard_active = False
        goal_jump_clipped = False
        goal_bearing_clipped = False
        robot_xy = np.array([float(obs.robot.x), float(obs.robot.y)], dtype=float)
        robot_yaw = float(obs.robot.yaw_rad)
        bearing_limit_rad = (
            None if self._side_follow_goal_bearing_limit_deg is None else math.radians(float(self._side_follow_goal_bearing_limit_deg))
        )

        goal_bearing_rad = _goal_bearing(robot_xy=robot_xy, robot_yaw_rad=robot_yaw, goal_xy=stable_goal_xy)
        if bearing_limit_rad is not None and abs(goal_bearing_rad) > bearing_limit_rad:
            clipped_goal_xy = _clip_goal_bearing(
                robot_xy=robot_xy,
                robot_yaw_rad=robot_yaw,
                goal_xy=stable_goal_xy,
                bearing_limit_rad=bearing_limit_rad,
            )
            goal_bearing_clipped = bool(np.linalg.norm(clipped_goal_xy - stable_goal_xy) > 1e-6)
            stable_goal_xy = clipped_goal_xy
            goal_bearing_rad = _goal_bearing(robot_xy=robot_xy, robot_yaw_rad=robot_yaw, goal_xy=stable_goal_xy)

        if self._side_follow_tick <= self._side_follow_startup_guard_steps:
            startup_guard_active = True
            if self._last_stable_goal_xy is not None:
                stable_goal_xy = self._last_stable_goal_xy.copy()
                goal_bearing_rad = _goal_bearing(robot_xy=robot_xy, robot_yaw_rad=robot_yaw, goal_xy=stable_goal_xy)
        elif self._last_stable_goal_xy is not None and self._side_follow_goal_jump_limit_m is not None:
            clipped_goal_xy = _clip_goal_jump(
                previous_goal_xy=self._last_stable_goal_xy,
                new_goal_xy=stable_goal_xy,
                jump_limit_m=float(self._side_follow_goal_jump_limit_m),
            )
            goal_jump_clipped = bool(np.linalg.norm(clipped_goal_xy - stable_goal_xy) > 1e-6)
            stable_goal_xy = clipped_goal_xy
            goal_bearing_rad = _goal_bearing(robot_xy=robot_xy, robot_yaw_rad=robot_yaw, goal_xy=stable_goal_xy)

        adjusted_target_xy = _target_xy_from_side_goal(
            goal_xy=stable_goal_xy,
            target_yaw_rad=smoothed_target_yaw,
            desired_distance=float(self._inner.desired_distance),
            follow_position=self._follow_position,
        )
        adjusted_vx = raw_speed * math.cos(smoothed_target_yaw)
        adjusted_vy = raw_speed * math.sin(smoothed_target_yaw)
        adjusted_target = dataclasses.replace(
            target,
            x=float(adjusted_target_xy[0]),
            y=float(adjusted_target_xy[1]),
            vx=float(adjusted_vx),
            vy=float(adjusted_vy),
            yaw_deg=float(math.degrees(smoothed_target_yaw)),
            speed=float(raw_speed),
        )
        adjusted_obs = dataclasses.replace(obs, target=adjusted_target)

        self._last_smoothed_target_xy = adjusted_target_xy.copy()
        self._last_smoothed_target_yaw_rad = float(smoothed_target_yaw)
        self._last_stable_goal_xy = stable_goal_xy.copy()

        debug.update({
            "active": True,
            "follow_position": self._follow_position,
            "tick_counter": int(self._side_follow_tick),
            "smoothing_alpha": self._side_follow_goal_smoothing_alpha,
            "startup_guard_steps": int(self._side_follow_startup_guard_steps),
            "goal_bearing_limit_deg": self._side_follow_goal_bearing_limit_deg,
            "goal_jump_limit_m": self._side_follow_goal_jump_limit_m,
            "raw_target_xy": raw_target_xy.tolist(),
            "smoothed_target_xy": adjusted_target_xy.tolist(),
            "raw_target_yaw_rad": float(raw_target_yaw),
            "smoothed_target_yaw_rad": float(smoothed_target_yaw),
            "raw_goal_xy": raw_goal_xy.tolist(),
            "stable_goal_xy": stable_goal_xy.tolist(),
            "goal_bearing_rad": float(goal_bearing_rad),
            "goal_jump_clipped": bool(goal_jump_clipped),
            "goal_bearing_clipped": bool(goal_bearing_clipped),
            "startup_guard_active": bool(startup_guard_active),
        })
        return adjusted_obs, debug


def _wrap_pi(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _blend_angle(prev_angle_rad: float, next_angle_rad: float, alpha: float) -> float:
    delta = _wrap_pi(float(next_angle_rad) - float(prev_angle_rad))
    return _wrap_pi(float(prev_angle_rad) + float(alpha) * delta)


def _side_offset_rad(follow_position: str) -> float:
    return -math.pi * 0.5 if str(follow_position) == "left_side" else math.pi * 0.5


def _side_goal_xy(
    *,
    target_xy: np.ndarray,
    target_yaw_rad: float,
    desired_distance: float,
    follow_position: str,
) -> np.ndarray:
    angle = _wrap_pi(float(target_yaw_rad) + _side_offset_rad(follow_position))
    return np.array([
        float(target_xy[0]) + float(desired_distance) * math.cos(angle),
        float(target_xy[1]) + float(desired_distance) * math.sin(angle),
    ], dtype=float)


def _target_xy_from_side_goal(
    *,
    goal_xy: np.ndarray,
    target_yaw_rad: float,
    desired_distance: float,
    follow_position: str,
) -> np.ndarray:
    angle = _wrap_pi(float(target_yaw_rad) + _side_offset_rad(follow_position))
    return np.array([
        float(goal_xy[0]) - float(desired_distance) * math.cos(angle),
        float(goal_xy[1]) - float(desired_distance) * math.sin(angle),
    ], dtype=float)


def _goal_bearing(*, robot_xy: np.ndarray, robot_yaw_rad: float, goal_xy: np.ndarray) -> float:
    delta = np.asarray(goal_xy, dtype=float)[:2] - np.asarray(robot_xy, dtype=float)[:2]
    return _wrap_pi(math.atan2(float(delta[1]), float(delta[0])) - float(robot_yaw_rad))


def _clip_goal_jump(*, previous_goal_xy: np.ndarray, new_goal_xy: np.ndarray, jump_limit_m: float) -> np.ndarray:
    prev_xy = np.asarray(previous_goal_xy, dtype=float)[:2]
    next_xy = np.asarray(new_goal_xy, dtype=float)[:2]
    delta = next_xy - prev_xy
    dist = float(np.linalg.norm(delta))
    if dist <= max(float(jump_limit_m), 0.0) or dist <= 1e-9:
        return next_xy
    return prev_xy + delta * (float(jump_limit_m) / dist)


def _clip_goal_bearing(
    *,
    robot_xy: np.ndarray,
    robot_yaw_rad: float,
    goal_xy: np.ndarray,
    bearing_limit_rad: float,
) -> np.ndarray:
    robot_xy = np.asarray(robot_xy, dtype=float)[:2]
    goal_xy = np.asarray(goal_xy, dtype=float)[:2]
    delta = goal_xy - robot_xy
    radius = float(np.linalg.norm(delta))
    if radius <= 1e-9:
        return goal_xy
    bearing = _goal_bearing(robot_xy=robot_xy, robot_yaw_rad=robot_yaw_rad, goal_xy=goal_xy)
    clipped_bearing = float(np.clip(bearing, -float(bearing_limit_rad), float(bearing_limit_rad)))
    if abs(clipped_bearing - bearing) <= 1e-9:
        return goal_xy
    world_angle = float(robot_yaw_rad) + clipped_bearing
    return robot_xy + radius * np.array([math.cos(world_angle), math.sin(world_angle)], dtype=float)
