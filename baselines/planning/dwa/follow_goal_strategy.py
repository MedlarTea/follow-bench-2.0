from __future__ import annotations

import math

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.planning.dwa.types import AgentPrediction, KinematicState


class DwaFollowGoalStrategy:
    def __init__(
        self,
        back_goal_mode: str = "target_heading",
        target_yaw_rate_limit: float = 1.0,
        slot_angle_rate_limit: float = 0.8,
        min_heading_speed: float = 0.30,
    ) -> None:
        self.back_goal_mode = str(back_goal_mode)
        self.target_yaw_rate_limit = float(target_yaw_rate_limit)
        self.slot_angle_rate_limit = float(slot_angle_rate_limit)
        self.min_heading_speed = float(min_heading_speed)
        self._filtered_yaw: float | None = None
        self._slot_angle: float | None = None
        self._side_sign: float | None = None
        self._last_debug: dict = {}

    def reset(self) -> None:
        self._filtered_yaw = None
        self._slot_angle = None
        self._side_sign = None
        self._last_debug = {}

    def get_debug_info(self) -> dict:
        return dict(self._last_debug)

    def build_goal_traj(
        self,
        robot_state: KinematicState,
        target_prediction: AgentPrediction,
        follow_position: str,
        desired_distance: float,
        tick_dt: float,
    ) -> np.ndarray:
        if len(target_prediction.positions) == 0:
            self._last_debug = {"goal_strategy_error": "empty_target_prediction"}
            return np.zeros((0, 3), dtype=float)

        dt = max(float(tick_dt), 1e-3)
        target_xy0 = np.asarray(target_prediction.positions[0], dtype=float).reshape(2)
        raw_yaw0 = float(target_prediction.yaws[0]) if len(target_prediction.yaws) else 0.0
        vel0 = target_prediction.velocities[0] if len(target_prediction.velocities) else np.zeros(2, dtype=float)
        filtered_yaw, yaw_measure = self._update_filtered_yaw(raw_yaw0, vel0, dt)
        desired_slot0 = self._desired_slot_angle(robot_state, target_xy0, filtered_yaw, follow_position)
        previous_slot = self._slot_angle
        if self._slot_angle is None:
            if follow_position in {"left_side", "right_side"}:
                self._slot_angle = desired_slot0
            else:
                self._slot_angle = self._initial_slot_angle(robot_state, target_xy0, desired_slot0)
        else:
            self._slot_angle = self._step_angle(self._slot_angle, desired_slot0, self.slot_angle_rate_limit * dt)

        goals = np.zeros((len(target_prediction.positions), 3), dtype=float)
        slot = float(self._slot_angle)
        yaw_k = filtered_yaw
        yaws = np.asarray(target_prediction.yaws, dtype=float).reshape(-1)
        for k, target_xy in enumerate(target_prediction.positions):
            raw_yaw = float(yaws[min(k, len(yaws) - 1)]) if len(yaws) else float(yaw_k)
            if k > 0:
                yaw_k = self._step_angle(yaw_k, raw_yaw, self.target_yaw_rate_limit * dt)
                desired_slot = self._desired_slot_angle(robot_state, target_xy, yaw_k, follow_position)
                slot = self._step_angle(slot, desired_slot, self.slot_angle_rate_limit * dt)
            gx = float(target_xy[0] + desired_distance * math.cos(slot))
            gy = float(target_xy[1] + desired_distance * math.sin(slot))
            gyaw = yaw_k
            goals[k] = np.array([gx, gy, gyaw], dtype=float)

        slot_error = wrap_pi(desired_slot0 - float(self._slot_angle))
        robot_xy = np.array([robot_state.x, robot_state.y], dtype=float)
        self._last_debug = {
            "target_raw_yaw": float(raw_yaw0),
            "target_yaw_measure": float(yaw_measure),
            "target_filtered_yaw": float(filtered_yaw),
            "slot_angle": float(self._slot_angle),
            "previous_slot_angle": None if previous_slot is None else float(previous_slot),
            "desired_slot_angle": float(desired_slot0),
            "slot_error": float(slot_error),
            "side_sign": None if self._side_sign is None else float(self._side_sign),
            "side_locked": bool(follow_position in {"left_side", "right_side"} and self._side_sign is not None),
            "segment_hits_target_safety": bool(
                _segment_hits_circle(robot_xy, goals[0, :2], target_xy0, max(float(desired_distance) * 0.45, 0.8))
            ),
        }
        return goals

    def _update_filtered_yaw(self, raw_yaw: float, velocity: np.ndarray, dt: float) -> tuple[float, float]:
        vel = np.asarray(velocity, dtype=float).reshape(2)
        speed = float(np.linalg.norm(vel))
        if speed >= self.min_heading_speed:
            yaw_measure = math.atan2(float(vel[1]), float(vel[0]))
        elif self._filtered_yaw is not None:
            yaw_measure = self._filtered_yaw
        else:
            yaw_measure = float(raw_yaw)

        if self._filtered_yaw is None:
            self._filtered_yaw = float(yaw_measure)
        else:
            self._filtered_yaw = self._step_angle(
                self._filtered_yaw,
                yaw_measure,
                self.target_yaw_rate_limit * dt,
            )
        return float(self._filtered_yaw), float(yaw_measure)

    def _desired_slot_angle(
        self,
        robot_state: KinematicState,
        target_xy: np.ndarray,
        yaw: float,
        follow_position: str,
    ) -> float:
        if follow_position == "back":
            if self.back_goal_mode == "bearing":
                return math.atan2(robot_state.y - float(target_xy[1]), robot_state.x - float(target_xy[0]))
            return wrap_pi(float(yaw) + math.pi)
        if follow_position == "front":
            return float(yaw)
        if follow_position in {"left_side", "right_side"}:
            side_sign = self._resolve_side_sign(robot_state, target_xy, yaw, follow_position)
            return wrap_pi(float(yaw) + side_sign * math.pi * 0.5)
        raise ValueError(f"Unsupported follow position: {follow_position}")

    def _resolve_side_sign(
        self,
        robot_state: KinematicState,
        target_xy: np.ndarray,
        yaw: float,
        follow_position: str,
    ) -> float:
        # CARLA target-body sides are mirrored relative to the usual
        # right-handed math normal: visual-left is yaw - 90 deg.
        requested = -1.0 if follow_position == "left_side" else 1.0
        if self._side_sign is None:
            self._side_sign = requested
        return float(self._side_sign)

    def _initial_slot_angle(
        self,
        robot_state: KinematicState,
        target_xy: np.ndarray,
        desired_slot: float,
    ) -> float:
        rel = np.array([robot_state.x - float(target_xy[0]), robot_state.y - float(target_xy[1])], dtype=float)
        if float(np.linalg.norm(rel)) > 0.20:
            return math.atan2(float(rel[1]), float(rel[0]))
        return float(desired_slot)

    @staticmethod
    def _step_angle(current: float, desired: float, max_step: float) -> float:
        err = wrap_pi(float(desired) - float(current))
        step = float(np.clip(err, -abs(float(max_step)), abs(float(max_step))))
        return wrap_pi(float(current) + step)


def _segment_hits_circle(start_xy: np.ndarray, end_xy: np.ndarray, center_xy: np.ndarray, radius: float) -> bool:
    start = np.asarray(start_xy, dtype=float).reshape(2)
    end = np.asarray(end_xy, dtype=float).reshape(2)
    center = np.asarray(center_xy, dtype=float).reshape(2)
    seg = end - start
    denom = float(np.dot(seg, seg))
    if denom <= 1e-9:
        return float(np.linalg.norm(start - center)) <= float(radius)
    t = float(np.clip(np.dot(center - start, seg) / denom, 0.0, 1.0))
    closest = start + t * seg
    return float(np.linalg.norm(closest - center)) <= float(radius)
