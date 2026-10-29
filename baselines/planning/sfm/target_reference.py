from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.planning.sfm.types import AgentPrediction


@dataclass
class TargetReferenceConfig:
    target_pos_alpha: float = 0.35
    target_vel_alpha: float = 0.25
    target_yaw_alpha: float = 0.22
    max_target_acc: float = 1.6
    max_target_yaw_rate: float = 1.0
    min_tangent_speed: float = 0.12
    history_size: int = 18
    history_min_span: float = 0.25
    goal_pos_alpha: float = 0.24
    goal_lateral_alpha: float = 0.14
    goal_yaw_alpha: float = 0.22
    max_goal_longitudinal_speed: float = 1.4
    max_goal_lateral_speed: float = 0.45
    max_goal_speed: float = 1.6
    goal_speed_margin: float = 0.45
    max_goal_yaw_rate: float = 0.8


class TargetReferenceFilter:
    """Stabilizes target observations and SFM follow goals across ticks."""

    def __init__(self, config: TargetReferenceConfig | None = None) -> None:
        self.config = config or TargetReferenceConfig()
        self._raw_xy: np.ndarray | None = None
        self._target_xy: np.ndarray | None = None
        self._target_vel: np.ndarray | None = None
        self._target_yaw: float | None = None
        self._goal_xy: np.ndarray | None = None
        self._goal_yaw: float | None = None
        self._history: list[np.ndarray] = []
        self._target_speed = 0.0

    def reset(self) -> None:
        self._raw_xy = None
        self._target_xy = None
        self._target_vel = None
        self._target_yaw = None
        self._goal_xy = None
        self._goal_yaw = None
        self._history = []
        self._target_speed = 0.0

    def target_prediction(
        self,
        target_state,
        tick_dt: float,
        num_steps: int,
        plan_dt: float,
    ) -> tuple[AgentPrediction, dict]:
        arr = np.asarray(target_state, dtype=float).reshape(-1)
        raw_xy = arr[:2].astype(float)
        raw_vel = np.array(
            [
                float(arr[2]) if arr.size >= 3 and math.isfinite(float(arr[2])) else 0.0,
                float(arr[3]) if arr.size >= 4 and math.isfinite(float(arr[3])) else 0.0,
            ],
            dtype=float,
        )
        raw_yaw = (
            float(arr[4])
            if arr.size >= 5 and math.isfinite(float(arr[4]))
            else _yaw_from_velocity(raw_vel, fallback=0.0)
        )
        dt = max(float(tick_dt), 1e-3)

        if self._target_xy is None or self._target_vel is None or self._target_yaw is None:
            filtered_xy = raw_xy.copy()
            filtered_vel = raw_vel.copy()
            filtered_yaw = _preferred_yaw(raw_vel, raw_yaw, raw_yaw, self.config.min_tangent_speed)
        else:
            predicted_xy = self._target_xy + self._target_vel * dt
            filtered_xy = predicted_xy + self.config.target_pos_alpha * (raw_xy - predicted_xy)

            if self._raw_xy is not None:
                observed_vel = (raw_xy - self._raw_xy) / dt
                measured_vel = 0.65 * raw_vel + 0.35 * observed_vel
            else:
                measured_vel = raw_vel
            desired_vel = self._target_vel + self.config.target_vel_alpha * (measured_vel - self._target_vel)
            filtered_vel = _limit_delta_vector(desired_vel, self._target_vel, self.config.max_target_acc * dt)

            history_yaw = self._history_yaw(fallback=self._target_yaw)
            yaw_measure = _preferred_yaw(
                filtered_vel,
                raw_yaw,
                fallback=self._target_yaw,
                min_speed=self.config.min_tangent_speed,
                path_yaw=history_yaw,
            )
            yaw_step = self.config.target_yaw_alpha * wrap_pi(yaw_measure - self._target_yaw)
            yaw_step = float(np.clip(yaw_step, -self.config.max_target_yaw_rate * dt, self.config.max_target_yaw_rate * dt))
            filtered_yaw = wrap_pi(self._target_yaw + yaw_step)

        self._raw_xy = raw_xy.copy()
        self._target_xy = filtered_xy.copy()
        self._target_vel = filtered_vel.copy()
        self._target_yaw = float(filtered_yaw)
        self._append_history(filtered_xy)
        self._target_speed = float(np.linalg.norm(filtered_vel))
        history_yaw = self._history_yaw(fallback=None)

        times = np.arange(max(int(num_steps), 1), dtype=float) * float(plan_dt)
        positions = filtered_xy[None, :] + times[:, None] * filtered_vel[None, :]
        velocities = np.tile(filtered_vel.reshape(1, 2), (len(times), 1))
        yaws = np.full((len(times),), filtered_yaw, dtype=float)
        debug = {
            "target_raw_xy": raw_xy.tolist(),
            "target_filtered_xy": filtered_xy.tolist(),
            "target_raw_velocity": raw_vel.tolist(),
            "target_filtered_velocity": filtered_vel.tolist(),
            "target_raw_speed": float(np.linalg.norm(raw_vel)),
            "target_filtered_speed": self._target_speed,
            "target_raw_yaw": float(raw_yaw),
            "target_filtered_yaw": float(filtered_yaw),
            "target_history_yaw": None if history_yaw is None else float(history_yaw),
        }
        return AgentPrediction(positions=positions, velocities=velocities, yaws=yaws), debug

    def smooth_goal_traj(
        self,
        raw_goal_traj: np.ndarray,
        tick_dt: float,
        plan_dt: float,
        follow_position: str = "back",
    ) -> tuple[np.ndarray, dict]:
        raw = np.asarray(raw_goal_traj, dtype=float).reshape(-1, 3)
        if raw.size == 0:
            return raw, {"goal_jump": 0.0}

        smooth = np.zeros_like(raw)
        goal_speed_limit = min(
            self.config.max_goal_speed,
            max(0.3, self._target_speed + self.config.goal_speed_margin),
        )
        lateral_speed_limit = self.config.max_goal_lateral_speed
        if follow_position not in {"left_side", "right_side"}:
            lateral_speed_limit = min(goal_speed_limit, lateral_speed_limit * 1.4)
        longitudinal_speed_limit = min(goal_speed_limit, self.config.max_goal_longitudinal_speed)

        raw0_xy = raw[0, :2]
        raw0_yaw = float(raw[0, 2])
        if self._goal_xy is None or self._goal_yaw is None:
            first_xy = raw0_xy.copy()
            first_yaw = raw0_yaw
            goal_jump = 0.0
        else:
            goal_jump = float(np.linalg.norm(raw0_xy - self._goal_xy))
            first_xy = self._smooth_goal_xy(
                previous_xy=self._goal_xy,
                raw_xy=raw0_xy,
                frame_yaw=raw0_yaw,
                dt=max(float(tick_dt), 1e-3),
                longitudinal_speed_limit=longitudinal_speed_limit,
                lateral_speed_limit=lateral_speed_limit,
            )
            yaw_step = self.config.goal_yaw_alpha * wrap_pi(raw0_yaw - self._goal_yaw)
            yaw_step = float(np.clip(yaw_step, -self.config.max_goal_yaw_rate * tick_dt, self.config.max_goal_yaw_rate * tick_dt))
            first_yaw = wrap_pi(self._goal_yaw + yaw_step)

        smooth[0, :2] = first_xy
        smooth[0, 2] = first_yaw
        for k in range(1, len(raw)):
            smooth[k, :2] = self._smooth_goal_xy(
                previous_xy=smooth[k - 1, :2],
                raw_xy=raw[k, :2],
                frame_yaw=float(raw[k, 2]),
                dt=max(float(plan_dt), 1e-3),
                longitudinal_speed_limit=longitudinal_speed_limit,
                lateral_speed_limit=lateral_speed_limit,
            )
            yaw_step = self.config.goal_yaw_alpha * wrap_pi(float(raw[k, 2]) - float(smooth[k - 1, 2]))
            yaw_step = float(np.clip(yaw_step, -self.config.max_goal_yaw_rate * plan_dt, self.config.max_goal_yaw_rate * plan_dt))
            smooth[k, 2] = wrap_pi(float(smooth[k - 1, 2]) + yaw_step)

        self._goal_xy = smooth[0, :2].copy()
        self._goal_yaw = float(smooth[0, 2])
        debug = {
            "raw_goal_point": raw0_xy.tolist(),
            "filtered_goal_point": smooth[0, :2].tolist(),
            "goal_jump": goal_jump,
            "goal_speed_limit": float(goal_speed_limit),
            "goal_longitudinal_speed_limit": float(longitudinal_speed_limit),
            "goal_lateral_speed_limit": float(lateral_speed_limit),
            "goal_filter_alpha": float(self.config.goal_pos_alpha),
        }
        return smooth, debug

    def _append_history(self, xy: np.ndarray) -> None:
        self._history.append(np.asarray(xy, dtype=float).reshape(2).copy())
        max_len = max(int(self.config.history_size), 2)
        if len(self._history) > max_len:
            self._history = self._history[-max_len:]

    def _history_yaw(self, fallback: float | None) -> float | None:
        if len(self._history) < 2:
            return fallback
        start = self._history[0]
        end = self._history[-1]
        delta = end - start
        if float(np.linalg.norm(delta)) < self.config.history_min_span:
            return fallback
        return math.atan2(float(delta[1]), float(delta[0]))

    def _smooth_goal_xy(
        self,
        previous_xy: np.ndarray,
        raw_xy: np.ndarray,
        frame_yaw: float,
        dt: float,
        longitudinal_speed_limit: float,
        lateral_speed_limit: float,
    ) -> np.ndarray:
        prev = np.asarray(previous_xy, dtype=float).reshape(2)
        raw = np.asarray(raw_xy, dtype=float).reshape(2)
        forward = np.array([math.cos(frame_yaw), math.sin(frame_yaw)], dtype=float)
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        delta = raw - prev
        longitudinal_delta = float(np.dot(delta, forward))
        lateral_delta = float(np.dot(delta, lateral))
        longitudinal_delta *= self.config.goal_pos_alpha
        lateral_delta *= self.config.goal_lateral_alpha
        longitudinal_delta = float(
            np.clip(
                longitudinal_delta,
                -float(longitudinal_speed_limit) * dt,
                float(longitudinal_speed_limit) * dt,
            )
        )
        lateral_delta = float(
            np.clip(
                lateral_delta,
                -float(lateral_speed_limit) * dt,
                float(lateral_speed_limit) * dt,
            )
        )
        return prev + longitudinal_delta * forward + lateral_delta * lateral


def _preferred_yaw(
    velocity: np.ndarray,
    raw_yaw: float,
    fallback: float,
    min_speed: float,
    path_yaw: float | None = None,
) -> float:
    if path_yaw is not None and math.isfinite(float(path_yaw)):
        return float(path_yaw)
    if float(np.linalg.norm(velocity)) >= float(min_speed):
        return _yaw_from_velocity(velocity, fallback=fallback)
    return float(raw_yaw) if math.isfinite(float(raw_yaw)) else float(fallback)


def _yaw_from_velocity(velocity: np.ndarray, fallback: float) -> float:
    vel = np.asarray(velocity, dtype=float).reshape(2)
    if float(np.linalg.norm(vel)) <= 1e-9:
        return float(fallback)
    return math.atan2(float(vel[1]), float(vel[0]))


def _limit_delta_vector(value: np.ndarray, previous: np.ndarray, max_delta: float) -> np.ndarray:
    value_arr = np.asarray(value, dtype=float).reshape(2)
    previous_arr = np.asarray(previous, dtype=float).reshape(2)
    delta = value_arr - previous_arr
    norm = float(np.linalg.norm(delta))
    if norm <= float(max_delta) or norm <= 1e-9:
        return value_arr
    return previous_arr + delta * (float(max_delta) / norm)
