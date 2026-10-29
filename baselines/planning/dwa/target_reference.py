from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from baselines.common.geometry import wrap_pi


@dataclass
class DwaTargetReferenceConfig:
    goal_pos_alpha: float = 0.24
    goal_lateral_alpha: float = 0.14
    goal_yaw_alpha: float = 0.22
    max_goal_longitudinal_speed: float = 1.4
    max_goal_lateral_speed: float = 0.45
    max_goal_speed: float = 1.6
    goal_speed_margin: float = 0.45
    max_goal_yaw_rate: float = 0.8


class DwaTargetReferenceFilter:
    """Stabilizes DWA follow goals across ticks."""

    def __init__(self, config: DwaTargetReferenceConfig | None = None) -> None:
        self.config = config or DwaTargetReferenceConfig()
        self._goal_xy: np.ndarray | None = None
        self._goal_yaw: float | None = None
        self._target_speed = 0.0

    def reset(self) -> None:
        self._goal_xy = None
        self._goal_yaw = None
        self._target_speed = 0.0

    def smooth_goal_traj(
        self,
        raw_goal_traj: np.ndarray,
        tick_dt: float,
        plan_dt: float,
        follow_position: str = "back",
        target_speed: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        raw = np.asarray(raw_goal_traj, dtype=float).reshape(-1, 3)
        if raw.size == 0:
            return raw, {"goal_jump": 0.0}

        self._target_speed = float(max(target_speed, 0.0))
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
