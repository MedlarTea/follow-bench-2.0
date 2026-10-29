from __future__ import annotations

import math

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.common.types import AgentState2D, RobotState2D


_POSITION_OFFSETS = {
    "front": 0.0,
    "back": math.pi,
    # CARLA/UE uses a left-handed world yaw convention. For a target person's
    # body frame, visual-left is yaw - 90 deg and visual-right is yaw + 90 deg.
    "left_side": math.pi * 1.5,
    "right_side": math.pi * 0.5,
}


def compute_follow_goal(
    robot: RobotState2D,
    target: AgentState2D,
    position: str,
    distance: float,
) -> np.ndarray:
    if position not in _POSITION_OFFSETS:
        raise ValueError(f"Unsupported follow position: {position}")

    if position == "back":
        bearing = math.atan2(target.y - robot.y, target.x - robot.x)
        gx = target.x - distance * math.cos(bearing)
        gy = target.y - distance * math.sin(bearing)
        yaw = bearing
        return np.array([[gx], [gy], [yaw]], dtype=float)

    target_yaw = _target_heading(target)
    if position == "front":
        angle = wrap_pi(target_yaw + _POSITION_OFFSETS[position])
        gx = target.x + distance * math.cos(angle)
        gy = target.y + distance * math.sin(angle)
        return np.array([[gx], [gy], [target_yaw]], dtype=float)

    if position in ("left_side", "right_side"):
        gx, gy = _side_goal_xy(target, target_yaw, position, distance)
        return np.array([[gx], [gy], [target_yaw]], dtype=float)

    raise ValueError(f"Unsupported follow position: {position}")


def _side_goal_xy(target: AgentState2D, target_yaw: float, position: str, distance: float) -> np.ndarray:
    angle = wrap_pi(target_yaw + _POSITION_OFFSETS[position])
    return np.array(
        [
            target.x + distance * math.cos(angle),
            target.y + distance * math.sin(angle),
        ],
        dtype=float,
    )


def _target_heading(target: AgentState2D) -> float:
    if math.isfinite(float(target.yaw)):
        return float(target.yaw)
    if target.speed > 0.05 or math.hypot(target.vx, target.vy) > 0.05:
        return math.atan2(target.vy, target.vx)
    return 0.0
