from __future__ import annotations

from typing import Optional, Sequence


def is_target_visible(obs, min_pixel_count: int = 1) -> bool:
    if obs is None or getattr(obs, "target", None) is None:
        return False
    if not bool(getattr(obs, "target_visible", False)):
        return False
    return int(getattr(obs, "target_pixel_count", 0) or 0) >= int(min_pixel_count)


def gt_target_xy(obs) -> Optional[tuple[float, float]]:
    target = getattr(obs, "target", None)
    if target is None:
        return None
    return float(target.x), float(target.y)


def gt_robot_xy(obs) -> Optional[tuple[float, float]]:
    robot = getattr(obs, "robot", None)
    if robot is None:
        return None
    return float(robot.x), float(robot.y)


def gt_distance_xy(obs, point_xy: Sequence[float]) -> Optional[float]:
    target_xy = gt_target_xy(obs)
    if target_xy is None or point_xy is None:
        return None
    dx = float(point_xy[0]) - float(target_xy[0])
    dy = float(point_xy[1]) - float(target_xy[1])
    return (dx * dx + dy * dy) ** 0.5


__all__ = ["gt_distance_xy", "gt_robot_xy", "gt_target_xy", "is_target_visible"]
