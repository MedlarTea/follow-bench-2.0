"""Observation-to-planner helpers for platform adapters."""

from __future__ import annotations

import numpy as np

from baselines.behavior.follow_goal import compute_follow_goal
from baselines.perception.gt.gt_scene_provider import robot_state_from_obs, target_state_from_obs


def follow_goal_pose_from_obs(obs, follow_position: str, desired_distance: float, target_radius: float = 0.35) -> np.ndarray:
    robot = robot_state_from_obs(obs)
    target = target_state_from_obs(obs, target_radius)
    if target is None:
        raise ValueError("follow goal requires obs.target")
    return compute_follow_goal(robot, target, follow_position, desired_distance).reshape(-1)[:3]


__all__ = ["follow_goal_pose_from_obs"]
