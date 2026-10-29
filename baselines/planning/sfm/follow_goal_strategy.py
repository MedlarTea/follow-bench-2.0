from __future__ import annotations

import math

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.planning.sfm.types import AgentPrediction, KinematicState


class SfmFollowGoalStrategy:
    def __init__(
        self,
        back_goal_mode: str = "target_heading",
    ) -> None:
        self.back_goal_mode = str(back_goal_mode)

    def reset(self) -> None:
        pass

    def build_goal_traj(
        self,
        robot_state: KinematicState,
        target_prediction: AgentPrediction,
        follow_position: str,
        desired_distance: float,
    ) -> np.ndarray:
        goals = np.zeros((len(target_prediction.positions), 3), dtype=float)
        for k, target_xy in enumerate(target_prediction.positions):
            yaw = float(target_prediction.yaws[min(k, len(target_prediction.yaws) - 1)])
            if follow_position == "back":
                gx, gy, gyaw = self._back_goal(robot_state, target_xy, yaw, desired_distance)
            elif follow_position == "front":
                gx = target_xy[0] + desired_distance * math.cos(yaw)
                gy = target_xy[1] + desired_distance * math.sin(yaw)
                gyaw = yaw
            elif follow_position == "left_side":
                gx, gy = self._side_goal(target_xy, yaw, desired_distance, left=True)
                gyaw = yaw
            elif follow_position == "right_side":
                gx, gy = self._side_goal(target_xy, yaw, desired_distance, left=False)
                gyaw = yaw
            else:
                raise ValueError(f"Unsupported follow position: {follow_position}")
            goals[k] = np.array([gx, gy, gyaw], dtype=float)
        return goals

    def _back_goal(
        self,
        robot_state: KinematicState,
        target_xy: np.ndarray,
        target_yaw: float,
        desired_distance: float,
    ) -> tuple[float, float, float]:
        if self.back_goal_mode == "bearing":
            bearing = math.atan2(target_xy[1] - robot_state.y, target_xy[0] - robot_state.x)
            return (
                float(target_xy[0] - desired_distance * math.cos(bearing)),
                float(target_xy[1] - desired_distance * math.sin(bearing)),
                bearing,
            )
        return (
            float(target_xy[0] - desired_distance * math.cos(target_yaw)),
            float(target_xy[1] - desired_distance * math.sin(target_yaw)),
            target_yaw,
        )

    @staticmethod
    def _side_goal(target_xy: np.ndarray, target_yaw: float, desired_distance: float, left: bool) -> tuple[float, float]:
        # CARLA target-body sides are left-handed: visual-left is yaw - 90 deg.
        offset = -math.pi * 0.5 if left else math.pi * 0.5
        angle = wrap_pi(target_yaw + offset)
        return (
            float(target_xy[0] + desired_distance * math.cos(angle)),
            float(target_xy[1] + desired_distance * math.sin(angle)),
        )
