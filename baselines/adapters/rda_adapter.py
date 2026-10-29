"""RDA point-goal follower adapter for the Follow-Bench platform."""

from __future__ import annotations

import math

import numpy as np

from baselines.planning.rda.rda import RdaPointPlanner
from baselines.adapters.rda_obstacles import NPC_OBS_HALF_SIZE_M, obstacles_debug, target_box_obstacle
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.perception.gt.gt_scene_provider import target_state_from_obs

from baselines.adapters.observation_utils import follow_goal_pose_from_obs


class RdaFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.8,
        follow_position: str = "back",
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        protect_target_for_side_follow: bool = True,
        target_protect_half_size: float = NPC_OBS_HALF_SIZE_M,
    ) -> None:
        self.desired_distance = desired_distance
        self.follow_position = follow_position
        self.protect_target_for_side_follow = bool(protect_target_for_side_follow)
        self.target_protect_half_size = float(target_protect_half_size)
        self._planner = RdaPointPlanner(
            dt=dt,
            receding=receding,
            iter_num=iter_num,
            process_num=process_num,
        )
        self._last_debug: dict = {"goal_point": None, "traj_points": [], "obstacles": []}

    def reset(self) -> None:
        self._planner.reset()
        self._last_debug = {"goal_point": None, "traj_points": [], "obstacles": []}

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        rx, ry = float(obs.robot.x), float(obs.robot.y)
        tx, ty = float(obs.target.x), float(obs.target.y)
        robot_yaw = float(obs.robot.yaw_rad)

        dist = math.hypot(tx - rx, ty - ry)
        goal_pose = follow_goal_pose_from_obs(obs, self.follow_position, self.desired_distance)
        target_speed = max(float(obs.target.speed), 0.3)
        obstacle_list = []
        if self.protect_target_for_side_follow and self.follow_position in ("left_side", "right_side"):
            target = target_state_from_obs(obs, self.target_protect_half_size)
            if target is not None:
                obstacle_list.append(target_box_obstacle(target, half_size=self.target_protect_half_size))

        result = self._planner.compute(
            robot_pose=np.array([rx, ry, robot_yaw], dtype=float),
            goal_pose=goal_pose,
            ref_speed=target_speed,
            obstacle_list=obstacle_list,
        )
        v = float(result["v_mps"])
        w = float(result["w_radps"])
        self._last_debug = {
            "goal_point": goal_pose[:2].tolist(),
            "traj_points": result.get("traj_points", []) if isinstance(result, dict) else [],
            "obstacles": obstacles_debug(obstacle_list),
        }
        print(
            f"[RDA] mode={self.follow_position} v={v:.3f} w={w:.3f}  "
            f"dist={dist:.2f}m  target_spd={target_speed:.2f}"
        )
        return FollowAction(v_mps=v, w_radps=w)

__all__ = ["RdaFollowerPolicy"]
