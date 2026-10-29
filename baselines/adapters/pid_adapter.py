"""PID follower adapter for the Follow-Bench platform."""

from __future__ import annotations

import math

from baselines.planning.pid.pid import PIDFollowerController
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter

from baselines.adapters.observation_utils import follow_goal_pose_from_obs


class PIDFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 3.0,
        follow_position: str = "back",
        max_vx: float = 2.5,
        max_va: float = 2.0,
        enable_back: bool = True,
        scale_a: float = 2.5,
        scale_v: float = 2.0,
    ) -> None:
        self.desired_distance = float(desired_distance)
        self.follow_position = follow_position
        self._controller = PIDFollowerController(
            dt=dt,
            desired_distance=desired_distance,
            max_vx=max_vx,
            max_va=max_va,
            enable_back=enable_back,
            scale_a=scale_a,
            scale_v=scale_v,
        )
        self._goal_controller = PIDFollowerController(
            dt=dt,
            desired_distance=0.0,
            max_vx=max_vx,
            max_va=max_va,
            enable_back=enable_back,
            scale_a=scale_a,
            scale_v=scale_v,
        )
        self._last_debug: dict = {"goal_point": None}

    def reset(self) -> None:
        self._controller.reset()
        self._goal_controller.reset()
        self._last_debug = {"goal_point": None}

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        controller = self._controller
        target_x = float(obs.target.x)
        target_y = float(obs.target.y)
        if self.follow_position != "back":
            goal = follow_goal_pose_from_obs(obs, self.follow_position, self.desired_distance)
            target_x = float(goal[0])
            target_y = float(goal[1])
            controller = self._goal_controller
        else:
            goal = follow_goal_pose_from_obs(obs, self.follow_position, self.desired_distance)

        command = controller.compute(
            robot_x=float(obs.robot.x),
            robot_y=float(obs.robot.y),
            robot_yaw=float(obs.robot.yaw_rad),
            target_x=target_x,
            target_y=target_y,
        )

        print(
            f"[PID] mode={self.follow_position} px={command['px']:.3f} py={command['py']:.3f} "
            f"robot_speed={obs.robot.speed:.3f} target_speed={obs.target.speed:.3f}"
        )
        print(
            f"[PID] vx={command['v_mps']:.3f} va={command['w_radps']:.3f} "
            f"th_err={math.degrees(command['th_err']):.1f}deg p_err={command['p_err']:.3f}"
        )

        self._last_debug = {"goal_point": goal[:2].tolist()}
        return FollowAction(v_mps=command["v_mps"], w_radps=command["w_radps"])

__all__ = ["PIDFollowerPolicy"]
