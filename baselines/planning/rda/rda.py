"""
RDA point-goal MPC planner.

This is the platform-neutral RDA point controller. Adapters provide robot state,
goal pose, and reference speed.
"""
from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np

from baselines.common.geometry import wrap_pi

_ROBOT_RADIUS = 0.5
_CarTuple = namedtuple("car_tuple", "G h cone_type wheelbase max_speed max_acce dynamics")
_SCOUT_CAR = _CarTuple(
    G=np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
    h=np.array([[0.0], [0.0], [-_ROBOT_RADIUS]]),
    cone_type="norm2",
    wheelbase=0.5,
    max_speed=[2.5, 2.5],
    max_acce=[2.0, 4.0],
    dynamics="diff",
)

_EMA_ALPHA = 0.5
_EMA_ALPHA_YAW = 0.15


class RdaPointPlanner:
    def __init__(
        self,
        dt: float,
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
    ) -> None:
        from baselines.planning.rda.core.mpc_chasing_point2 import MPC

        self._mpc = MPC(
            car_tuple=_SCOUT_CAR,
            receding=receding,
            sample_time=dt,
            iter_num=iter_num,
            process_num=process_num,
            max_obs_num=1,
            accelerated=True,
            time_print=False,
            ws=5.0,
            wu=1.0,
        )
        self._goal_ema: np.ndarray | None = None

    def reset(self) -> None:
        self._mpc.reset()
        self._goal_ema = None

    def compute(self, robot_pose, goal_pose, ref_speed: float, obstacle_list=None) -> dict:
        robot_pose_arr = np.asarray(robot_pose, dtype=float).reshape(3)
        goal_pose_arr = np.asarray(goal_pose, dtype=float).reshape(3)
        smoothed_goal = self._smooth_goal(goal_pose_arr)
        state = robot_pose_arr.reshape(3, 1)
        goal = smoothed_goal.reshape(3, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            u_opt, info = self._mpc.control(
                state=state,
                goal=goal,
                ref_speed=float(ref_speed),
                obstacle_list=[] if obstacle_list is None else obstacle_list,
            )
        return {
            "v_mps": float(np.clip(u_opt[0, 0], -2.5, 2.5)),
            "w_radps": float(np.clip(u_opt[1, 0], -2.5, 2.5)),
            "goal_pose": goal,
            "info": info,
        }

    def _smooth_goal(self, goal_pose: np.ndarray) -> np.ndarray:
        gx, gy, gyaw = goal_pose
        if self._goal_ema is None:
            self._goal_ema = np.array([gx, gy, gyaw], dtype=float)
        else:
            d_yaw = wrap_pi(float(gyaw - self._goal_ema[2]))
            self._goal_ema[0] += _EMA_ALPHA * (gx - self._goal_ema[0])
            self._goal_ema[1] += _EMA_ALPHA * (gy - self._goal_ema[1])
            self._goal_ema[2] = wrap_pi(float(self._goal_ema[2] + _EMA_ALPHA_YAW * d_yaw))
        return self._goal_ema.copy()


__all__ = ["RdaPointPlanner"]
