"""
RDA lidar-obstacle MPC planner.
"""
from __future__ import annotations

import math
import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import List

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.planning.rda.rda_obstacles import ROBOT_RADIUS

_CarTuple = namedtuple("car_tuple", "G h cone_type wheelbase max_speed max_acce dynamics")
_SCOUT_CAR = _CarTuple(
    G=np.array([[1, 0], [0, 1], [0, 0]], dtype=float),
    h=np.array([[0.0], [0.0], [-ROBOT_RADIUS]]),
    cone_type="norm2",
    wheelbase=0.5,
    max_speed=[2.5, 2.5],
    max_acce=[2.0, 4.0],
    dynamics="diff",
)

_EMA_ALPHA = 0.5
_EMA_ALPHA_YAW = 0.15


@dataclass
class RdaLidarControlResult:
    success: bool
    v_mps: float = 0.0
    w_radps: float = 0.0
    goal_pose: np.ndarray | None = None
    info: dict | None = None
    error: str | None = None
    mpc_ms: float = 0.0


def predict_traj(
    x: float,
    y: float,
    yaw: float,
    v: float,
    w: float,
    N: int = 16,
    dt: float = 0.1,
) -> List[List[float]]:
    pts = []
    for _ in range(N):
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        pts.append([x, y])
    return pts


class RdaLidarPlanner:
    def __init__(
        self,
        dt: float,
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        max_obs_num: int = 5,
    ) -> None:
        from baselines.planning.rda.core.mpc_chasing_point2 import MPC

        self._mpc = MPC(
            car_tuple=_SCOUT_CAR,
            receding=receding,
            sample_time=dt,
            iter_num=iter_num,
            process_num=process_num,
            max_obs_num=max_obs_num,
            max_edge_num=4,
            obstacle_order=True,
            accelerated=True,
            time_print=False,
            ws=5.0,
            wu=1.0,
            slack_gain=10,
        )
        self._goal_ema: np.ndarray | None = None

    def reset(self) -> None:
        self._mpc.reset()
        self._goal_ema = None

    def compute(self, robot_pose, goal_pose, ref_speed: float, obstacle_list) -> RdaLidarControlResult:
        robot_pose_arr = np.asarray(robot_pose, dtype=float).reshape(3)
        goal_pose_arr = np.asarray(goal_pose, dtype=float).reshape(3)
        smoothed_goal = self._smooth_goal(goal_pose_arr)
        state = robot_pose_arr.reshape(3, 1)
        goal = smoothed_goal.reshape(3, 1)

        t_start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                u_opt, info = self._mpc.control(
                    state=state,
                    goal=goal,
                    ref_speed=float(ref_speed),
                    obstacle_list=obstacle_list,
                )
        except Exception as exc:
            return RdaLidarControlResult(
                success=False,
                goal_pose=goal,
                error=str(exc),
                mpc_ms=(time.perf_counter() - t_start) * 1000.0,
            )

        return RdaLidarControlResult(
            success=True,
            v_mps=float(np.clip(u_opt[0, 0], -2.5, 2.5)),
            w_radps=float(np.clip(u_opt[1, 0], -2.5, 2.5)),
            goal_pose=goal,
            info=info,
            mpc_ms=(time.perf_counter() - t_start) * 1000.0,
        )

    def _smooth_goal(self, goal_pose: np.ndarray) -> np.ndarray:
        gx, gy, gyaw = goal_pose
        if self._goal_ema is None:
            self._goal_ema = np.array([gx, gy, gyaw], dtype=float)
        else:
            d_yaw = wrap_pi(float(gyaw - self._goal_ema[2]))
            self._goal_ema[0] += _EMA_ALPHA * (gx - self._goal_ema[0])
            self._goal_ema[1] += _EMA_ALPHA * (gy - self._goal_ema[1])
            self._goal_ema[2] += _EMA_ALPHA_YAW * d_yaw
        return self._goal_ema.copy()


__all__ = ["RdaLidarControlResult", "RdaLidarPlanner", "predict_traj"]
