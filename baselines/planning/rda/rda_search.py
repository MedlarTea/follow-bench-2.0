"""
RDA search follower algorithm core.

This module owns the solver-facing RDA MPC execution used by the search planner:
goal smoothing, solver invocation, clipping, and optimized trajectory extraction.
"""
from __future__ import annotations

import time
import warnings
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

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
class SearchControlResult:
    success: bool
    goal_pose: np.ndarray
    v: float = 0.0
    w: float = 0.0
    traj_points: list | None = None
    error: str | None = None
    mpc_ms: float = 0.0


class RdaSearchFollowerAlgorithm:
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
        self._goal_ema: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._mpc.reset()
        self._goal_ema = None

    def execute(self, robot, goal: np.ndarray, ref_speed: float, obstacle_list) -> SearchControlResult:
        smoothed_goal = self._smooth_goal(goal, float(robot.yaw))
        state = np.array([[float(robot.x)], [float(robot.y)], [float(robot.yaw)]], dtype=float)

        t_mpc_start = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                u_opt, info = self._mpc.control(
                    state=state,
                    goal=smoothed_goal,
                    ref_speed=float(ref_speed),
                    obstacle_list=obstacle_list,
                )
        except Exception as exc:
            return SearchControlResult(
                success=False,
                goal_pose=smoothed_goal,
                error=str(exc),
                mpc_ms=(time.perf_counter() - t_mpc_start) * 1000.0,
                traj_points=[],
            )

        mpc_ms = (time.perf_counter() - t_mpc_start) * 1000.0
        v = float(np.clip(u_opt[0, 0], -2.5, 2.5))
        w = float(np.clip(u_opt[1, 0], -2.5, 2.5))
        return SearchControlResult(
            success=True,
            goal_pose=smoothed_goal,
            v=v,
            w=w,
            traj_points=_extract_traj_points(info),
            mpc_ms=mpc_ms,
        )

    def _smooth_goal(self, goal: np.ndarray, robot_yaw: float) -> np.ndarray:
        if goal.shape != (3, 1):
            return np.array([[0.0], [0.0], [robot_yaw]], dtype=float)

        raw = goal[:, 0].astype(float)
        raw[2] = robot_yaw + wrap_pi(raw[2] - robot_yaw)
        if self._goal_ema is None:
            self._goal_ema = raw.copy()
        else:
            d_yaw = wrap_pi(raw[2] - self._goal_ema[2])
            self._goal_ema[0] += _EMA_ALPHA * (raw[0] - self._goal_ema[0])
            self._goal_ema[1] += _EMA_ALPHA * (raw[1] - self._goal_ema[1])
            self._goal_ema[2] += _EMA_ALPHA_YAW * d_yaw
        return self._goal_ema.reshape(3, 1)


def _extract_traj_points(info) -> list:
    states = []
    if isinstance(info, dict):
        states = info.get("opt_state_list") or []
    points = []
    for state in states:
        arr = np.asarray(state).reshape(-1)
        if arr.size >= 2:
            points.append([float(arr[0]), float(arr[1])])
    return points


__all__ = ["RdaSearchFollowerAlgorithm", "SearchControlResult"]
