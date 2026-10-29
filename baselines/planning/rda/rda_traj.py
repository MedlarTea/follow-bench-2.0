"""
RDA trajectory-chasing MPC planner.
"""
from __future__ import annotations

import warnings
from collections import namedtuple

import numpy as np

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


class RdaTrajPlanner:
    def __init__(
        self,
        dt: float,
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        enable_uncertainty_brake: bool = False,
    ) -> None:
        from baselines.planning.rda.core.mpc_chasing_traj import MPC

        self._mpc = MPC(
            car_tuple=_SCOUT_CAR,
            receding=receding,
            sample_time=dt,
            iter_num=iter_num,
            process_num=process_num,
            max_edge_num=4,
            max_obs_num=5,
            obstacle_order=True,
            ws=10.0,
            wu=2.0,
            wo=1.0,
            sigma_o=0.5,
            slack_gain=10,
            accelerated=True,
            time_print=False,
        )
        self._nis_threshold = 4.0 if enable_uncertainty_brake else float("inf")

    def reset(self) -> None:
        self._mpc.reset()

    def compute(self, robot_pose, goal_traj, target_traj, ref_speed: float, obstacle_list=None) -> dict:
        state = np.asarray(robot_pose, dtype=float).reshape(3, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            u_opt, info = self._mpc.control(
                state,
                goal_traj,
                target_traj,
                0.0,
                self._nis_threshold,
                float(ref_speed),
                [] if obstacle_list is None else obstacle_list,
            )
        return {
            "v_mps": float(np.clip(u_opt[0, 0], -2.5, 2.5)),
            "w_radps": float(np.clip(u_opt[1, 0], -2.5, 2.5)),
            "info": info,
        }


__all__ = ["RdaTrajPlanner"]
