"""RDA lidar follower adapter for the Follow-Bench platform."""

from __future__ import annotations

import time

import numpy as np

from baselines.planning.rda.rda_lidar import RdaLidarPlanner, predict_traj
from baselines.adapters.rda_obstacles import (
    LIDAR_RANGE_MAX_M,
    NPC_OBS_HALF_SIZE_M,
    TARGET_MASK_RADIUS_M,
    build_rda_obstacles,
    obstacles_debug,
    target_box_obstacle,
)
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.perception.gt.gt_scene_provider import target_state_from_obs

from baselines.adapters.observation_utils import follow_goal_pose_from_obs

_predict_traj = predict_traj


class RdaLidarFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        max_obs_num: int = 5,
        lidar_range_max: float = LIDAR_RANGE_MAX_M,
        target_mask_radius: float = TARGET_MASK_RADIUS_M,
        protect_target_for_side_follow: bool = True,
        target_protect_half_size: float = NPC_OBS_HALF_SIZE_M,
        use_npc_gt_obstacles: bool = True,
    ) -> None:
        self.desired_distance = desired_distance
        self.follow_position = follow_position
        self.dt = dt
        self.lidar_range_max = float(lidar_range_max)
        self.target_mask_radius = float(target_mask_radius)
        self.protect_target_for_side_follow = bool(protect_target_for_side_follow)
        self.target_protect_half_size = float(target_protect_half_size)
        self.use_npc_gt_obstacles = bool(use_npc_gt_obstacles)
        self._planner = RdaLidarPlanner(
            dt=dt,
            receding=receding,
            iter_num=iter_num,
            process_num=process_num,
            max_obs_num=max_obs_num,
        )
        self._last_debug: dict = {"obstacles": [], "traj_points": []}

    def reset(self) -> None:
        self._planner.reset()
        self._last_debug = {"obstacles": [], "traj_points": []}

    def get_debug_info(self) -> dict:
        return self._last_debug

    def build_obstacles(self, obs: FollowObservation):
        obstacle_list = build_rda_obstacles(
            obs,
            lidar_range_max=self.lidar_range_max,
            target_mask_radius=self.target_mask_radius,
            use_npc_gt=self.use_npc_gt_obstacles,
        )
        if self.protect_target_for_side_follow and self.follow_position in ("left_side", "right_side"):
            target = target_state_from_obs(obs, self.target_protect_half_size)
            if target is not None:
                obstacle_list.append(target_box_obstacle(target, half_size=self.target_protect_half_size))
        return obstacle_list

    def act(self, obs: FollowObservation) -> FollowAction:
        rx, ry = float(obs.robot.x), float(obs.robot.y)
        tx, ty = float(obs.target.x), float(obs.target.y)
        robot_yaw = float(obs.robot.yaw_rad)

        dist = float(np.hypot(tx - rx, ty - ry))
        goal_pose = follow_goal_pose_from_obs(obs, self.follow_position, self.desired_distance)
        target_speed = max(float(obs.target.speed), 0.3)

        t_obs_start = time.perf_counter()
        obstacle_list = self.build_obstacles(obs)
        t_obs_ms = (time.perf_counter() - t_obs_start) * 1000.0

        control = self._planner.compute(
            robot_pose=np.array([rx, ry, robot_yaw], dtype=float),
            goal_pose=goal_pose,
            ref_speed=target_speed,
            obstacle_list=obstacle_list,
        )
        if not control.success:
            print(
                f"[RDA] MPC error: {control.error} -- braking. n_obs={len(obstacle_list)} "
                f"t_obs={t_obs_ms:.1f}ms t_mpc={control.mpc_ms:.1f}ms"
            )
            return FollowAction(v_mps=0.0, w_radps=0.0)

        v = float(control.v_mps)
        w = float(control.w_radps)
        t_total_ms = t_obs_ms + float(control.mpc_ms)
        print(
            f"[RDA] mode={self.follow_position} v={v:.3f} w={w:.3f}  dist={dist:.2f}m  "
            f"target_spd={target_speed:.2f}  obs={len(obstacle_list)}  "
            f"t_obs={t_obs_ms:.1f}ms t_mpc={control.mpc_ms:.1f}ms t_total={t_total_ms:.1f}ms"
        )

        self._last_debug = {
            "obstacles": obstacles_debug(obstacle_list),
            "traj_points": predict_traj(rx, ry, robot_yaw, v, w),
            "goal_point": goal_pose[:2].tolist(),
        }
        return FollowAction(v_mps=v, w_radps=w)
__all__ = ["RdaLidarFollowerPolicy", "_predict_traj"]
