"""GT expert follower: RDA-Lidar MPC with GT dynamics + ROI static map obstacles."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from baselines.adapters.rda_obstacles import (
    NPC_OBS_HALF_SIZE_M,
    TARGET_MASK_RADIUS_M,
    obstacles_debug,
    target_box_obstacle,
    to_rda_obstacle,
)
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.maps.roi_static_map import RoiStaticMap
from baselines.perception.gt.gt_scene_provider import target_state_from_obs
from baselines.perception.gt.npc_obstacles import npc_box_obstacles
from baselines.planning.rda.rda_lidar import RdaLidarPlanner, predict_traj

from baselines.adapters.observation_utils import follow_goal_pose_from_obs

GT_NPC_OBS_RANGE_M = 12.0
GT_STATIC_OBS_RANGE_M = 8.0


class GtPlannerFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        grid_npz: Optional[str] = None,
        receding: int = 20,
        iter_num: int = 5,
        process_num: int = 8,
        max_obs_num: int = 5,
        static_obs_range_m: float = GT_STATIC_OBS_RANGE_M,
        target_mask_radius: float = TARGET_MASK_RADIUS_M,
        npc_obs_range_m: float = GT_NPC_OBS_RANGE_M,
        protect_target_for_side_follow: bool = True,
        target_protect_half_size: float = NPC_OBS_HALF_SIZE_M,
    ) -> None:
        if not grid_npz:
            raise ValueError("gt_planner requires grid_npz= path to gridmap_roi.npz")
        self.desired_distance = desired_distance
        self.follow_position = follow_position
        self.dt = dt
        self.static_obs_range_m = float(static_obs_range_m)
        self.target_mask_radius = float(target_mask_radius)
        self.npc_obs_range_m = float(npc_obs_range_m)
        self.protect_target_for_side_follow = bool(protect_target_for_side_follow)
        self.target_protect_half_size = float(target_protect_half_size)
        self._static_map = RoiStaticMap(grid_npz)
        self._planner = RdaLidarPlanner(
            dt=dt,
            receding=receding,
            iter_num=iter_num,
            process_num=process_num,
            max_obs_num=max_obs_num,
        )
        self._last_debug: dict = {
            "obstacles": [], "traj_points": [], "goal_point": None, "global_path": [],
        }

    def reset(self) -> None:
        self._planner.reset()
        self._last_debug = {
            "obstacles": [], "traj_points": [], "goal_point": None, "global_path": [],
        }

    def get_debug_info(self) -> dict:
        return self._last_debug

    def build_obstacles(self, obs: FollowObservation):
        rx, ry = float(obs.robot.x), float(obs.robot.y)
        target_xy = None
        target_track_id = None
        if obs.target is not None:
            target_xy = (float(obs.target.x), float(obs.target.y))
            target_track_id = str(obs.target.track_id)

        static_items = self._static_map.static_obstacles_near(
            rx,
            ry,
            self.static_obs_range_m,
            target_xy=target_xy,
            target_mask_radius=self.target_mask_radius,
        )
        obstacle_list = [to_rda_obstacle(item) for item in static_items]

        if obs.npcs:
            obstacle_list.extend(
                to_rda_obstacle(item)
                for item in npc_box_obstacles(
                    obs.npcs,
                    target_track_id=target_track_id,
                    robot_xy=(rx, ry),
                    range_m=self.npc_obs_range_m,
                    half_size=NPC_OBS_HALF_SIZE_M,
                )
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
        global_path = self._static_map.plan_path_world(
            rx, ry, float(goal_pose[0]), float(goal_pose[1]),
        )

        t_obs_start = time.perf_counter()
        # obstacle_list = self.build_obstacles(obs)
        obstacle_list = []
        t_obs_ms = (time.perf_counter() - t_obs_start) * 1000.0

        control = self._planner.compute(
            robot_pose=np.array([rx, ry, robot_yaw], dtype=float),
            goal_pose=goal_pose,
            ref_speed=target_speed,
            obstacle_list=obstacle_list,
        )
        if not control.success:
            print(
                f"[GT_PLANNER] MPC error: {control.error} -- braking. n_obs={len(obstacle_list)} "
                f"t_obs={t_obs_ms:.1f}ms t_mpc={control.mpc_ms:.1f}ms"
            )
            self._last_debug = {
                "obstacles": obstacles_debug(obstacle_list),
                "traj_points": [],
                "goal_point": goal_pose[:2].tolist(),
                "global_path": global_path,
            }
            return FollowAction(v_mps=0.0, w_radps=0.0)

        v = float(control.v_mps)
        w = float(control.w_radps)
        print(
            f"[GT_PLANNER] mode={self.follow_position} v={v:.3f} w={w:.3f}  dist={dist:.2f}m  "
            f"target_spd={target_speed:.2f}  obs={len(obstacle_list)}  "
            f"t_obs={t_obs_ms:.1f}ms t_mpc={control.mpc_ms:.1f}ms"
        )
        self._last_debug = {
            "obstacles": obstacles_debug(obstacle_list),
            "traj_points": predict_traj(rx, ry, robot_yaw, v, w),
            "goal_point": goal_pose[:2].tolist(),
            "global_path": global_path,
        }
        return FollowAction(v_mps=v, w_radps=w)


__all__ = ["GtPlannerFollowerPolicy", "GT_NPC_OBS_RANGE_M", "GT_STATIC_OBS_RANGE_M"]
