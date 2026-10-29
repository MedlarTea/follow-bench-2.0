from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from typing import List

import numpy as np


from baselines.planning.rda.rda_search import RdaSearchFollowerAlgorithm
from baselines.adapters.rda_obstacles import NPC_OBS_HALF_SIZE_M, build_rda_obstacles, obstacles_debug, target_box_obstacle
from baselines.behavior.search.config import SearchConfig
from baselines.behavior.search.mode_manager import SearchMode
from baselines.behavior.search.target_search import TargetSearchPlanner
from baselines.common.types import PredictionBundle
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.maps.lidar_grid_builder import build_local_occupancy_grid
from baselines.maps.null_map import NullMapQuery
from baselines.perception.gt.gt_scene_provider import agent_states_from_obs, robot_state_from_obs, target_state_from_obs
from baselines.perception.gt.gt_visibility import is_target_visible
from baselines.prediction import TrajectoryHistory, TrajectoryPredictionService


class RdaSearchFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        predictor_name: str = "cvkf",
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        max_obs_num: int = 5,
        overtake_sample_count: int = 20,
        fluid_sample_count: int = 30,
        lidar_range_max: float = 5.0,
        local_map_resolution: float = 0.2,
        local_map_width_cells: int | None = None,
        local_map_height_cells: int | None = None,
        local_map_inflation_radius: float = 0.45,
        human_radius: float = 0.35,
        target_radius: float = 0.35,
        enable_search: bool = True,
        gt_target_always_visible: bool = False,
        protect_target_for_side_follow: bool = True,
        target_protect_half_size: float = NPC_OBS_HALF_SIZE_M,
        use_npc_gt_obstacles: bool = True,
        enable_profiling: bool = False,
        profiling_window: int = 100,
        profiling_print_interval: int = 10,
    ) -> None:
        self.dt = float(dt)
        self.desired_distance = float(desired_distance)
        self.follow_position = follow_position
        self.predictor_name = predictor_name
        self.human_radius = float(human_radius)
        self.target_radius = float(target_radius)
        self.enable_search = bool(enable_search)
        self.gt_target_always_visible = bool(gt_target_always_visible)
        self.protect_target_for_side_follow = bool(protect_target_for_side_follow)
        self.target_protect_half_size = float(target_protect_half_size)
        self.use_npc_gt_obstacles = bool(use_npc_gt_obstacles)
        self.lidar_range_max = float(lidar_range_max)
        self.local_map_resolution = float(local_map_resolution)
        self.local_map_width_cells = None if local_map_width_cells is None else int(local_map_width_cells)
        self.local_map_height_cells = None if local_map_height_cells is None else int(local_map_height_cells)
        self.local_map_inflation_radius = float(local_map_inflation_radius)
        self.enable_profiling = bool(enable_profiling)
        self.profiling_window = max(1, int(profiling_window))
        self.profiling_print_interval = max(1, int(profiling_print_interval))

        self._algorithm = RdaSearchFollowerAlgorithm(
            dt=dt,
            receding=receding,
            iter_num=iter_num,
            process_num=process_num,
            max_obs_num=max_obs_num,
        )
        self._predictor = TrajectoryPredictionService(predictor_name, dt)
        self._history = TrajectoryHistory(self._predictor.history_length)
        self._search = TargetSearchPlanner(
            SearchConfig(
                overtake_sample_count=int(overtake_sample_count),
                fluid_sample_count=int(fluid_sample_count),
            ),
            target_radius=target_radius,
        )
        self._map_query = NullMapQuery()
        self._last_debug: dict = {"obstacles": [], "traj_points": []}
        self._profile_records = deque(maxlen=self.profiling_window)
        self._tick_counter = 0

    def reset(self) -> None:
        self._algorithm.reset()
        self._predictor.reset()
        self._history.reset()
        self._search.reset()
        self._last_debug = {"obstacles": [], "traj_points": []}
        self._profile_records.clear()
        self._tick_counter = 0

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.target is None:
            return FollowAction(v_mps=0.0, w_radps=0.0)

        self._tick_counter += 1
        t_total_start = time.perf_counter()
        t_stage = t_total_start

        robot = robot_state_from_obs(obs)
        target = target_state_from_obs(obs, self.human_radius)
        agents = agent_states_from_obs(obs, self.human_radius)
        t_state_ms = _elapsed_ms(t_stage)
        t_stage = time.perf_counter()

        raw_target_visible = is_target_visible(obs, min_pixel_count=0)
        if not self.enable_search or (self.gt_target_always_visible and obs.target is not None):
            target_visible = True
        else:
            target_visible = raw_target_visible
        self._map_query = build_local_occupancy_grid(
            lidar_points=obs.lidar_points,
            lidar_extrinsics=obs.lidar_extrinsics_robot_to_sensor,
            robot_x=robot.x,
            robot_y=robot.y,
            robot_yaw=robot.yaw,
            resolution=self.local_map_resolution,
            width_cells=self.local_map_width_cells,
            height_cells=self.local_map_height_cells,
            inflation_radius_m=self.local_map_inflation_radius,
            lidar_range_max=self.lidar_range_max,
        )
        t_map_ms = _elapsed_ms(t_stage)
        t_stage = time.perf_counter()

        self._history.update(robot, target, agents, target_visible)
        t_history_ms = _elapsed_ms(t_stage)
        t_stage = time.perf_counter()

        prediction = None
        occluder_id = None
        search_cost = float("inf")
        search_samples = np.empty((0, 2), dtype=float)
        search_anchor_xy = None
        search_hidden_steps = 0
        search_detail_mode = SearchMode.FOLLOW
        t_prediction_ms = 0.0
        t_search_ms = 0.0

        if not target_visible:
            pred_start = time.perf_counter()
            prediction = self._predictor.predict(self._history)
            t_prediction_ms = _elapsed_ms(pred_start)
        search_start = time.perf_counter()
        result = self._search.plan_goal(
            robot=robot,
            target=target,
            target_visible=target_visible,
            agents=agents,
            prediction=prediction if prediction is not None else self._empty_prediction(),
            map_query=self._map_query,
            search_direction=self._fallback_search_direction(robot),
            follow_position=self.follow_position,
            desired_distance=self.desired_distance,
        )
        goal = result.goal_pose
        mode = result.mode
        search_detail_mode = result.detail_mode or result.mode
        search_samples = result.samples
        occluder_id = result.occluder_id
        search_cost = result.cost
        search_anchor_xy = result.anchor_xy
        search_hidden_steps = result.hidden_steps
        t_search_ms = 0.0 if target_visible else _elapsed_ms(search_start)

        ref_speed = max(float(target.speed), 0.3) if target_visible else 0.8

        t_obs_start = time.perf_counter()
        obstacle_list = build_rda_obstacles(
            obs,
            lidar_range_max=self.lidar_range_max,
            use_npc_gt=self.use_npc_gt_obstacles,
        )
        target_protected = self._append_target_protection(obstacle_list, target)
        t_obs_ms = (time.perf_counter() - t_obs_start) * 1000.0

        control = self._algorithm.execute(robot=robot, goal=goal, ref_speed=ref_speed, obstacle_list=obstacle_list)
        if not control.success:
            t_mpc_ms = float(control.mpc_ms)
            print(f"[RDA-Search] MPC error: {control.error} -- braking. mode={mode} obs={len(obstacle_list)} "
                  f"t_obs={t_obs_ms:.1f}ms t_mpc={t_mpc_ms:.1f}ms")
            profiling = self._build_profile(
                mode=mode,
                target_visible=target_visible,
                raw_target_visible=raw_target_visible,
                t_state_ms=t_state_ms,
                t_map_ms=t_map_ms,
                t_history_ms=t_history_ms,
                t_prediction_ms=t_prediction_ms,
                t_search_ms=t_search_ms,
                t_obstacle_ms=t_obs_ms,
                t_mpc_ms=t_mpc_ms,
                t_total_ms=_elapsed_ms(t_total_start),
                num_agents=len(agents),
                num_obstacles=len(obstacle_list),
                num_search_samples=len(search_samples),
                target_protected=target_protected,
            )
            self._update_debug(
                obstacle_list,
                [],
                mode,
                control.goal_pose,
                search_samples,
                prediction,
                occluder_id,
                search_cost,
                search_anchor_xy,
                search_detail_mode,
                search_hidden_steps,
                profiling,
            )
            self._record_profile(profiling)
            return FollowAction(v_mps=0.0, w_radps=0.0)

        t_mpc_ms = float(control.mpc_ms)
        v = float(control.v)
        w = float(control.w)
        traj_points = control.traj_points or []
        goal = control.goal_pose
        t_total_ms = _elapsed_ms(t_total_start)
        profiling = self._build_profile(
            mode=mode,
            target_visible=target_visible,
            raw_target_visible=raw_target_visible,
            t_state_ms=t_state_ms,
            t_map_ms=t_map_ms,
            t_history_ms=t_history_ms,
            t_prediction_ms=t_prediction_ms,
            t_search_ms=t_search_ms,
            t_obstacle_ms=t_obs_ms,
            t_mpc_ms=t_mpc_ms,
            t_total_ms=t_total_ms,
            num_agents=len(agents),
            num_obstacles=len(obstacle_list),
            num_search_samples=len(search_samples),
            target_protected=target_protected,
        )

        dist = math.hypot(float(target.x) - robot.x, float(target.y) - robot.y)
        goal_xy = np.asarray(goal[:2, 0], dtype=float) if goal.shape == (3, 1) else np.asarray([np.nan, np.nan])
        goal_target_dist = float(np.linalg.norm(goal_xy - target.xy)) if np.all(np.isfinite(goal_xy)) else float("nan")
        robot_goal_dist = float(np.linalg.norm(goal_xy - robot.xy)) if np.all(np.isfinite(goal_xy)) else float("nan")
        print(f"[RDA-Search] mode={mode} pos={self.follow_position} v={v:.3f} w={w:.3f} "
              f"dist={dist:.2f}m goal_t={goal_target_dist:.2f}m goal_r={robot_goal_dist:.2f}m "
              f"visible={int(target_visible)} raw_visible={int(raw_target_visible)} "
              f"protect={int(target_protected)} obs={len(obstacle_list)} "
              f"t_obs={t_obs_ms:.1f}ms t_mpc={t_mpc_ms:.1f}ms")

        self._update_debug(
            obstacle_list,
            traj_points,
            mode,
            goal,
            search_samples,
            prediction,
            occluder_id,
            search_cost,
            search_anchor_xy,
            search_detail_mode,
            search_hidden_steps,
            profiling,
        )
        self._record_profile(profiling)
        return FollowAction(v_mps=v, w_radps=w)

    def _fallback_search_direction(self, robot) -> np.ndarray:
        last_target = self._history.last_target_xy()
        if last_target is None:
            return np.array([math.cos(robot.yaw), math.sin(robot.yaw)], dtype=float)

        direction = last_target - robot.xy
        if np.linalg.norm(direction) <= 1e-6:
            return np.array([math.cos(robot.yaw), math.sin(robot.yaw)], dtype=float)
        return direction

    def _empty_prediction(self):
        return PredictionBundle()

    def _append_target_protection(self, obstacle_list, target) -> bool:
        if not self.protect_target_for_side_follow:
            return False
        if self.follow_position not in ("left_side", "right_side"):
            return False
        obstacle_list.append(target_box_obstacle(target, half_size=self.target_protect_half_size))
        return True

    def _update_debug(
        self,
        obstacle_list,
        traj_points,
        mode: str,
        goal: np.ndarray,
        search_samples: np.ndarray,
        prediction,
        occluder_id,
        search_cost: float,
        search_anchor_xy,
        search_detail_mode: str | None,
        search_hidden_steps: int,
        profiling: dict | None,
    ) -> None:
        predicted_target = []
        if prediction is not None and prediction.target_traj is not None:
            predicted_target = prediction.target_traj[:, :2].tolist()
        self._last_debug = {
            "obstacles": obstacles_debug(obstacle_list),
            "traj_points": traj_points,
            "goal_point": goal[:2, 0].tolist() if goal.shape == (3, 1) else None,
            "search_goal": goal[:2, 0].tolist() if goal.shape == (3, 1) else None,
            "search_anchor": np.asarray(search_anchor_xy, dtype=float).tolist() if search_anchor_xy is not None else None,
            "search_samples": search_samples.tolist() if search_samples is not None else [],
            "search_mode": mode,
            "search_detail_mode": search_detail_mode,
            "identity_recovery_active": search_detail_mode == SearchMode.REACQUIRE_TRANSITION or mode == SearchMode.REACQUIRE_TRANSITION,
            "identity_recovery_reason": "reacquire_transition" if (search_detail_mode == SearchMode.REACQUIRE_TRANSITION or mode == SearchMode.REACQUIRE_TRANSITION) else None,
            "search_hidden_steps": int(search_hidden_steps),
            "search_occluder_id": occluder_id,
            "search_cost": float(search_cost) if np.isfinite(search_cost) else None,
            "target_protected": bool(profiling.get("target_protected", False)) if profiling else False,
            "raw_target_visible": bool(profiling.get("raw_target_visible", False)) if profiling else False,
            "gt_target_always_visible": self.gt_target_always_visible,
            "enable_search": self.enable_search,
            "predicted_target_traj": predicted_target,
            "map_occupied_cells": self._map_query.export_debug_cells() if hasattr(self._map_query, "export_debug_cells") else [],
            "map_observed_free_cells": self._map_query.export_debug_observed_free_cells() if hasattr(self._map_query, "export_debug_observed_free_cells") else [],
            "map_outline": self._map_query.export_debug_outline() if hasattr(self._map_query, "export_debug_outline") else [],
            "map_occupancy_rgba": self._map_query.export_occupancy_rgba() if hasattr(self._map_query, "export_occupancy_rgba") else None,
            "map_esdf_rgba": self._map_query.export_esdf_rgba() if hasattr(self._map_query, "export_esdf_rgba") else None,
            "map_hybrid_rgba": self._map_query.export_hybrid_rgba() if hasattr(self._map_query, "export_hybrid_rgba") else None,
            "map_debug_extent": self._map_query.export_debug_extent() if hasattr(self._map_query, "export_debug_extent") else None,
            "map_debug_mode": "hybrid" if hasattr(self._map_query, "export_hybrid_rgba") else "occupancy",
            "lidar_range_max": self.lidar_range_max,
            "map_resolution": getattr(self._map_query, "resolution", None),
            "map_origin": getattr(self._map_query, "origin_xy", None).tolist() if hasattr(getattr(self._map_query, "origin_xy", None), "tolist") else None,
            "predictor_name": self.predictor_name,
            "profiling": profiling,
            "profiling_summary": self._profile_summary() if self.enable_profiling else None,
        }

    def _build_profile(
        self,
        mode: str,
        target_visible: bool,
        raw_target_visible: bool,
        t_state_ms: float,
        t_map_ms: float,
        t_history_ms: float,
        t_prediction_ms: float,
        t_search_ms: float,
        t_obstacle_ms: float,
        t_mpc_ms: float,
        t_total_ms: float,
        num_agents: int,
        num_obstacles: int,
        num_search_samples: int,
        target_protected: bool,
    ) -> dict:
        return {
            "tick": self._tick_counter,
            "mode": mode,
            "target_visible": bool(target_visible),
            "raw_target_visible": bool(raw_target_visible),
            "target_protected": bool(target_protected),
            "follow_position": self.follow_position,
            "enable_search": self.enable_search,
            "gt_target_always_visible": self.gt_target_always_visible,
            "t_state_ms": float(t_state_ms),
            "t_map_ms": float(t_map_ms),
            "t_history_ms": float(t_history_ms),
            "t_prediction_ms": float(t_prediction_ms),
            "t_search_ms": float(t_search_ms),
            "t_obstacle_ms": float(t_obstacle_ms),
            "t_mpc_ms": float(t_mpc_ms),
            "t_total_ms": float(t_total_ms),
            "num_agents": int(num_agents),
            "num_obstacles": int(num_obstacles),
            "num_search_samples": int(num_search_samples),
            "map_size": [int(getattr(self._map_query, "width_cells", 0)), int(getattr(self._map_query, "height_cells", 0))],
            "predictor_name": self.predictor_name,
        }

    def _record_profile(self, record: dict) -> None:
        if not self.enable_profiling:
            return
        self._profile_records.append(record)
        if self._tick_counter % self.profiling_print_interval != 0:
            return
        summary = self._profile_summary()
        if not summary:
            return
        print(
            "[RDA-Search/Profile] "
            f"n={summary['n']} "
            f"mean_total={summary['t_total_ms_mean']:.1f}ms "
            f"p95_total={summary['t_total_ms_p95']:.1f}ms "
            f"mean_map={summary['t_map_ms_mean']:.1f}ms "
            f"mean_search={summary['t_search_ms_mean']:.1f}ms "
            f"mean_obs={summary['t_obstacle_ms_mean']:.1f}ms "
            f"mean_mpc={summary['t_mpc_ms_mean']:.1f}ms"
        )

    def _profile_summary(self) -> dict | None:
        if not self._profile_records:
            return None
        keys = (
            "t_state_ms",
            "t_map_ms",
            "t_history_ms",
            "t_prediction_ms",
            "t_search_ms",
            "t_obstacle_ms",
            "t_mpc_ms",
            "t_total_ms",
        )
        summary = {"n": len(self._profile_records)}
        for key in keys:
            values = np.asarray([record.get(key, 0.0) for record in self._profile_records], dtype=float)
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_p95"] = float(np.percentile(values, 95))
            summary[f"{key}_max"] = float(np.max(values))
        return summary


def _elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0
