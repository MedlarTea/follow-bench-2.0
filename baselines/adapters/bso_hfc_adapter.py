"""BSO-HFC follower adapter for the Follow-Bench/CARLA platform."""

from __future__ import annotations

import math
import os
import sys
from collections import deque

import numpy as np


from baselines.common.types import AgentState2D, RobotState2D
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.behavior.follow_goal import compute_follow_goal
from baselines.maps.lidar_grid_builder import build_local_occupancy_grid
from baselines.perception.gt.gt_scene_provider import target_yaw_from_obs
from baselines.perception.sensor.lidar_obstacles import LIDAR_RANGE_MAX_M
from baselines.planning.bso_hfc import BSOHFCPlanner, DiscSpec, load_bso_hfc_config


_SCOUT_ROBOT_RADIUS_M = 0.5
_SCOUT_MAX_V_MPS = 2.5
_SCOUT_MAX_W_RADPS = 2.5
_SCOUT_MAX_ACC_V = 2.0
_SCOUT_MAX_ACC_W = 4.0


class BsoHfcFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        robot_radius: float = _SCOUT_ROBOT_RADIUS_M,
        target_radius: float = 0.35,
        human_radius: float = 0.35,
        max_v_mps: float = _SCOUT_MAX_V_MPS,
        max_w_radps: float = _SCOUT_MAX_W_RADPS,
        max_acc_v: float = _SCOUT_MAX_ACC_V,
        max_acc_w: float = _SCOUT_MAX_ACC_W,
        lidar_range_max: float = LIDAR_RANGE_MAX_M,
        local_map_resolution: float = 0.2,
        local_map_width_cells: int | None = None,
        local_map_height_cells: int | None = None,
        local_map_window_size_m: float | None = None,
        local_map_inflation_radius: float = 0.0,
        use_npc_gt_obstacles: bool = False,
        target_history_maxlen: int = 256,
        include_edt_debug: bool = False,
        target_stop_speed_thr: float = 0.05,
        hold_distance_eps: float = 0.15,
    ) -> None:
        self.dt = float(dt)
        self.desired_distance = float(desired_distance)
        self.follow_position = follow_position
        self.human_radius = float(human_radius)
        self.target_radius = float(target_radius)
        self.lidar_range_max = float(lidar_range_max)
        self.local_map_resolution = float(local_map_resolution)
        self.local_map_window_size_m = (
            # Use the square inscribed in the lidar detection circle: the
            # planning window stays inside sensed range, and its corner-to-
            # center distance equals lidar_range_max.
            math.sqrt(2.0) * self.lidar_range_max
            if local_map_window_size_m is None
            else float(local_map_window_size_m)
        )
        default_map_cells = None
        if self.local_map_window_size_m is not None:
            # The backing grid is world-axis aligned, while BSO-HFC searches in
            # robot-local axes. Allocate the backing grid around the local
            # window's circumscribed circle so the requested square fits at
            # every robot yaw.
            grid_span = self.local_map_window_size_m * math.sqrt(2.0)
            default_map_cells = max(int(math.ceil(grid_span / max(self.local_map_resolution, 1e-6))) + 4, 5)
        self.local_map_width_cells = (
            default_map_cells if local_map_width_cells is None else max(int(local_map_width_cells), 5)
        )
        self.local_map_height_cells = (
            default_map_cells if local_map_height_cells is None else max(int(local_map_height_cells), 5)
        )
        self.local_map_inflation_radius = float(local_map_inflation_radius)
        self.use_npc_gt_obstacles = bool(use_npc_gt_obstacles)
        self.include_edt_debug = bool(include_edt_debug)
        self.target_stop_speed_thr = float(target_stop_speed_thr)
        self.hold_distance_eps = float(hold_distance_eps)

        cfg = load_bso_hfc_config(
            sample_time=self.dt,
            d_desired=self.desired_distance,
            target_radius=self.target_radius,
            robot_radius=float(robot_radius),
            linear_min=-abs(float(max_v_mps)),
            linear_max=float(max_v_mps),
            omega_min=-abs(float(max_w_radps)),
            omega_max=abs(float(max_w_radps)),
            acc_v=float(max_acc_v),
            acc_omega=float(max_acc_w),
        )
        cfg.local_map_window_size_m = self.local_map_window_size_m
        self._planner = BSOHFCPlanner(cfg)
        self._target_history: deque[AgentState2D] = deque(maxlen=max(int(target_history_maxlen), 1))
        self._last_cmd = np.zeros((2,), dtype=float)
        self._last_velocity_estimate = np.zeros((2,), dtype=float)
        self._last_debug: dict = {}

    def reset(self) -> None:
        self._planner.reset()
        self._target_history.clear()
        self._last_cmd = np.zeros((2,), dtype=float)
        self._last_velocity_estimate = np.zeros((2,), dtype=float)
        self._last_debug = {}

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.target is None:
            self._last_debug = {"planning_success": False, "reason": "missing_target"}
            return FollowAction(v_mps=0.0, w_radps=0.0)

        robot_vel = self._estimate_robot_velocity(obs)
        robot = self._robot_from_obs(obs, robot_vel)
        target = self._target_from_obs(obs)
        self._target_history.append(target)
        hold, hold_error = self._should_hold(robot, target)
        if hold:
            self._last_cmd = np.zeros((2,), dtype=float)
            desired_xy = self._desired_follow_xy(robot, target)
            self._last_debug = {
                "planning_success": False,
                "hold": True,
                "follow_mode": self.follow_position,
                "d_current": float(hold_error),
                "d_desired": self.desired_distance,
                "goal_point": desired_xy.tolist(),
                "bso_hfc_velocity_estimate": robot_vel.tolist(),
                "bso_hfc_last_cmd": self._last_cmd.tolist(),
            }
            print(
                f"[BSO-HFC] mode={self.follow_position} HOLD v=0.000 w=0.000 "
                f"vel=({robot_vel[0]:.3f},{robot_vel[1]:.3f}) err={hold_error:.2f}m"
            )
            return FollowAction(v_mps=0.0, w_radps=0.0)

        map_query = build_local_occupancy_grid(
            lidar_points=obs.lidar_points,
            lidar_extrinsics=obs.lidar_extrinsics_robot_to_sensor,
            robot_x=robot.x,
            robot_y=robot.y,
            robot_yaw=robot.yaw,
            resolution=self.local_map_resolution,
            width_cells=self.local_map_width_cells,
            height_cells=self.local_map_height_cells,
            inflation_radius_m=0.0,
            lidar_range_max=self.lidar_range_max,
        )
        extra_discs = self._npc_discs(obs) if self.use_npc_gt_obstacles else []

        try:
            opt_vel, info = self._planner.control_follow(
                robot=robot,
                target=target,
                target_history=list(self._target_history),
                map_query=map_query,
                follow_position=self.follow_position,
                desired_distance=self.desired_distance,
                lidar_range_max=self.lidar_range_max,
                include_edt_debug=self.include_edt_debug,
                map_overlay_inflation_radius=self.local_map_inflation_radius,
                extra_occupied_discs_world=extra_discs,
                robot_vel=robot_vel,
            )
        except Exception as exc:
            self._last_debug = self._build_debug(map_query, {}, [], error=str(exc))
            print(f"[BSO-HFC] control error: {exc}")
            return FollowAction(v_mps=0.0, w_radps=0.0)

        v = float(np.asarray(opt_vel, dtype=float).reshape(-1)[0]) if np.asarray(opt_vel).size > 0 else 0.0
        w = float(np.asarray(opt_vel, dtype=float).reshape(-1)[1]) if np.asarray(opt_vel).size > 1 else 0.0
        self._last_cmd = np.array([v, w], dtype=float)
        self._last_debug = self._build_debug(map_query, info, extra_discs)
        dist = math.hypot(target.x - robot.x, target.y - robot.y)
        map_shape = info.get("local_map_shape")
        map_shape_text = np.asarray(map_shape).reshape(-1).astype(int).tolist() if map_shape is not None else None
        tracker_dbg = info.get("mpc_tracker") if isinstance(info.get("mpc_tracker"), dict) else {}
        print(
            f"[BSO-HFC] mode={self.follow_position} v={v:.3f} w={w:.3f} "
            f"vel=({robot_vel[0]:.3f},{robot_vel[1]:.3f}) dist={dist:.2f}m "
            f"v_ref={float(info.get('v_ref', 0.0)):.2f} dt_ref={float(info.get('delta_t', 0.0)):.2f} "
            f"mpc_raw_v={float(tracker_dbg.get('raw_v_cmd', 0.0)):.3f} "
            f"ref=({float(tracker_dbg.get('ref_x', 0.0)):.2f},{float(tracker_dbg.get('ref_y', 0.0)):.2f},"
            f"{float(tracker_dbg.get('ref_speed', 0.0)):.2f}) "
            f"goal=({np.asarray(info.get('target_raw_local', [0.0, 0.0])).reshape(-1)[0]:.2f},"
            f"{np.asarray(info.get('target_raw_local', [0.0, 0.0])).reshape(-1)[1]:.2f}->"
            f"{np.asarray(info.get('local_goal_projected_local', [0.0, 0.0])).reshape(-1)[0]:.2f},"
            f"{np.asarray(info.get('local_goal_projected_local', [0.0, 0.0])).reshape(-1)[1]:.2f}) "
            f"map={map_shape_text} success={int(info.get('planning_success', False))}"
        )
        return FollowAction(v_mps=v, w_radps=w)

    def _should_hold(self, robot: RobotState2D, target: AgentState2D) -> tuple[bool, float]:
        robot_xy = np.array([robot.x, robot.y], dtype=float)
        if self.follow_position == "back":
            target_xy = np.array([target.x, target.y], dtype=float)
            d_current = float(np.linalg.norm(robot_xy - target_xy))
            if d_current <= self.desired_distance + self.hold_distance_eps:
                return True, d_current
            return False, d_current

        target_speed = float(target.speed)
        if target_speed >= self.target_stop_speed_thr:
            return False, float("inf")

        desired = self._desired_follow_xy(robot, target)
        error = float(np.linalg.norm(robot_xy - desired))
        return error <= self.hold_distance_eps, error

    def _desired_follow_xy(self, robot: RobotState2D, target: AgentState2D) -> np.ndarray:
        return compute_follow_goal(robot, target, self.follow_position, self.desired_distance).reshape(-1)[:2]

    def _robot_from_obs(self, obs: FollowObservation, robot_vel: np.ndarray) -> RobotState2D:
        yaw = float(obs.robot.yaw_rad)
        speed = float(robot_vel[0])
        vx = speed * math.cos(yaw)
        vy = speed * math.sin(yaw)
        return RobotState2D(
            x=float(obs.robot.x),
            y=float(obs.robot.y),
            yaw=yaw,
            vx=vx,
            vy=vy,
            speed=speed,
        )

    def _estimate_robot_velocity(self, obs: FollowObservation) -> np.ndarray:
        estimate = self._last_cmd.copy()
        estimate[0] = float(np.clip(estimate[0], self._planner.cfg.mpc.min_v, self._planner.cfg.mpc.max_v))
        estimate[1] = float(np.clip(estimate[1], self._planner.cfg.mpc.min_omega, self._planner.cfg.mpc.max_omega))
        self._last_velocity_estimate = estimate.copy()
        return estimate

    def _target_from_obs(self, obs: FollowObservation) -> AgentState2D:
        yaw = self._target_yaw(obs)
        return AgentState2D(
            track_id=str(obs.target.track_id),
            x=float(obs.target.x),
            y=float(obs.target.y),
            vx=float(obs.target.vx),
            vy=float(obs.target.vy),
            yaw=yaw,
            speed=float(obs.target.speed),
            radius=self.target_radius,
            is_target=True,
        )

    def _target_yaw(self, obs: FollowObservation) -> float:
        return target_yaw_from_obs(obs)

    def _npc_discs(self, obs: FollowObservation) -> list[DiscSpec]:
        discs = []
        target_id = str(obs.target.track_id)
        for npc in obs.npcs:
            if str(npc.track_id) == target_id:
                continue
            discs.append(DiscSpec(center_world_xy=np.array([npc.x, npc.y], dtype=float), radius=self.human_radius))
        return discs

    def _build_debug(self, map_query, info: dict, extra_discs: list[DiscSpec], error: str | None = None) -> dict:
        mpc = np.asarray(info.get("mpc_path_list", np.empty((2, 0), dtype=float)), dtype=float)
        traj_points = mpc.T.tolist() if mpc.ndim == 2 and mpc.shape[0] == 2 else []
        local_outline = _plot_array_to_points(info.get("local_map_square_path_list"))
        debug = {
            **info,
            "obstacles": [
                {"center": disc.center_world_xy.tolist(), "radius": float(disc.radius)}
                for disc in extra_discs
            ],
            "traj_points": traj_points,
            "goal_point": _debug_goal_point(info),
            "map_occupied_cells": map_query.export_debug_cells() if hasattr(map_query, "export_debug_cells") else [],
            "map_observed_free_cells": map_query.export_debug_observed_free_cells() if hasattr(map_query, "export_debug_observed_free_cells") else [],
            "map_outline": local_outline or (map_query.export_debug_outline() if hasattr(map_query, "export_debug_outline") else []),
            "map_occupancy_rgba": _mask_rgba_to_polygon(
                map_query.export_occupancy_rgba() if hasattr(map_query, "export_occupancy_rgba") else None,
                map_query,
                local_outline,
            ),
            "map_esdf_rgba": _mask_rgba_to_polygon(
                map_query.export_esdf_rgba() if hasattr(map_query, "export_esdf_rgba") else None,
                map_query,
                local_outline,
            ),
            "map_hybrid_rgba": _mask_rgba_to_polygon(
                map_query.export_hybrid_rgba() if hasattr(map_query, "export_hybrid_rgba") else None,
                map_query,
                local_outline,
            ),
            "map_debug_extent": map_query.export_debug_extent() if hasattr(map_query, "export_debug_extent") else None,
            "map_debug_mode": "hybrid" if hasattr(map_query, "export_hybrid_rgba") else "occupancy",
            "lidar_range_max": self.lidar_range_max,
            "bso_hfc_velocity_estimate": self._last_velocity_estimate.tolist(),
            "bso_hfc_last_cmd": self._last_cmd.tolist(),
        }
        if error is not None:
            debug["planning_success"] = False
            debug["error"] = error
        return debug


def _debug_goal_point(info: dict):
    for key in ("desired_follow_pose_world", "local_goal_projected_world", "planning_target_world"):
        value = info.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size >= 2:
            return arr[:2].tolist()
    return None


def _plot_array_to_points(path_like) -> list[list[float]]:
    arr = np.asarray(path_like, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return []
    if arr.shape[0] == 2 and arr.shape[1] >= 2:
        arr = arr.T
    if arr.shape[1] < 2:
        return []
    return arr[:, :2].tolist()


def _mask_rgba_to_polygon(rgba, map_query, polygon_world: list[list[float]]):
    if rgba is None or len(polygon_world) < 3:
        return rgba
    if not hasattr(map_query, "grid_to_world_points"):
        return rgba
    image = np.asarray(rgba, dtype=np.uint8).copy()
    if image.ndim != 3 or image.shape[2] < 4:
        return rgba

    height, width = image.shape[:2]
    rows, cols = np.indices((height, width), dtype=int)
    world = map_query.grid_to_world_points(cols.reshape(-1), rows.reshape(-1))
    inside = _points_in_polygon(world, np.asarray(polygon_world, dtype=float)).reshape(height, width)
    image[~inside, 3] = 0
    return image


def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=float)
    polygon = np.asarray(polygon_xy, dtype=float)
    if len(polygon) >= 2 and np.linalg.norm(polygon[0] - polygon[-1]) <= 1e-9:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        return np.zeros((len(points),), dtype=bool)

    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros((len(points),), dtype=bool)
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        denom = by - ay
        if abs(denom) <= 1e-12:
            continue
        crosses = ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / denom + ax)
        inside ^= crosses
    return inside
__all__ = ["BsoHfcFollowerPolicy"]
