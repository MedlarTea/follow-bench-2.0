"""DWA trajectory follower adapter for the Follow-Bench platform."""

from __future__ import annotations

import math

import numpy as np

from baselines.common.geometry import wrap_pi
from followbench.sim.core_types import FollowAction, FollowObservation, NpcState
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.perception.gt.gt_scene_provider import agent_states_from_obs, robot_state_from_obs, target_state_from_obs, target_yaw_from_obs
from baselines.planning.dwa.follow_goal_strategy import DwaFollowGoalStrategy
from baselines.planning.dwa.local_map import build_dwa_local_map
from baselines.planning.dwa.target_reference import DwaTargetReferenceFilter
from baselines.planning.dwa.dwa_traj import DwaTrajPlanner
from baselines.planning.dwa.types import AgentPrediction, KinematicState
from baselines.perception.sensor.lidar_obstacles import LIDAR_RANGE_MAX_M
from baselines.prediction import TrajectoryHistory, TrajectoryPredictionService


_SCOUT_ROBOT_RADIUS_M = 0.5


def _npcs_to_obstacles(npcs: list[NpcState], target_track_id: str | None = None, npc_radius: float = 0.35) -> np.ndarray:
    pts = []
    for npc in npcs:
        if target_track_id is not None and str(npc.track_id) == str(target_track_id):
            continue
        pts.append([npc.x, npc.y, npc.vx, npc.vy, npc_radius])
    return np.array(pts, dtype=float) if pts else np.empty((0, 5), dtype=float)


class DwaTrajFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        obstacle_radius: float = 0.3,
        predict_horizon: float = 2.0,
        weight_heading: float = 0.7,
        weight_distance: float = 0.8,
        weight_obstacle: float = 1.5,
        weight_velocity: float = 1.2,
        predictor_name: str = "cv",
        use_prediction: bool = True,
        target_radius: float = 0.35,
        lidar_range_max: float = LIDAR_RANGE_MAX_M,
        local_map_resolution: float = 0.2,
        local_map_width_cells: int | None = None,
        local_map_height_cells: int | None = None,
        target_clear_margin: float = 0.15,
        enable_spacing_reverse: bool = False,
        visibility_aware_reverse: bool = False,
        perception_goal_behind_deadband: float = 0.20,
        goal_behind_confirm_steps: int = 1,
        startup_guard_steps: int = 0,
        spacing_reverse_min_speed: float = 0.08,
        spacing_reverse_max_speed: float = 0.22,
        spacing_reverse_max_yawrate: float = 0.35,
        severe_goal_behind_m: float = 0.80,
    ) -> None:
        self.dt = float(dt)
        self.desired_distance = float(desired_distance)
        self.follow_position = follow_position
        self.target_radius = float(target_radius)
        self.use_prediction = bool(use_prediction)
        self.lidar_range_max = float(lidar_range_max)
        self.local_map_resolution = float(local_map_resolution)
        self.local_map_width_cells = None if local_map_width_cells is None else int(local_map_width_cells)
        self.local_map_height_cells = None if local_map_height_cells is None else int(local_map_height_cells)
        self.target_clear_margin = float(target_clear_margin)
        self.visibility_aware_reverse = bool(visibility_aware_reverse)
        self._planner = DwaTrajPlanner(
            dt=dt,
            obstacle_radius=obstacle_radius,
            predict_horizon=predict_horizon,
            weight_heading=weight_heading,
            weight_distance=weight_distance,
            weight_obstacle=weight_obstacle,
            weight_velocity=weight_velocity,
            enable_spacing_reverse=enable_spacing_reverse,
            perception_goal_behind_deadband=perception_goal_behind_deadband,
            goal_behind_confirm_steps=goal_behind_confirm_steps,
            startup_guard_steps=startup_guard_steps,
            spacing_reverse_min_speed=spacing_reverse_min_speed,
            spacing_reverse_max_speed=spacing_reverse_max_speed,
            spacing_reverse_max_yawrate=spacing_reverse_max_yawrate,
            severe_goal_behind_m=severe_goal_behind_m,
        )
        self._predictor_error: str | None = None
        try:
            self._predictor = TrajectoryPredictionService(predictor_name, dt)
        except Exception as exc:
            if predictor_name == "cv":
                raise
            self._predictor_error = str(exc)
            self._predictor = TrajectoryPredictionService("cv", dt)
        self._history = TrajectoryHistory(self._predictor.history_length)
        self._target_reference = DwaTargetReferenceFilter()
        self._goal_strategy = DwaFollowGoalStrategy(back_goal_mode="target_heading")
        self._last_robot_pose: np.ndarray | None = None
        self._last_velocity_estimate = np.zeros(2, dtype=float)
        self._last_debug: dict = {"goal_point": None, "traj_points": []}

    def reset(self) -> None:
        self._planner.reset()
        self._predictor.reset()
        self._history.reset()
        self._target_reference.reset()
        self._goal_strategy.reset()
        self._last_robot_pose = None
        self._last_velocity_estimate = np.zeros(2, dtype=float)
        self._last_debug = {"goal_point": None, "traj_points": []}

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.target is None:
            self._last_debug = {"goal_point": None, "traj_points": [], "error": "missing_target"}
            return FollowAction(v_mps=0.0, w_radps=0.0)

        robot_pose = np.array([obs.robot.x, obs.robot.y, obs.robot.yaw_rad], dtype=float)
        robot_vel = self._estimate_robot_velocity(obs)
        robot = robot_state_from_obs(obs)
        target = target_state_from_obs(obs, self.target_radius)
        agents = agent_states_from_obs(obs, radius=self.target_radius)
        self._history.update(robot, target, agents, target_visible=True)

        prediction_debug: dict = {}
        target_prediction = self._build_target_prediction(obs, prediction_debug)
        robot_state = KinematicState(
            x=float(obs.robot.x),
            y=float(obs.robot.y),
            yaw=float(obs.robot.yaw_rad),
            v=float(robot_vel[0]),
            w=float(robot_vel[1]),
        )
        raw_goal_traj = self._goal_strategy.build_goal_traj(
            robot_state=robot_state,
            target_prediction=target_prediction,
            follow_position=self.follow_position,
            desired_distance=self.desired_distance,
            tick_dt=self.dt,
        )
        strategy_debug = self._goal_strategy.get_debug_info()
        goal_traj, goal_debug = self._target_reference.smooth_goal_traj(
            raw_goal_traj=raw_goal_traj,
            tick_dt=self.dt,
            plan_dt=self.dt,
            follow_position=self.follow_position,
            target_speed=float(obs.target.speed),
        )
        if len(goal_traj) == 0:
            self._last_debug = {
                **prediction_debug,
                **strategy_debug,
                **goal_debug,
                "goal_point": None,
                "traj_points": [],
                "error": "empty_goal_traj",
            }
            return FollowAction(v_mps=0.0, w_radps=0.0)
        tracking_goal_traj = goal_traj[:, :2]
        target_track_id = str(obs.target.track_id)
        obstacles = _npcs_to_obstacles(obs.npcs, target_track_id=target_track_id, npc_radius=self.target_radius)
        target_xy = np.array([float(obs.target.x), float(obs.target.y)], dtype=float)
        target_distance = float(np.linalg.norm(target_xy - robot_pose[:2]))
        target_closing_speed = _target_closing_speed(obs, robot_vel)
        visibility_ctx = self._reverse_visibility_context(obs, target_xy)
        if obs.lidar_points is not None and len(obs.lidar_points) > 0:
            map_query, map_debug = build_dwa_local_map(
                lidar_points=obs.lidar_points,
                lidar_extrinsics=obs.lidar_extrinsics_robot_to_sensor,
                robot_x=float(obs.robot.x),
                robot_y=float(obs.robot.y),
                robot_yaw=float(obs.robot.yaw_rad),
                target_xy=target_xy,
                target_radius=self.target_radius,
                robot_radius=_SCOUT_ROBOT_RADIUS_M,
                resolution=self.local_map_resolution,
                width_cells=self.local_map_width_cells,
                height_cells=self.local_map_height_cells,
                lidar_range_max=self.lidar_range_max,
                target_clear_margin=self.target_clear_margin,
            )
            map_debug["local_map_available"] = True
        else:
            map_query = None
            map_debug = {"local_map_available": False}

        try:
            v, w, info = self._planner.compute(
                robot_pose=robot_pose,
                robot_vel=robot_vel,
                goal_traj=tracking_goal_traj,
                obstacles=obstacles,
                target_speed=float(obs.target.speed),
                desired_distance=self.desired_distance,
                target_distance=target_distance,
                target_closing_speed=target_closing_speed,
                map_query=map_query,
                visibility_yaw_error=visibility_ctx.get("visibility_yaw_error"),
                target_visible_for_reverse=bool(visibility_ctx.get("target_visible_for_reverse", True)),
                preferred_view=visibility_ctx.get("preferred_view"),
                target_view=visibility_ctx.get("target_view"),
            )
        except Exception as e:
            self._last_debug = {
                **_map_debug_payload(map_query),
                **map_debug,
                "goal_point": _first_xy(raw_goal_traj),
                "traj_points": [],
                "error": str(e),
            }
            print(f"[DWA-Traj] control error: {e}")
            return FollowAction(v_mps=0.0, w_radps=0.0)

        goal_point = goal_traj[0, :2].tolist()
        self._last_debug = {
            **prediction_debug,
            **strategy_debug,
            **goal_debug,
            **info,
            "goal_point": goal_point,
            "traj_points": _states_to_points(info.get("opt_state_list")),
            "goal_traj": tracking_goal_traj.tolist(),
            "raw_goal_point": raw_goal_traj[0, :2].tolist(),
            "filtered_goal_point": goal_traj[0, :2].tolist(),
            "predicted_target_traj": target_prediction.positions.tolist(),
            "robot_vel_estimate": robot_vel.tolist(),
            "target_distance": target_distance,
            "target_closing_speed": target_closing_speed,
            **visibility_ctx,
            "dynamic_obstacle_points": obstacles[:, :2].tolist() if obstacles.size else [],
            **map_debug,
            **_map_debug_payload(map_query),
        }
        print(
            f"[DWA-Traj] mode={info.get('mode', 'n/a')} follow={self.follow_position} "
            f"v={v:.3f} w={w:.3f} vref={float(info.get('v_ref', 0.0)):.2f} "
            f"dg={float(info.get('distance_to_goal', 0.0)):.2f} tdist={target_distance:.2f} "
            f"goal=({float(goal_point[0]):.2f},{float(goal_point[1]):.2f}) "
            f"hint=({float(info.get('hint_point', goal_point)[0]):.2f},{float(info.get('hint_point', goal_point)[1]):.2f}) "
            f"slot={float(strategy_debug.get('slot_error', 0.0)):.2f} "
            f"rev={info.get('reverse_reason', '')} "
            f"map_clear={float(info.get('min_static_clearance', float('inf'))):.2f} "
            f"dyn_clear={float(info.get('min_dynamic_clearance', float('inf'))):.2f} "
            f"hard={int(info.get('hard_invalid_count', 0))}/{int(info.get('candidate_count', 0))}"
        )
        return FollowAction(v_mps=v, w_radps=w)

    def _reverse_visibility_context(self, obs: FollowObservation, target_xy: np.ndarray) -> dict:
        preferred = _preferred_target_view(self.follow_position)
        extras = dict(obs.extras or {})
        target_view = extras.get("perception_target_view")
        target_bbox = extras.get("perception_target_bbox")
        ctx = {
            "preferred_view": preferred,
            "target_view": target_view,
            "target_bbox": target_bbox,
            "target_visible_for_reverse": extras.get("perception_has_target", True) is not False,
            "visibility_yaw_error": None,
            "visibility_yaw_source": "disabled",
        }
        if not self.visibility_aware_reverse or preferred is None:
            return ctx
        if extras.get("perception_has_target") is False:
            ctx["visibility_yaw_source"] = "target_lost"
            return ctx
        robot_xy = np.array([float(obs.robot.x), float(obs.robot.y)], dtype=float)
        delta = np.asarray(target_xy, dtype=float).reshape(2) - robot_xy
        if float(np.linalg.norm(delta)) <= 1e-6:
            ctx["visibility_yaw_source"] = "target_at_robot"
            return ctx
        bearing = float(wrap_pi(math.atan2(float(delta[1]), float(delta[0])) - float(obs.robot.yaw_rad)))
        desired = _desired_target_bearing(self.follow_position)
        if desired is None:
            return ctx
        ctx["visibility_yaw_error"] = float(wrap_pi(bearing - desired))
        ctx["visibility_yaw_source"] = "target_bearing"
        ctx["visibility_target_bearing_rad"] = bearing
        ctx["visibility_desired_bearing_rad"] = float(desired)
        return ctx

    def _build_target_prediction(self, obs: FollowObservation, debug: dict) -> AgentPrediction:
        num_steps = max(self._planner.num_steps, 1)
        pred_positions = None
        pred_velocities = None
        if self.use_prediction and self._history.ready():
            prediction = self._predictor.predict(self._history)
            target_traj = prediction.target_traj
            if target_traj is not None and np.asarray(target_traj).size:
                arr = np.asarray(target_traj, dtype=float)
                pred_positions = arr[:, :2]
                if arr.shape[1] >= 4:
                    pred_velocities = arr[:, 2:4]
                debug["prediction_backend"] = self._predictor.predictor_name

        if pred_positions is None:
            tx, ty = float(obs.target.x), float(obs.target.y)
            vx, vy = float(obs.target.vx), float(obs.target.vy)
            times = np.arange(num_steps, dtype=float) * self.dt
            pred_positions = np.column_stack((tx + vx * times, ty + vy * times))
            pred_velocities = np.tile(np.array([vx, vy], dtype=float), (num_steps, 1))
            debug["prediction_backend"] = "constant_velocity_fallback"

        pred_positions = _fit_rows(np.asarray(pred_positions, dtype=float).reshape(-1, 2), num_steps)
        if pred_velocities is None:
            pred_velocities = _velocity_from_positions(pred_positions, self.dt)
        pred_velocities = _fit_rows(np.asarray(pred_velocities, dtype=float).reshape(-1, 2), num_steps)
        fallback_yaw = target_yaw_from_obs(obs)
        yaws = np.array(
            [
                math.atan2(vy, vx) if math.hypot(float(vx), float(vy)) > 0.05 else fallback_yaw
                for vx, vy in pred_velocities
            ],
            dtype=float,
        )
        debug["prediction_steps"] = int(num_steps)
        if self._predictor_error:
            debug["prediction_fallback_reason"] = self._predictor_error
        return AgentPrediction(positions=pred_positions, velocities=pred_velocities, yaws=yaws)

    def _estimate_robot_velocity(self, obs: FollowObservation) -> np.ndarray:
        pose = np.array([obs.robot.x, obs.robot.y, obs.robot.yaw_rad], dtype=float)
        if self._last_robot_pose is None:
            estimate = np.array([float(obs.robot.speed), 0.0], dtype=float)
        else:
            dt = max(float(getattr(obs, "dt", self.dt) or self.dt), 1e-3)
            delta_xy = pose[:2] - self._last_robot_pose[:2]
            forward = np.array([math.cos(pose[2]), math.sin(pose[2])], dtype=float)
            v_signed = float(np.dot(delta_xy / dt, forward))
            w_actual = float(wrap_pi(pose[2] - self._last_robot_pose[2]) / dt)
            measured = np.array([v_signed, w_actual], dtype=float)
            estimate = 0.65 * measured + 0.35 * self._last_velocity_estimate
        self._last_robot_pose = pose
        self._last_velocity_estimate = estimate.copy()
        return estimate


def _fit_rows(values: np.ndarray, rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1, values.shape[-1])
    if len(arr) == rows:
        return arr.copy()
    if len(arr) > rows:
        return arr[:rows].copy()
    if len(arr) == 0:
        return np.zeros((rows, 2), dtype=float)
    pad = np.repeat(arr[-1:, :], rows - len(arr), axis=0)
    return np.vstack((arr, pad))


def _velocity_from_positions(positions: np.ndarray, dt: float) -> np.ndarray:
    if len(positions) <= 1:
        return np.zeros_like(positions)
    vel = np.zeros_like(positions)
    vel[:-1] = np.diff(positions, axis=0) / max(float(dt), 1e-3)
    vel[-1] = vel[-2]
    return vel


def _target_closing_speed(obs: FollowObservation, robot_vel: np.ndarray) -> float:
    robot_xy = np.array([float(obs.robot.x), float(obs.robot.y)], dtype=float)
    target_xy = np.array([float(obs.target.x), float(obs.target.y)], dtype=float)
    line = target_xy - robot_xy
    dist = float(np.linalg.norm(line))
    if dist <= 1e-6:
        return 0.0
    line /= dist
    robot_forward = np.array([math.cos(float(obs.robot.yaw_rad)), math.sin(float(obs.robot.yaw_rad))], dtype=float)
    robot_vxy = robot_forward * float(robot_vel[0])
    target_vxy = np.array([float(obs.target.vx), float(obs.target.vy)], dtype=float)
    relative_v = target_vxy - robot_vxy
    return float(max(0.0, -np.dot(relative_v, line)))


def _preferred_target_view(follow_position: str) -> str | None:
    if follow_position == "back":
        return "front"
    if follow_position == "left_side":
        return "right"
    if follow_position == "right_side":
        return "left"
    return None


def _desired_target_bearing(follow_position: str) -> float | None:
    if follow_position == "back":
        return 0.0
    if follow_position == "left_side":
        return -math.pi * 0.5
    if follow_position == "right_side":
        return math.pi * 0.5
    return None


def _map_debug_payload(map_query) -> dict:
    if map_query is None:
        return {
            "map_occupied_cells": [],
            "map_observed_free_cells": [],
            "map_outline": [],
            "map_occupancy_rgba": None,
            "map_esdf_rgba": None,
            "map_hybrid_rgba": None,
            "map_debug_extent": None,
            "map_debug_mode": "none",
        }
    return {
        "map_occupied_cells": map_query.export_debug_cells() if hasattr(map_query, "export_debug_cells") else [],
        "map_observed_free_cells": (
            map_query.export_debug_observed_free_cells()
            if hasattr(map_query, "export_debug_observed_free_cells")
            else []
        ),
        "map_outline": map_query.export_debug_outline() if hasattr(map_query, "export_debug_outline") else [],
        "map_occupancy_rgba": map_query.export_occupancy_rgba() if hasattr(map_query, "export_occupancy_rgba") else None,
        "map_esdf_rgba": map_query.export_esdf_rgba() if hasattr(map_query, "export_esdf_rgba") else None,
        "map_hybrid_rgba": map_query.export_hybrid_rgba() if hasattr(map_query, "export_hybrid_rgba") else None,
        "map_debug_extent": map_query.export_debug_extent() if hasattr(map_query, "export_debug_extent") else None,
        "map_debug_mode": "hybrid" if hasattr(map_query, "export_hybrid_rgba") else "occupancy",
    }


def _first_xy(points) -> list[float] | None:
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 2:
        return None
    return arr[0, :2].tolist()


def _states_to_points(states) -> list[list[float]]:
    arr = np.asarray(states, dtype=float)
    if arr.size == 0:
        return []
    if arr.ndim == 2 and arr.shape[0] == 2:
        return arr.T.tolist()
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].tolist()
    return []


__all__ = ["DwaTrajFollowerPolicy"]
