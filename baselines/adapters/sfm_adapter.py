"""SFM follower adapter for the Follow-Bench platform."""

from __future__ import annotations

import math

import numpy as np

from baselines.planning.sfm.sfm import SfmPlanner
from baselines.planning.rda.rda_obstacles import ROBOT_RADIUS
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.maps.lidar_grid_builder import build_local_occupancy_grid
from baselines.perception.sensor.lidar_obstacles import LIDAR_RANGE_MAX_M


_SCOUT_ROBOT_RADIUS_M = ROBOT_RADIUS
_SCOUT_MAX_V_MPS = 2.5
_SCOUT_MAX_W_RADPS = 1.8
_SCOUT_MAX_ACC_V = 2.0
_SCOUT_MAX_ACC_W = 2.8


class SfmFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        robot_radius: float = _SCOUT_ROBOT_RADIUS_M,
        max_v_mps: float = _SCOUT_MAX_V_MPS,
        max_w_radps: float = _SCOUT_MAX_W_RADPS,
        max_acc_v: float = _SCOUT_MAX_ACC_V,
        max_acc_w: float = _SCOUT_MAX_ACC_W,
        max_jerk_v: float = 12.0,
        max_jerk_w: float = 9.0,
        lidar_range_max: float = LIDAR_RANGE_MAX_M,
        local_map_resolution: float = 0.2,
        local_map_width_cells: int | None = None,
        local_map_height_cells: int | None = None,
        local_map_inflation_radius: float | None = None,
        kinematics: str = "diff_drive",
        predict_horizon: float = 2.0,
        plan_dt: float = 0.1,
        back_goal_mode: str = "target_heading",
        debug_print: bool = True,
        debug_print_interval: int = 10,
        vmax: float | None = None,
        w_max: float | None = None,
        startup_goal_behind_guard: bool = False,
        startup_guard_steps: int = 0,
        goal_behind_deadband: float = 0.45,
    ) -> None:
        self.desired_distance = float(desired_distance)
        self.follow_position = follow_position
        self.robot_radius = float(robot_radius)
        self.max_v_mps = float(max_v_mps if vmax is None else vmax)
        self.max_w_radps = float(max_w_radps if w_max is None else w_max)
        self.max_acc_v = float(max_acc_v)
        self.max_acc_w = float(max_acc_w)
        self.max_jerk_v = float(max_jerk_v)
        self.max_jerk_w = float(max_jerk_w)
        self.lidar_range_max = float(lidar_range_max)
        self.local_map_resolution = float(local_map_resolution)
        self.local_map_width_cells = None if local_map_width_cells is None else int(local_map_width_cells)
        self.local_map_height_cells = None if local_map_height_cells is None else int(local_map_height_cells)
        self.local_map_inflation_radius = (
            self.robot_radius if local_map_inflation_radius is None else float(local_map_inflation_radius)
        )
        self.debug_print = bool(debug_print)
        self.debug_print_interval = max(int(debug_print_interval), 1)
        self.startup_goal_behind_guard = bool(startup_goal_behind_guard)
        self.startup_guard_steps = max(int(startup_guard_steps), 0)
        self.goal_behind_deadband = max(float(goal_behind_deadband), 0.0)
        self._tick_counter = 0
        planner_kinematics = _adapter_kinematics(kinematics)
        self._planner = SfmPlanner(
            dt=dt,
            robot_radius=self.robot_radius,
            max_v_mps=self.max_v_mps,
            max_w_radps=self.max_w_radps,
            max_acc_v=self.max_acc_v,
            max_acc_w=self.max_acc_w,
            predict_horizon=predict_horizon,
            plan_dt=plan_dt,
            kinematics=planner_kinematics,
            back_goal_mode=back_goal_mode,
            max_jerk_v=self.max_jerk_v,
            max_jerk_w=self.max_jerk_w,
        )
        self._last_debug: dict = {"goal_point": None}

    def reset(self) -> None:
        self._planner.reset()
        self._last_debug = {"goal_point": None}
        self._tick_counter = 0

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.target is None:
            self._last_debug = {"goal_point": None, "error": "missing_target"}
            return FollowAction(v_mps=0.0, w_radps=0.0)

        self._tick_counter += 1
        rx, ry = float(obs.robot.x), float(obs.robot.y)
        yaw = float(obs.robot.yaw_rad)
        humans = [
            np.array([n.x, n.y, n.vx, n.vy, math.radians(float(n.yaw_deg))], dtype=float)
            for n in obs.npcs
            if n.track_id != obs.target.track_id
        ]
        target_state = np.array(
            [
                obs.target.x,
                obs.target.y,
                obs.target.vx,
                obs.target.vy,
                math.radians(float(obs.target.yaw_deg)),
            ],
            dtype=float,
        )
        map_query = build_local_occupancy_grid(
            lidar_points=obs.lidar_points,
            lidar_extrinsics=obs.lidar_extrinsics_robot_to_sensor,
            robot_x=rx,
            robot_y=ry,
            robot_yaw=yaw,
            resolution=self.local_map_resolution,
            width_cells=self.local_map_width_cells,
            height_cells=self.local_map_height_cells,
            inflation_radius_m=self.local_map_inflation_radius,
            lidar_range_max=self.lidar_range_max,
        )

        try:
            result = self._planner.compute(
                robot_pose=np.array([rx, ry, yaw], dtype=float),
                goal=None,
                humans=humans,
                map_query=map_query,
                target_state=target_state,
                follow_position=self.follow_position,
                desired_distance=self.desired_distance,
            )
        except Exception as exc:
            self._last_debug = self._build_debug(map_query, {"goal_point": None, "error": str(exc)})
            return FollowAction(v_mps=0.0, w_radps=0.0)

        planner_info = dict(result.get("info", {}))
        opt_state_list = planner_info.pop("opt_state_list", [])
        goal_traj = planner_info.get("goal_traj", [])
        goal_point = goal_traj[0][:2] if goal_traj else None
        guard_debug = self._startup_goal_behind_guard(obs, rx, ry, yaw, goal_point)
        v_cmd = float(result["v_mps"])
        w_cmd = float(result["w_radps"])
        if guard_debug["startup_goal_behind_guard_active"]:
            v_cmd = 0.0
            w_cmd = 0.0
        self._last_debug = self._build_debug(
            map_query,
            {
                **planner_info,
                **guard_debug,
                "goal_point": goal_point,
                "traj_points": _states_to_points(opt_state_list),
                "target_predicted_traj": planner_info.get("predicted_target_traj", []),
                "goal_traj": planner_info.get("goal_traj", []),
                "force_traj": planner_info.get("force_traj", []),
                "clearance_traj": planner_info.get("clearance_traj", []),
                "heading_err": float(result["heading_err"]),
                "vx_sfm": float(result["vx_sfm"]),
                "vy_sfm": float(result["vy_sfm"]),
                "v_mps": v_cmd,
                "w_radps": w_cmd,
            },
        )
        self._maybe_print_debug()
        return FollowAction(v_mps=v_cmd, w_radps=w_cmd)

    def _startup_goal_behind_guard(
        self,
        obs: FollowObservation,
        rx: float,
        ry: float,
        yaw: float,
        goal_point,
    ) -> dict:
        debug = {
            "startup_goal_behind_guard_enabled": bool(self.startup_goal_behind_guard),
            "startup_goal_behind_guard_active": False,
            "ahead_component": None,
            "goal_bearing_rad": None,
            "goal_behind_deadband": float(self.goal_behind_deadband),
            "startup_guard_steps": int(self.startup_guard_steps),
        }
        if not self.startup_goal_behind_guard:
            return debug
        if self.follow_position != "back":
            return debug
        if self._tick_counter > self.startup_guard_steps:
            return debug
        extras = dict(obs.extras or {})
        if extras.get("perception_has_target") is False:
            return debug
        if goal_point is None:
            return debug
        goal_xy = np.asarray(goal_point, dtype=float).reshape(-1)
        if goal_xy.size < 2:
            return debug
        robot_xy = np.array([float(rx), float(ry)], dtype=float)
        delta = goal_xy[:2] - robot_xy
        distance_to_goal = float(np.linalg.norm(delta))
        if distance_to_goal <= 1e-9:
            return debug
        forward = np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=float)
        ahead_component = float(np.dot(delta, forward))
        goal_bearing = float(math.atan2(float(delta[1]), float(delta[0])) - float(yaw))
        goal_bearing = float(math.atan2(math.sin(goal_bearing), math.cos(goal_bearing)))
        debug.update({
            "ahead_component": ahead_component,
            "goal_bearing_rad": goal_bearing,
            "distance_to_goal_for_guard": distance_to_goal,
        })
        if (
            ahead_component < 0.0
            and abs(ahead_component) < self.goal_behind_deadband
            and distance_to_goal < self.goal_behind_deadband
        ):
            debug["startup_goal_behind_guard_active"] = True
        return debug

    def _build_debug(self, map_query, info: dict) -> dict:
        return {
            **info,
            "map_occupied_cells": map_query.export_debug_cells() if hasattr(map_query, "export_debug_cells") else [],
            "map_observed_free_cells": (
                map_query.export_debug_observed_free_cells()
                if hasattr(map_query, "export_debug_observed_free_cells")
                else []
            ),
            "map_outline": map_query.export_debug_outline() if hasattr(map_query, "export_debug_outline") else [],
            "map_occupancy_rgba": (
                map_query.export_occupancy_rgba() if hasattr(map_query, "export_occupancy_rgba") else None
            ),
            "map_esdf_rgba": map_query.export_esdf_rgba() if hasattr(map_query, "export_esdf_rgba") else None,
            "map_hybrid_rgba": map_query.export_hybrid_rgba() if hasattr(map_query, "export_hybrid_rgba") else None,
            "map_debug_extent": map_query.export_debug_extent() if hasattr(map_query, "export_debug_extent") else None,
            "map_debug_mode": "hybrid" if hasattr(map_query, "export_hybrid_rgba") else "occupancy",
            "lidar_range_max": self.lidar_range_max,
            "robot_radius": self.robot_radius,
            "max_v_mps": self.max_v_mps,
            "max_w_radps": self.max_w_radps,
            "max_acc_v": self.max_acc_v,
            "max_acc_w": self.max_acc_w,
            "max_jerk_v": self.max_jerk_v,
            "max_jerk_w": self.max_jerk_w,
        }

    def _maybe_print_debug(self) -> None:
        if not self.debug_print or self._tick_counter % self.debug_print_interval != 0:
            return
        dbg = self._last_debug
        force = np.asarray(dbg.get("total_force", [0.0, 0.0]), dtype=float).reshape(-1)
        goal_force = np.asarray(dbg.get("goal_force", [0.0, 0.0]), dtype=float).reshape(-1)
        human_force = np.asarray(dbg.get("human_force", [0.0, 0.0]), dtype=float).reshape(-1)
        map_force = np.asarray(dbg.get("map_force", [0.0, 0.0]), dtype=float).reshape(-1)
        print(
            f"[SFM] mode={self.follow_position} kin={dbg.get('kinematics', 'n/a')} "
            f"v={float(dbg.get('v_mps', 0.0)):.3f} w={float(dbg.get('w_radps', 0.0)):.3f} "
            f"d_goal={float(dbg.get('distance_to_goal', 0.0)):.2f} "
            f"clear={_fmt_float(dbg.get('map_clearance'))} "
            f"herr={math.degrees(float(dbg.get('heading_err', 0.0))):+.1f}deg"
        )
        print(
            f"[SFM] |F| goal={np.linalg.norm(goal_force):.2f} "
            f"human={np.linalg.norm(human_force):.2f} "
            f"map={np.linalg.norm(map_force):.2f} "
            f"total={np.linalg.norm(force):.2f} "
            f"crowd={int(dbg.get('crowd_count', 0))} "
            f"pred={int(dbg.get('num_steps', 0))}"
        )
        cmd_dbg = dbg.get("command_debug", {}) or {}
        print(
            f"[SFM] target_speed raw={float(dbg.get('target_raw_speed', 0.0)):.2f} "
            f"filt={float(dbg.get('target_filtered_speed', 0.0)):.2f} "
            f"goal_jump={float(dbg.get('goal_jump', 0.0)):.2f} "
            f"w_raw={float(cmd_dbg.get('w_raw', 0.0)):.3f}"
        )


def _states_to_points(states) -> list[list[float]]:
    points = []
    for state in states or []:
        arr = np.asarray(state, dtype=float).reshape(-1)
        if arr.size >= 2:
            points.append(arr[:2].tolist())
    return points


def _fmt_float(value) -> str:
    if value is None:
        return "inf"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "inf"
    return f"{number:.2f}"


def _adapter_kinematics(name: str) -> str:
    key = str(name).lower()
    if key in ("omni", "omnidirectional"):
        raise ValueError("SfmFollowerPolicy outputs FollowAction(v_mps, w_radps); use diff_drive or ackermann here")
    return key


__all__ = ["SfmFollowerPolicy"]
