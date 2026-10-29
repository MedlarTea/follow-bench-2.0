from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from baselines.common.geometry import wrap_pi


def _wrap_pi_array(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def _clip_cost(cost, cap=5.0):
    return np.minimum(np.asarray(cost, dtype=float), float(cap))


def _coerce_obstacles(obstacles, default_radius: float = 0.3) -> np.ndarray:
    if obstacles is None:
        return np.empty((0, 5), dtype=float)
    arr = np.asarray(obstacles, dtype=float)
    if arr.size == 0:
        return np.empty((0, 5), dtype=float)
    if arr.ndim == 0:
        return np.empty((0, 5), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] >= 5:
        out = arr[:, :5].copy()
    elif arr.shape[1] >= 4:
        out = np.column_stack((arr[:, :4], np.full((len(arr),), float(default_radius), dtype=float)))
    elif arr.shape[1] >= 2:
        out = np.column_stack(
            (
                arr[:, :2],
                np.zeros((len(arr), 2), dtype=float),
                np.full((len(arr),), float(default_radius), dtype=float),
            )
        )
    else:
        return np.empty((0, 5), dtype=float)
    out[:, 4] = np.maximum(out[:, 4], 0.0)
    return out


class DWA:
    def __init__(
        self,
        robot_tuple: tuple,
        obstacle_radius: float,
        sample_time: float = 0.1,
        predict_horizon: float = 2.0,
        resolution_scale: tuple[float, float] = (0.02, 0.02),
        weight_heading: float = 0.7,
        weight_distance: float = 0.8,
        weight_obstacle: float = 1.5,
        weight_velocity: float = 1.2,
        weight_progress: float = 0.45,
        weight_turn: float = 0.18,
        weight_smooth: float = 0.12,
        weight_reverse: float = 0.7,
        enable_spacing_reverse: bool = False,
        perception_goal_behind_deadband: float = 0.20,
        goal_behind_confirm_steps: int = 1,
        startup_guard_steps: int = 0,
        spacing_reverse_min_speed: float = 0.08,
        spacing_reverse_max_speed: float = 0.22,
        spacing_reverse_max_yawrate: float = 0.35,
        severe_goal_behind_m: float = 0.80,
    ) -> None:
        self.dt = float(sample_time)
        self.predict_horizon = float(predict_horizon)
        self.robot_radius = float(robot_tuple.radius)
        self.obstacle_radius = float(obstacle_radius)

        self.vel_cons = [
            float(robot_tuple.min_speed[0]),
            float(robot_tuple.max_speed[0]),
            float(robot_tuple.min_speed[1]),
            float(robot_tuple.max_speed[1]),
        ]
        self.acc_cons = [float(robot_tuple.max_acce[0]), float(robot_tuple.max_acce[1])]
        self.vel_resolution = [
            max(float(resolution_scale[0]) * abs(float(robot_tuple.max_speed[0])), 1e-3),
            max(float(resolution_scale[1]) * abs(float(robot_tuple.max_speed[1])), 1e-3),
        ]

        self.weight_heading = float(weight_heading)
        self.weight_distance = float(weight_distance)
        self.weight_obstacle = float(weight_obstacle)
        self.weight_velocity = float(weight_velocity)
        self.weight_progress = float(weight_progress)
        self.weight_turn = float(weight_turn)
        self.weight_smooth = float(weight_smooth)
        self.weight_reverse = float(weight_reverse)

        self.align_enter_angle = math.radians(45.0)
        self.align_exit_angle = math.radians(20.0)
        self.max_align_yawrate = 1.2
        self.min_align_yawrate = 0.35

        self.reverse_enter_margin = 0.20
        self.reverse_max_speed = 0.45
        self.reverse_max_yawrate = 1.1
        self.enable_spacing_reverse = bool(enable_spacing_reverse)
        self.perception_goal_behind_deadband = max(float(perception_goal_behind_deadband), 0.0)
        self.goal_behind_confirm_steps = max(int(goal_behind_confirm_steps), 1)
        self.startup_guard_steps = max(int(startup_guard_steps), 0)
        self.spacing_reverse_min_speed = max(float(spacing_reverse_min_speed), 0.0)
        self.spacing_reverse_max_speed = max(float(spacing_reverse_max_speed), self.spacing_reverse_min_speed)
        self.spacing_reverse_max_yawrate = max(float(spacing_reverse_max_yawrate), 0.05)
        self.severe_goal_behind_m = max(float(severe_goal_behind_m), self.perception_goal_behind_deadband)

        self.deadlock_speed_threshold = 0.03
        self.deadlock_progress_threshold = 0.03
        self.deadlock_frames = 8
        self.recovery_frames = 12
        self.recovery_yawrate = 0.65
        self.recovery_reverse_speed = -0.12

        self.reaction_time = 0.25
        self.hard_horizon_min = 0.25
        self.hard_horizon_max = 1.0
        self.clearance_sigma = 0.45
        self.obstacle_tau = 0.8
        self.soft_collision_penalty = 4.0

        self.hint_time = 0.45
        self.hint_base = 0.20
        self.hint_min = 0.15
        self.hint_max = 0.55
        self.distance_scale = 1.2
        self.progress_scale = 0.8
        self.velocity_scale = 0.8

        self.mode = "FOLLOW"
        self._deadlock_count = 0
        self._recovery_count = 0
        self._recovery_dir = 1.0
        self._last_cmd = np.zeros(2, dtype=float)
        self._last_distance_to_goal: float | None = None
        self._last_reverse_reason = ""
        self._last_reverse_submode = ""
        self._last_ahead_component = 0.0
        self._goal_behind_count = 0
        self._control_steps = 0

    def reset(self) -> None:
        self.mode = "FOLLOW"
        self._deadlock_count = 0
        self._recovery_count = 0
        self._recovery_dir = 1.0
        self._last_cmd = np.zeros(2, dtype=float)
        self._last_distance_to_goal = None
        self._last_reverse_reason = ""
        self._last_reverse_submode = ""
        self._last_ahead_component = 0.0
        self._goal_behind_count = 0
        self._control_steps = 0

    def control(
        self,
        robot_pose,
        robot_vel,
        goal_traj,
        obstacles,
        target_speed: float = 0.0,
        desired_distance: float = 1.5,
        target_distance: float | None = None,
        target_closing_speed: float = 0.0,
        map_query=None,
        visibility_yaw_error: float | None = None,
        target_visible_for_reverse: bool = True,
        preferred_view: str | None = None,
        target_view: str | None = None,
    ):
        self._control_steps += 1
        robot_pose = np.asarray(robot_pose, dtype=float).reshape(3)
        robot_vel = np.asarray(robot_vel, dtype=float).reshape(2)
        goal_arr = np.asarray(goal_traj, dtype=float)
        if goal_arr.ndim == 1:
            goal_arr = goal_arr.reshape(1, -1)
        goal_traj = goal_arr.reshape(-1, goal_arr.shape[-1])
        goal_xy = goal_traj[:, :2]
        obstacles = _coerce_obstacles(obstacles, default_radius=self.obstacle_radius)

        current_goal = goal_xy[0].copy() if goal_xy.size else robot_pose[:2].copy()
        hint_point, hint_index = self._select_hint_point(current_goal, robot_vel, goal_xy)
        angle_to_goal = self._angle_to_point(robot_pose, current_goal)
        distance_to_goal = float(np.linalg.norm(current_goal - robot_pose[:2]))
        hint_vector = hint_point - current_goal
        hint_norm = float(np.linalg.norm(hint_vector))
        mode = self._update_mode(
            robot_pose,
            robot_vel,
            current_goal,
            angle_to_goal,
            distance_to_goal,
            desired_distance,
            target_distance=target_distance,
            target_closing_speed=target_closing_speed,
            target_visible_for_reverse=target_visible_for_reverse,
        )

        allowable_vel = self._calc_dynamic_window(robot_vel, mode=mode, angle_to_goal=angle_to_goal)
        candidate_trajectory = self._generate_trajectory(robot_pose, allowable_vel, robot_vel=robot_vel, mode=mode)
        opt_vel, opt_trajectory, debug = self._evaluate_trajectory(
            candidate_trajectory,
            robot_vel,
            current_goal,
            hint_point,
            angle_to_goal,
            obstacles,
            mode=mode,
            target_speed=target_speed,
            desired_distance=desired_distance,
            distance_to_goal=distance_to_goal,
            target_distance=target_distance,
            visibility_yaw_error=visibility_yaw_error,
            map_query=map_query,
        )

        self._last_cmd = np.asarray(opt_vel, dtype=float).reshape(2)
        self._last_distance_to_goal = distance_to_goal

        opt_vel_arr = np.array(opt_vel, dtype=float).reshape(2, 1)
        opt_state_list = opt_trajectory[:, :2].T
        info = {
            "arrive": bool(distance_to_goal <= 0.15),
            "opt_state_list": opt_state_list,
            "mode": mode,
            "current_goal_point": current_goal.tolist(),
            "hint_point": hint_point.tolist(),
            "hint_index": int(hint_index),
            "hint_distance_from_goal": hint_norm,
            "angle_to_goal": float(angle_to_goal),
            "distance_to_goal": float(distance_to_goal),
            "target_distance": None if target_distance is None else float(target_distance),
            "target_closing_speed": float(target_closing_speed),
            "reverse_reason": self._last_reverse_reason,
            "reverse_submode": self._last_reverse_submode,
            "ahead_component": float(self._last_ahead_component),
            "goal_behind_confirm_count": int(self._goal_behind_count),
            "goal_behind_deadband": float(self.perception_goal_behind_deadband),
            "preferred_view": preferred_view,
            "target_view": target_view,
            "target_visible_for_reverse": bool(target_visible_for_reverse),
            "visibility_yaw_error": None if visibility_yaw_error is None else float(visibility_yaw_error),
            **debug,
        }

        return opt_vel_arr, info

    def _update_mode(
        self,
        robot_pose: np.ndarray,
        robot_vel: np.ndarray,
        current_goal: np.ndarray,
        angle_to_goal: float,
        distance_to_goal: float,
        desired_distance: float,
        target_distance: float | None = None,
        target_closing_speed: float = 0.0,
        target_visible_for_reverse: bool = True,
    ) -> str:
        if self._recovery_count > 0:
            self._recovery_count -= 1
            self.mode = "RECOVERY"
            self._last_reverse_reason = ""
            self._last_reverse_submode = ""
            return self.mode

        robot_to_goal = current_goal - robot_pose[:2]
        forward = np.array([math.cos(robot_pose[2]), math.sin(robot_pose[2])], dtype=float)
        ahead_component = float(np.dot(robot_to_goal, forward))
        self._last_ahead_component = ahead_component
        too_close = target_distance is not None and float(target_distance) < float(desired_distance) - self.reverse_enter_margin
        closing_close = (
            target_distance is not None
            and float(target_distance) < float(desired_distance) + 0.25
            and float(target_closing_speed) > 0.15
            and ahead_component < 0.20
        )
        goal_behind = ahead_component < -self.perception_goal_behind_deadband and distance_to_goal > 0.18
        severe_goal_behind = ahead_component < -self.severe_goal_behind_m and distance_to_goal > 0.18

        if goal_behind:
            self._goal_behind_count += 1
        else:
            self._goal_behind_count = 0

        self._last_reverse_reason = ""
        self._last_reverse_submode = ""
        if self.enable_spacing_reverse:
            startup_guard = bool(self._control_steps <= self.startup_guard_steps)
            confirmed_goal_behind = self._goal_behind_count >= self.goal_behind_confirm_steps
            if not target_visible_for_reverse:
                if self.mode == "REVERSE":
                    self.mode = "FOLLOW"
            elif too_close:
                self.mode = "REVERSE"
                self._last_reverse_reason = "too_close"
                self._last_reverse_submode = "spacing_backoff"
            elif closing_close:
                self.mode = "REVERSE"
                self._last_reverse_reason = "target_closing"
                self._last_reverse_submode = "spacing_backoff"
            elif severe_goal_behind and confirmed_goal_behind and not startup_guard:
                self.mode = "REVERSE"
                self._last_reverse_reason = "goal_behind"
                self._last_reverse_submode = "maneuver_reverse"
            elif goal_behind and confirmed_goal_behind and not startup_guard:
                self.mode = "REVERSE"
                self._last_reverse_reason = "goal_behind"
                self._last_reverse_submode = "spacing_backoff"
            elif self.mode == "REVERSE":
                self.mode = "FOLLOW"
        elif too_close:
            self.mode = "REVERSE"
            self._last_reverse_reason = "too_close"
        elif goal_behind:
            self.mode = "REVERSE"
            self._last_reverse_reason = "goal_behind"
        elif closing_close:
            self.mode = "REVERSE"
            self._last_reverse_reason = "target_closing"
        elif self.mode == "REVERSE" and ahead_component >= -0.05 and not too_close:
            self.mode = "FOLLOW"

        if self.mode == "ALIGN":
            if abs(angle_to_goal) <= self.align_exit_angle:
                self.mode = "FOLLOW"
        elif self.mode != "REVERSE" and abs(angle_to_goal) >= self.align_enter_angle and distance_to_goal > 0.45:
            self.mode = "ALIGN"

        if self.mode == "REVERSE" and ahead_component >= -0.05 and not too_close and distance_to_goal <= 0.25:
            self.mode = "FOLLOW"
            self._last_reverse_submode = ""

        progress = 0.0
        if self._last_distance_to_goal is not None:
            progress = self._last_distance_to_goal - distance_to_goal
        if (
            self.mode == "FOLLOW"
            and abs(robot_vel[0]) < self.deadlock_speed_threshold
            and abs(progress) < self.deadlock_progress_threshold
            and distance_to_goal > 0.7
        ):
            self._deadlock_count += 1
        else:
            self._deadlock_count = 0
        if self._deadlock_count >= self.deadlock_frames:
            self._deadlock_count = 0
            self._recovery_count = self.recovery_frames
            self._recovery_dir = 1.0 if angle_to_goal >= 0.0 else -1.0
            self.mode = "RECOVERY"

        return self.mode

    def _calc_dynamic_window(self, robot_vel, mode: str, angle_to_goal: float):
        acc_v = self.acc_cons[0] * (1.8 if mode == "FOLLOW" else 1.2)
        acc_w = self.acc_cons[1] * (1.8 if mode == "ALIGN" else 1.0)
        min_v = max(self.vel_cons[0], float(robot_vel[0]) - acc_v * self.dt)
        max_v = min(self.vel_cons[1], float(robot_vel[0]) + acc_v * self.dt)
        min_w = max(self.vel_cons[2], float(robot_vel[1]) - acc_w * self.dt)
        max_w = min(self.vel_cons[3], float(robot_vel[1]) + acc_w * self.dt)

        if mode == "ALIGN":
            min_v = -0.05
            max_v = 0.05
            max_abs_w = min(abs(self.vel_cons[2]), abs(self.vel_cons[3]), self.max_align_yawrate)
            min_w = max(min_w, -max_abs_w)
            max_w = min(max_w, max_abs_w)
        elif mode == "REVERSE":
            reverse_max_speed = self.spacing_reverse_max_speed if self._last_reverse_submode == "spacing_backoff" else self.reverse_max_speed
            min_v = max(min_v, -reverse_max_speed)
            max_v = min(max_v, 0.08)
            if min_v > max_v:
                max_v = min_v
            reverse_max_yawrate = self.spacing_reverse_max_yawrate if self._last_reverse_submode == "spacing_backoff" else self.reverse_max_yawrate
            max_abs_w = min(abs(self.vel_cons[2]), abs(self.vel_cons[3]), reverse_max_yawrate)
            min_w = max(min_w, -max_abs_w)
            max_w = min(max_w, max_abs_w)
        elif mode == "RECOVERY":
            min_v = max(min_v, self.recovery_reverse_speed - 0.05)
            max_v = min(max_v, 0.05)
            min_w = max(min_w, -self.recovery_yawrate)
            max_w = min(max_w, self.recovery_yawrate)
        else:
            angle_gate = max(0.25, math.cos(min(abs(float(angle_to_goal)), math.pi * 0.5)))
            max_v = min(max_v, self.vel_cons[1] * angle_gate)
            min_v = max(min_v, -0.20)

        if min_v > max_v or min_w > max_w:
            return np.zeros(4, dtype=float)
        return np.array([min_v, max_v, min_w, max_w], dtype=float)

    def _generate_trajectory(self, robot_pose, allowable_vel, robot_vel, mode: str):
        v_forwards = self._sample_axis(
            allowable_vel[0],
            allowable_vel[1],
            self.vel_resolution[0],
            extras=[0.0, float(robot_vel[0]), float(self._last_cmd[0])],
        )
        w_extras = [0.0, float(robot_vel[1]), float(self._last_cmd[1])]
        if mode == "ALIGN":
            w_extras.extend([self.min_align_yawrate, -self.min_align_yawrate])
        elif mode == "REVERSE":
            yaw_max = self.spacing_reverse_max_yawrate if self._last_reverse_submode == "spacing_backoff" else self.reverse_max_yawrate
            w_extras.extend([0.35, -0.35, yaw_max * 0.7, -yaw_max * 0.7])
            if self._last_reverse_submode == "spacing_backoff":
                v_extras = [0.0, -self.spacing_reverse_min_speed, -0.15, -self.spacing_reverse_max_speed]
            else:
                v_extras = [0.0, -0.15, -0.30]
            v_forwards = self._sample_axis(
                allowable_vel[0],
                allowable_vel[1],
                self.vel_resolution[0],
                extras=[*v_extras, float(robot_vel[0]), float(self._last_cmd[0])],
            )
        elif mode == "RECOVERY":
            w_extras.append(self._recovery_dir * self.recovery_yawrate)
            v_forwards = self._sample_axis(
                allowable_vel[0],
                allowable_vel[1],
                self.vel_resolution[0],
                extras=[0.0, self.recovery_reverse_speed],
            )
        v_laterals = self._sample_axis(allowable_vel[2], allowable_vel[3], self.vel_resolution[1], extras=w_extras)

        v_forward, v_lateral = np.meshgrid(v_forwards, v_laterals, indexing="ij")
        v_forward = v_forward.flatten()
        v_lateral = v_lateral.flatten()
        steps = int(math.ceil(self.predict_horizon / self.dt)) + 1
        candidate_trajectory = np.zeros((len(v_forward), steps, 5), dtype=float)
        candidate_trajectory[:, 0, :3] = robot_pose[:3]
        candidate_trajectory[:, 0, 3] = v_forward
        candidate_trajectory[:, 0, 4] = v_lateral

        for t in range(1, steps):
            prev = candidate_trajectory[:, t - 1, :]
            candidate_trajectory[:, t, 0] = prev[:, 0] + prev[:, 3] * np.cos(prev[:, 2]) * self.dt
            candidate_trajectory[:, t, 1] = prev[:, 1] + prev[:, 3] * np.sin(prev[:, 2]) * self.dt
            candidate_trajectory[:, t, 2] = prev[:, 2] + prev[:, 4] * self.dt
            candidate_trajectory[:, t, 3] = prev[:, 3]
            candidate_trajectory[:, t, 4] = prev[:, 4]

        return candidate_trajectory

    @staticmethod
    def _sample_axis(low: float, high: float, resolution: float, extras) -> np.ndarray:
        if high < low:
            return np.array([0.0], dtype=float)
        count = max(int(math.ceil((high - low) / max(float(resolution), 1e-6))) + 1, 2)
        samples = np.linspace(low, high, count)
        for value in extras:
            value = float(value)
            if low - 1e-9 <= value <= high + 1e-9:
                samples = np.append(samples, value)
        return np.unique(np.round(samples, decimals=6))

    def _evaluate_trajectory(
        self,
        candidate_trajectory,
        robot_vel,
        current_goal,
        hint_point,
        angle_to_goal,
        obstacles,
        mode: str,
        target_speed: float,
        desired_distance: float,
        distance_to_goal: float,
        target_distance: float | None = None,
        visibility_yaw_error: float | None = None,
        map_query=None,
    ):
        obstacle_info = self._obstacle_cost(candidate_trajectory, robot_vel, obstacles, map_query=map_query)
        hard_invalid = obstacle_info["hard_invalid"]
        v_ref = self._reference_speed(
            mode,
            target_speed,
            distance_to_goal,
            desired_distance,
            angle_to_goal,
            target_distance=target_distance,
        )

        distance_cost = self._goal_cost(candidate_trajectory, current_goal)
        progress_cost = self._progress_cost(candidate_trajectory, current_goal, distance_to_goal)
        heading_cost = self._heading_cost(candidate_trajectory, current_goal)
        hint_cost = self._hint_cost(candidate_trajectory, current_goal, hint_point)
        velocity_cost = _clip_cost(((candidate_trajectory[:, -1, 3] - v_ref) / self.velocity_scale) ** 2)
        turn_cost = _clip_cost(np.abs(candidate_trajectory[:, -1, 4]) / max(abs(self.vel_cons[3]), 1e-6))
        smooth_cost = _clip_cost(
            np.abs(candidate_trajectory[:, -1, 3] - robot_vel[0]) / max(self.acc_cons[0] * 0.5, 1e-6)
            + np.abs(candidate_trajectory[:, -1, 4] - robot_vel[1]) / max(self.acc_cons[1] * 0.5, 1e-6),
            cap=4.0,
        )
        if mode == "REVERSE":
            reverse_cost = np.maximum(0.0, candidate_trajectory[:, -1, 3]) / max(self.reverse_max_speed, 1e-6)
        else:
            reverse_cost = np.maximum(0.0, -candidate_trajectory[:, -1, 3]) / max(abs(self.vel_cons[0]), 1e-6)
        if mode == "REVERSE":
            reverse_cost *= 0.35

        total_costs = (
            self.weight_distance * distance_cost
            + self.weight_progress * progress_cost
            + self.weight_heading * heading_cost
            + 0.30 * hint_cost
            + self.weight_obstacle * obstacle_info["soft_cost"]
            + self.weight_velocity * velocity_cost
            + self.weight_turn * turn_cost
            + self.weight_smooth * smooth_cost
            + self.weight_reverse * reverse_cost
        )

        selectable = ~hard_invalid
        fallback_used = False
        if not np.any(selectable):
            fallback_used = True
            total_costs = (
                2.0 * np.maximum(0.0, -obstacle_info["min_clearance"])
                + 0.7 * np.abs(candidate_trajectory[:, -1, 3])
                + 0.2 * np.abs(candidate_trajectory[:, -1, 4])
                + 0.5 * obstacle_info["soft_cost"]
            )
        else:
            total_costs = np.where(selectable, total_costs, np.inf)

        if mode == "ALIGN":
            desired_w = self._align_yawrate(angle_to_goal)
            total_costs += 0.8 * np.abs(candidate_trajectory[:, -1, 3])
            total_costs += 3.0 * np.abs(candidate_trajectory[:, -1, 4] - desired_w) / max(self.max_align_yawrate, 1e-6)
        elif mode == "REVERSE":
            if self._last_reverse_submode == "spacing_backoff":
                desired_w = self._spacing_reverse_yawrate(visibility_yaw_error)
                yaw_scale = self.spacing_reverse_max_yawrate
            else:
                desired_w = self._reverse_yawrate(angle_to_goal)
                yaw_scale = self.reverse_max_yawrate
            total_costs += 2.0 * np.abs(candidate_trajectory[:, -1, 4] - desired_w) / max(yaw_scale, 1e-6)
        elif mode == "RECOVERY":
            desired_w = self._recovery_dir * self.recovery_yawrate
            total_costs += np.abs(candidate_trajectory[:, -1, 4] - desired_w)

        opt_idx = int(np.argmin(total_costs))
        opt_traj = candidate_trajectory[opt_idx]
        opt_vel = [float(opt_traj[-1, 3]), float(opt_traj[-1, 4])]
        if mode == "ALIGN":
            opt_vel[0] = 0.0

        debug = {
            "v_ref": float(v_ref),
            "fallback_used": bool(fallback_used),
            "hard_invalid_count": int(np.count_nonzero(hard_invalid)),
            "candidate_count": int(len(candidate_trajectory)),
            "min_clearance": float(obstacle_info["min_clearance"][opt_idx]),
            "first_collision_time": None
            if not np.isfinite(obstacle_info["first_collision_time"][opt_idx])
            else float(obstacle_info["first_collision_time"][opt_idx]),
            "cost_distance": float(distance_cost[opt_idx]),
            "cost_progress": float(progress_cost[opt_idx]),
            "cost_heading": float(heading_cost[opt_idx]),
            "cost_hint": float(hint_cost[opt_idx]),
            "cost_obstacle": float(obstacle_info["soft_cost"][opt_idx]),
            "cost_static_obstacle": float(obstacle_info["static_soft_cost"][opt_idx]),
            "cost_dynamic_obstacle": float(obstacle_info["dynamic_soft_cost"][opt_idx]),
            "cost_velocity": float(velocity_cost[opt_idx]),
            "cost_turn": float(turn_cost[opt_idx]),
            "cost_smooth": float(smooth_cost[opt_idx]),
            "min_static_clearance": float(obstacle_info["static_min_clearance"][opt_idx]),
            "min_dynamic_clearance": float(obstacle_info["dynamic_min_clearance"][opt_idx]),
            "reverse_desired_w": None if mode != "REVERSE" else float(desired_w),
        }
        return opt_vel, opt_traj, debug

    def _reference_speed(
        self,
        mode: str,
        target_speed: float,
        distance_to_goal: float,
        desired_distance: float,
        angle_to_goal: float,
        target_distance: float | None = None,
    ) -> float:
        if mode == "ALIGN":
            return 0.0
        if mode == "RECOVERY":
            return self.recovery_reverse_speed
        if mode == "REVERSE":
            spacing_error = 0.0 if target_distance is None else max(float(desired_distance) - float(target_distance), 0.0)
            goal_error = max(float(distance_to_goal) - 0.10, 0.0)
            if self._last_reverse_submode == "spacing_backoff":
                speed_mag = max(0.45 * spacing_error, 0.18 * goal_error, self.spacing_reverse_min_speed)
                return -float(np.clip(speed_mag, self.spacing_reverse_min_speed, self.spacing_reverse_max_speed))
            speed_mag = max(0.60 * spacing_error, 0.35 * goal_error, 0.15)
            return -float(np.clip(speed_mag, 0.15, self.reverse_max_speed))
        distance_error = max(float(distance_to_goal) - 0.20, 0.0)
        angle_gate = max(0.0, math.cos(min(abs(float(angle_to_goal)), math.pi * 0.5)))
        v_ref = max(float(target_speed), 0.25) + 0.45 * distance_error
        return float(np.clip(v_ref * angle_gate, 0.0, self.vel_cons[1]))

    def _align_yawrate(self, angle_to_goal: float) -> float:
        if abs(float(angle_to_goal)) <= 1e-6:
            return 0.0
        magnitude = float(np.clip(abs(float(angle_to_goal)), self.min_align_yawrate, self.max_align_yawrate))
        return math.copysign(magnitude, float(angle_to_goal))

    def _reverse_yawrate(self, angle_to_goal: float) -> float:
        rear_error = float(wrap_pi(float(angle_to_goal) - math.pi))
        if abs(rear_error) <= math.radians(3.0):
            return 0.0
        magnitude = float(np.clip(abs(rear_error), 0.12, self.reverse_max_yawrate))
        return math.copysign(magnitude, rear_error)

    def _spacing_reverse_yawrate(self, visibility_yaw_error: float | None) -> float:
        if visibility_yaw_error is None:
            return 0.0
        err = float(wrap_pi(float(visibility_yaw_error)))
        if abs(err) <= math.radians(3.0):
            return 0.0
        magnitude = float(np.clip(abs(err), 0.08, self.spacing_reverse_max_yawrate))
        return math.copysign(magnitude, err)

    def _select_hint_point(self, current_goal: np.ndarray, robot_vel: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray, int]:
        if goal_xy.size == 0:
            return current_goal.copy(), 0
        goals = np.asarray(goal_xy, dtype=float).reshape(-1, 2)
        current_goal = np.asarray(current_goal, dtype=float).reshape(2)
        hint_dist = float(
            np.clip(abs(float(robot_vel[0])) * self.hint_time + self.hint_base, self.hint_min, self.hint_max)
        )
        if len(goals) == 1:
            return goals[0].copy(), 0
        diffs = np.diff(goals, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg)))
        idx = int(np.searchsorted(cum, hint_dist, side="left"))
        idx = int(np.clip(idx, 0, len(goals) - 1))
        trend = goals[idx] - goals[0]
        trend_norm = float(np.linalg.norm(trend))
        if trend_norm <= 1e-6:
            return current_goal.copy(), 0
        limited = current_goal + trend / trend_norm * min(hint_dist, self.hint_max)
        return limited, idx

    @staticmethod
    def _angle_to_point(robot_pose: np.ndarray, point: np.ndarray) -> float:
        delta = np.asarray(point, dtype=float).reshape(2) - robot_pose[:2]
        return float(wrap_pi(math.atan2(float(delta[1]), float(delta[0])) - float(robot_pose[2])))

    def _goal_cost(self, candidate_trajectory, current_goal):
        eval_steps = min(max(int(round(1.0 / max(self.dt, 1e-6))) + 1, 2), candidate_trajectory.shape[1])
        dists = np.linalg.norm(candidate_trajectory[:, 1:eval_steps, :2] - current_goal[None, None, :], axis=2)
        return _clip_cost(np.min(dists, axis=1) / self.distance_scale)

    def _progress_cost(self, candidate_trajectory, current_goal, current_distance: float):
        eval_steps = min(max(int(round(1.0 / max(self.dt, 1e-6))) + 1, 2), candidate_trajectory.shape[1])
        dists = np.linalg.norm(candidate_trajectory[:, 1:eval_steps, :2] - current_goal[None, None, :], axis=2)
        progress = float(current_distance) - np.min(dists, axis=1)
        return _clip_cost(np.maximum(0.0, -progress) / self.progress_scale)

    def _heading_cost(self, candidate_trajectory, current_goal):
        first_pose = candidate_trajectory[:, 1, :2]
        delta = current_goal[None, :] - first_pose
        target_angle = np.arctan2(delta[:, 1], delta[:, 0])
        effective_yaw = candidate_trajectory[:, 1, 2] + np.where(candidate_trajectory[:, 1, 3] < 0.0, np.pi, 0.0)
        return _clip_cost(np.abs(_wrap_pi_array(target_angle - effective_yaw)) / np.pi)

    def _hint_cost(self, candidate_trajectory, current_goal, hint_point):
        hint_vec = np.asarray(hint_point, dtype=float).reshape(2) - np.asarray(current_goal, dtype=float).reshape(2)
        hint_norm = float(np.linalg.norm(hint_vec))
        if hint_norm <= 1e-6:
            return np.zeros(candidate_trajectory.shape[0], dtype=float)
        eval_idx = min(max(int(round(0.6 / max(self.dt, 1e-6))), 1), candidate_trajectory.shape[1] - 1)
        move_vec = candidate_trajectory[:, eval_idx, :2] - candidate_trajectory[:, 0, :2]
        move_norm = np.linalg.norm(move_vec, axis=1)
        dot = np.sum(move_vec * hint_vec[None, :], axis=1) / np.maximum(move_norm * hint_norm, 1e-6)
        return _clip_cost((1.0 - dot) * 0.5)

    def _obstacle_cost(self, candidate_trajectory, robot_vel, obstacles, map_query=None):
        dynamic_info = self._dynamic_obstacle_cost(candidate_trajectory, robot_vel, obstacles)
        static_info = self._static_map_cost(candidate_trajectory, robot_vel, map_query)
        soft_cost = np.maximum(dynamic_info["soft_cost"], static_info["soft_cost"])
        hard_invalid = dynamic_info["hard_invalid"] | static_info["hard_invalid"]
        min_clearance = np.minimum(dynamic_info["min_clearance"], static_info["min_clearance"])
        first_collision_time = np.minimum(dynamic_info["first_collision_time"], static_info["first_collision_time"])
        return {
            "soft_cost": _clip_cost(soft_cost),
            "hard_invalid": hard_invalid,
            "min_clearance": min_clearance,
            "first_collision_time": first_collision_time,
            "static_soft_cost": static_info["soft_cost"],
            "dynamic_soft_cost": dynamic_info["soft_cost"],
            "static_min_clearance": static_info["min_clearance"],
            "dynamic_min_clearance": dynamic_info["min_clearance"],
        }

    def _dynamic_obstacle_cost(self, candidate_trajectory, robot_vel, obstacles):
        n_traj, steps, _ = candidate_trajectory.shape
        if len(obstacles) == 0:
            return self._empty_obstacle_info(n_traj)

        obs = _coerce_obstacles(obstacles, default_radius=self.obstacle_radius)
        times_full = np.arange(steps, dtype=float) * self.dt
        centers = obs[:, :2][None, :, :] + times_full[:, None, None] * obs[:, 2:4][None, :, :]
        radii = obs[:, 4]
        collision_dist = self.robot_radius + radii[None, None, :] + 0.15
        diffs = candidate_trajectory[:, :, None, :2] - centers[None, :, :, :]
        dists = np.hypot(diffs[..., 0], diffs[..., 1])
        clearance_by_step = np.min(dists - collision_dist, axis=2)
        return self._clearance_cost_from_steps(candidate_trajectory, robot_vel, clearance_by_step)

    def _static_map_cost(self, candidate_trajectory, robot_vel, map_query):
        n_traj, steps, _ = candidate_trajectory.shape
        if map_query is None or not hasattr(map_query, "distance_to_obstacle_points"):
            return self._empty_obstacle_info(n_traj)

        points = candidate_trajectory[:, :, :2].reshape(-1, 2)
        try:
            clearance = self._map_clearance_points(map_query, points).reshape(n_traj, steps)
        except Exception:
            return self._empty_obstacle_info(n_traj)
        return self._clearance_cost_from_steps(candidate_trajectory, robot_vel, clearance)

    @staticmethod
    def _map_clearance_points(map_query, points: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(map_query.distance_to_obstacle_points(points, unknown_is_occupied=False), dtype=float)
        except TypeError:
            return np.asarray(map_query.distance_to_obstacle_points(points), dtype=float)

    def _clearance_cost_from_steps(self, candidate_trajectory, robot_vel, clearance_by_step):
        n_traj, steps, _ = candidate_trajectory.shape
        clearance_future = clearance_by_step[:, 1:]
        min_clearance = np.min(clearance_future, axis=1)

        collision_mask = clearance_future <= 0.05
        has_collision = np.any(collision_mask, axis=1)
        first_collision_idx = np.argmax(collision_mask, axis=1) + 1
        first_collision_time = np.where(has_collision, first_collision_idx * self.dt, np.inf)

        t_stop = abs(float(robot_vel[0])) / max(self.acc_cons[0], 1e-6)
        hard_horizon = float(np.clip(t_stop + self.reaction_time, self.hard_horizon_min, self.hard_horizon_max))
        hard_invalid = first_collision_time <= hard_horizon

        times = np.arange(1, steps, dtype=float) * self.dt
        time_weight = np.exp(-times / self.obstacle_tau)
        risk = np.where(
            clearance_future <= 0.05,
            self.soft_collision_penalty,
            np.exp(-np.maximum(clearance_future, 0.0) / self.clearance_sigma),
        )
        soft_cost = np.sum(risk * time_weight[None, :], axis=1) / max(float(np.sum(time_weight)), 1e-6)
        initial_collision = clearance_by_step[:, 0] <= 0.05
        hard_invalid |= initial_collision
        soft_cost = np.where(initial_collision, self.soft_collision_penalty, soft_cost)

        return {
            "soft_cost": _clip_cost(soft_cost),
            "hard_invalid": hard_invalid,
            "min_clearance": min_clearance,
            "first_collision_time": first_collision_time,
        }

    @staticmethod
    def _empty_obstacle_info(n_traj: int):
        return {
            "soft_cost": np.zeros(n_traj, dtype=float),
            "hard_invalid": np.zeros(n_traj, dtype=bool),
            "min_clearance": np.full(n_traj, np.inf, dtype=float),
            "first_collision_time": np.full(n_traj, np.inf, dtype=float),
        }


_robot_nt = namedtuple("robot", "radius min_speed max_speed max_acce")
_SCOUT_ROBOT = _robot_nt(
    radius=0.5,
    min_speed=[-0.6, -3.14],
    max_speed=[2.0, 3.14],
    max_acce=[1.5, 2.4],
)


class DwaTrajPlanner:
    def __init__(
        self,
        dt: float,
        obstacle_radius: float = 0.3,
        predict_horizon: float = 2.0,
        weight_heading: float = 0.7,
        weight_distance: float = 0.8,
        weight_obstacle: float = 1.5,
        weight_velocity: float = 1.2,
        enable_spacing_reverse: bool = False,
        perception_goal_behind_deadband: float = 0.20,
        goal_behind_confirm_steps: int = 1,
        startup_guard_steps: int = 0,
        spacing_reverse_min_speed: float = 0.08,
        spacing_reverse_max_speed: float = 0.22,
        spacing_reverse_max_yawrate: float = 0.35,
        severe_goal_behind_m: float = 0.80,
    ) -> None:
        self.dt = float(dt)
        self._robot_vel = np.array([0.0, 0.0], dtype=float)
        self._dwa = DWA(
            robot_tuple=_SCOUT_ROBOT,
            obstacle_radius=obstacle_radius,
            sample_time=dt,
            predict_horizon=predict_horizon,
            resolution_scale=(0.02, 0.02),
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

    @property
    def num_steps(self) -> int:
        return int(math.ceil(self._dwa.predict_horizon / max(self.dt, 1e-6))) + 1

    def reset(self) -> None:
        self._robot_vel = np.array([0.0, 0.0], dtype=float)
        self._dwa.reset()

    def compute(
        self,
        robot_pose,
        goal_traj,
        obstacles,
        robot_vel=None,
        target_speed: float = 0.0,
        desired_distance: float = 1.5,
        target_distance: float | None = None,
        target_closing_speed: float = 0.0,
        map_query=None,
        visibility_yaw_error: float | None = None,
        target_visible_for_reverse: bool = True,
        preferred_view: str | None = None,
        target_view: str | None = None,
    ):
        robot_pose_arr = np.asarray(robot_pose, dtype=float).reshape(3)
        if robot_vel is None:
            robot_vel_arr = self._robot_vel.copy()
        else:
            robot_vel_arr = np.asarray(robot_vel, dtype=float).reshape(-1)[:2]
        goal_traj_arr = np.asarray(goal_traj, dtype=float)
        if goal_traj_arr.ndim == 1:
            goal_traj_arr = goal_traj_arr.reshape(1, -1)
        goal_traj_arr = goal_traj_arr[:, :2]
        obstacles_arr = _coerce_obstacles(obstacles, default_radius=self._dwa.obstacle_radius)
        opt_vel, info = self._dwa.control(
            robot_pose_arr,
            robot_vel_arr,
            goal_traj_arr,
            obstacles_arr,
            target_speed=target_speed,
            desired_distance=desired_distance,
            target_distance=target_distance,
            target_closing_speed=target_closing_speed,
            map_query=map_query,
            visibility_yaw_error=visibility_yaw_error,
            target_visible_for_reverse=target_visible_for_reverse,
            preferred_view=preferred_view,
            target_view=target_view,
        )
        v = float(opt_vel[0])
        w = float(opt_vel[1])
        self._robot_vel = np.array([v, w], dtype=float)
        return v, w, info
