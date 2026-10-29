from __future__ import annotations

import math

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.planning.sfm.follow_goal_strategy import SfmFollowGoalStrategy
from baselines.planning.sfm.force_field import SocialForceConfig, SocialForceField
from baselines.planning.sfm.kinematics import make_kinematics
from baselines.planning.sfm.target_reference import TargetReferenceFilter
from baselines.planning.sfm.types import AgentPrediction, KinematicCommand, KinematicState, SfmPlanResult


class SfmPlanner:
    def __init__(
        self,
        dt: float,
        robot_radius: float = 0.5,
        max_v_mps: float = 2.5,
        max_w_radps: float = 2.5,
        max_acc_v: float = 2.0,
        max_acc_w: float = 4.0,
        predict_horizon: float = 2.0,
        plan_dt: float = 0.1,
        kinematics: str = "diff_drive",
        back_goal_mode: str = "target_heading",
        vmax: float | None = None,
        w_max: float | None = None,
        max_jerk_v: float = 12.0,
        max_jerk_w: float = 16.0,
    ) -> None:
        self.dt = float(dt)
        self.robot_radius = float(robot_radius)
        self.vmax = float(max_v_mps if vmax is None else vmax)
        self.w_max = float(max_w_radps if w_max is None else w_max)
        self.max_acc_v = float(max_acc_v)
        self.max_acc_w = float(max_acc_w)
        self.predict_horizon = float(predict_horizon)
        self.plan_dt = float(plan_dt)
        self.num_steps = max(int(math.ceil(self.predict_horizon / max(self.plan_dt, 1e-6))), 1)
        self._last_v = 0.0
        self._last_w = 0.0
        self._last_acc_v = 0.0
        self._last_acc_w = 0.0
        self.max_jerk_v = float(max_jerk_v)
        self.max_jerk_w = float(max_jerk_w)

        self.force_field = SocialForceField(
            SocialForceConfig(
                max_speed=self.vmax,
                safe_clearance=self.robot_radius + 0.2,
            )
        )
        self.kinematics = make_kinematics(
            kinematics,
            max_v=self.vmax,
            max_w=self.w_max,
            max_acc_v=self.max_acc_v,
            max_acc_w=self.max_acc_w,
        )
        self.goal_strategy = SfmFollowGoalStrategy(back_goal_mode=back_goal_mode)
        self.target_reference = TargetReferenceFilter()

    def reset(self) -> None:
        self._last_v = 0.0
        self._last_w = 0.0
        self._last_acc_v = 0.0
        self._last_acc_w = 0.0
        self.force_field.reset()
        self.kinematics.reset()
        self.goal_strategy.reset()
        self.target_reference.reset()

    def compute(
        self,
        robot_pose,
        goal=None,
        humans=None,
        map_query=None,
        target_state=None,
        follow_position: str = "back",
        desired_distance: float = 1.5,
    ) -> dict:
        robot_pose_arr = np.asarray(robot_pose, dtype=float).reshape(3)
        robot_state = self._initial_state(robot_pose_arr)
        humans_list = [] if humans is None else list(humans)

        reference_debug: dict = {}
        if target_state is not None:
            target_prediction, target_debug = self.target_reference.target_prediction(
                target_state=target_state,
                tick_dt=self.dt,
                num_steps=self.num_steps,
                plan_dt=self.plan_dt,
            )
            reference_debug.update(target_debug)
        else:
            target_prediction = self._prediction_from_goal(goal, robot_state)
        human_predictions = [self._prediction_from_human_state(h) for h in humans_list]

        if target_state is not None:
            raw_goal_traj = self.goal_strategy.build_goal_traj(
                robot_state=robot_state,
                target_prediction=target_prediction,
                follow_position=follow_position,
                desired_distance=desired_distance,
            )
            goal_traj, goal_debug = self.target_reference.smooth_goal_traj(
                raw_goal_traj=raw_goal_traj,
                tick_dt=self.dt,
                plan_dt=self.plan_dt,
                follow_position=follow_position,
            )
            reference_debug.update(goal_debug)
        else:
            goal_arr = np.asarray(goal, dtype=float).reshape(-1)
            goal_traj = np.tile(goal_arr[:3], (self.num_steps, 1))

        plan = self._rollout(
            initial_state=robot_state,
            target_prediction=target_prediction,
            human_predictions=human_predictions,
            goal_traj=goal_traj,
            map_query=map_query,
            desired_distance=desired_distance,
            reference_debug=reference_debug,
        )

        command = plan.command
        self._last_v = float(command.v)
        self._last_w = float(command.w)

        first_desired_velocity = np.asarray(plan.debug.get("first_desired_velocity", np.zeros(2)), dtype=float)
        heading_err = float(command.debug.get("heading_err", 0.0))
        info = self._build_info(plan)
        return {
            "v_mps": float(command.v),
            "w_radps": float(command.w),
            "vx_sfm": float(first_desired_velocity[0]),
            "vy_sfm": float(first_desired_velocity[1]),
            "heading_err": heading_err,
            "info": info,
        }

    def _initial_state(self, robot_pose: np.ndarray) -> KinematicState:
        yaw = float(robot_pose[2])
        return KinematicState(
            x=float(robot_pose[0]),
            y=float(robot_pose[1]),
            yaw=yaw,
            vx=self._last_v * math.cos(yaw),
            vy=self._last_v * math.sin(yaw),
            v=self._last_v,
            w=self._last_w,
        )

    def _prediction_from_goal(self, goal, robot_state: KinematicState) -> AgentPrediction:
        goal_arr = np.asarray(goal, dtype=float).reshape(-1)
        xy = np.tile(goal_arr[:2], (self.num_steps, 1))
        yaw = float(goal_arr[2]) if goal_arr.size >= 3 else float(robot_state.yaw)
        return AgentPrediction(
            positions=xy,
            velocities=np.zeros_like(xy),
            yaws=np.full((self.num_steps,), yaw, dtype=float),
        )

    def _prediction_from_human_state(self, human_state) -> AgentPrediction:
        arr = np.asarray(human_state, dtype=float).reshape(-1)
        x, y = float(arr[0]), float(arr[1])
        vx = float(arr[2]) if arr.size >= 3 else 0.0
        vy = float(arr[3]) if arr.size >= 4 else 0.0
        yaw = float(arr[4]) if arr.size >= 5 and math.isfinite(float(arr[4])) else _yaw_from_velocity(vx, vy)
        times = np.arange(self.num_steps, dtype=float) * self.plan_dt
        positions = np.column_stack((x + vx * times, y + vy * times))
        velocities = np.tile(np.array([vx, vy], dtype=float), (self.num_steps, 1))
        yaws = np.full((self.num_steps,), yaw, dtype=float)
        return AgentPrediction(positions=positions, velocities=velocities, yaws=yaws)

    def _rollout(
        self,
        initial_state: KinematicState,
        target_prediction: AgentPrediction,
        human_predictions: list[AgentPrediction],
        goal_traj: np.ndarray,
        map_query,
        desired_distance: float,
        reference_debug: dict | None = None,
    ) -> SfmPlanResult:
        state = initial_state
        robot_traj = np.zeros((self.num_steps, 3), dtype=float)
        force_traj = np.zeros((self.num_steps, 2), dtype=float)
        clearance_traj = np.full((self.num_steps,), float("inf"), dtype=float)
        commands: list[KinematicCommand] = []
        first_desired_velocity = np.zeros(2, dtype=float)
        first_force_debug: dict = {}

        for k in range(self.num_steps):
            humans_xy = _humans_at_step(human_predictions, k)
            target_xy = target_prediction.positions[min(k, len(target_prediction.positions) - 1)]
            target_velocity = target_prediction.velocities[min(k, len(target_prediction.velocities) - 1)]
            goal_xy = goal_traj[min(k, len(goal_traj) - 1), :2]
            desired_velocity, force_debug = self.force_field.desired_velocity(
                state=state,
                goal_xy=goal_xy,
                humans_xy=humans_xy,
                target_xy=target_xy,
                target_velocity=target_velocity,
                map_query=map_query,
                desired_distance=desired_distance,
                temporal_filter=(k == 0),
                dt=self.dt if k == 0 else self.plan_dt,
            )
            command_dt = self.dt if k == 0 else self.plan_dt
            command = self.kinematics.command_from_desired_velocity(state, desired_velocity, command_dt)
            if k == 0:
                command = self._limit_first_command_jerk(command, state, command_dt)
                first_desired_velocity = desired_velocity.copy()
                first_force_debug = force_debug
            force_traj[k] = np.asarray(force_debug["total_force"], dtype=float)
            clearance = force_debug.get("map_clearance")
            clearance_traj[k] = float("inf") if clearance is None else float(clearance)
            state = self.kinematics.step(state, command, self.plan_dt)
            robot_traj[k] = np.array([state.x, state.y, state.yaw], dtype=float)
            commands.append(command)

        command0 = commands[0] if commands else KinematicCommand()
        debug = {
            **first_force_debug,
            **(reference_debug or {}),
            "first_desired_velocity": first_desired_velocity.tolist(),
            "kinematics": self.kinematics.name,
            "num_steps": self.num_steps,
            "plan_dt": self.plan_dt,
            "predict_horizon": self.predict_horizon,
            "command_debug": command0.debug,
        }
        return SfmPlanResult(
            command=command0,
            robot_traj=robot_traj,
            goal_traj=np.asarray(goal_traj, dtype=float),
            target_traj=target_prediction.positions,
            force_traj=force_traj,
            clearance_traj=clearance_traj,
            debug=debug,
        )

    def _limit_first_command_jerk(
        self,
        command: KinematicCommand,
        state: KinematicState,
        dt: float,
    ) -> KinematicCommand:
        dt_safe = max(float(dt), 1e-3)
        acc_v = (float(command.v) - float(state.v)) / dt_safe
        acc_w = (float(command.w) - float(state.w)) / dt_safe
        acc_v = _limit_scalar_delta(acc_v, self._last_acc_v, self.max_jerk_v * dt_safe)
        acc_w = _limit_scalar_delta(acc_w, self._last_acc_w, self.max_jerk_w * dt_safe)
        v = float(state.v) + acc_v * dt_safe
        w = float(state.w) + acc_w * dt_safe
        self._last_acc_v = acc_v
        self._last_acc_w = acc_w
        debug = {
            **command.debug,
            "acc_v": acc_v,
            "acc_w": acc_w,
            "jerk_limited": True,
        }
        return KinematicCommand(v=v, w=w, vx=command.vx, vy=command.vy, steer=command.steer, debug=debug)

    def _build_info(self, plan: SfmPlanResult) -> dict:
        robot_traj = np.asarray(plan.robot_traj, dtype=float)
        goal_traj = np.asarray(plan.goal_traj, dtype=float)
        first_goal = goal_traj[0, :2] if len(goal_traj) else robot_traj[0, :2]
        first_robot = robot_traj[0, :2] if len(robot_traj) else first_goal
        distance_to_goal = float(np.linalg.norm(first_goal - first_robot))
        clearances = np.asarray(plan.clearance_traj, dtype=float)
        finite_clearances = clearances[np.isfinite(clearances)]
        min_clearance = float(np.min(finite_clearances)) if finite_clearances.size else None
        return {
            **plan.debug,
            "arrive": bool(distance_to_goal <= 0.15),
            "distance_to_goal": distance_to_goal,
            "map_clearance": min_clearance,
            "opt_state_list": [np.array([[p[0]], [p[1]]], dtype=float) for p in robot_traj],
            "robot_traj": robot_traj.tolist(),
            "goal_traj": goal_traj.tolist(),
            "predicted_target_traj": plan.target_traj.tolist(),
            "force_traj": plan.force_traj.tolist(),
            "clearance_traj": [
                None if not np.isfinite(clearance) else float(clearance)
                for clearance in plan.clearance_traj
            ],
        }

def _humans_at_step(human_predictions: list[AgentPrediction], step: int) -> np.ndarray:
    if not human_predictions:
        return np.empty((0, 2), dtype=float)
    points = []
    for prediction in human_predictions:
        idx = min(step, len(prediction.positions) - 1)
        points.append(prediction.positions[idx])
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _yaw_from_velocity(vx: float, vy: float) -> float:
    if math.hypot(vx, vy) > 0.05:
        return math.atan2(vy, vx)
    return 0.0


def _limit_scalar_delta(value: float, previous: float, max_delta: float) -> float:
    if max_delta <= 0.0:
        return float(previous)
    return float(previous + np.clip(float(value) - float(previous), -float(max_delta), float(max_delta)))
