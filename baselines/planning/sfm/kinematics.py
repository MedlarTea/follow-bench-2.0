from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from baselines.common.geometry import wrap_pi
from baselines.planning.sfm.types import KinematicCommand, KinematicState


class KinematicsAdapter(ABC):
    name = "base"

    def __init__(
        self,
        max_v: float,
        max_w: float,
        max_acc_v: float,
        max_acc_w: float,
        wheelbase: float = 0.5,
    ) -> None:
        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.max_acc_v = float(max_acc_v)
        self.max_acc_w = float(max_acc_w)
        self.wheelbase = float(wheelbase)

    def reset(self) -> None:
        pass

    @abstractmethod
    def command_from_desired_velocity(
        self,
        state: KinematicState,
        desired_velocity_world: np.ndarray,
        dt: float,
    ) -> KinematicCommand:
        raise NotImplementedError

    @abstractmethod
    def step(self, state: KinematicState, command: KinematicCommand, dt: float) -> KinematicState:
        raise NotImplementedError

    @staticmethod
    def _limit_delta(value: float, previous: float, max_delta: float) -> float:
        if max_delta <= 0.0:
            return float(previous)
        return float(previous + np.clip(value - previous, -max_delta, max_delta))


class DifferentialDriveKinematics(KinematicsAdapter):
    name = "diff_drive"

    def __init__(
        self,
        max_v: float,
        max_w: float,
        max_acc_v: float,
        max_acc_w: float,
        wheelbase: float = 0.5,
        k_heading: float = 1.0,
        heading_deadband: float = 0.05,
        w_filter_alpha: float = 0.60,
        v_filter_alpha: float = 0.25,
        turn_slowdown_gain: float = 0.55,
        reverse_enabled: bool = False,
    ) -> None:
        super().__init__(max_v, max_w, max_acc_v, max_acc_w, wheelbase)
        self.k_heading = float(k_heading)
        self.heading_deadband = float(max(heading_deadband, 0.0))
        self.w_filter_alpha = float(np.clip(w_filter_alpha, 0.0, 1.0))
        self.v_filter_alpha = float(np.clip(v_filter_alpha, 0.0, 1.0))
        self.turn_slowdown_gain = float(max(turn_slowdown_gain, 0.0))
        self.reverse_enabled = bool(reverse_enabled)

    def command_from_desired_velocity(
        self,
        state: KinematicState,
        desired_velocity_world: np.ndarray,
        dt: float,
    ) -> KinematicCommand:
        desired = np.asarray(desired_velocity_world, dtype=float).reshape(2)
        speed = float(np.linalg.norm(desired))
        if speed <= 1e-6:
            v_raw = 0.0
            w_raw = 0.0
            heading_err = 0.0
            desired_heading = float(state.yaw)
        else:
            desired_heading = math.atan2(float(desired[1]), float(desired[0]))
            heading_err = wrap_pi(desired_heading - float(state.yaw))
            if abs(heading_err) < self.heading_deadband:
                heading_err = 0.0
            heading_scale = math.cos(heading_err)
            if not self.reverse_enabled:
                heading_scale = max(heading_scale, 0.0)
            turn_scale = max(0.2, 1.0 - self.turn_slowdown_gain * abs(heading_err))
            v_raw = speed * heading_scale * turn_scale
            w_raw = self.k_heading * heading_err

        v_raw = float(np.clip(v_raw, -self.max_v if self.reverse_enabled else 0.0, self.max_v))
        w_raw = float(np.clip(w_raw, -self.max_w, self.max_w))
        v_raw = self.v_filter_alpha * state.v + (1.0 - self.v_filter_alpha) * v_raw
        w_raw = self.w_filter_alpha * state.w + (1.0 - self.w_filter_alpha) * w_raw
        v = self._limit_delta(v_raw, state.v, self.max_acc_v * dt)
        w = self._limit_delta(w_raw, state.w, self.max_acc_w * dt)
        return KinematicCommand(
            v=v,
            w=w,
            debug={
                "desired_heading": desired_heading,
                "heading_err": heading_err,
                "desired_speed": speed,
                "v_raw": v_raw,
                "w_raw": w_raw,
            },
        )

    def step(self, state: KinematicState, command: KinematicCommand, dt: float) -> KinematicState:
        yaw = wrap_pi(state.yaw + command.w * dt)
        x = state.x + command.v * math.cos(state.yaw) * dt
        y = state.y + command.v * math.sin(state.yaw) * dt
        return KinematicState(
            x=x,
            y=y,
            yaw=yaw,
            vx=command.v * math.cos(yaw),
            vy=command.v * math.sin(yaw),
            v=command.v,
            w=command.w,
        )


class OmniKinematics(KinematicsAdapter):
    name = "omni"

    def command_from_desired_velocity(
        self,
        state: KinematicState,
        desired_velocity_world: np.ndarray,
        dt: float,
    ) -> KinematicCommand:
        desired = np.asarray(desired_velocity_world, dtype=float).reshape(2)
        speed = float(np.linalg.norm(desired))
        if speed > self.max_v:
            desired = desired * (self.max_v / max(speed, 1e-9))
        return KinematicCommand(vx=float(desired[0]), vy=float(desired[1]), w=0.0)

    def step(self, state: KinematicState, command: KinematicCommand, dt: float) -> KinematicState:
        yaw = wrap_pi(state.yaw + command.w * dt)
        return KinematicState(
            x=state.x + command.vx * dt,
            y=state.y + command.vy * dt,
            yaw=yaw,
            vx=command.vx,
            vy=command.vy,
            v=float(math.hypot(command.vx, command.vy)),
            w=command.w,
        )


class AckermannKinematics(KinematicsAdapter):
    name = "ackermann"

    def command_from_desired_velocity(
        self,
        state: KinematicState,
        desired_velocity_world: np.ndarray,
        dt: float,
    ) -> KinematicCommand:
        diff = DifferentialDriveKinematics(
            max_v=self.max_v,
            max_w=self.max_w,
            max_acc_v=self.max_acc_v,
            max_acc_w=self.max_acc_w,
            wheelbase=self.wheelbase,
        )
        cmd = diff.command_from_desired_velocity(state, desired_velocity_world, dt)
        if abs(cmd.v) <= 1e-6:
            steer = 0.0
        else:
            steer = math.atan2(cmd.w * self.wheelbase, cmd.v)
        cmd.steer = steer
        return cmd

    def step(self, state: KinematicState, command: KinematicCommand, dt: float) -> KinematicState:
        yaw_rate = command.v * math.tan(command.steer) / max(self.wheelbase, 1e-6)
        yaw = wrap_pi(state.yaw + yaw_rate * dt)
        x = state.x + command.v * math.cos(state.yaw) * dt
        y = state.y + command.v * math.sin(state.yaw) * dt
        return KinematicState(
            x=x,
            y=y,
            yaw=yaw,
            vx=command.v * math.cos(yaw),
            vy=command.v * math.sin(yaw),
            v=command.v,
            w=yaw_rate,
        )


def make_kinematics(name: str, **kwargs) -> KinematicsAdapter:
    key = str(name).lower()
    if key in ("diff", "diff_drive", "differential"):
        return DifferentialDriveKinematics(**kwargs)
    if key in ("omni", "omnidirectional"):
        return OmniKinematics(**kwargs)
    if key in ("ackermann", "car"):
        return AckermannKinematics(**kwargs)
    raise ValueError(f"Unsupported SFM kinematics: {name}")
