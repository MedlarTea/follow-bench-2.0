"""
PID follower controller logic.

This module keeps the PID control implementation platform-neutral; adapters are
responsible for reading platform observations and turning the controller output
into platform actions.
"""
from __future__ import annotations

import math

import numpy as np


class _PIDController:
    """Pure-Python port of the ROS PID_controller class."""

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        deadband: float,
        u_min: float,
        u_max: float,
        e_int_min: float,
        e_int_max: float,
        dt: float = 0.1,
    ) -> None:
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.deadband = deadband
        self.u_min, self.u_max = u_min, u_max
        self.e_int_min, self.e_int_max = e_int_min, e_int_max
        self.dt = dt
        self.e_cur = self.e_old = self.e_int = self.e_der = self.u = 0.0
        self.ref = 0.0

    def reset(self) -> None:
        self.e_cur = self.e_old = self.e_int = self.e_der = self.u = 0.0

    def calc_output(self, x: float, dt: float) -> float:
        self.dt = dt
        self.e_cur = self.ref - x

        if -self.deadband / 2 <= self.e_cur <= self.deadband / 2:
            self.e_cur = 0.0

        self.e_der = (self.e_cur - self.e_old) / max(self.dt, 1e-6)
        self.e_int = float(np.clip(self.e_int + self.e_cur * self.dt, self.e_int_min, self.e_int_max))
        self.u = float(
            np.clip(
                self.Kp * self.e_cur + self.Kd * self.e_der + self.Ki * self.e_int,
                self.u_min,
                self.u_max,
            )
        )
        self.e_old = self.e_cur
        return self.u


class PIDFollowerController:
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        max_vx: float = 2.5,
        max_va: float = 2.0,
        enable_back: bool = True,
        scale_a: float = 2.5,
        scale_v: float = 2.0,
    ) -> None:
        self.dt = dt
        self.desired_distance = desired_distance
        self.max_vx = max_vx
        self.max_va = max_va
        self.enable_back = enable_back
        self.scale_a = scale_a
        self.scale_v = scale_v

        self.th_pid = _PIDController(
            Kp=1.5,
            Ki=0.0,
            Kd=0.1,
            deadband=0.0,
            u_min=-max_va,
            u_max=max_va,
            e_int_min=-0.2,
            e_int_max=0.2,
            dt=dt,
        )
        self.xy_pid = _PIDController(
            Kp=3.0,
            Ki=0.0,
            Kd=0.1,
            deadband=0.0,
            u_min=-max_vx,
            u_max=max_vx,
            e_int_min=-0.1,
            e_int_max=0.1,
            dt=dt,
        )

    def reset(self) -> None:
        self.th_pid.reset()
        self.xy_pid.reset()

    def compute(self, robot_x: float, robot_y: float, robot_yaw: float, target_x: float, target_y: float) -> dict:
        dx = float(target_x - robot_x)
        dy = float(target_y - robot_y)
        yaw = float(robot_yaw)

        px = dx * math.cos(yaw) + dy * math.sin(yaw)
        py = -dx * math.sin(yaw) + dy * math.cos(yaw)

        th_err = math.atan2(py, px)
        p_err = px - self.desired_distance

        w = self.th_pid.calc_output(-th_err, self.dt) * self.scale_a
        if abs(th_err) <= math.pi / 2:
            v = self.xy_pid.calc_output(-p_err, self.dt) * self.scale_v
            vx = float(np.clip(v, -self.max_vx, self.max_vx))
            vx *= float(max(0.3, math.cos(th_err)))
        else:
            vx = 0.35
            self.xy_pid.reset()

        return {
            "v_mps": float(vx),
            "w_radps": float(w),
            "px": float(px),
            "py": float(py),
            "th_err": float(th_err),
            "p_err": float(p_err),
        }


__all__ = ["PIDFollowerController"]
