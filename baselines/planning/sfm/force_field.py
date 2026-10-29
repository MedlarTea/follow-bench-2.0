from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from baselines.planning.sfm.types import KinematicState


@dataclass
class SocialForceConfig:
    max_speed: float = 2.5
    tau: float = 0.6
    stop_radius: float = 0.15
    slow_radius: float = 1.5
    follow_speed_margin: float = 0.35
    max_goal_speed_without_error: float = 1.4
    crowd_slowdown_per_human: float = 0.08
    human_a: float = 2.5
    human_b: float = 0.65
    human_safe_dist: float = 0.75
    human_influence_dist: float = 3.0
    max_human_force: float = 3.0
    human_force_alpha: float = 0.70
    max_human_force_delta: float = 3.0
    desired_velocity_alpha: float = 0.60
    max_desired_velocity_delta: float = 2.5
    target_a: float = 1.2
    target_b: float = 0.45
    static_a: float = 5.0
    static_b: float = 0.45
    static_influence_dist: float = 1.8
    safe_clearance: float = 0.7
    gradient_eps: float = 0.1


class SocialForceField:
    def __init__(self, config: SocialForceConfig) -> None:
        self.config = config
        self._last_human_force: np.ndarray | None = None
        self._last_desired_velocity: np.ndarray | None = None

    def reset(self) -> None:
        self._last_human_force = None
        self._last_desired_velocity = None

    def desired_velocity(
        self,
        state: KinematicState,
        goal_xy: np.ndarray,
        humans_xy: np.ndarray,
        target_xy: np.ndarray | None,
        target_velocity: np.ndarray | None,
        map_query,
        desired_distance: float,
        temporal_filter: bool = False,
        dt: float = 0.1,
    ) -> tuple[np.ndarray, dict]:
        robot_xy = state.xy
        robot_vel = np.array([state.vx, state.vy], dtype=float)
        humans_arr = np.asarray(humans_xy, dtype=float).reshape(-1, 2)
        crowd_count = _crowd_count(robot_xy, humans_arr, self.config.human_influence_dist)
        goal_force, goal_debug = self._goal_force(robot_xy, robot_vel, goal_xy, target_velocity, crowd_count)
        raw_human_force = self._human_force(robot_xy, humans_arr)
        human_force = _limit_norm(raw_human_force, self.config.max_human_force)
        if temporal_filter:
            human_force = self._temporal_filter(
                value=human_force,
                previous=self._last_human_force,
                alpha=self.config.human_force_alpha,
                max_delta=self.config.max_human_force_delta * max(float(dt), 1e-3),
            )
            self._last_human_force = human_force.copy()
        target_force = self._target_personal_space_force(robot_xy, target_xy, desired_distance)
        static_force, clearance = self._static_force(robot_xy, map_query)
        total_force = goal_force + human_force + target_force + static_force
        desired_velocity = robot_vel + total_force * self.config.tau
        desired_velocity = _limit_norm(desired_velocity, self.config.max_speed)
        if temporal_filter:
            desired_velocity = self._temporal_filter(
                value=desired_velocity,
                previous=self._last_desired_velocity,
                alpha=self.config.desired_velocity_alpha,
                max_delta=self.config.max_desired_velocity_delta * max(float(dt), 1e-3),
            )
            self._last_desired_velocity = desired_velocity.copy()
        debug = {
            **goal_debug,
            "goal_force": goal_force.tolist(),
            "human_force": human_force.tolist(),
            "raw_human_force": raw_human_force.tolist(),
            "target_force": target_force.tolist(),
            "map_force": static_force.tolist(),
            "total_force": total_force.tolist(),
            "map_clearance": None if not np.isfinite(clearance) else float(clearance),
            "crowd_count": int(crowd_count),
        }
        return desired_velocity, debug

    def _goal_force(
        self,
        robot_xy: np.ndarray,
        robot_vel: np.ndarray,
        goal_xy: np.ndarray,
        target_velocity: np.ndarray | None,
        crowd_count: int,
    ) -> tuple[np.ndarray, dict]:
        delta = np.asarray(goal_xy, dtype=float).reshape(2) - robot_xy
        distance = float(np.linalg.norm(delta))
        target_speed = 0.0 if target_velocity is None else float(np.linalg.norm(target_velocity))
        if distance <= 1e-9:
            desired_speed = 0.0
            direction = np.zeros(2, dtype=float)
        else:
            desired_speed = self._speed_profile(distance, target_speed, crowd_count)
            direction = delta / distance
        desired_velocity = desired_speed * direction
        force = (desired_velocity - robot_vel) / max(self.config.tau, 1e-6)
        return force, {
            "distance_to_goal": distance,
            "desired_speed": desired_speed,
            "target_speed_for_goal": target_speed,
            "goal_velocity": desired_velocity.tolist(),
        }

    def _speed_profile(self, distance: float, target_speed: float, crowd_count: int) -> float:
        if distance <= self.config.stop_radius:
            return 0.0
        comfortable_speed = min(
            self.config.max_speed,
            max(self.config.max_goal_speed_without_error, target_speed + self.config.follow_speed_margin),
        )
        if distance < self.config.slow_radius:
            span = max(self.config.slow_radius - self.config.stop_radius, 1e-6)
            desired = comfortable_speed * (distance - self.config.stop_radius) / span
        else:
            extra = min(max(distance - self.config.slow_radius, 0.0) * 0.35, 0.8)
            desired = min(self.config.max_speed, comfortable_speed + extra)
        crowd_scale = max(0.55, 1.0 - float(crowd_count) * self.config.crowd_slowdown_per_human)
        return desired * crowd_scale

    def _human_force(self, robot_xy: np.ndarray, humans_xy: np.ndarray) -> np.ndarray:
        humans = np.asarray(humans_xy, dtype=float).reshape(-1, 2)
        if humans.size == 0:
            return np.zeros(2, dtype=float)
        rel = robot_xy - humans
        dist = np.linalg.norm(rel, axis=1)
        valid = (dist > 1e-6) & (dist <= self.config.human_influence_dist)
        if not np.any(valid):
            return np.zeros(2, dtype=float)
        unit = rel[valid] / dist[valid, None]
        mag = self.config.human_a * np.exp((self.config.human_safe_dist - dist[valid, None]) / self.config.human_b)
        return np.sum(mag * unit, axis=0)

    def _target_personal_space_force(
        self,
        robot_xy: np.ndarray,
        target_xy: np.ndarray | None,
        desired_distance: float,
    ) -> np.ndarray:
        if target_xy is None:
            return np.zeros(2, dtype=float)
        rel = robot_xy - np.asarray(target_xy, dtype=float).reshape(2)
        dist = float(np.linalg.norm(rel))
        active_dist = max(0.7 * float(desired_distance), 1e-3)
        if dist <= 1e-6 or dist >= active_dist:
            return np.zeros(2, dtype=float)
        unit = rel / dist
        mag = self.config.target_a * math.exp((active_dist - dist) / self.config.target_b)
        return mag * unit

    def _static_force(self, robot_xy: np.ndarray, map_query) -> tuple[np.ndarray, float]:
        if map_query is None:
            return np.zeros(2, dtype=float), float("inf")
        clearance = _distance_to_obstacle(map_query, float(robot_xy[0]), float(robot_xy[1]))
        if not np.isfinite(clearance) or clearance >= self.config.static_influence_dist:
            return np.zeros(2, dtype=float), clearance
        eps = max(float(getattr(map_query, "resolution", self.config.gradient_eps)), self.config.gradient_eps, 1e-3)
        grad = np.array(
            [
                _distance_to_obstacle(map_query, float(robot_xy[0] + eps), float(robot_xy[1]))
                - _distance_to_obstacle(map_query, float(robot_xy[0] - eps), float(robot_xy[1])),
                _distance_to_obstacle(map_query, float(robot_xy[0]), float(robot_xy[1] + eps))
                - _distance_to_obstacle(map_query, float(robot_xy[0]), float(robot_xy[1] - eps)),
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(grad))
        if norm <= 1e-9:
            return np.zeros(2, dtype=float), clearance
        direction = grad / norm
        mag = self.config.static_a * math.exp((self.config.safe_clearance - clearance) / self.config.static_b)
        return mag * direction, clearance

    @staticmethod
    def _temporal_filter(
        value: np.ndarray,
        previous: np.ndarray | None,
        alpha: float,
        max_delta: float,
    ) -> np.ndarray:
        value_arr = np.asarray(value, dtype=float).reshape(2)
        if previous is None:
            return value_arr
        previous_arr = np.asarray(previous, dtype=float).reshape(2)
        blended = float(alpha) * previous_arr + (1.0 - float(alpha)) * value_arr
        delta = blended - previous_arr
        norm = float(np.linalg.norm(delta))
        if norm <= float(max_delta) or norm <= 1e-9:
            return blended
        return previous_arr + delta * (float(max_delta) / norm)


def _distance_to_obstacle(map_query, x: float, y: float) -> float:
    try:
        return float(map_query.distance_to_obstacle(x, y, unknown_is_occupied=False))
    except TypeError:
        return float(map_query.distance_to_obstacle(x, y))


def _limit_norm(vector: np.ndarray, limit: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= float(limit) or norm <= 1e-9:
        return np.asarray(vector, dtype=float)
    return np.asarray(vector, dtype=float) * (float(limit) / norm)


def _crowd_count(robot_xy: np.ndarray, humans_xy: np.ndarray, influence_dist: float) -> int:
    humans = np.asarray(humans_xy, dtype=float).reshape(-1, 2)
    if humans.size == 0:
        return 0
    dist = np.linalg.norm(humans - np.asarray(robot_xy, dtype=float).reshape(1, 2), axis=1)
    return int(np.count_nonzero(dist <= float(influence_dist)))
