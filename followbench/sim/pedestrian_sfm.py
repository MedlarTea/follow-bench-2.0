from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SFMPlannerConfig:
    tau: float
    desired_speed_scale: float
    ped_A: float
    ped_B: float
    ped_radius: float
    robot_A: float
    robot_B: float
    robot_radius: float
    wall_A: float
    wall_B: float
    wall_influence_dist: float
    tangential_weight: float
    robot_tangential_weight: float
    anisotropy_lambda: float
    prediction_enabled: bool
    prediction_horizon: float
    close_risk_margin: float
    headon_cos_threshold: float
    headon_front_dot_threshold: float
    headon_lateral_trigger_m: float
    headon_bias_weight: float
    headon_min_risk: float
    max_force: float
    max_speed: float
    neighbor_radius: float
    max_neighbors: int


@dataclass
class AgentKinematics:
    track_id: str
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    radius: float
    desired_speed: float
    goal_xy: np.ndarray
    reference_forward_xy: np.ndarray
    pass_side_pref: int


@dataclass
class DynamicObstacle:
    kind: str
    position_xy: np.ndarray
    velocity_xy: np.ndarray
    radius: float


@dataclass
class WallQueryResult:
    valid: bool
    distance: float
    normal_xy: np.ndarray


@dataclass
class SFMPlannerResult:
    velocity_xy: np.ndarray
    desired_velocity_xy: np.ndarray
    driving_force_xy: np.ndarray
    interaction_force_xy: np.ndarray
    tangential_force_xy: np.ndarray
    wall_force_xy: np.ndarray
    total_force_xy: np.ndarray
    max_conflict_risk: float
    min_neighbor_distance: float
    min_surface_distance: float
    suggested_pass_side: int
    dominant_obstacle_kind: str


class SFMPlanner:
    def __init__(self, config: SFMPlannerConfig) -> None:
        self.config = config

    def compute_velocity(
        self,
        agent: AgentKinematics,
        neighbors: List[DynamicObstacle],
        wall: WallQueryResult,
        dt: float,
    ) -> SFMPlannerResult:
        desired_velocity = self._compute_desired_velocity(agent)
        driving_force = self._compute_driving_force(agent, desired_velocity)
        interaction_force = np.zeros(2, dtype=np.float32)
        tangential_force = np.zeros(2, dtype=np.float32)
        max_conflict_risk = 0.0
        min_neighbor_distance = float("inf")
        min_surface_distance = float("inf")
        suggested_pass_side = int(agent.pass_side_pref) if int(agent.pass_side_pref) in (-1, 1) else 1
        dominant_obstacle_kind = ""

        counted = 0
        for obstacle in neighbors:
            center_dist = float(np.linalg.norm(agent.position_xy - obstacle.position_xy))
            if center_dist > 1e-6:
                min_neighbor_distance = min(min_neighbor_distance, center_dist)
                min_surface_distance = min(
                    min_surface_distance,
                    center_dist - float(agent.radius + obstacle.radius),
                )
            force, tangential, risk, side, kind = self._compute_neighbor_interaction(agent, obstacle, desired_velocity)
            if float(np.linalg.norm(force)) <= 1e-6:
                continue
            interaction_force += force
            tangential_force += tangential
            if risk > max_conflict_risk:
                max_conflict_risk = float(risk)
                suggested_pass_side = int(side)
                dominant_obstacle_kind = str(kind)
            counted += 1
            if counted >= self.config.max_neighbors:
                break

        wall_force = self._compute_wall_force(agent, wall)
        total_force = driving_force + interaction_force + wall_force
        force_norm = float(np.linalg.norm(total_force))
        if force_norm > self.config.max_force > 1e-6:
            total_force = total_force * (self.config.max_force / force_norm)

        dt_eff = max(float(dt), 1e-3)
        velocity = agent.velocity_xy + total_force * dt_eff
        speed = float(np.linalg.norm(velocity))
        if speed > self.config.max_speed > 1e-6:
            velocity = velocity * (self.config.max_speed / speed)

        return SFMPlannerResult(
            velocity_xy=velocity.astype(np.float32),
            desired_velocity_xy=desired_velocity.astype(np.float32),
            driving_force_xy=driving_force.astype(np.float32),
            interaction_force_xy=interaction_force.astype(np.float32),
            tangential_force_xy=tangential_force.astype(np.float32),
            wall_force_xy=wall_force.astype(np.float32),
            total_force_xy=total_force.astype(np.float32),
            max_conflict_risk=float(max_conflict_risk),
            min_neighbor_distance=float(min_neighbor_distance),
            min_surface_distance=float(min_surface_distance),
            suggested_pass_side=int(suggested_pass_side),
            dominant_obstacle_kind=dominant_obstacle_kind,
        )

    def _compute_desired_velocity(self, agent: AgentKinematics) -> np.ndarray:
        goal_vec = agent.goal_xy - agent.position_xy
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        desired_speed = max(0.0, float(agent.desired_speed) * self.config.desired_speed_scale)
        return (goal_vec / goal_dist * desired_speed).astype(np.float32)

    def _compute_driving_force(
        self,
        agent: AgentKinematics,
        desired_velocity: np.ndarray,
    ) -> np.ndarray:
        tau = max(float(self.config.tau), 1e-3)
        return ((desired_velocity - agent.velocity_xy) / tau).astype(np.float32)

    def _compute_neighbor_interaction(
        self,
        agent: AgentKinematics,
        obstacle: DynamicObstacle,
        desired_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int, str]:
        rel_now = agent.position_xy - obstacle.position_xy
        dist_now = float(np.linalg.norm(rel_now))
        if dist_now <= 1e-6 or dist_now > self.config.neighbor_radius:
            zero = np.zeros(2, dtype=np.float32)
            return zero, zero, 0.0, 0, ""

        rel_vel = agent.velocity_xy - obstacle.velocity_xy
        risk, d_min = self._compute_conflict_risk(agent, obstacle, rel_now, rel_vel, dist_now)
        headon = obstacle.kind == "pedestrian" and self._is_headon_conflict(agent, obstacle, rel_now)
        if headon:
            risk = max(float(risk), float(self.config.headon_min_risk))
        dist_for_force = float(d_min if risk > 1e-6 else dist_now)
        if dist_for_force <= 1e-6:
            dist_for_force = max(dist_now, 1e-3)
        direction = (rel_now / max(dist_now, 1e-3)).astype(np.float32)

        if obstacle.kind == "robot":
            strength = float(self.config.robot_A)
            decay = float(self.config.robot_B)
        else:
            strength = float(self.config.ped_A)
            decay = float(self.config.ped_B)

        decay = max(decay, 1e-3)
        combined_radius = float(agent.radius + obstacle.radius)
        base = strength * np.exp((combined_radius - dist_for_force) / decay)
        anisotropy = self._anisotropy_weight(agent, direction, desired_velocity)
        normal_force = (direction * base * anisotropy * (1.0 + 0.75 * risk)).astype(np.float32)
        pass_side = 1 if headon else self._suggest_pass_side(agent)
        tangential_force = self._compute_tangential_force(
            agent,
            direction,
            base,
            risk,
            pass_side,
            tangential_weight=(
                float(self.config.robot_tangential_weight)
                if obstacle.kind == "robot"
                else float(self.config.tangential_weight)
            ),
            bias_weight=float(self.config.headon_bias_weight) if headon else 1.0,
        )
        return (
            (normal_force + tangential_force).astype(np.float32),
            tangential_force.astype(np.float32),
            float(risk),
            int(pass_side),
            str(obstacle.kind),
        )

    def _is_headon_conflict(
        self,
        agent: AgentKinematics,
        obstacle: DynamicObstacle,
        rel_now: np.ndarray,
    ) -> bool:
        facing = agent.reference_forward_xy
        facing_norm = float(np.linalg.norm(facing))
        if facing_norm <= 1e-6:
            return False
        facing = facing / facing_norm
        obstacle_speed = float(np.linalg.norm(obstacle.velocity_xy))
        if obstacle_speed <= 1e-4:
            return False
        obstacle_dir = obstacle.velocity_xy / obstacle_speed
        if float(np.dot(facing, obstacle_dir)) > float(self.config.headon_cos_threshold):
            return False

        agent_to_obstacle = -rel_now
        dist = float(np.linalg.norm(agent_to_obstacle))
        if dist <= 1e-6:
            return False
        ahead = float(np.dot(facing, agent_to_obstacle / dist))
        if ahead < float(self.config.headon_front_dot_threshold):
            return False
        right = np.array([facing[1], -facing[0]], dtype=np.float32)
        lateral_abs = abs(float(np.dot(agent_to_obstacle, right)))
        return lateral_abs <= float(self.config.headon_lateral_trigger_m)

    def _compute_conflict_risk(
        self,
        agent: AgentKinematics,
        obstacle: DynamicObstacle,
        rel_now: np.ndarray,
        rel_vel: np.ndarray,
        dist_now: float,
    ) -> tuple[float, float]:
        combined_radius = float(agent.radius + obstacle.radius)
        close_margin = max(
            combined_radius * 2.35,
            combined_radius + max(float(self.config.close_risk_margin), 0.0),
        )
        close_risk = float(np.clip((close_margin - dist_now) / max(close_margin, 1e-3), 0.0, 1.0))
        if not bool(self.config.prediction_enabled):
            return close_risk, dist_now
        horizon = max(float(self.config.prediction_horizon), 1e-3)
        speed_sq = float(np.dot(rel_vel, rel_vel))
        if speed_sq <= 1e-6:
            return close_risk, dist_now

        rel_dir = rel_now / max(dist_now, 1e-3)
        radial_speed = float(np.dot(rel_dir, rel_vel))
        t_unclipped = -float(np.dot(rel_now, rel_vel)) / speed_sq
        t_star = float(np.clip(t_unclipped, 0.0, horizon))
        closest = rel_now + rel_vel * t_star
        d_min = float(np.linalg.norm(closest))

        collision_margin = max(combined_radius * 2.2, combined_radius + 0.35)
        collision_factor = float(np.clip((collision_margin - d_min) / max(collision_margin, 1e-3), 0.0, 1.0))
        if collision_factor <= 1e-6 and (t_unclipped < 0.0 or radial_speed >= 0.0):
            return close_risk, d_min

        distance_factor = float(np.clip(1.0 - dist_now / max(self.config.neighbor_radius, 1e-3), 0.0, 1.0))
        distance_factor = 0.35 + 0.65 * distance_factor
        time_factor = float(np.clip(1.0 - t_star / horizon, 0.0, 1.0) ** 0.7)
        closing_speed = max(0.0, -radial_speed)
        closing_factor = float(np.clip(closing_speed / max(self.config.max_speed, 1e-3), 0.25, 1.0))
        risk = distance_factor * closing_factor * (0.75 * collision_factor + 0.25 * time_factor)
        return float(np.clip(max(risk, close_risk), 0.0, 1.0)), d_min

    def _suggest_pass_side(
        self,
        agent: AgentKinematics,
    ) -> int:
        if int(agent.pass_side_pref) in (-1, 1):
            return int(agent.pass_side_pref)
        return 1

    def _compute_tangential_force(
        self,
        agent: AgentKinematics,
        obstacle_to_agent_dir: np.ndarray,
        base: float,
        risk: float,
        pass_side: int,
        tangential_weight: float,
        bias_weight: float = 1.0,
    ) -> np.ndarray:
        if risk <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        facing = agent.reference_forward_xy
        facing_norm = float(np.linalg.norm(facing))
        if facing_norm <= 1e-6:
            facing = agent.velocity_xy
            facing_norm = float(np.linalg.norm(facing))
        if facing_norm <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        facing = facing / facing_norm
        right = np.array([facing[1], -facing[0]], dtype=np.float32)
        tangent_dir = right if pass_side >= 0 else -right
        agent_to_obstacle_dir = -obstacle_to_agent_dir
        ahead = max(0.0, float(np.dot(facing, agent_to_obstacle_dir)))
        strength = float(tangential_weight) * float(bias_weight) * base * risk * (0.25 + 0.75 * ahead)
        return (tangent_dir * strength).astype(np.float32)

    def _anisotropy_weight(
        self,
        agent: AgentKinematics,
        obstacle_to_agent_dir: np.ndarray,
        desired_velocity: np.ndarray,
    ) -> float:
        facing = desired_velocity
        facing_norm = float(np.linalg.norm(facing))
        if facing_norm <= 1e-6:
            facing = agent.velocity_xy
            facing_norm = float(np.linalg.norm(facing))
        if facing_norm <= 1e-6:
            return 1.0
        facing = facing / facing_norm
        agent_to_obstacle_dir = -obstacle_to_agent_dir
        cos_phi = float(np.clip(np.dot(facing, agent_to_obstacle_dir), -1.0, 1.0))
        lam = float(np.clip(self.config.anisotropy_lambda, 0.0, 1.0))
        return lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)

    def _compute_wall_force(
        self,
        agent: AgentKinematics,
        wall: WallQueryResult,
    ) -> np.ndarray:
        if not wall.valid:
            return np.zeros(2, dtype=np.float32)
        if wall.distance > self.config.wall_influence_dist:
            return np.zeros(2, dtype=np.float32)
        decay = max(float(self.config.wall_B), 1e-3)
        envelope = max(0.0, 1.0 - float(wall.distance) / max(self.config.wall_influence_dist, 1e-3))
        base = float(self.config.wall_A) * np.exp((float(agent.radius) - float(wall.distance)) / decay) * (envelope ** 2)
        return (wall.normal_xy * base).astype(np.float32)
