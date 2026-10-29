"""Density/velocity-field search fallback used when overtake search fails."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from baselines.common.geometry import yaw_between
from baselines.behavior.search.config import SearchConfig
from baselines.behavior.search.mode_manager import SearchMode
from baselines.behavior.search.sampler import generate_korobov_semicircle_samples
from baselines.maps.map_query import MapQuery
from baselines.common.types import AgentState2D, RobotState2D, SearchResult


def select_fluid_goal(
    robot: RobotState2D,
    agents: List[AgentState2D],
    search_direction: Sequence[float],
    map_query: MapQuery,
    config: SearchConfig,
) -> SearchResult:
    robot_xy = robot.xy
    direction = np.asarray(search_direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array([np.cos(robot.yaw), np.sin(robot.yaw)], dtype=float)
    else:
        direction = direction / norm

    samples = generate_korobov_semicircle_samples(
        robot_xy,
        robot_xy + 0.5 * direction,
        config.fluid_inner_radius,
        config.fluid_outer_radius,
        config.fluid_sample_count,
    )
    if len(samples) == 0:
        goal = np.array([[robot.x], [robot.y], [robot.yaw]], dtype=float)
        return SearchResult(goal_pose=goal, mode=SearchMode.FLUID, detail_mode=SearchMode.FLUID, samples=samples)

    objectives = np.full(len(samples), np.inf, dtype=float)
    free_mask = map_query.is_free_points(samples)
    clearance_all = map_query.distance_to_obstacle_points(samples)
    candidate_mask = free_mask & (clearance_all >= config.min_obstacle_clearance)
    if np.any(candidate_mask):
        candidate_idx = np.flatnonzero(candidate_mask)
        ray_mask = np.zeros(len(candidate_idx), dtype=bool)
        for local_idx, sample_idx in enumerate(candidate_idx):
            point = samples[sample_idx]
            ray_mask[local_idx] = map_query.ray_clear(
                float(robot_xy[0]),
                float(robot_xy[1]),
                float(point[0]),
                float(point[1]),
            )
        candidate_idx = candidate_idx[ray_mask]

        if len(candidate_idx) > 0:
            candidate_samples = samples[candidate_idx]
            clearance = clearance_all[candidate_idx]
            density = np.maximum(_density(agents, candidate_samples, config.sigma), config.min_objective_density)
            v_field = _velocity_field(agents, candidate_samples, config.gamma)
            robot_vel = np.array([robot.vx, robot.vy], dtype=float)
            velocity_diff = np.linalg.norm(v_field - robot_vel[None, :], axis=1)
            position_diff = np.linalg.norm(candidate_samples - robot_xy[None, :], axis=1)
            obstacle_penalty = config.obstacle_penalty_weight / np.maximum(clearance, 1e-3)
            objectives[candidate_idx] = (
                density
                * config.gamma
                * np.maximum(velocity_diff, config.min_velocity_diff)
                * np.maximum(position_diff, config.min_position_diff)
            ) + obstacle_penalty

    if not np.any(np.isfinite(objectives)):
        fallback_xy = robot_xy + direction
        yaw = yaw_between(robot_xy, fallback_xy)
        goal = np.array([[fallback_xy[0]], [fallback_xy[1]], [yaw]], dtype=float)
        return SearchResult(
            goal_pose=goal,
            mode=SearchMode.FLUID_FALLBACK,
            detail_mode=SearchMode.FLUID_FALLBACK,
            samples=samples,
        )

    best_idx = int(np.nanargmin(objectives))
    best_xy = samples[best_idx]
    yaw = yaw_between(robot_xy, best_xy)
    goal = np.array([[best_xy[0]], [best_xy[1]], [yaw]], dtype=float)
    return SearchResult(
        goal_pose=goal,
        mode=SearchMode.FLUID,
        detail_mode=SearchMode.FLUID,
        samples=samples,
        cost=float(objectives[best_idx]),
    )


def _velocity_field(agents: List[AgentState2D], points: np.ndarray, gamma: float) -> np.ndarray:
    # Batch implementation over candidate point arrays.
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    if not agents:
        return np.zeros((len(pts), 2), dtype=float)
    agent_xy = np.asarray([agent.xy for agent in agents], dtype=float)
    agent_vel = np.asarray([agent.velocity_xy for agent in agents], dtype=float)
    deltas = pts[:, None, :] - agent_xy[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    alpha = np.exp(-float(gamma) * dist_sq)
    vel_sum = alpha @ agent_vel
    alpha_sum = np.sum(alpha, axis=1, keepdims=True)
    out = np.zeros_like(vel_sum)
    np.divide(vel_sum, alpha_sum, out=out, where=alpha_sum > 0.0)
    return out


def _density(agents: List[AgentState2D], points: np.ndarray, sigma: float) -> np.ndarray:
    # Batch implementation over candidate point arrays.
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=float)
    if not agents:
        return np.zeros((len(pts),), dtype=float)
    agent_xy = np.asarray([agent.xy for agent in agents], dtype=float)
    deltas = pts[:, None, :] - agent_xy[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    sigma_sq = float(sigma) * float(sigma)
    kernel = np.exp(-dist_sq / (2.0 * sigma_sq))
    return np.sum(kernel, axis=1) / (2.0 * np.pi * sigma_sq)
