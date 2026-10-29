"""Occluder-aware overtake goal selection for temporary target loss."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from baselines.common.geometry import inverse_distance_costs, polygon_iou, tangent_triangle, yaw_between
from baselines.behavior.search.config import SearchConfig
from baselines.behavior.search.mode_manager import SearchMode
from baselines.behavior.search.occlusion import find_occluder
from baselines.behavior.search.sampler import generate_korobov_semicircle_samples
from baselines.maps.map_query import MapQuery
from baselines.common.types import AgentState2D, RobotState2D, SearchResult


def select_overtake_goal(
    robot: RobotState2D,
    search_anchor_xy: Sequence[float],
    agents: List[AgentState2D],
    predicted_agents: Dict[str, np.ndarray],
    map_query: MapQuery,
    target_radius: float,
    config: SearchConfig,
) -> Optional[SearchResult]:
    robot_xy = robot.xy
    anchor_xy = np.asarray(search_anchor_xy, dtype=float)
    occluder_id = find_occluder(robot_xy, anchor_xy, agents)
    if occluder_id is None:
        return None

    occluder = next((agent for agent in agents if agent.track_id == occluder_id), None)
    if occluder is None:
        return None

    predicted = predicted_agents.get(occluder_id)
    if predicted is None or len(predicted) == 0:
        predicted_xy = occluder.xy.reshape(1, 2)
    else:
        predicted_xy = np.asarray(predicted[:, :2], dtype=float)

    samples = generate_korobov_semicircle_samples(
        robot_xy,
        occluder.xy,
        config.overtake_inner_radius,
        config.overtake_outer_radius,
        config.overtake_sample_count,
    )

    best_pose = None
    best_cost = float("inf")
    free_mask = map_query.is_free_points(samples)
    clearance_all = map_query.distance_to_obstacle_points(samples)
    cost_to_occluder_all = inverse_distance_costs(samples, predicted_xy, config.occluder_max_distance)
    candidate_idx = np.flatnonzero(free_mask & (clearance_all >= config.min_obstacle_clearance))

    for sample_idx in candidate_idx:
        sample = samples[sample_idx]
        if not map_query.ray_clear(float(robot_xy[0]), float(robot_xy[1]), float(sample[0]), float(sample[1])):
            continue
        if not map_query.ray_clear(float(sample[0]), float(sample[1]), float(anchor_xy[0]), float(anchor_xy[1])):
            continue
        clearance = float(clearance_all[sample_idx])
        cost_to_occluder = float(cost_to_occluder_all[sample_idx])
        triangle_to_target = tangent_triangle(sample, anchor_xy, target_radius)
        triangle_to_occluder = tangent_triangle(sample, predicted_xy[0], target_radius)
        if triangle_to_target is None or triangle_to_occluder is None:
            continue

        cost_iou = polygon_iou(triangle_to_target, triangle_to_occluder)
        if cost_iou > config.iou_threshold:
            continue

        cost_attractive = float(np.linalg.norm(sample - robot_xy) / max(1.0 - cost_iou, 1e-3))
        obstacle_penalty = config.obstacle_penalty_weight / max(clearance, 1e-3)
        total_cost = cost_to_occluder + cost_attractive + obstacle_penalty
        if total_cost > config.overtake_cost_threshold:
            continue

        if total_cost < best_cost:
            yaw = yaw_between(sample, anchor_xy)
            best_pose = np.array([[sample[0]], [sample[1]], [yaw]], dtype=float)
            best_cost = total_cost

    if best_pose is None:
        return SearchResult(
            goal_pose=np.empty((0, 1), dtype=float),
            mode=SearchMode.OVERTAKE_FAILED,
            detail_mode=SearchMode.OVERTAKE_FAILED,
            samples=samples,
            occluder_id=occluder_id,
        )

    return SearchResult(
        goal_pose=best_pose,
        mode=SearchMode.OVERTAKE,
        detail_mode=SearchMode.OVERTAKE,
        samples=samples,
        occluder_id=occluder_id,
        cost=best_cost,
    )
