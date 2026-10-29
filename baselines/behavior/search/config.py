"""Search hyperparameters shared by the reusable target-search modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchConfig:
    gamma: float = 0.1
    sigma: float = 1.0
    overtake_sample_count: int = 20
    fluid_sample_count: int = 30
    overtake_inner_radius: float = 0.5
    overtake_outer_radius: float = 3.0
    fluid_inner_radius: float = 0.0
    fluid_outer_radius: float = 2.5
    occluder_max_distance: float = 2.0
    iou_threshold: float = 0.3
    overtake_cost_threshold: float = 3.5
    min_objective_density: float = 0.1
    min_velocity_diff: float = 0.1
    min_position_diff: float = 1.5
    min_obstacle_clearance: float = 0.35
    obstacle_penalty_weight: float = 0.6
    anchor_prediction_index: int = 2
    anchor_smoothing_alpha: float = 0.5
    anchor_min_distance: float = 0.75
    anchor_max_distance: float = 3.5
    reacquire_transition_ticks: int = 1
