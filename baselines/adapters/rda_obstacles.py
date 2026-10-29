"""RDA obstacle assembly for platform adapters."""

from __future__ import annotations

import numpy as np

from baselines.planning.rda.rda_obstacles import (
    Obstacle,
    ROBOT_RADIUS,
    obstacles_debug,
    to_rda_obstacle,
)
from baselines.perception.gt.npc_obstacles import NPC_OBS_HALF_SIZE_M, NPC_OBS_RANGE_M, npc_box_obstacles
from baselines.perception.sensor.obstacle_pipeline import build_obstacle_inputs
from baselines.perception.sensor.lidar_obstacles import (
    LIDAR_RANGE_MAX_M,
    LIDAR_RANGE_MIN_M,
    TARGET_MASK_RADIUS_M,
    filter_lidar_2d,
    lidar_obstacles_world,
    lidar_to_world,
)


def build_rda_obstacles(
    obs,
    lidar_range_max: float = LIDAR_RANGE_MAX_M,
    target_mask_radius: float = TARGET_MASK_RADIUS_M,
    use_npc_gt: bool = True,
    npc_range_m: float = NPC_OBS_RANGE_M,
    npc_half_size: float = NPC_OBS_HALF_SIZE_M,
) -> list[Obstacle]:
    return [
        to_rda_obstacle(item)
        for item in build_obstacle_inputs(
            obs,
            lidar_range_max=lidar_range_max,
            target_mask_radius=target_mask_radius,
            use_npc_gt=use_npc_gt,
            npc_range_m=npc_range_m,
            npc_half_size=npc_half_size,
        )
    ]


def target_box_obstacle(target, half_size: float = NPC_OBS_HALF_SIZE_M) -> Obstacle:
    """Represent the followed target with the same footprint as other NPCs."""
    cx = float(target.x)
    cy = float(target.y)
    h = float(half_size)
    vertex = np.array(
        [
            [cx - h, cy - h],
            [cx + h, cy - h],
            [cx + h, cy + h],
            [cx - h, cy + h],
        ],
        dtype=np.float32,
    ).T
    center = np.array([[cx], [cy]], dtype=np.float32)
    velocity = np.array([[float(target.vx)], [float(target.vy)]], dtype=np.float32)
    return Obstacle(
        center=center,
        radius=None,
        vertex=vertex,
        cone_type="Rpositive",
        velocity=velocity,
    )


__all__ = [
    "NPC_OBS_HALF_SIZE_M",
    "NPC_OBS_RANGE_M",
    "LIDAR_RANGE_MAX_M",
    "LIDAR_RANGE_MIN_M",
    "Obstacle",
    "ROBOT_RADIUS",
    "TARGET_MASK_RADIUS_M",
    "build_rda_obstacles",
    "filter_lidar_2d",
    "lidar_obstacles_world",
    "lidar_to_world",
    "npc_box_obstacles",
    "obstacles_debug",
    "target_box_obstacle",
    "to_rda_obstacle",
]
