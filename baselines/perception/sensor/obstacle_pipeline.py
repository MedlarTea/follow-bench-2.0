from __future__ import annotations

from typing import Optional

from baselines.perception.gt.npc_obstacles import NPC_OBS_HALF_SIZE_M, NPC_OBS_RANGE_M, npc_box_obstacles
from baselines.perception.sensor.lidar_obstacles import (
    LIDAR_RANGE_MAX_M,
    LIDAR_RANGE_MIN_M,
    TARGET_MASK_RADIUS_M,
    filter_lidar_2d,
    lidar_obstacles_world,
    lidar_to_world,
)


def build_obstacle_inputs(
    obs,
    lidar_range_max: float = LIDAR_RANGE_MAX_M,
    target_mask_radius: float = TARGET_MASK_RADIUS_M,
    use_npc_gt: bool = True,
    npc_range_m: float = NPC_OBS_RANGE_M,
    npc_half_size: float = NPC_OBS_HALF_SIZE_M,
) -> list[dict]:
    rx = float(obs.robot.x)
    ry = float(obs.robot.y)
    robot_yaw = float(obs.robot.yaw_rad)

    out: list[dict] = []
    if obs.lidar_points is not None and obs.lidar_points.shape[0] > 0:
        xy_lidar = filter_lidar_2d(obs.lidar_points, lidar_range_max, LIDAR_RANGE_MIN_M)
        xy_world = lidar_to_world(xy_lidar, obs.lidar_extrinsics_robot_to_sensor, rx, ry, robot_yaw)
        target_xy = _target_xy(obs)
        out.extend(lidar_obstacles_world(xy_world, target_xy, target_mask_radius))

    if use_npc_gt and obs.npcs:
        target_track_id = getattr(obs.target, "track_id", None) if obs.target is not None else None
        out.extend(
            npc_box_obstacles(
                obs.npcs,
                target_track_id=target_track_id,
                robot_xy=(rx, ry),
                range_m=npc_range_m,
                half_size=npc_half_size,
            )
        )

    return out


def _target_xy(obs) -> Optional[tuple[float, float]]:
    if obs.target is None:
        return None
    return float(obs.target.x), float(obs.target.y)


__all__ = [
    "LIDAR_RANGE_MAX_M",
    "LIDAR_RANGE_MIN_M",
    "NPC_OBS_HALF_SIZE_M",
    "NPC_OBS_RANGE_M",
    "TARGET_MASK_RADIUS_M",
    "build_obstacle_inputs",
]
