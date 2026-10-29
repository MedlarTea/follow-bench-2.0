from __future__ import annotations

import math

import numpy as np

from baselines.maps.occupancy_grid import OccupancyGridMap, grid_from_points, inflate_grid, raycast_observed_grid
from baselines.perception.sensor.lidar_obstacles import filter_lidar_2d, lidar_to_world


def build_local_occupancy_grid(
    lidar_points: np.ndarray | None,
    lidar_extrinsics: dict | None,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    resolution: float,
    width_cells: int | None,
    height_cells: int | None,
    inflation_radius_m: float,
    lidar_range_max: float,
    lidar_range_min: float = 0.2,
) -> OccupancyGridMap:
    width_cells, height_cells = _resolve_grid_dims(
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        lidar_range_max=lidar_range_max,
        inflation_radius_m=inflation_radius_m,
    )
    origin_xy = _grid_origin(robot_x, robot_y, resolution, width_cells, height_cells)
    if lidar_points is None or len(lidar_points) == 0:
        return OccupancyGridMap.empty(
            resolution=resolution,
            width_cells=width_cells,
            height_cells=height_cells,
            origin_xy=origin_xy,
            robot_xy=(robot_x, robot_y),
            robot_yaw=robot_yaw,
        )

    xy_lidar = filter_lidar_2d(lidar_points, lidar_range_max, lidar_range_min)
    xy_world = lidar_to_world(xy_lidar, lidar_extrinsics, robot_x, robot_y, robot_yaw)
    sensor_world_xy = _sensor_world_xy(lidar_extrinsics, robot_x, robot_y, robot_yaw)
    occupied = grid_from_points(
        points_world_xy=xy_world,
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        origin_xy=origin_xy,
    )
    observed = raycast_observed_grid(
        sensor_world_xy=sensor_world_xy,
        hit_points_world_xy=xy_world,
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        origin_xy=origin_xy,
    )
    inflated = inflate_grid(occupied, inflation_radius_m, resolution)
    return OccupancyGridMap(
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        origin_xy=origin_xy,
        robot_xy=(robot_x, robot_y),
        robot_yaw=robot_yaw,
        occupied_grid=occupied,
        observed_grid=observed,
        inflated_grid=inflated,
        raw_version=1,
        inflation_version=1,
        esdf_version=1,
    )


def _grid_origin(
    robot_x: float,
    robot_y: float,
    resolution: float,
    width_cells: int,
    height_cells: int,
):
    half_w = 0.5 * float(width_cells) * float(resolution)
    half_h = 0.5 * float(height_cells) * float(resolution)
    return np.array([float(robot_x) - half_w, float(robot_y) - half_h], dtype=float)


def _resolve_grid_dims(
    resolution: float,
    width_cells: int | None,
    height_cells: int | None,
    lidar_range_max: float,
    inflation_radius_m: float,
):
    margin_m = max(0.5, inflation_radius_m + resolution)
    span_m = 2.0 * (float(lidar_range_max) + margin_m)
    auto_cells = int(math.ceil(span_m / float(resolution)))
    auto_cells = max(auto_cells, 5)
    if width_cells is None or int(width_cells) <= 0:
        width_cells = auto_cells
    if height_cells is None or int(height_cells) <= 0:
        height_cells = auto_cells
    return int(width_cells), int(height_cells)


def _sensor_world_xy(
    lidar_extrinsics: dict | None,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
) -> np.ndarray:
    tx = float(lidar_extrinsics.get("x", 0.0)) if lidar_extrinsics else 0.0
    ty = float(lidar_extrinsics.get("y", 0.0)) if lidar_extrinsics else 0.0
    yaw_l = math.radians(float(lidar_extrinsics.get("yaw_deg", 0.0))) if lidar_extrinsics else 0.0
    cl = math.cos(yaw_l)
    sl = math.sin(yaw_l)
    bx = cl * 0.0 - sl * 0.0 + tx
    by = sl * 0.0 + cl * 0.0 + ty
    cy = math.cos(robot_yaw)
    sy = math.sin(robot_yaw)
    wx = robot_x + cy * bx - sy * by
    wy = robot_y + sy * bx + cy * by
    return np.array([wx, wy], dtype=float)
