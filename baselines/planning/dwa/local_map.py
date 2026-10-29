from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from baselines.maps.lidar_grid_builder import build_local_occupancy_grid
from baselines.maps.occupancy_grid import OccupancyGridMap, inflate_grid


def build_dwa_local_map(
    *,
    lidar_points: np.ndarray | None,
    lidar_extrinsics: dict | None,
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    target_xy: np.ndarray,
    target_radius: float,
    robot_radius: float,
    resolution: float,
    width_cells: int | None,
    height_cells: int | None,
    lidar_range_max: float,
    target_clear_margin: float = 0.15,
) -> tuple[OccupancyGridMap, dict[str, Any]]:
    raw_map = build_local_occupancy_grid(
        lidar_points=lidar_points,
        lidar_extrinsics=lidar_extrinsics,
        robot_x=robot_x,
        robot_y=robot_y,
        robot_yaw=robot_yaw,
        resolution=resolution,
        width_cells=width_cells,
        height_cells=height_cells,
        inflation_radius_m=0.0,
        lidar_range_max=lidar_range_max,
    )
    occupied = np.asarray(raw_map.occupied_grid, dtype=np.uint8).copy()
    observed = np.asarray(raw_map.observed_grid, dtype=np.uint8).copy()
    clear_radius = max(float(target_radius), 0.0) + max(float(target_clear_margin), 0.0)
    cleared, clear_cells = _clear_target_connected_component(
        occupancy=occupied,
        grid=raw_map,
        target_xy=np.asarray(target_xy, dtype=float).reshape(2),
        clear_radius=clear_radius,
    )
    inflated = inflate_grid(occupied, robot_radius, raw_map.resolution)
    map_query = OccupancyGridMap(
        resolution=raw_map.resolution,
        width_cells=raw_map.width_cells,
        height_cells=raw_map.height_cells,
        origin_xy=raw_map.origin_xy,
        robot_xy=raw_map.robot_xy,
        robot_yaw=raw_map.robot_yaw,
        occupied_grid=occupied,
        observed_grid=observed,
        inflated_grid=inflated,
        raw_version=raw_map.raw_version + int(cleared),
        inflation_version=raw_map.inflation_version + 1,
        esdf_version=raw_map.esdf_version + 1,
    )
    debug = {
        "target_blob_cleared": bool(cleared),
        "target_blob_clear_cells": int(clear_cells),
        "target_clear_radius": float(clear_radius),
        "local_map_resolution": float(raw_map.resolution),
    }
    return map_query, debug


def _clear_target_connected_component(
    *,
    occupancy: np.ndarray,
    grid: OccupancyGridMap,
    target_xy: np.ndarray,
    clear_radius: float,
) -> tuple[bool, int]:
    if clear_radius <= 0.0 or occupancy.size == 0:
        return False, 0
    gx, gy = grid.world_to_grid(float(target_xy[0]), float(target_xy[1]))
    if not grid.in_bounds(gx, gy):
        return False, 0

    margin_cells = max(int(math.ceil(clear_radius / max(grid.resolution, 1e-6))), 1)
    if gx < margin_cells or gy < margin_cells or gx >= grid.width_cells - margin_cells or gy >= grid.height_cells - margin_cells:
        return False, 0

    seed = _find_blob_seed(occupancy, gx, gy, clear_radius, grid.resolution)
    if seed is None:
        return False, 0

    max_blob_radius = max(float(clear_radius) * 1.75, float(clear_radius) + 2.0 * float(grid.resolution))
    radius_cells = max(int(math.ceil(max_blob_radius / max(grid.resolution, 1e-6))), 1)
    max_cells = max(24, int(math.ceil(math.pi * radius_cells * radius_cells * 1.5)))
    component_mask = _extract_connected_component(occupancy, seed, radius_cells, max_cells)
    if component_mask is None or not np.any(component_mask):
        return False, 0

    count = int(np.count_nonzero(component_mask))
    occupancy[component_mask] = 0
    return True, count


def _find_blob_seed(
    occupancy: np.ndarray,
    gx: int,
    gy: int,
    search_radius: float,
    resolution: float,
) -> tuple[int, int] | None:
    radius_cells = max(int(math.ceil(max(search_radius, 0.0) / max(resolution, 1e-6))), 1)
    x0 = max(gx - radius_cells, 0)
    x1 = min(gx + radius_cells + 1, occupancy.shape[1])
    y0 = max(gy - radius_cells, 0)
    y1 = min(gy + radius_cells + 1, occupancy.shape[0])
    if x0 >= x1 or y0 >= y1:
        return None
    window = occupancy[y0:y1, x0:x1]
    rows, cols = np.nonzero(window)
    if len(rows) == 0:
        return None
    rows = rows + y0
    cols = cols + x0
    dist_sq = (rows - gy) ** 2 + (cols - gx) ** 2
    best_idx = int(np.argmin(dist_sq))
    return int(rows[best_idx]), int(cols[best_idx])


def _extract_connected_component(
    occupancy: np.ndarray,
    seed: tuple[int, int],
    max_radius_cells: int,
    max_cells: int,
) -> np.ndarray | None:
    seed_row, seed_col = int(seed[0]), int(seed[1])
    if not occupancy[seed_row, seed_col]:
        return None

    visited = np.zeros_like(occupancy, dtype=bool)
    component_mask = np.zeros_like(occupancy, dtype=bool)
    queue = deque([(seed_row, seed_col)])
    visited[seed_row, seed_col] = True
    count = 0

    while queue:
        row, col = queue.popleft()
        if (row - seed_row) ** 2 + (col - seed_col) ** 2 > max_radius_cells ** 2:
            continue
        if not occupancy[row, col]:
            continue

        component_mask[row, col] = True
        count += 1
        if count > max_cells:
            return None

        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                nxt_row = row + d_row
                nxt_col = col + d_col
                if 0 <= nxt_row < occupancy.shape[0] and 0 <= nxt_col < occupancy.shape[1] and not visited[nxt_row, nxt_col]:
                    visited[nxt_row, nxt_col] = True
                    queue.append((nxt_row, nxt_col))

    return component_mask if count > 0 else None
