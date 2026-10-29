from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from baselines.maps.esdf import compute_unsigned_esdf, sample_line_min_clearance
from baselines.maps.map_query import MapQuery


class OccupancyGridMap(MapQuery):
    def __init__(
        self,
        resolution: float,
        width_cells: int,
        height_cells: int,
        origin_xy: Sequence[float],
        robot_xy: Sequence[float],
        robot_yaw: float,
        occupied_grid: np.ndarray | None = None,
        observed_grid: np.ndarray | None = None,
        inflated_grid: np.ndarray | None = None,
        distance_field: np.ndarray | None = None,
        raw_version: int = 0,
        inflation_version: int = 0,
        esdf_version: int = 0,
    ) -> None:
        self.resolution = float(resolution)
        self.width_cells = int(width_cells)
        self.height_cells = int(height_cells)
        self.origin_xy = np.asarray(origin_xy, dtype=float)
        self.robot_xy = np.asarray(robot_xy, dtype=float)
        self.robot_yaw = float(robot_yaw)
        self.raw_version = int(raw_version)
        self.inflation_version = int(inflation_version)
        self.esdf_version = int(esdf_version)
        self._occupancy_rgba_cache: np.ndarray | None = None
        self._esdf_rgba_cache: np.ndarray | None = None
        self._hybrid_rgba_cache: np.ndarray | None = None
        self._unknown_free_distance_field_cache: np.ndarray | None = None
        self.occupied_grid = (
            np.asarray(occupied_grid, dtype=np.uint8)
            if occupied_grid is not None
            else np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        )
        self.observed_grid = (
            np.asarray(observed_grid, dtype=np.uint8)
            if observed_grid is not None
            else np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        )
        self.inflated_grid = (
            np.asarray(inflated_grid, dtype=np.uint8)
            if inflated_grid is not None
            else self.occupied_grid.copy()
        )
        self.distance_field = (
            np.asarray(distance_field, dtype=np.float32)
            if distance_field is not None
            else compute_unsigned_esdf(self.inflated_grid, self.observed_grid, self.resolution)
        )

    def distance_field_for_unknown_policy(self, unknown_is_occupied: bool = True) -> np.ndarray:
        """Return an ESDF for conservative or optimistic unknown-space planning.

        The stored ``distance_field`` is conservative and only gives clearance
        inside observed free cells. Some local planners intentionally treat
        unknown cells as free and only avoid known occupied/inflated cells; those
        planners should request ``unknown_is_occupied=False`` explicitly.
        """
        if bool(unknown_is_occupied):
            return self.distance_field
        if self._unknown_free_distance_field_cache is None:
            self._unknown_free_distance_field_cache = compute_unsigned_esdf(
                self.inflated_grid,
                None,
                self.resolution,
            )
        return self._unknown_free_distance_field_cache

    @classmethod
    def empty(
        cls,
        resolution: float,
        width_cells: int,
        height_cells: int,
        origin_xy: Sequence[float],
        robot_xy: Sequence[float],
        robot_yaw: float,
    ) -> "OccupancyGridMap":
        return cls(
            resolution=resolution,
            width_cells=width_cells,
            height_cells=height_cells,
            origin_xy=origin_xy,
            robot_xy=robot_xy,
            robot_yaw=robot_yaw,
        )

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(math.floor((float(x) - float(self.origin_xy[0])) / self.resolution))
        gy = int(math.floor((float(y) - float(self.origin_xy[1])) / self.resolution))
        return gx, gy

    def world_to_grid_points(self, points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_xy, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
        gx = np.floor((pts[:, 0] - float(self.origin_xy[0])) / self.resolution).astype(np.int32)
        gy = np.floor((pts[:, 1] - float(self.origin_xy[1])) / self.resolution).astype(np.int32)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = float(self.origin_xy[0] + (float(gx) + 0.5) * self.resolution)
        y = float(self.origin_xy[1] + (float(gy) + 0.5) * self.resolution)
        return x, y

    def grid_to_world_points(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        gx_arr = np.asarray(gx, dtype=float)
        gy_arr = np.asarray(gy, dtype=float)
        x = float(self.origin_xy[0]) + (gx_arr + 0.5) * self.resolution
        y = float(self.origin_xy[1]) + (gy_arr + 0.5) * self.resolution
        return np.column_stack((x, y))

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width_cells and 0 <= gy < self.height_cells

    def in_bounds_points(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        gx_arr = np.asarray(gx, dtype=np.int32)
        gy_arr = np.asarray(gy, dtype=np.int32)
        return (gx_arr >= 0) & (gy_arr >= 0) & (gx_arr < self.width_cells) & (gy_arr < self.height_cells)

    def is_free(self, x: float, y: float) -> bool:
        gx, gy = self.world_to_grid(x, y)
        if not self.in_bounds(gx, gy):
            return False
        return bool(self.observed_grid[gy, gx] != 0 and self.inflated_grid[gy, gx] == 0)

    def is_observed(self, x: float, y: float) -> bool:
        gx, gy = self.world_to_grid(x, y)
        if not self.in_bounds(gx, gy):
            return False
        return bool(self.observed_grid[gy, gx] != 0)

    def is_unknown(self, x: float, y: float) -> bool:
        gx, gy = self.world_to_grid(x, y)
        if not self.in_bounds(gx, gy):
            return True
        return bool(self.observed_grid[gy, gx] == 0)

    def distance_to_obstacle(self, x: float, y: float, unknown_is_occupied: bool = True) -> float:
        gx, gy = self.world_to_grid(x, y)
        if not self.in_bounds(gx, gy):
            return 0.0
        field = self.distance_field_for_unknown_policy(unknown_is_occupied)
        return float(field[gy, gx])

    def is_free_points(self, points_xy: np.ndarray) -> np.ndarray:
        # Batch implementation for candidate point filtering.
        gx, gy = self.world_to_grid_points(points_xy)
        mask = self.in_bounds_points(gx, gy)
        out = np.zeros((len(gx),), dtype=bool)
        if np.any(mask):
            out[mask] = (self.observed_grid[gy[mask], gx[mask]] != 0) & (self.inflated_grid[gy[mask], gx[mask]] == 0)
        return out

    def is_observed_points(self, points_xy: np.ndarray) -> np.ndarray:
        # Batch implementation for observed/unknown checks.
        gx, gy = self.world_to_grid_points(points_xy)
        mask = self.in_bounds_points(gx, gy)
        out = np.zeros((len(gx),), dtype=bool)
        if np.any(mask):
            out[mask] = self.observed_grid[gy[mask], gx[mask]] != 0
        return out

    def is_unknown_points(self, points_xy: np.ndarray) -> np.ndarray:
        return ~self.is_observed_points(points_xy)

    def distance_to_obstacle_points(self, points_xy: np.ndarray, unknown_is_occupied: bool = True) -> np.ndarray:
        # Batch ESDF lookup for sample arrays.
        gx, gy = self.world_to_grid_points(points_xy)
        mask = self.in_bounds_points(gx, gy)
        out = np.zeros((len(gx),), dtype=np.float32)
        if np.any(mask):
            field = self.distance_field_for_unknown_policy(unknown_is_occupied)
            out[mask] = field[gy[mask], gx[mask]]
        return out

    def ray_clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)
        for gx, gy in _bresenham(gx0, gy0, gx1, gy1):
            if not self.in_bounds(gx, gy):
                return False
            if self.inflated_grid[gy, gx] != 0:
                return False
        return True

    def line_min_clearance(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        unknown_is_occupied: bool = True,
    ) -> float:
        return sample_line_min_clearance(
            distance_field=self.distance_field_for_unknown_policy(unknown_is_occupied),
            origin_xy=self.origin_xy,
            resolution=self.resolution,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        )

    def export_debug_cells(self) -> List[List[float]]:
        ys, xs = np.where(self.occupied_grid != 0)
        if len(xs) == 0:
            return []
        return self.grid_to_world_points(xs, ys).tolist()

    def export_debug_observed_free_cells(self, stride: int = 2) -> List[List[float]]:
        free_mask = (self.observed_grid != 0) & (self.inflated_grid == 0)
        ys, xs = np.where(free_mask)
        if stride > 1 and len(xs) > 0:
            ys = ys[::stride]
            xs = xs[::stride]
        if len(xs) == 0:
            return []
        return self.grid_to_world_points(xs, ys).tolist()

    def export_debug_outline(self) -> List[List[float]]:
        x0 = float(self.origin_xy[0])
        y0 = float(self.origin_xy[1])
        x1 = float(self.origin_xy[0] + self.width_cells * self.resolution)
        y1 = float(self.origin_xy[1] + self.height_cells * self.resolution)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]

    def export_debug_extent(self) -> List[float]:
        x0 = float(self.origin_xy[0])
        y0 = float(self.origin_xy[1])
        x1 = float(self.origin_xy[0] + self.width_cells * self.resolution)
        y1 = float(self.origin_xy[1] + self.height_cells * self.resolution)
        return [x0, x1, y0, y1]

    def export_occupancy_rgba(self) -> np.ndarray:
        if self._occupancy_rgba_cache is not None:
            return self._occupancy_rgba_cache
        rgba = np.zeros((self.height_cells, self.width_cells, 4), dtype=np.uint8)

        unknown_mask = self.observed_grid == 0
        free_mask = (self.observed_grid != 0) & (self.inflated_grid == 0)
        occupied_mask = self.occupied_grid != 0
        inflated_only_mask = (self.inflated_grid != 0) & (~occupied_mask)

        rgba[unknown_mask] = np.array([196, 202, 210, 55], dtype=np.uint8)
        rgba[free_mask] = np.array([94, 184, 230, 125], dtype=np.uint8)
        rgba[inflated_only_mask] = np.array([255, 170, 0, 95], dtype=np.uint8)
        rgba[occupied_mask] = np.array([56, 56, 56, 205], dtype=np.uint8)
        self._occupancy_rgba_cache = rgba
        return rgba

    def export_esdf_rgba(self, max_distance: float | None = None) -> np.ndarray:
        if max_distance is None and self._esdf_rgba_cache is not None:
            return self._esdf_rgba_cache
        rgba = np.zeros((self.height_cells, self.width_cells, 4), dtype=np.uint8)
        observed_free = (self.observed_grid != 0) & (self.inflated_grid == 0)
        if max_distance is None:
            max_distance = max(float(np.max(self.distance_field)), self.resolution)
        clipped = np.clip(self.distance_field, 0.0, float(max_distance))
        scaled = np.round((clipped / max(float(max_distance), 1e-6)) * 255.0).astype(np.uint8)
        bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)

        rgba[:, :, :3] = bgr[:, :, ::-1]
        rgba[:, :, 3] = 0
        rgba[observed_free, 3] = 185
        rgba[self.observed_grid == 0] = np.array([196, 202, 210, 40], dtype=np.uint8)
        rgba[(self.inflated_grid != 0) & (self.occupied_grid == 0)] = np.array([255, 170, 0, 85], dtype=np.uint8)
        rgba[self.occupied_grid != 0] = np.array([56, 56, 56, 205], dtype=np.uint8)
        if max_distance is None:
            self._esdf_rgba_cache = rgba
        return rgba

    def export_hybrid_rgba(
        self,
        max_distance: float | None = None,
        esdf_alpha: int = 135,
    ) -> np.ndarray:
        if max_distance is None and self._hybrid_rgba_cache is not None:
            return self._hybrid_rgba_cache
        occupancy_rgba = self.export_occupancy_rgba().copy()
        observed_free = (self.observed_grid != 0) & (self.inflated_grid == 0)

        if np.any(observed_free):
            if max_distance is None:
                max_distance = max(float(np.max(self.distance_field[observed_free])), self.resolution)
            clipped = np.clip(self.distance_field, 0.0, float(max_distance))
            scaled = np.round((clipped / max(float(max_distance), 1e-6)) * 255.0).astype(np.uint8)
            esdf_rgb = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[:, :, ::-1]
            alpha = np.float32(np.clip(esdf_alpha, 0, 255)) / 255.0
            blended = np.round(
                occupancy_rgba[:, :, :3].astype(np.float32) * (1.0 - alpha)
                + esdf_rgb.astype(np.float32) * alpha
            ).astype(np.uint8)
            occupancy_rgba[observed_free, :3] = blended[observed_free]
            occupancy_rgba[observed_free, 3] = np.maximum(occupancy_rgba[observed_free, 3], np.uint8(135))

        if max_distance is None:
            self._hybrid_rgba_cache = occupancy_rgba
        return occupancy_rgba


def inflate_grid(occupied_grid: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    radius_cells = int(math.ceil(float(radius_m) / float(resolution)))
    if radius_cells <= 0:
        return np.asarray(occupied_grid, dtype=np.uint8)
    kernel = _disk_kernel(radius_cells)
    inflated = cv2.dilate(np.asarray(occupied_grid, dtype=np.uint8), kernel, iterations=1)
    return (inflated > 0).astype(np.uint8)


def with_disc_overlays(
    grid: OccupancyGridMap,
    clear_discs_world: Iterable[Tuple[Sequence[float], float]] | None = None,
    occupied_discs_world: Iterable[Tuple[Sequence[float], float]] | None = None,
    inflation_radius_m: float = 0.0,
) -> OccupancyGridMap:
    occupied = np.asarray(grid.occupied_grid, dtype=np.uint8).copy()
    observed = np.asarray(grid.observed_grid, dtype=np.uint8).copy()

    for center_world_xy, radius in clear_discs_world or []:
        _apply_world_disc(occupied, observed, grid, center_world_xy, radius, occupied_value=0, observed_value=1)

    for center_world_xy, radius in occupied_discs_world or []:
        _apply_world_disc(occupied, observed, grid, center_world_xy, radius, occupied_value=1, observed_value=1)

    inflated = inflate_grid(occupied, inflation_radius_m, grid.resolution)
    return OccupancyGridMap(
        resolution=grid.resolution,
        width_cells=grid.width_cells,
        height_cells=grid.height_cells,
        origin_xy=grid.origin_xy,
        robot_xy=grid.robot_xy,
        robot_yaw=grid.robot_yaw,
        occupied_grid=occupied,
        observed_grid=observed,
        inflated_grid=inflated,
        raw_version=grid.raw_version + 1,
        inflation_version=grid.inflation_version + 1,
        esdf_version=grid.esdf_version + 1,
    )


def _apply_world_disc(
    occupied: np.ndarray,
    observed: np.ndarray,
    grid: OccupancyGridMap,
    center_world_xy: Sequence[float],
    radius_m: float,
    occupied_value: int,
    observed_value: int,
) -> None:
    center = np.asarray(center_world_xy, dtype=float).reshape(-1)
    if center.size < 2:
        return
    radius = max(float(radius_m), 0.0)
    if radius <= 0.0:
        return

    gx, gy = grid.world_to_grid(float(center[0]), float(center[1]))
    radius_cells = max(int(math.ceil(radius / max(grid.resolution, 1e-6))), 1)
    x0 = max(gx - radius_cells, 0)
    x1 = min(gx + radius_cells + 1, grid.width_cells)
    y0 = max(gy - radius_cells, 0)
    y1 = min(gy + radius_cells + 1, grid.height_cells)
    if x0 >= x1 or y0 >= y1:
        return

    yy = np.arange(y0, y1)[:, None]
    xx = np.arange(x0, x1)[None, :]
    mask = ((xx - gx) ** 2 + (yy - gy) ** 2) * (grid.resolution ** 2) <= radius ** 2
    occupied[y0:y1, x0:x1][mask] = np.uint8(occupied_value)
    observed[y0:y1, x0:x1][mask] = np.uint8(observed_value)


def grid_from_points(
    points_world_xy: Iterable[Sequence[float]],
    resolution: float,
    width_cells: int,
    height_cells: int,
    origin_xy: Sequence[float],
) -> np.ndarray:
    grid = np.zeros((int(height_cells), int(width_cells)), dtype=np.uint8)
    pts = np.asarray(list(points_world_xy) if not isinstance(points_world_xy, np.ndarray) else points_world_xy, dtype=float)
    if pts.size == 0:
        return grid
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    res = float(resolution)
    gx = np.floor((pts[:, 0] - ox) / res).astype(np.int32)
    gy = np.floor((pts[:, 1] - oy) / res).astype(np.int32)
    mask = (gx >= 0) & (gy >= 0) & (gx < int(width_cells)) & (gy < int(height_cells))
    if np.any(mask):
        grid[gy[mask], gx[mask]] = 1
    return grid


def raycast_observed_grid(
    sensor_world_xy: Sequence[float],
    hit_points_world_xy: Iterable[Sequence[float]],
    resolution: float,
    width_cells: int,
    height_cells: int,
    origin_xy: Sequence[float],
) -> np.ndarray:
    observed = np.zeros((int(height_cells), int(width_cells)), dtype=np.uint8)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    res = float(resolution)
    sx = int(math.floor((float(sensor_world_xy[0]) - ox) / res))
    sy = int(math.floor((float(sensor_world_xy[1]) - oy) / res))
    if not (0 <= sx < int(width_cells) and 0 <= sy < int(height_cells)):
        return observed
    pts = np.asarray(list(hit_points_world_xy) if not isinstance(hit_points_world_xy, np.ndarray) else hit_points_world_xy, dtype=float)
    if pts.size == 0:
        return observed
    gx = np.floor((pts[:, 0] - ox) / res).astype(np.int32)
    gy = np.floor((pts[:, 1] - oy) / res).astype(np.int32)
    mask = (gx >= 0) & (gy >= 0) & (gx < int(width_cells)) & (gy < int(height_cells))
    if not np.any(mask):
        return observed
    endpoints = np.unique(np.column_stack((gx[mask], gy[mask])), axis=0)
    for ex, ey in endpoints:
        cv2.line(observed, (sx, sy), (int(ex), int(ey)), 1, thickness=1)
    return observed


@lru_cache(maxsize=32)
def _disk_kernel(radius_cells: int) -> np.ndarray:
    size = radius_cells * 2 + 1
    yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    mask = (xx * xx + yy * yy) <= radius_cells * radius_cells
    return mask.astype(np.uint8)


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
