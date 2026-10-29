from __future__ import annotations

import math
from typing import Any

import numpy as np


class BsoHfcMapView:
    """Robot-local view over Follow-Bench's world-frame map query/ESDF API."""

    def __init__(
        self,
        map_query: Any,
        robot_pose_world: np.ndarray,
        window_size_m: float | None = None,
        unknown_is_occupied: bool = False,
    ) -> None:
        self.map_query = map_query
        self.robot_pose_world = np.asarray(robot_pose_world, dtype=float).reshape(-1)[:3]
        self.unknown_is_occupied = bool(unknown_is_occupied)
        self.resolution = float(getattr(map_query, "resolution", 0.2))
        self.width = int(getattr(map_query, "width_cells", 0))
        self.height = int(getattr(map_query, "height_cells", 0))
        self.distance_field = self._resolve_distance_field()
        if self.width <= 0 or self.height <= 0:
            span = float(window_size_m) if window_size_m is not None else 20.0
            cells = max(int(math.ceil(span / max(self.resolution, 1e-6))), 5)
            self.width = cells
            self.height = cells
        self.col_center = self.width // 2
        self.row_center = self.height // 2

        half_w = 0.5 * self.width * self.resolution
        half_h = 0.5 * self.height * self.resolution
        # The shared occupancy grid is world-axis aligned while BSO-HFC searches in
        # robot-local axes. The inscribed local square stays inside the backing grid
        # for all robot yaws.
        backing_half_span = min(half_w, half_h) / math.sqrt(2.0)
        half_span = backing_half_span
        if window_size_m is not None:
            requested_half_span = 0.5 * max(float(window_size_m), self.resolution)
            half_span = min(backing_half_span, requested_half_span)
            if backing_half_span < requested_half_span:
                half_span -= self.resolution
        else:
            half_span -= self.resolution
        half_span = max(half_span, self.resolution)
        self.x_min = -half_span
        self.x_max = half_span
        self.y_min = -half_span
        self.y_max = half_span

    def _resolve_distance_field(self) -> np.ndarray | None:
        if self.unknown_is_occupied:
            field = getattr(self.map_query, "distance_field", None)
            return None if field is None else np.asarray(field, dtype=np.float32)

        # BSO-HFC follows the original planner assumption that unknown space is
        # not a hard obstacle. Ask the shared map for the optimistic ESDF instead
        # of sampling the conservative observed-only field.
        if hasattr(self.map_query, "distance_field_for_unknown_policy"):
            field = self.map_query.distance_field_for_unknown_policy(False)
            return np.asarray(field, dtype=np.float32)

        field = getattr(self.map_query, "distance_field", None)
        return None if field is None else np.asarray(field, dtype=np.float32)

    def world_to_local(self, points_world: np.ndarray) -> np.ndarray:
        pose = self.robot_pose_world
        points = np.asarray(points_world, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        dx = points[:, 0] - pose[0]
        dy = points[:, 1] - pose[1]
        cos_yaw = math.cos(float(pose[2]))
        sin_yaw = math.sin(float(pose[2]))
        x_local = cos_yaw * dx + sin_yaw * dy
        y_local = -sin_yaw * dx + cos_yaw * dy
        return np.column_stack((x_local, y_local))

    def local_to_world(self, points_local: np.ndarray) -> np.ndarray:
        pose = self.robot_pose_world
        points = np.asarray(points_local, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        cos_yaw = math.cos(float(pose[2]))
        sin_yaw = math.sin(float(pose[2]))
        x_world = pose[0] + cos_yaw * points[:, 0] - sin_yaw * points[:, 1]
        y_world = pose[1] + sin_yaw * points[:, 0] + cos_yaw * points[:, 1]
        return np.column_stack((x_world, y_world))

    def local_to_grid(self, x_local: float, y_local: float) -> tuple[int, int]:
        x_world, y_world = self.local_to_world(np.array([x_local, y_local], dtype=float))[0]
        if hasattr(self.map_query, "world_to_grid"):
            gx, gy = self.map_query.world_to_grid(float(x_world), float(y_world))
            return int(gy), int(gx)
        row = int(round(float(y_local) / self.resolution + self.row_center))
        col = int(round(float(x_local) / self.resolution + self.col_center))
        return row, col

    def in_bounds(self, x_local: float, y_local: float) -> bool:
        if not (self.x_min <= x_local <= self.x_max and self.y_min <= y_local <= self.y_max):
            return False
        x_world, y_world = self.local_to_world(np.array([x_local, y_local], dtype=float))[0]
        if hasattr(self.map_query, "world_to_grid") and hasattr(self.map_query, "in_bounds"):
            gx, gy = self.map_query.world_to_grid(float(x_world), float(y_world))
            return bool(self.map_query.in_bounds(gx, gy))
        return True

    def sample_distance_bilinear(self, x_local: float, y_local: float) -> float:
        if not self.in_bounds(x_local, y_local):
            return 0.0
        if self.distance_field is not None and hasattr(self.map_query, "origin_xy"):
            return self._sample_distance_field_bilinear(x_local, y_local)
        x_world, y_world = self.local_to_world(np.array([x_local, y_local], dtype=float))[0]
        return float(self.map_query.distance_to_obstacle(float(x_world), float(y_world)))

    def sample_distances(self, points_local: np.ndarray) -> np.ndarray:
        points = np.asarray(points_local, dtype=float)
        if points.size == 0:
            return np.empty((0,), dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if self.distance_field is not None and hasattr(self.map_query, "origin_xy"):
            distances = np.asarray(
                [self.sample_distance_bilinear(float(x), float(y)) for x, y in points[:, :2]],
                dtype=float,
            )
        else:
            points_world = self.local_to_world(points[:, :2])
            if hasattr(self.map_query, "distance_to_obstacle_points"):
                distances = np.asarray(self.map_query.distance_to_obstacle_points(points_world), dtype=float)
            else:
                distances = np.asarray(
                    [self.map_query.distance_to_obstacle(float(x), float(y)) for x, y in points_world],
                    dtype=float,
                )
        in_window = (
            (points[:, 0] >= self.x_min)
            & (points[:, 0] <= self.x_max)
            & (points[:, 1] >= self.y_min)
            & (points[:, 1] <= self.y_max)
        )
        distances[~in_window] = 0.0
        return distances

    def map_limits(self) -> tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max

    def occupancy_to_local_points(self, stride: int = 1) -> np.ndarray:
        stride = max(int(stride), 1)
        grid = getattr(self.map_query, "inflated_grid", getattr(self.map_query, "occupied_grid", None))
        if grid is None or not hasattr(self.map_query, "grid_to_world_points"):
            return np.empty((0, 2), dtype=float)
        rows, cols = np.nonzero(np.asarray(grid) != 0)
        if len(rows) == 0:
            return np.empty((0, 2), dtype=float)
        rows = rows[::stride]
        cols = cols[::stride]
        world = self.map_query.grid_to_world_points(cols, rows)
        return self.world_to_local(world)

    def sample_edt_local_points(self, stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
        stride = max(int(stride), 1)
        field = self.distance_field
        if field is None or not hasattr(self.map_query, "grid_to_world_points"):
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
        distance_field = np.asarray(field, dtype=float)
        rows = np.arange(0, distance_field.shape[0], stride, dtype=int)
        cols = np.arange(0, distance_field.shape[1], stride, dtype=int)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr = rr.reshape(-1)
        cc = cc.reshape(-1)
        values = distance_field[rr, cc]
        free = values > 0.0
        if not np.any(free):
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
        world = self.map_query.grid_to_world_points(cc[free], rr[free])
        return self.world_to_local(world), values[free].astype(float, copy=False)

    @staticmethod
    def circle_to_local_points(radius: float, num_points: int = 100) -> np.ndarray:
        radius = max(float(radius), 0.0)
        if radius <= 0.0:
            return np.empty((0, 2), dtype=float)
        angles = np.linspace(0.0, 2.0 * np.pi, max(int(num_points), 12), endpoint=True, dtype=float)
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles))).astype(float, copy=False)

    @staticmethod
    def to_plot_array(points_world: np.ndarray | list) -> np.ndarray:
        points = np.asarray(points_world, dtype=float)
        if points.size == 0:
            return np.empty((2, 0), dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return points[:, :2].T

    def _sample_distance_field_bilinear(self, x_local: float, y_local: float) -> float:
        x_world, y_world = self.local_to_world(np.array([x_local, y_local], dtype=float))[0]
        origin = np.asarray(self.map_query.origin_xy, dtype=float)
        row_f = (float(y_world) - float(origin[1])) / self.resolution
        col_f = (float(x_world) - float(origin[0])) / self.resolution
        field = np.asarray(self.distance_field, dtype=float)
        if row_f < 0.0 or row_f > field.shape[0] - 1 or col_f < 0.0 or col_f > field.shape[1] - 1:
            return 0.0
        row0 = int(np.floor(row_f))
        col0 = int(np.floor(col_f))
        row1 = min(row0 + 1, field.shape[0] - 1)
        col1 = min(col0 + 1, field.shape[1] - 1)
        wy = row_f - row0
        wx = col_f - col0
        v00 = field[row0, col0]
        v01 = field[row0, col1]
        v10 = field[row1, col0]
        v11 = field[row1, col1]
        return float((1.0 - wy) * (1.0 - wx) * v00 + (1.0 - wy) * wx * v01 + wy * (1.0 - wx) * v10 + wy * wx * v11)
