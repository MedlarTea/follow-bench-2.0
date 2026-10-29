from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MapQuery(ABC):
    @abstractmethod
    def is_free(self, x: float, y: float) -> bool:
        pass

    @abstractmethod
    def distance_to_obstacle(self, x: float, y: float) -> float:
        pass

    @abstractmethod
    def is_observed(self, x: float, y: float) -> bool:
        pass

    @abstractmethod
    def is_unknown(self, x: float, y: float) -> bool:
        pass

    @abstractmethod
    def ray_clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        pass

    @abstractmethod
    def line_min_clearance(self, x0: float, y0: float, x1: float, y1: float) -> float:
        pass

    def is_free_points(self, points_xy: np.ndarray) -> np.ndarray:
        # Default batch implementation for map backends without vectorized overrides.
        pts = np.asarray(points_xy, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.asarray([self.is_free(float(x), float(y)) for x, y in pts], dtype=bool)

    def is_observed_points(self, points_xy: np.ndarray) -> np.ndarray:
        # Default batch implementation for map backends without vectorized overrides.
        pts = np.asarray(points_xy, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.asarray([self.is_observed(float(x), float(y)) for x, y in pts], dtype=bool)

    def is_unknown_points(self, points_xy: np.ndarray) -> np.ndarray:
        # Default batch implementation for map backends without vectorized overrides.
        pts = np.asarray(points_xy, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=bool)
        return np.asarray([self.is_unknown(float(x), float(y)) for x, y in pts], dtype=bool)

    def distance_to_obstacle_points(self, points_xy: np.ndarray) -> np.ndarray:
        # Default batch implementation for map backends without vectorized overrides.
        pts = np.asarray(points_xy, dtype=float)
        if pts.size == 0:
            return np.zeros((0,), dtype=float)
        return np.asarray([self.distance_to_obstacle(float(x), float(y)) for x, y in pts], dtype=float)
