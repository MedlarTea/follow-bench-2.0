from __future__ import annotations

import numpy as np

from baselines.maps.map_query import MapQuery


class NullMapQuery(MapQuery):
    def is_free(self, x: float, y: float) -> bool:
        return True

    def distance_to_obstacle(self, x: float, y: float) -> float:
        return float("inf")

    def is_observed(self, x: float, y: float) -> bool:
        return True

    def is_unknown(self, x: float, y: float) -> bool:
        return False

    def ray_clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        return True

    def line_min_clearance(self, x0: float, y0: float, x1: float, y1: float) -> float:
        return float("inf")

    def is_free_points(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=float)
        return np.ones((len(pts),), dtype=bool)

    def is_observed_points(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=float)
        return np.ones((len(pts),), dtype=bool)

    def is_unknown_points(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=float)
        return np.zeros((len(pts),), dtype=bool)

    def distance_to_obstacle_points(self, points_xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_xy, dtype=float)
        return np.full((len(pts),), float("inf"), dtype=float)
