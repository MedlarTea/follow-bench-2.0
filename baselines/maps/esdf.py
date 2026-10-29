from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np


def compute_unsigned_esdf(
    occupied_grid: np.ndarray,
    observed_grid: np.ndarray | None,
    resolution: float,
) -> np.ndarray:
    occupied = np.asarray(occupied_grid, dtype=np.uint8)
    observed = None if observed_grid is None else np.asarray(observed_grid, dtype=np.uint8)

    if observed is None:
        traversable = occupied == 0
    else:
        traversable = (occupied == 0) & (observed != 0)

    free_mask = traversable.astype(np.uint8)
    dist_cells = cv2.distanceTransform(free_mask, cv2.DIST_L2, 3)
    return dist_cells.astype(np.float32) * float(resolution)


def sample_line_min_clearance(
    distance_field: np.ndarray,
    origin_xy: Sequence[float],
    resolution: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    oversample: float = 2.0,
) -> float:
    field = np.asarray(distance_field, dtype=np.float32)
    if field.ndim != 2 or field.size == 0:
        return 0.0

    length = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    steps = max(2, int(math.ceil(length / max(float(resolution), 1e-6) * float(oversample))) + 1)
    ts = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    xs = np.float32(x0) + ts * np.float32(x1 - x0)
    ys = np.float32(y0) + ts * np.float32(y1 - y0)

    ox = float(origin_xy[0])
    oy = float(origin_xy[1])
    gx = np.floor((xs - ox) / float(resolution)).astype(np.int32)
    gy = np.floor((ys - oy) / float(resolution)).astype(np.int32)

    in_bounds = (gx >= 0) & (gy >= 0) & (gx < field.shape[1]) & (gy < field.shape[0])
    if not np.all(in_bounds):
        return 0.0
    return float(np.min(field[gy, gx]))
