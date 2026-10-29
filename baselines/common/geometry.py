from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from shapely.geometry import Polygon


def wrap_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def yaw_between(start_xy: Sequence[float], end_xy: Sequence[float]) -> float:
    dx = float(end_xy[0]) - float(start_xy[0])
    dy = float(end_xy[1]) - float(start_xy[1])
    return math.atan2(dy, dx)


def distance_to_closest_point(point: Sequence[float], positions: np.ndarray) -> float:
    pts = np.asarray(positions, dtype=float)
    if pts.size == 0:
        return float("inf")
    p = np.asarray(point, dtype=float)
    return float(np.min(np.linalg.norm(pts - p, axis=1)))


def distance_to_closest_points(points: np.ndarray, positions: np.ndarray) -> np.ndarray:
    # Batch implementation for sample arrays used by search scoring.
    pts = np.asarray(points, dtype=float)
    ref = np.asarray(positions, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=float)
    if ref.size == 0:
        return np.full((len(pts),), float("inf"), dtype=float)
    deltas = pts[:, None, :] - ref[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    return np.sqrt(np.min(dist_sq, axis=1))


def inverse_distance_cost(point: Sequence[float], positions: np.ndarray, max_dist: float) -> float:
    dist = distance_to_closest_point(point, positions)
    if not np.isfinite(dist) or dist >= max_dist:
        return 0.0
    dist = max(dist, 1e-3)
    return float((1.0 / dist - 1.0 / max_dist) / (dist * dist))


def inverse_distance_costs(points: np.ndarray, positions: np.ndarray, max_dist: float) -> np.ndarray:
    # Batch implementation of the scalar inverse-distance cost.
    dists = distance_to_closest_points(points, positions)
    out = np.zeros_like(dists, dtype=float)
    keep = np.isfinite(dists) & (dists < float(max_dist))
    if not np.any(keep):
        return out
    safe = np.maximum(dists[keep], 1e-3)
    out[keep] = (1.0 / safe - 1.0 / float(max_dist)) / (safe * safe)
    return out


def tangent_triangle(sample_point: Sequence[float], target_point: Sequence[float], radius: float) -> Optional[Polygon]:
    p = np.asarray(sample_point, dtype=float)
    c = np.asarray(target_point, dtype=float)
    d = c - p
    dist = float(np.linalg.norm(d))
    if dist <= radius or dist <= 1e-6:
        return None

    d_hat = d / dist
    n_hat = np.array([-d_hat[1], d_hat[0]], dtype=float)
    theta = math.asin(radius / dist)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    v1 = cos_theta * d_hat + sin_theta * n_hat
    v2 = cos_theta * d_hat - sin_theta * n_hat
    t1 = c - radius * np.array([-v1[1], v1[0]], dtype=float)
    t2 = c + radius * np.array([-v2[1], v2[0]], dtype=float)
    return Polygon([p.tolist(), t1.tolist(), t2.tolist()])


def polygon_iou(a: Polygon, b: Polygon) -> float:
    if a is None or b is None:
        return 0.0
    intersection = a.intersection(b)
    if intersection.is_empty:
        return 0.0
    union = a.union(b)
    if union.area <= 1e-9:
        return 0.0
    return float(intersection.area / union.area)
