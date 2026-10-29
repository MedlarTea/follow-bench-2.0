from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

LIDAR_RANGE_MIN_M = 0.2
LIDAR_RANGE_MAX_M = 5.0
DBSCAN_EPS_M = 0.4
DBSCAN_MIN_SAMPLES = 3
TARGET_MASK_RADIUS_M = 0.6

_DBSCAN_CLS = None


def filter_lidar_2d(points: np.ndarray, max_range: float, min_range: float) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    xy = points[:, :2].astype(np.float32, copy=False)
    z = points[:, 2] if points.shape[1] >= 3 else np.zeros(points.shape[0], dtype=np.float32)
    radii = np.hypot(xy[:, 0], xy[:, 1])
    keep = (radii >= min_range) & (radii <= max_range) & (np.abs(z) <= 0.20)
    return xy[keep]


def lidar_to_world(
    points_xy_lidar: np.ndarray,
    lidar_extr: Optional[dict],
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
) -> np.ndarray:
    if points_xy_lidar.shape[0] == 0:
        return points_xy_lidar

    tx = float(lidar_extr.get("x", 0.0)) if lidar_extr else 0.0
    ty = float(lidar_extr.get("y", 0.0)) if lidar_extr else 0.0
    yaw_l = math.radians(float(lidar_extr.get("yaw_deg", 0.0))) if lidar_extr else 0.0

    cl = math.cos(yaw_l)
    sl = math.sin(yaw_l)
    bx = cl * points_xy_lidar[:, 0] - sl * points_xy_lidar[:, 1] + tx
    by = sl * points_xy_lidar[:, 0] + cl * points_xy_lidar[:, 1] + ty

    cy = math.cos(robot_yaw)
    sy = math.sin(robot_yaw)
    wx = robot_x + cy * bx - sy * by
    wy = robot_y + sy * bx + cy * by
    return np.column_stack([wx, wy]).astype(np.float32, copy=False)


def lidar_obstacles_world(
    points_world_xy: np.ndarray,
    target_xy: Optional[Sequence[float]],
    target_mask_radius: float,
) -> list[dict[str, np.ndarray | None | str]]:
    if points_world_xy.shape[0] < DBSCAN_MIN_SAMPLES:
        return []

    dbscan_cls = _get_dbscan_cls()
    labels = dbscan_cls(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(points_world_xy)
    out: list[dict[str, np.ndarray | None | str]] = []
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster = points_world_xy[labels == label]
        if cluster.shape[0] < 3:
            continue

        rect = cv2.minAreaRect(cluster.astype(np.float32))
        cx = float(rect[0][0])
        cy = float(rect[0][1])
        if target_xy is not None and math.hypot(cx - float(target_xy[0]), cy - float(target_xy[1])) < target_mask_radius:
            continue

        vertex = cv2.boxPoints(rect).T.astype(np.float32)
        center = np.array([[cx], [cy]], dtype=np.float32)
        out.append(
            {
                "center": center,
                "radius": None,
                "vertex": vertex,
                "cone_type": "Rpositive",
                "velocity": np.zeros((2, 1), dtype=np.float32),
            }
        )
    return out


def _get_dbscan_cls():
    global _DBSCAN_CLS
    if _DBSCAN_CLS is None:
        from sklearn.cluster import DBSCAN

        _DBSCAN_CLS = DBSCAN
    return _DBSCAN_CLS


__all__ = [
    "DBSCAN_EPS_M",
    "DBSCAN_MIN_SAMPLES",
    "LIDAR_RANGE_MAX_M",
    "LIDAR_RANGE_MIN_M",
    "TARGET_MASK_RADIUS_M",
    "filter_lidar_2d",
    "lidar_obstacles_world",
    "lidar_to_world",
]
