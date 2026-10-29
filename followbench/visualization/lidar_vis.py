from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

LIDAR_BEV_RANGE_M = 10.0
LIDAR_DEBUG_RANGE_M = 10.0
LIDAR_RANGE_MIN_M = 0.2
LIDAR_Z_TOL_M = 0.20


def lidar_raw_from_carla(lidar_data) -> Optional[np.ndarray]:
    if lidar_data is None:
        return None
    return np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4).copy()


def _filter_lidar_sensor_xy(
    points: np.ndarray,
    max_range: float,
    min_range: float = LIDAR_RANGE_MIN_M,
) -> np.ndarray:
    if points is None or points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    xy = points[:, :2].astype(np.float32, copy=False)
    z = points[:, 2] if points.shape[1] >= 3 else np.zeros(points.shape[0], dtype=np.float32)
    radii = np.hypot(xy[:, 0], xy[:, 1])
    keep = (radii >= min_range) & (radii < max_range) & (np.abs(z) <= LIDAR_Z_TOL_M)
    return xy[keep]


def lidar_points_world_xy(
    lidar_points: Optional[np.ndarray],
    lidar_extr: Optional[dict],
    robot_x: float,
    robot_y: float,
    robot_yaw: float,
    *,
    max_range: float = LIDAR_DEBUG_RANGE_M,
    min_range: float = LIDAR_RANGE_MIN_M,
) -> List[List[float]]:
    if lidar_points is None or lidar_points.shape[0] == 0:
        return []
    try:
        from baselines.perception.sensor.lidar_obstacles import lidar_to_world
    except ImportError:
        return []
    xy = _filter_lidar_sensor_xy(lidar_points, max_range, min_range)
    if xy.shape[0] == 0:
        return []
    world = lidar_to_world(xy, lidar_extr, robot_x, robot_y, robot_yaw)
    return world.tolist()


def debug_lidar_world_points(
    robot,
    robot_state,
    lidar_img,
    last_lidar_np: Optional[np.ndarray],
    lidar_extr_cached: Optional[dict],
    *,
    max_range: float = LIDAR_DEBUG_RANGE_M,
) -> List[List[float]]:
    extr = lidar_extr_cached
    if extr is None:
        extr = robot.get_calibration().get("lidar", {}).get("extrinsics_robot_to_sensor")
    pts = last_lidar_np if last_lidar_np is not None else lidar_raw_from_carla(lidar_img)
    return lidar_points_world_xy(
        pts,
        extr,
        float(robot_state.x),
        float(robot_state.y),
        float(robot_state.yaw_rad),
        max_range=max_range,
    )


def lidar_to_bev(
    lidar_data,
    w: int = 400,
    h: int = 260,
    rng: float = LIDAR_BEV_RANGE_M,
) -> Optional[np.ndarray]:
    if lidar_data is None:
        return None
    img = np.zeros((h, w, 3), dtype=np.uint8)
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    radii = np.hypot(xs, ys)
    m = (radii < float(rng)) & (np.abs(zs) <= LIDAR_Z_TOL_M)
    xs, ys, zs = xs[m], ys[m], zs[m]
    if xs.size == 0:
        cx, cy = w // 2, h // 2
        cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)
        return img
    sc = min(w, h) / (2.0 * float(rng))
    cx, cy = w // 2, h // 2
    px = (cx - ys * sc).astype(int)
    py = (cy - xs * sc).astype(int)
    v = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    zn = np.clip((zs[v] + LIDAR_Z_TOL_M) / (2.0 * LIDAR_Z_TOL_M), 0.0, 1.0)
    img[py[v], px[v]] = np.stack(
        [
            (zn * 255).astype(np.uint8),
            ((1.0 - zn) * 180).astype(np.uint8),
            np.full(v.sum(), 40, dtype=np.uint8),
        ],
        axis=1,
    )
    cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)
    return img


__all__ = [
    "LIDAR_BEV_RANGE_M",
    "LIDAR_DEBUG_RANGE_M",
    "debug_lidar_world_points",
    "lidar_points_world_xy",
    "lidar_raw_from_carla",
    "lidar_to_bev",
]
