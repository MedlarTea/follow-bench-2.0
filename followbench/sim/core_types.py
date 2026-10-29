from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NpcState:
    track_id: str
    actor_id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    yaw_deg: float
    speed: float
    # Optional debug/runtime annotations used by park scenario.
    npc_state: str = "transit"
    wp_x: Optional[float] = None
    wp_y: Optional[float] = None


@dataclass
class RobotState:
    x: float
    y: float
    z: float
    yaw_rad: float
    speed: float


@dataclass
class FollowObservation:
    tick: int
    dt: float
    robot: RobotState
    target: NpcState
    npcs: List[NpcState]
    target_visible: bool
    target_pixel_count: int
    extras: Dict
    # Raw RGB image in HxWx3 uint8 (RGB order), already decoded from CARLA buffer.
    rgb_image: Optional[np.ndarray] = None
    # Depth image in HxW float32 metres (decoded from CARLA's RGBA depth encoding).
    # Shares pixel grid + intrinsics + extrinsics with rgb_image (cameras are co-mounted).
    depth_image: Optional[np.ndarray] = None
    # Raw LiDAR point cloud in the LiDAR sensor frame, shape (N, 4) float32:
    # columns are [x, y, z, intensity] following CARLA's convention.
    lidar_points: Optional[np.ndarray] = None
    # RGB camera intrinsics: {fx, fy, cx, cy, width, height, fov_deg}.
    rgb_intrinsics: Optional[Dict[str, Any]] = None
    # Sensor pose relative to the robot base (extrinsics_robot_to_sensor):
    # {x, y, z, roll_deg, pitch_deg, yaw_deg}.
    rgb_extrinsics_robot_to_sensor: Optional[Dict[str, Any]] = None
    lidar_extrinsics_robot_to_sensor: Optional[Dict[str, Any]] = None
    # Side cameras (yaw=-90° left, +90° right relative to robot forward).
    rgb_image_left: Optional[np.ndarray] = None
    rgb_image_right: Optional[np.ndarray] = None
    depth_image_left: Optional[np.ndarray] = None
    depth_image_right: Optional[np.ndarray] = None
    rgb_intrinsics_left: Optional[Dict[str, Any]] = None
    rgb_intrinsics_right: Optional[Dict[str, Any]] = None
    rgb_extrinsics_left_robot_to_sensor: Optional[Dict[str, Any]] = None
    rgb_extrinsics_right_robot_to_sensor: Optional[Dict[str, Any]] = None


@dataclass
class FollowAction:
    v_mps: float
    w_radps: float


CalibrationDict = Dict[str, Dict]
Point2D = Tuple[float, float]
