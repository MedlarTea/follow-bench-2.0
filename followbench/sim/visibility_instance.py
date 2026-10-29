from __future__ import annotations

from typing import Optional, Tuple

import carla
import numpy as np


def _instance_id_map(instance_image) -> np.ndarray:
    arr = np.frombuffer(instance_image.raw_data, dtype=np.uint8).reshape(instance_image.height, instance_image.width, 4)
    b = arr[:, :, 0].astype(np.uint32)
    g = arr[:, :, 1].astype(np.uint32)
    r = arr[:, :, 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def _project_world_to_image(world_point, sensor_transform, image_w: int, image_h: int, fov_deg: float):
    w2c = np.array(sensor_transform.get_inverse_matrix())
    p = np.array([world_point.x, world_point.y, world_point.z, 1.0])
    pc = np.dot(w2c, p)
    # Unreal->camera axis conversion
    x_cam = pc[1]
    y_cam = -pc[2]
    z_cam = pc[0]
    if z_cam <= 1e-3:
        return None
    f = image_w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    u = f * (x_cam / z_cam) + image_w / 2.0
    v = f * (y_cam / z_cam) + image_h / 2.0
    return float(u), float(v), float(z_cam)


def _target_bbox_roi(target_actor, sensor_actor, image_w: int, image_h: int, fov_deg: float) -> Optional[Tuple[int, int, int, int]]:
    bb = target_actor.bounding_box
    tf = target_actor.get_transform()
    ex = bb.extent.x
    ey = bb.extent.y
    ez = bb.extent.z
    verts_local = [
        (ex, ey, ez),
        (ex, ey, -ez),
        (ex, -ey, ez),
        (ex, -ey, -ez),
        (-ex, ey, ez),
        (-ex, ey, -ez),
        (-ex, -ey, ez),
        (-ex, -ey, -ez),
    ]
    pts = []
    sensor_tf = sensor_actor.get_transform()
    for x, y, z in verts_local:
        wp = tf.transform(
            carla.Location(
                x=float(x + bb.location.x),
                y=float(y + bb.location.y),
                z=float(z + bb.location.z),
            )
        )
        uvz = _project_world_to_image(wp, sensor_tf, image_w, image_h, fov_deg)
        if uvz is not None:
            pts.append(uvz)
    if not pts:
        return None
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    u0 = max(0, int(np.floor(min(us))))
    v0 = max(0, int(np.floor(min(vs))))
    u1 = min(image_w - 1, int(np.ceil(max(us))))
    v1 = min(image_h - 1, int(np.ceil(max(vs))))
    if u1 <= u0 or v1 <= v0:
        return None
    return u0, v0, u1, v1


class InstanceVisibilityChecker:
    """
    Instance-pixel visibility checker.
    Rule: visible = pixel_count > threshold.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = int(threshold)
        self._tracked_instance_id: Optional[int] = None

    def reset(self) -> None:
        self._tracked_instance_id = None

    def evaluate(self, instance_image, instance_sensor_actor, target_actor, image_w: int, image_h: int, fov_deg: float):
        if instance_image is None or instance_sensor_actor is None or target_actor is None:
            return False, 0
        id_map = _instance_id_map(instance_image)
        roi = _target_bbox_roi(target_actor, instance_sensor_actor, image_w, image_h, fov_deg)
        if roi is None:
            return False, 0
        u0, v0, u1, v1 = roi
        patch = id_map[v0 : v1 + 1, u0 : u1 + 1]
        if patch.size == 0:
            return False, 0
        vals, counts = np.unique(patch, return_counts=True)
        nonzero = vals != 0
        vals = vals[nonzero]
        counts = counts[nonzero]
        if len(vals) == 0:
            return False, 0
        if self._tracked_instance_id is None or self._tracked_instance_id not in vals:
            self._tracked_instance_id = int(vals[np.argmax(counts)])
        pix_count = int(np.sum(id_map == self._tracked_instance_id))
        return pix_count > self.threshold, pix_count

    def evaluate_multi(self, views, target_actor, image_w: int, image_h: int, fov_deg: float):
        """Evaluate visibility across multiple instance-segmentation views.

        views: iterable of (instance_image, instance_sensor_actor) pairs.
        Returns (visible, total_pix_count) — pix_count is the sum across all
        views where the tracked instance ID appears.
        """
        if target_actor is None:
            return False, 0
        id_maps = []
        roi_patches = []
        for img, sensor in views:
            if img is None or sensor is None:
                continue
            id_map = _instance_id_map(img)
            id_maps.append(id_map)
            roi = _target_bbox_roi(target_actor, sensor, image_w, image_h, fov_deg)
            if roi is None:
                continue
            u0, v0, u1, v1 = roi
            patch = id_map[v0 : v1 + 1, u0 : u1 + 1]
            if patch.size > 0:
                roi_patches.append(patch)
        if not id_maps or not roi_patches:
            return False, 0
        agg: dict[int, int] = {}
        for patch in roi_patches:
            vals, counts = np.unique(patch, return_counts=True)
            for v, c in zip(vals, counts):
                if int(v) != 0:
                    agg[int(v)] = agg.get(int(v), 0) + int(c)
        if not agg:
            return False, 0
        if self._tracked_instance_id is None or self._tracked_instance_id not in agg:
            self._tracked_instance_id = max(agg, key=agg.get)
        pix_count = sum(int(np.sum(id_map == self._tracked_instance_id)) for id_map in id_maps)
        return pix_count > self.threshold, pix_count
