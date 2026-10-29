from __future__ import annotations

import json
import math
import queue

import numpy as np


def _impulse_norm(vec) -> float:
    return float(math.sqrt(float(vec.x) ** 2 + float(vec.y) ** 2 + float(vec.z) ** 2))


class PassageCollisionMonitor:
    """Collision sensor wrapper that keeps pedestrian and obstacle contacts."""

    def __init__(self, world, robot_actor, enabled: bool = True) -> None:
        self.world = world
        self.robot_actor = robot_actor
        self.enabled = bool(enabled)
        self.sensor = None
        self._events: "queue.Queue[dict]" = queue.Queue()
        if self.enabled and self.robot_actor is not None:
            self._spawn_sensor()

    def _spawn_sensor(self) -> None:
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, self.robot_actor.get_transform(), attach_to=self.robot_actor)

        def _callback(event):
            other = getattr(event, "other_actor", None)
            other_id = -1 if other is None else int(getattr(other, "id", -1))
            if other_id == int(getattr(self.robot_actor, "id", -2)):
                return
            other_type = "" if other is None else str(getattr(other, "type_id", ""))
            collision_type = "human" if other_type.startswith("walker.pedestrian") else "obstacle"
            self._events.put(
                {
                    "source": "collision_sensor",
                    "collision_type": collision_type,
                    "robot_actor_id": int(getattr(event.actor, "id", -1)),
                    "other_actor_id": other_id,
                    "other_type_id": other_type,
                    "impulse_norm": _impulse_norm(event.normal_impulse),
                }
            )

        self.sensor.listen(_callback)

    def drain(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def destroy(self) -> None:
        if self.sensor is None:
            return
        try:
            self.sensor.stop()
        except Exception:
            pass
        try:
            self.sensor.destroy()
        except Exception:
            pass
        self.sensor = None


class PassageMapObstacleMonitor:
    """Grid-map based raw-obstacle overlap / clearance checker for kinematic robot motion."""

    def __init__(
        self,
        *,
        grid_npz: str,
        robot_radius_m: float,
        overlap_confirm_steps: int = 2,
        clearance_probe_m: float = 2.0,
    ) -> None:
        data = np.load(grid_npz, allow_pickle=True)
        self.obstacle_raw = np.array(data["obstacle_raw"], dtype=np.uint8) > 0
        self.occupied = np.array(data["occupied_grid"], dtype=np.uint8) > 0
        self.free = np.array(data["free_grid"], dtype=np.uint8) > 0
        self.roi = np.array(data["roi_mask"], dtype=np.uint8) > 0 if "roi_mask" in data else np.ones_like(self.occupied)
        self.world_min = np.array(data["world_min"], dtype=np.float64)[:2]
        meta: dict = {}
        if "__meta__" in data:
            try:
                raw = data["__meta__"]
                item = raw.item() if isinstance(raw, np.ndarray) else raw
                if isinstance(item, bytes):
                    item = item.decode()
                meta = json.loads(item) if isinstance(item, str) else {}
            except Exception:
                meta = {}
        self.resolution = float(meta.get("resolution", 0.5))
        self.robot_radius_m = max(float(robot_radius_m), self.resolution * 0.5)
        self.overlap_confirm_steps = max(int(overlap_confirm_steps), 1)
        self.clearance_probe_m = max(float(clearance_probe_m), self.robot_radius_m)
        self._overlap_count = 0

    def reset(self) -> None:
        self._overlap_count = 0

    def update(self, *, x: float, y: float) -> dict:
        overlap = self._footprint_overlaps_obstacle(x=x, y=y)
        self._overlap_count = self._overlap_count + 1 if overlap else 0
        clearance_m = self._clearance_to_obstacle(x=x, y=y)
        return {
            "grid_overlap": bool(overlap),
            "grid_overlap_confirm_count": int(self._overlap_count),
            "grid_overlap_confirm_steps": int(self.overlap_confirm_steps),
            "grid_clearance_m": clearance_m,
            "termination_triggered": bool(self._overlap_count >= self.overlap_confirm_steps),
            "termination_reason": "obstacle_collision" if self._overlap_count >= self.overlap_confirm_steps else None,
        }

    def make_event(self, *, tick: int, time_s: float, episode_active: bool, status: dict) -> dict:
        return {
            "tick": int(tick),
            "time_s": float(time_s),
            "episode_active": bool(episode_active),
            "source": "map_obstacle",
            "collision_type": "obstacle",
            "other_actor_id": -1,
            "other_type_id": "gridmap_obstacle",
            "grid_clearance_m": status.get("grid_clearance_m"),
            "grid_overlap": bool(status.get("grid_overlap", False)),
        }

    def _footprint_overlaps_obstacle(self, *, x: float, y: float) -> bool:
        gx_f = (float(x) - self.world_min[0]) / self.resolution - 0.5
        gy_f = (float(y) - self.world_min[1]) / self.resolution - 0.5
        r_cells = int(math.ceil(self.robot_radius_m / self.resolution))
        gx0 = max(int(math.floor(gx_f)) - r_cells, 0)
        gx1 = min(int(math.floor(gx_f)) + r_cells + 1, self.occupied.shape[1])
        gy0 = max(int(math.floor(gy_f)) - r_cells, 0)
        gy1 = min(int(math.floor(gy_f)) + r_cells + 1, self.occupied.shape[0])
        for gy in range(gy0, gy1):
            cy = self.world_min[1] + (gy + 0.5) * self.resolution
            for gx in range(gx0, gx1):
                cx = self.world_min[0] + (gx + 0.5) * self.resolution
                if math.hypot(cx - x, cy - y) > self.robot_radius_m:
                    continue
                if self.obstacle_raw[gy, gx]:
                    return True
        return False

    def _clearance_to_obstacle(self, *, x: float, y: float) -> float | None:
        gx_f = (float(x) - self.world_min[0]) / self.resolution - 0.5
        gy_f = (float(y) - self.world_min[1]) / self.resolution - 0.5
        pad_cells = int(math.ceil((self.clearance_probe_m + self.robot_radius_m) / self.resolution))
        gx0 = max(int(math.floor(gx_f)) - pad_cells, 0)
        gx1 = min(int(math.floor(gx_f)) + pad_cells + 1, self.occupied.shape[1])
        gy0 = max(int(math.floor(gy_f)) - pad_cells, 0)
        gy1 = min(int(math.floor(gy_f)) + pad_cells + 1, self.occupied.shape[0])
        best = None
        cell_radius = 0.5 * math.sqrt(2.0) * self.resolution
        for gy in range(gy0, gy1):
            cy = self.world_min[1] + (gy + 0.5) * self.resolution
            for gx in range(gx0, gx1):
                if not self.obstacle_raw[gy, gx]:
                    continue
                cx = self.world_min[0] + (gx + 0.5) * self.resolution
                dist = math.hypot(cx - x, cy - y) - self.robot_radius_m - cell_radius
                if best is None or dist < best:
                    best = dist
        if best is None:
            return None
        return float(max(best, 0.0))
