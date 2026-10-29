"""Load gridmap_roi.npz: static obstacles + global A* on walkable grid."""

from __future__ import annotations

import heapq
import json
import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

STATIC_OBS_MIN_COMPONENT_CELLS = 3
ASTAR_MAX_ITERS = 30000
ASTAR_WALL_CLEARANCE_WEIGHT = 0.5
ASTAR_WALL_EPS = 0.5
ASTAR_NEIGHBORS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)


class RoiStaticMap:
    def __init__(self, npz_path: str, resolution_fallback: float = 0.5) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.world_min = np.array(data["world_min"], dtype=np.float64)
        self.world_max = np.array(data["world_max"], dtype=np.float64)
        if "free_grid" in data:
            self.free_grid = np.array(data["free_grid"], dtype=np.uint8)
        else:
            occ = np.array(data["occupied_grid"], dtype=np.uint8) if "occupied_grid" in data else None
            self.free_grid = (occ == 0).astype(np.uint8) if occ is not None else np.ones_like(data["obstacle_inflated"], dtype=np.uint8)
        if "roi_mask" in data:
            self.roi_mask = np.array(data["roi_mask"], dtype=np.uint8)
        else:
            self.roi_mask = np.ones_like(self.free_grid, dtype=np.uint8)
        if "obstacle_inflated" in data:
            self.obstacle_grid = np.array(data["obstacle_inflated"], dtype=np.uint8)
        elif "occupied_grid" in data:
            self.obstacle_grid = np.array(data["occupied_grid"], dtype=np.uint8)
        else:
            self.obstacle_grid = (self.free_grid == 0).astype(np.uint8)
        self.walkable = (self.free_grid > 0) & (self.roi_mask > 0)
        self.resolution = self._read_resolution(data, self.free_grid.shape, resolution_fallback)
        self._wall_dist_cache: Optional[np.ndarray] = None

    @staticmethod
    def _read_resolution(data: np.lib.npyio.NpzFile, grid_shape: Tuple[int, ...], fallback: float) -> float:
        try:
            meta_raw = data["__meta__"].item() if "__meta__" in data else None
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8", errors="ignore")
            if isinstance(meta_raw, str) and meta_raw.strip():
                return float(json.loads(meta_raw).get("resolution", fallback))
        except Exception:
            pass
        nrows, ncols = grid_shape
        return float((float(data["world_max"][0]) - float(data["world_min"][0])) / max(ncols, 1))

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.floor((float(x) - float(self.world_min[0])) / self.resolution))
        gy = int(np.floor((float(y) - float(self.world_min[1])) / self.resolution))
        return gx, gy

    def cell_to_world_xy(self, gx: int, gy: int) -> Tuple[float, float]:
        x = float(self.world_min[0] + (float(gx) + 0.5) * self.resolution)
        y = float(self.world_min[1] + (float(gy) + 0.5) * self.resolution)
        return x, y

    def is_walkable_cell(self, gx: int, gy: int) -> bool:
        h, w = self.walkable.shape
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return False
        return bool(self.walkable[gy, gx])

    def nearest_walkable_cell(self, gx: int, gy: int) -> Tuple[int, int]:
        h, w = self.walkable.shape
        if self.is_walkable_cell(gx, gy):
            return gx, gy
        visited = set()
        q = deque([(gx, gy)])
        visited.add((gx, gy))
        while q:
            cx, cy = q.popleft()
            for dx, dy in ASTAR_NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                if 0 <= nx < w and 0 <= ny < h:
                    if self.walkable[ny, nx]:
                        return nx, ny
                    q.append((nx, ny))
        return max(0, min(gx, w - 1)), max(0, min(gy, h - 1))

    def _wall_distance_field(self) -> np.ndarray:
        if self._wall_dist_cache is not None:
            return self._wall_dist_cache
        occupied = (self.obstacle_grid > 0).astype(np.uint8)
        dist = cv2.distanceTransform((occupied == 0).astype(np.uint8), cv2.DIST_L2, 5)
        self._wall_dist_cache = dist.astype(np.float32)
        return self._wall_dist_cache

    def astar_cells(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iters: int = ASTAR_MAX_ITERS,
        wall_w: float = ASTAR_WALL_CLEARANCE_WEIGHT,
    ) -> Optional[List[Tuple[int, int]]]:
        h, w = self.walkable.shape
        sx, sy = start
        gx, gy = goal
        if not self.is_walkable_cell(sx, sy):
            sx, sy = self.nearest_walkable_cell(sx, sy)
        if not self.is_walkable_cell(gx, gy):
            gx, gy = self.nearest_walkable_cell(gx, gy)
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        def heur(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, (sx, sy))]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
        wall_dist = self._wall_distance_field()
        iters = 0

        while open_heap and iters < max_iters:
            iters += 1
            _f, cur = heapq.heappop(open_heap)
            if cur == (gx, gy):
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path
            cx, cy = cur
            for dx, dy in ASTAR_NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not self.is_walkable_cell(nx, ny):
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                clearance = float(wall_w / (float(wall_dist[ny, nx]) + ASTAR_WALL_EPS))
                ng = gscore[cur] + step + clearance
                nxt = (nx, ny)
                if ng < gscore.get(nxt, 1e18):
                    came_from[nxt] = cur
                    gscore[nxt] = ng
                    heapq.heappush(open_heap, (ng + heur(nxt, (gx, gy)), nxt))
        return None

    def _los_clear(self, ax: int, ay: int, bx: int, by: int, margin: int = 1) -> bool:
        h, w = self.walkable.shape
        wall_dist = self._wall_distance_field()
        dx, dy = abs(bx - ax), abs(by - ay)
        sx = 1 if bx > ax else -1
        sy_s = 1 if by > ay else -1
        x, y = ax, ay
        err = dx - dy
        for _ in range(max(dx, dy) + 1):
            if not (0 <= x < w and 0 <= y < h):
                return False
            if not self.walkable[y, x]:
                return False
            if wall_dist[y, x] < margin:
                return False
            if x == bx and y == by:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy_s
        return True

    def los_smooth_path(self, cell_path: List[Tuple[int, int]], margin: int = 1) -> List[Tuple[int, int]]:
        if not cell_path or len(cell_path) <= 2:
            return cell_path
        out = [cell_path[0]]
        anchor = 0
        idx = 2
        while idx < len(cell_path):
            if not self._los_clear(cell_path[anchor][0], cell_path[anchor][1],
                                   cell_path[idx][0], cell_path[idx][1], margin=margin):
                out.append(cell_path[idx - 1])
                anchor = idx - 1
            idx += 1
        out.append(cell_path[-1])
        return out

    def plan_path_world(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
        *,
        smooth: bool = True,
    ) -> List[List[float]]:
        sx, sy = self.world_to_cell(start_x, start_y)
        gx, gy = self.world_to_cell(goal_x, goal_y)
        cell_path = self.astar_cells((sx, sy), (gx, gy))
        if cell_path is None:
            return []
        if smooth:
            cell_path = self.los_smooth_path(cell_path)
        return [[self.cell_to_world_xy(cx, cy)[0], self.cell_to_world_xy(cx, cy)[1]] for cx, cy in cell_path]

    def static_obstacles_near(
        self,
        robot_x: float,
        robot_y: float,
        range_m: float,
        *,
        target_xy: Optional[Sequence[float]] = None,
        target_mask_radius: float = 0.6,
        max_obstacles: int = 12,
    ) -> List[dict]:
        h, w = self.obstacle_grid.shape
        margin = max(int(math.ceil(float(range_m) / self.resolution)), 1)
        gxc, gyc = self.world_to_cell(robot_x, robot_y)
        gx0 = max(0, gxc - margin)
        gx1 = min(w, gxc + margin + 1)
        gy0 = max(0, gyc - margin)
        gy1 = min(h, gyc + margin + 1)
        crop = self.obstacle_grid[gy0:gy1, gx0:gx1]
        if crop.size == 0 or not np.any(crop):
            return []

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (crop > 0).astype(np.uint8), connectivity=8,
        )
        candidates: List[Tuple[float, dict]] = []
        half_cell = 0.5 * self.resolution
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < STATIC_OBS_MIN_COMPONENT_CELLS:
                continue
            ys, xs = np.where(labels == label)
            wx = self.world_min[0] + (xs.astype(np.float64) + float(gx0) + 0.5) * self.resolution
            wy = self.world_min[1] + (ys.astype(np.float64) + float(gy0) + 0.5) * self.resolution
            points = np.column_stack([wx, wy]).astype(np.float32)
            if points.shape[0] < 3:
                cx = float(wx.mean())
                cy = float(wy.mean())
                vertex = np.array(
                    [
                        [cx - half_cell, cy - half_cell],
                        [cx + half_cell, cy - half_cell],
                        [cx + half_cell, cy + half_cell],
                        [cx - half_cell, cy + half_cell],
                    ],
                    dtype=np.float32,
                ).T
            else:
                rect = cv2.minAreaRect(points)
                cx, cy = float(rect[0][0]), float(rect[0][1])
                vertex = cv2.boxPoints(rect).T.astype(np.float32)
            if target_xy is not None:
                if math.hypot(cx - float(target_xy[0]), cy - float(target_xy[1])) < float(target_mask_radius):
                    continue
            dist = math.hypot(cx - float(robot_x), cy - float(robot_y))
            if dist > float(range_m) + self.resolution:
                continue
            center = np.array([[cx], [cy]], dtype=np.float32)
            candidates.append(
                (
                    dist,
                    {
                        "center": center,
                        "radius": None,
                        "vertex": vertex,
                        "cone_type": "Rpositive",
                        "velocity": np.zeros((2, 1), dtype=np.float32),
                    },
                )
            )
        candidates.sort(key=lambda item: item[0])
        return [item[1] for item in candidates[: max(0, int(max_obstacles))]]


__all__ = ["RoiStaticMap", "STATIC_OBS_MIN_COMPONENT_CELLS"]
