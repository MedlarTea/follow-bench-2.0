from __future__ import annotations

import heapq
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np

from followbench.sim.core_types import NpcState


# ── Zone capacity constants ───────────────────────────────────────────────────
# Number of *dispatched* (corner_flow / trunk_walk / shadow_follower) NPCs
# allowed on routes that pass through each zone type simultaneously.
# Orbit NPCs are exempt — they roam freely and rely on SFM for light avoidance.
_ZONE_CAP_NARROW   = 1   # single-lane corner: only 1 at a time
_ZONE_CAP_JUNCTION = 2   # junction: 2 allowed simultaneously

# ── Global rescue constants ───────────────────────────────────────────────────
_RESCUE_INTERVAL  = 5.0   # seconds between accumulated-distance checks
_RESCUE_MIN_MOVE  = 1.5   # metres — must accumulate at least this much travel per interval
_RESCUE_OPEN_DIST = 2     # min wall_dist (cells) for rescue destination

_ROBOT_SFM_KEY = "__robot__"  # sentinel id so robot enters SFM but never collides with NPC track_ids

_NON_HUMAN_WALKER_BLUEPRINTS = {
    "walker.pedestrian.robot",
    "walker.pedestrian.g1_unitree",
    "walker.pedestrian.dog",
}


# ── API-compat stub (no longer used internally) ──────────────────────────────

class PassageQueueConfig:
    """Kept for import compatibility — passage logic has been removed."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── RoiMap ───────────────────────────────────────────────────────────────────

class RoiMap:
    """Lightweight wrapper around gridmap_roi.npz."""

    def __init__(self, npz_path: str, resolution_fallback: float = 0.5) -> None:
        data = np.load(npz_path, allow_pickle=True)
        self.world_min = np.array(data["world_min"], dtype=np.float64)
        self.world_max = np.array(data["world_max"], dtype=np.float64)
        self.free_grid = np.array(data["free_grid"], dtype=np.uint8)
        self.roi_mask  = (np.array(data["roi_mask"], dtype=np.uint8)
                          if "roi_mask" in data
                          else np.ones_like(self.free_grid, dtype=np.uint8))
        try:
            import json as _json
            meta_raw = data["__meta__"] if "__meta__" in data else None
            if meta_raw is not None:
                if isinstance(meta_raw, np.ndarray):
                    meta_raw = meta_raw.item()
                if isinstance(meta_raw, bytes):
                    meta_raw = meta_raw.decode("utf-8", errors="ignore")
                meta = _json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw.strip() else {}
            else:
                meta = {}
            self.resolution = float(meta.get("resolution", resolution_fallback))
        except Exception:
            nrows, ncols = self.free_grid.shape
            self.resolution = float((self.world_max[0] - self.world_min[0]) / ncols)
        walkable = (self.free_grid > 0) & (self.roi_mask > 0)
        self._walkable_cells = np.argwhere(walkable)  # (N,2) [gy,gx]

    @property
    def walkable_cells(self):
        return self._walkable_cells

    def cell_to_world(self, gx: int, gy: int, z: float = 0.0,
                      ground_z: float = None) -> "carla.Location":
        _z = ground_z if ground_z is not None else z
        x = self.world_min[0] + (float(gx) + 0.5) * self.resolution
        y = self.world_min[1] + (float(gy) + 0.5) * self.resolution
        return carla.Location(x=float(x), y=float(y), z=float(_z))

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.floor((float(x) - float(self.world_min[0])) / self.resolution))
        gy = int(np.floor((float(y) - float(self.world_min[1])) / self.resolution))
        return gx, gy

    def is_world_walkable(self, x: float, y: float) -> bool:
        gx, gy = self.world_to_cell(x, y)
        h, w = self.free_grid.shape
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return False
        return bool(self.free_grid[gy, gx] > 0 and self.roi_mask[gy, gx] > 0)

    def nearest_walkable_cell(self, gx: int, gy: int) -> Tuple[int, int]:
        """Return the nearest walkable (gx, gy) to the given cell via BFS."""
        h, w = self.free_grid.shape
        walkable = (self.free_grid > 0) & (self.roi_mask > 0)
        if 0 <= gx < w and 0 <= gy < h and walkable[gy, gx]:
            return gx, gy
        from collections import deque
        visited = set()
        q = deque([(gx, gy)])
        visited.add((gx, gy))
        while q:
            cx, cy = q.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nx, ny = cx+dx, cy+dy
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                if 0 <= nx < w and 0 <= ny < h:
                    if walkable[ny, nx]:
                        return nx, ny
                    q.append((nx, ny))
        return max(0, min(gx, w-1)), max(0, min(gy, h-1))

    def sample_world_location(self, rng: random.Random,
                               z: float = 0.0, ground_z: float = None,
                               jitter: bool = True) -> "carla.Location":
        _z = ground_z if ground_z is not None else z
        gy, gx = self._walkable_cells[rng.randrange(len(self._walkable_cells))]
        x = self.world_min[0] + (float(gx) + 0.5) * self.resolution
        y = self.world_min[1] + (float(gy) + 0.5) * self.resolution
        if jitter:
            half = 0.45 * self.resolution
            x += rng.uniform(-half, half)
            y += rng.uniform(-half, half)
        return carla.Location(x=float(x), y=float(y), z=float(_z))


# ── Flow-point loader ────────────────────────────────────────────────────────

def _load_flow_points(
    json_path: str,
    roi_map,
    spawn_z: float = 0.5,
    min_points: int = 3,
) -> List[carla.Location]:
    with open(json_path) as f:
        d = json.load(f)
    pw = d.get("points_world", [])
    if isinstance(pw, list) and len(pw) >= min_points:
        return [carla.Location(x=float(p["x"]), y=float(p["y"]), z=spawn_z)
                for p in pw[:min_points + 1]]
    pg = d.get("points_grid", [])
    if isinstance(pg, list) and len(pg) >= min_points:
        return [roi_map.cell_to_world(int(gx), int(gy), z=spawn_z) for gx, gy in pg]
    raise RuntimeError(f"flow points json must have at least {min_points} points.")


# ── Difficulty presets ───────────────────────────────────────────────────────

DIFFICULTY_PRESETS: Dict[str, Dict] = {
    "easy": {
        "p_corner_flow":            0.3,
        "p_shadow_follower":          0.0,
        "shadow_delay_sec":           5.0,
        "max_per_seg":                2,
        "min_gap_sec":                8.0,
        "trunk_max_per_route":        1,
        "trunk_dispatch_interval_sec": 12.0,
    },
    "hard": {
        "p_corner_flow":            0.7,
        "p_shadow_follower":          0.5,
        "shadow_delay_sec":           5.0,
        "max_per_seg":                3,
        "min_gap_sec":                5.0,
        "trunk_max_per_route":        2,
        "trunk_dispatch_interval_sec": 8.0,
    },
    "boss": {
        "p_corner_flow":            1.0,
        "p_shadow_follower":          1.0,
        "shadow_delay_sec":           3.0,
        "max_per_seg":                4,
        "min_gap_sec":                3.0,
        "trunk_max_per_route":        3,
        "trunk_dispatch_interval_sec": 5.0,
    },
    "legend": {
        "p_corner_flow":            1.0,
        "p_shadow_follower":          1.0,
        "shadow_delay_sec":           1.5,
        "max_per_seg":                6,
        "min_gap_sec":                1.5,
        "trunk_max_per_route":        4,
        "trunk_dispatch_interval_sec": 3.0,
    },
}

# Target-look-alike distractors (reuse the target's appearance), legend only.
_NUM_LOOKALIKES: Dict[str, int] = {"easy": 0, "hard": 0, "boss": 0, "legend": 2}

# ── Agent dataclass ──────────────────────────────────────────────────────────

@dataclass
class CrowdAgent:
    walker: carla.Actor
    track_id: str
    speed: float
    target: carla.Location         # current final destination
    dest_loc: carla.Location       # forward endpoint (flow point B)
    origin_loc: carla.Location     # reverse endpoint  (flow point A)
    wps_fwd: List[Tuple[float, float, float]] = field(default_factory=list)
    wps_rev: List[Tuple[float, float, float]] = field(default_factory=list)
    wps: List[Tuple[float, float, float]] = field(default_factory=list)
    wp_idx: int = 0
    going_forward: bool = True
    loop_mode: bool = False          # True = circular P0→P1→P2→P0 loop
    # API compatibility — unused in simplified runtime
    is_interceptor: bool = False
    pair_endpoint: int = -1
    # Crowd scheduling
    agent_type: str = "distractor_orbit"  # target | distractor_orbit | corner_flow | shadow_follower | pending | trunk_walk | hidden_pending
    seg_idx: int = -1       # seg currently being walked (-1 = orbit)
    trunk_route_idx: int = -1   # which extra trunk route this NPC is on (-1 = none)
    seg_offset: int = 0     # rotation offset of this NPC's wps relative to global full_loop
    pending_start: float = 0.0   # wall-clock time to actually enter the seg (pending mode)
    # Zone token tracking (dispatched routes only — orbit NPCs are exempt)
    registered_zones: List[str] = field(default_factory=list)  # zone IDs this NPC holds tokens for
    # Global rescue — interval accumulated-distance check
    _rescue_check_t: float = field(default=0.0, repr=False)
    _rescue_accum_dist: float = field(default=0.0, repr=False)
    _pre_rescue_agent_type: str = field(default="", repr=False)
    # Debug
    is_currently_stuck: bool = False
    _last_pos: Optional[Tuple[float, float]] = field(default=None, repr=False)
    _stuck_since: float = field(default=0.0, repr=False)


# ── NpcRuntime ───────────────────────────────────────────────────────────────

class NpcRuntime:
    """
    Minimal NPC controller.

    Each NPC:
      1. Is teleported to its home flow point at setup.
      2. Follows a pre-planned A* + LoS-smoothed path to its destination.
      3. On arrival, reverses and walks back.
      4. Never replans mid-path.
    """

    def __init__(
        self,
        world: carla.World,
        grid_npz: str,
        flow_points_json: str,
        flow_mode: str = "free_3",   # accepted for API compat, ignored
        rng_seed: Optional[int] = None,
        spawn_z: float = 0.5,
        use_orca: bool = False,      # accepted for API compat, ignored
        passage_config=None,         # accepted for API compat, ignored
    ) -> None:
        self.world = world
        try:
            self._sim_dt = float(world.get_settings().fixed_delta_seconds or 0.1)
        except Exception:
            self._sim_dt = 0.1
        self.roi_map = RoiMap(grid_npz)
        self._trajectory_cache_path = os.path.join(
            os.path.dirname(os.path.abspath(grid_npz)),
            "npc_trajectory_cache.json",
        )
        self.spawn_z = spawn_z
        if rng_seed is None:
            rng_seed = int(time.time_ns() % 2_147_483_647)
        self.rng_seed = int(rng_seed)
        print(f"[NPC] rng_seed={self.rng_seed}")
        self.rng = random.Random(self.rng_seed)
        # Simulation clock advanced by step(); drives all schedule timers
        # (min gap, shadow delay, trunk dispatch, stuck/rescue) so episodes are
        # reproducible under a fixed seed.  Starts at an offset so 0.0 keeps
        # meaning "never" for last-entry / last-dispatch / sentinel timestamps.
        self._sim_t: float = 1000.0
        self.lookalike_track_ids: List[str] = []
        self.agents: List[CrowdAgent] = []
        self.flow_points = _load_flow_points(
            flow_points_json, self.roi_map, spawn_z=spawn_z, min_points=3
        )
        self._motion_active = False
        self._spawned = False
        self._episode_done = False
        self._difficulty: str = "easy"  # set via set_difficulty() before spawn
        # Params: set externally by the episode runner but only these are used.
        self.params: Dict = {
            "min_speed": 0.8,
            "max_speed": 1.4,
            "spawn_attempts_per_agent": 12,
            "spawn_min_clearance_m": 0.9,
            "arrive_threshold": 1.2,    # m — triggers bounce
            "waypoint_arrive_dist": 0.6, # m — advances to next waypoint
            "stuck_timeout_sec": 4.0,   # s — skip stuck waypoint after this long
            # SFM
            "sfm_radius": 2.5,
            "sfm_A": 2.0,
            "sfm_B": 0.8,
            "sfm_target_mult": 3.0,
            "sfm_robot_mult": 1.0,      # treat robot as a regular pedestrian
            # Crowd scheduling — difficulty knobs
            "p_corner_flow": 1.0,     # prob to spawn flow NPCs on other 2 segs when target enters a new seg
            "p_shadow_follower": 1.0,   # prob to spawn shadow NPC behind target
            "shadow_delay_sec": 3.0,    # s — shadow departs this long after target
            "max_per_seg": 4,           # max concurrent pedestrians in one seg (all types)
            "min_gap_sec": 4.0,         # s — min interval between two entries into same seg
            "segment_count_radius_m": 1.25,  # distance to connector centreline counted as in-segment
            "capture_radius": 5.0,      # m — NPC this close to dest flow point → orbit capture
            # Trunk-walk knobs
            "trunk_max_per_route":         1,
            "trunk_dispatch_interval_sec": 12.0,
        }
        self._wall_dist_cache: Optional[np.ndarray] = None
        self._walkable_cache: Optional[np.ndarray] = None
        self._open_mask_cache: Optional[np.ndarray] = None
        # Corner-flow scheduling state
        self._seg_wps: List[List[Tuple[float, float, float]]] = []
        self._seg_boundaries: List[int] = []    # [0, len0, len0+len1] global offsets
        self._distractor_routes: List[List[Tuple[float, float, float]]] = []
        self._target_seg: int = -1              # seg target was on last step
        # Reserved/dispatched corner walkers.  The crowd occupancy check
        # adds every other pedestrian physically on the segment at query time.
        self._seg_occupancy: Dict[int, int] = {0: 0, 1: 0, 2: 0}
        self._seg_last_entry: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
        self._pending_spawns: List[Dict] = []   # deferred shadow-follower entries
        self._hidden_pending_spawns: List[Dict] = []
        self._hidden_release_clearance_m: float = 2.0
        self._hidden_reserved_locs: List[carla.Location] = []
        self._distractor_cache_path = os.path.join(
            os.path.dirname(os.path.abspath(grid_npz)),
            "distractor_routes_cache.json",
        )
        self._extra_routes_cache_path = os.path.join(
            os.path.dirname(os.path.abspath(grid_npz)),
            "extra_routes.json",
        )
        self._trunk_routes: List[List[Tuple[float, float, float]]] = []
        self._trunk_occupancy: Dict[int, int] = {}
        self._trunk_last_dispatch: Dict[int, float] = {}
        # Bottleneck zones — token-based capacity system
        self._zones: list = []
        self._zone_active: Dict[str, int] = {}   # zone_id → count of dispatched NPCs holding a token
        self._zones_path = os.path.join(
            os.path.dirname(os.path.abspath(grid_npz)),
            "bottleneck_zones.json",
        )

    # ── Map helpers ──────────────────────────────────────────────────────────

    def _walkable_mask(self) -> np.ndarray:
        if self._walkable_cache is None:
            fg = self.roi_map.free_grid
            rm = self.roi_map.roi_mask
            self._walkable_cache = ((fg > 0) | (rm > 0)).astype(bool)
        return self._walkable_cache

    def _get_wall_dist(self) -> np.ndarray:
        """BFS distance-to-nearest-wall for each walkable cell."""
        if self._wall_dist_cache is not None:
            return self._wall_dist_cache
        walkable = self._walkable_mask()
        h, w = walkable.shape
        dist = np.full((h, w), 999.0, dtype=np.float32)
        queue: List[Tuple[int, int]] = []
        nbrs8 = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for y in range(h):
            for x in range(w):
                if not walkable[y, x]:
                    continue
                for dx, dy in nbrs8:
                    nx, ny = x+dx, y+dy
                    if not (0 <= nx < w and 0 <= ny < h) or not walkable[ny, nx]:
                        dist[y, x] = 0.0
                        queue.append((x, y))
                        break
        bfs = list(queue)
        while bfs:
            cx, cy = bfs.pop(0)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = cx+dx, cy+dy
                if 0 <= nx < w and 0 <= ny < h and walkable[ny, nx]:
                    nd = dist[cy, cx] + 1.0
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        bfs.append((nx, ny))
        self._wall_dist_cache = dist
        print(f"[NPC] wall_dist computed, max_clearance={dist[walkable].max():.1f} cells")
        return dist

    def _astar_cells(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iters: int = 30000,
        wall_w: float = 0.5,
        walkable_mask: Optional[np.ndarray] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """8-neighbour A* with soft wall-clearance penalty."""
        walkable = walkable_mask if walkable_mask is not None else self._walkable_mask()
        h, w = walkable.shape

        def valid(gx: int, gy: int) -> bool:
            return 0 <= gx < w and 0 <= gy < h and bool(walkable[gy, gx])

        sx, sy = start
        gx, gy = goal
        if not valid(sx, sy):
            sx, sy = self.roi_map.nearest_walkable_cell(sx, sy)
        if not valid(gx, gy):
            gx, gy = self.roi_map.nearest_walkable_cell(gx, gy)
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        def heur(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return float(np.hypot(a[0]-b[0], a[1]-b[1]))

        open_heap = [(0.0, (sx, sy))]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
        wall_dist = self._get_wall_dist()
        WALL_EPS = 0.5
        nbrs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
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
            for dx, dy in nbrs:
                nx, ny = cx+dx, cy+dy
                if not valid(nx, ny):
                    continue
                step = 1.4142 if (dx != 0 and dy != 0) else 1.0
                clearance = float(wall_w / (float(wall_dist[ny, nx]) + WALL_EPS))
                ng = gscore[cur] + step + clearance
                nxt = (nx, ny)
                if ng < gscore.get(nxt, 1e18):
                    came_from[nxt] = cur
                    gscore[nxt] = ng
                    heapq.heappush(open_heap, (ng + heur(nxt, (gx, gy)), nxt))
        return None

    def _los_smooth_path(
        self,
        cell_path: List[Tuple[int, int]],
        margin: int = 1,
        walkable_mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        """Greedy LoS shortcutting — removes redundant intermediate cells."""
        if not cell_path or len(cell_path) <= 2:
            return cell_path
        walkable = walkable_mask if walkable_mask is not None else self._walkable_mask()
        wall_dist = self._get_wall_dist()
        h, w = walkable.shape

        def los_clear(ax: int, ay: int, bx: int, by: int) -> bool:
            dx, dy = abs(bx-ax), abs(by-ay)
            sx = 1 if bx > ax else -1
            sy_s = 1 if by > ay else -1
            x, y = ax, ay
            err = dx - dy
            for _ in range(max(dx, dy) + 1):
                if not (0 <= x < w and 0 <= y < h):
                    return False
                if not walkable[y, x]:
                    return False
                if wall_dist[y, x] < margin:
                    return False
                if x == bx and y == by:
                    return True
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy; x += sx
                if e2 < dx:
                    err += dx; y += sy_s
            return True

        result = [cell_path[0]]
        anchor = 0
        while anchor < len(cell_path) - 1:
            ax, ay = cell_path[anchor]
            best = anchor + 1
            for j in range(len(cell_path)-1, anchor, -1):
                bx, by = cell_path[j]
                if los_clear(ax, ay, bx, by):
                    best = j
                    break
            result.append(cell_path[best])
            anchor = best
        return result

    # ── Spawn ────────────────────────────────────────────────────────────────

    def set_difficulty(self, level: str) -> None:
        """Apply a named difficulty preset. Safe to call before or after spawn."""
        preset = DIFFICULTY_PRESETS.get(level.lower())
        if preset is None:
            print(f"[NPC] unknown difficulty '{level}', choices: {list(DIFFICULTY_PRESETS)}")
            return
        self._difficulty = level.lower()
        self.params.update(preset)
        print(f"[NPC] difficulty → {level.upper()} "
              f"(p_flow={preset['p_corner_flow']:.1f} "
              f"p_shadow={preset['p_shadow_follower']:.1f} "
              f"max_per_seg={preset['max_per_seg']} "
              f"min_gap={preset['min_gap_sec']:.1f}s)")

    def _sample_spawn_location_near_flow_point(self, flow_point) -> carla.Location:
        cand = carla.Location(x=flow_point.x, y=flow_point.y, z=flow_point.z + 0.5)
        for _ in range(30):
            jx = self.rng.uniform(-1.5, 1.5)
            jy = self.rng.uniform(-1.5, 1.5)
            c = carla.Location(x=flow_point.x + jx, y=flow_point.y + jy, z=flow_point.z + 0.5)
            if self.roi_map.is_world_walkable(c.x, c.y):
                return c
        return cand

    def _spawn_location_is_clear(self, loc: carla.Location, min_clearance_m: float) -> bool:
        for ag in self.agents:
            try:
                ag_loc = ag.walker.get_location()
            except Exception:
                continue
            if float(np.hypot(float(loc.x) - float(ag_loc.x), float(loc.y) - float(ag_loc.y))) < min_clearance_m:
                return False
        return True

    def spawn(self, num_walkers: int) -> None:
        if self._spawned:
            return
        lib = self.world.get_blueprint_library()
        all_bps = [bp for bp in lib.filter("walker.pedestrian.*")
                   if bp.id not in _NON_HUMAN_WALKER_BLUEPRINTS]
        if not all_bps:
            all_bps = [lib.find("walker.pedestrian.0001")]
        for bp in all_bps:
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")

        # Always assign each walker (target + distractors) a unique blueprint.
        # If the requested count exceeds the pool, raise — duplicate appearances
        # would confuse perception/ReID baselines.
        if num_walkers > len(all_bps):
            raise RuntimeError(
                f"[NPC] num_walkers={num_walkers} exceeds blueprint pool "
                f"({len(all_bps)}); each NPC needs a unique appearance.")
        chosen_bps = self.rng.sample(all_bps, num_walkers)
        # Legend: some distractors reuse the target's appearance (target = N01 = index 0).
        n_look = min(int(_NUM_LOOKALIKES.get(self._difficulty, 0)), max(0, num_walkers - 1))
        lookalike_idx = set(self.rng.sample(range(1, num_walkers), n_look)) if n_look > 0 else set()
        for li in lookalike_idx:
            chosen_bps[li] = chosen_bps[0]
        self.lookalike_track_ids = []

        n_pts = min(3, len(self.flow_points))
        failed = 0
        spawn_attempts = int(max(1, self.params.get("spawn_attempts_per_agent", 12)))
        min_clearance_m = float(max(0.1, self.params.get("spawn_min_clearance_m", 0.9)))
        for i in range(num_walkers):
            bp = chosen_bps[i]
            fp = self.flow_points[i % n_pts]
            actor = None
            cand = carla.Location(x=fp.x, y=fp.y, z=fp.z + 0.5)
            for _ in range(spawn_attempts):
                cand = self._sample_spawn_location_near_flow_point(fp)
                if not self._spawn_location_is_clear(cand, min_clearance_m):
                    continue
                actor = self.world.try_spawn_actor(bp, carla.Transform(cand))
                if actor is not None:
                    break
            if actor is None:
                failed += 1
                continue
            tid = f"N{i+1:02d}"
            ag = CrowdAgent(
                walker=actor,
                track_id=tid,
                speed=float(self.rng.uniform(
                    self.params["min_speed"], self.params["max_speed"])),
                target=cand,
                dest_loc=cand,
                origin_loc=cand,
            )
            self.agents.append(ag)
            if i in lookalike_idx:
                self.lookalike_track_ids.append(tid)
        self._spawned = True
        print(f"[NPC] spawned={len(self.agents)} failed={failed} "
              f"difficulty={self._difficulty} bp_pool={len(all_bps)} "
              f"lookalikes={self.lookalike_track_ids}")

    # ── Scheduled-flow setup ─────────────────────────────────────────────────

    # ── Distractor helpers ───────────────────────────────────────────────────

    def _open_mask(self, min_dist: int = 3) -> np.ndarray:
        """Cells that are walkable AND at least min_dist cells away from any wall."""
        if self._open_mask_cache is not None:
            return self._open_mask_cache
        wall_dist = self._get_wall_dist()
        walkable  = self._walkable_mask()
        self._open_mask_cache = walkable & (wall_dist >= min_dist)
        n_open = int(self._open_mask_cache.sum())
        print(f"[Distractor] open_mask: {n_open} cells (wall_dist≥{min_dist})")
        return self._open_mask_cache

    def _find_open_anchors(
        self,
        fp,
        open_mask: np.ndarray,
        n: int = 3,
        radii: Tuple = (5, 8, 11, 15),
        n_angles: int = 12,
    ) -> Optional[List[Tuple[float, float]]]:
        """Find n well-separated open cells near flow-point fp."""
        H, W = open_mask.shape
        candidates = []
        for ang in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            for r in radii:
                wx = fp.x + r * float(np.cos(ang))
                wy = fp.y + r * float(np.sin(ang))
                gx = int(np.floor((wx - self.roi_map.world_min[0]) / self.roi_map.resolution))
                gy = int(np.floor((wy - self.roi_map.world_min[1]) / self.roi_map.resolution))
                if 0 <= gx < W and 0 <= gy < H and open_mask[gy, gx]:
                    candidates.append((wx, wy))
                    break
        if len(candidates) < n:
            return None
        # Greedy farthest-point selection for spread
        selected = [candidates[0]]
        remaining = candidates[1:]
        while len(selected) < n and remaining:
            best_d, best_c = -1.0, None
            for c in remaining:
                d = min(float(np.hypot(c[0]-s[0], c[1]-s[1])) for s in selected)
                if d > best_d:
                    best_d, best_c = d, c
            if best_c is None:
                break
            selected.append(best_c)
            remaining.remove(best_c)
        return selected if len(selected) == n else None

    def _plan_distractor_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        open_mask: np.ndarray,
        z: float = 0.5,
    ) -> List[Tuple[float, float, float]]:
        gx_s, gy_s = self.roi_map.world_to_cell(a[0], a[1])
        gx_g, gy_g = self.roi_map.world_to_cell(b[0], b[1])
        cell_path = self._astar_cells(
            (gx_s, gy_s), (gx_g, gy_g), wall_w=0.5, walkable_mask=open_mask)
        if cell_path is None or len(cell_path) < 2:
            return [(float(a[0]), float(a[1]), z), (float(b[0]), float(b[1]), z)]
        cell_path = self._los_smooth_path(cell_path, margin=1, walkable_mask=open_mask)
        wps = []
        for cx, cy in cell_path:
            wp = self.roi_map.cell_to_world(cx, cy, z=z)
            wps.append((float(wp.x), float(wp.y), float(wp.z)))
        return wps

    def _generate_distractor_routes(self) -> List[List[Tuple[float, float, float]]]:
        open_mask = self._open_mask(min_dist=3)
        routes = []
        for fi, fp in enumerate(self.flow_points[:3]):
            anchors = self._find_open_anchors(fp, open_mask)
            if anchors is None:
                print(f"[Distractor] P{fi}: not enough open cells, skipping route")
                continue
            # Build triangle loop A→B→C→A
            full: List[Tuple[float, float, float]] = []
            for ai in range(len(anchors)):
                seg = self._plan_distractor_segment(
                    anchors[ai], anchors[(ai + 1) % len(anchors)],
                    open_mask, z=self.spawn_z)
                full.extend(seg)
            routes.append(full)
            pts_str = "  ".join(f"({a[0]:.1f},{a[1]:.1f})" for a in anchors)
            print(f"[Distractor] P{fi} route: {len(full)} wps  anchors={pts_str}")
        return routes

    def _load_distractor_cache(self) -> Optional[List[List[Tuple[float, float, float]]]]:
        if not os.path.exists(self._distractor_cache_path):
            return None
        try:
            with open(self._distractor_cache_path) as f:
                data = json.load(f)
            routes = [[(float(p[0]), float(p[1]), float(p[2])) for p in r]
                      for r in data["routes"]]
            print(f"[Distractor] loaded cache: {self._distractor_cache_path}")
            for i, r in enumerate(routes):
                print(f"[Distractor]   route{i}: {len(r)} waypoints")
            return routes
        except Exception as e:
            print(f"[Distractor] cache load failed ({e}), re-generating")
            return None

    def _save_distractor_cache(self, routes: List[List[Tuple[float, float, float]]]) -> None:
        data = {"routes": [[[p[0], p[1], p[2]] for p in r] for r in routes]}
        tmp = self._distractor_cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._distractor_cache_path)
        print(f"[Distractor] cache saved: {self._distractor_cache_path}")

    def _load_extra_routes(self) -> None:
        """Load user-painted trunk routes from extra_routes.json."""
        if not os.path.exists(self._extra_routes_cache_path):
            print("[Trunk] no extra_routes.json found, trunk-walk disabled")
            return
        try:
            with open(self._extra_routes_cache_path) as f:
                data = json.load(f)
            routes_raw = data.get("routes", {})
            self._trunk_routes = []
            for k in sorted(routes_raw.keys(), key=lambda x: int(x)):
                pts = routes_raw[k]
                wps = [(float(p[0]), float(p[1]), float(self.spawn_z)) for p in pts]
                self._trunk_routes.append(wps)
            for ri in range(len(self._trunk_routes)):
                self._trunk_occupancy[ri]    = 0
                self._trunk_last_dispatch[ri] = 0.0
            print(f"[Trunk] loaded {len(self._trunk_routes)} trunk route(s) "
                  f"from {self._extra_routes_cache_path}")
            for ri, r in enumerate(self._trunk_routes):
                print(f"[Trunk]   trunk{ri}: {len(r)} wps "
                      f"start=({r[0][0]:.1f},{r[0][1]:.1f}) "
                      f"end=({r[-1][0]:.1f},{r[-1][1]:.1f})")
        except Exception as e:
            print(f"[Trunk] failed to load extra_routes.json: {e}")

    def _setup_distractors(self, target_track_id: str) -> None:
        """Assign fixed loop routes to all non-target NPCs."""
        distractors = [ag for ag in self.agents if ag.track_id != target_track_id]
        if not distractors:
            return
        routes = self._load_distractor_cache()
        if routes is None:
            routes = self._generate_distractor_routes()
            if routes:
                self._save_distractor_cache(routes)
            else:
                print("[Distractor] no routes available")
                return
        self._distractor_routes = routes  # stored for orbit-capture
        n_routes = len(routes)
        for i, ag in enumerate(distractors):
            route = routes[i % n_routes]
            if not route:
                continue
            # Stagger start positions across the route
            n = len(route)
            offset = (i * n // max(len(distractors), 1)) % n
            rotated = route[offset:] + route[:offset]
            start = rotated[0]
            home = carla.Location(x=start[0], y=start[1], z=start[2] + 0.3)
            ag.walker.set_location(home)
            ag.wps_fwd     = list(rotated)
            ag.wps_rev     = []
            ag.wps         = list(rotated)
            ag.wp_idx      = 0
            ag.going_forward = True
            ag.loop_mode   = True
            ag.agent_type  = "distractor_orbit"
            ag.seg_idx     = -1
            ag.dest_loc    = home
            ag.origin_loc  = home
            ag.target      = home
            ag.speed       = float(self.rng.uniform(
                self.params["min_speed"], self.params["max_speed"]))
            print(f"[Distractor] {ag.track_id} → route{i % n_routes} "
                  f"({len(rotated)} wps, offset={offset})")

    def reposition_distractors_away_from(
        self,
        reserved_locs: List[carla.Location],
        target_track_id: str = "",
        min_sep_m: float = 2.0,
    ) -> None:
        """Move initial distractor route offsets away from reserved spawn points."""
        reserved_xy = [
            (float(loc.x), float(loc.y))
            for loc in reserved_locs
            if loc is not None
        ]
        if not reserved_xy:
            return
        min_sep = max(float(min_sep_m), 0.0)

        def score_wp(wp: Tuple[float, float, float]) -> float:
            x, y = float(wp[0]), float(wp[1])
            return min(float(np.hypot(x - rx, y - ry)) for rx, ry in reserved_xy)

        for ag in self.agents:
            if ag.track_id == target_track_id or not ag.wps:
                continue
            current_score = score_wp(ag.wps[0])
            if current_score >= min_sep:
                continue

            best_idx = 0
            best_score = current_score
            chosen_idx: Optional[int] = None
            for idx, wp in enumerate(ag.wps):
                score = score_wp(wp)
                if score > best_score:
                    best_idx = idx
                    best_score = score
                if chosen_idx is None and score >= min_sep:
                    chosen_idx = idx
            idx = chosen_idx if chosen_idx is not None else best_idx
            if idx <= 0:
                continue

            rotated = ag.wps[idx:] + ag.wps[:idx]
            start = rotated[0]
            home = carla.Location(x=float(start[0]), y=float(start[1]), z=float(start[2]) + 0.3)
            ag.walker.set_location(home)
            ag.wps_fwd = list(rotated)
            ag.wps = list(rotated)
            ag.wp_idx = 0
            ag.dest_loc = home
            ag.origin_loc = home
            ag.target = home
            final_score = score_wp(rotated[0])
            print(
                f"[Distractor] {ag.track_id} initial offset adjusted "
                f"(reserved_sep={final_score:.2f}m)"
            )

    def hide_conflicting_initial_distractors(
        self,
        reserved_locs: List[carla.Location],
        target_track_id: str = "",
        min_sep_m: float = 2.0,
    ) -> None:
        """Hide non-target NPCs whose current initial point conflicts with reserved starts."""
        self._hidden_reserved_locs = [loc for loc in reserved_locs if loc is not None]
        self._hidden_release_clearance_m = max(float(min_sep_m), 0.0)
        reserved_xy = [(float(loc.x), float(loc.y)) for loc in self._hidden_reserved_locs]
        if not reserved_xy:
            return

        def nearest_reserved_dist(loc: carla.Location) -> float:
            return min(float(np.hypot(loc.x - rx, loc.y - ry)) for rx, ry in reserved_xy)

        now_t = self._sim_t
        for ag in self.agents:
            if ag.track_id == target_track_id or ag.agent_type == "hidden_pending":
                continue
            loc = ag.walker.get_location()
            sep = nearest_reserved_dist(loc)
            if sep >= self._hidden_release_clearance_m:
                continue
            route = list(ag.wps)
            if not route:
                route = [(float(loc.x), float(loc.y), float(loc.z))]
            self._queue_hidden_pending(
                ag,
                route,
                pending_type=ag.agent_type,
                start_at=now_t,
                seg_i=ag.seg_idx,
                trunk_route_idx=ag.trunk_route_idx,
                target_track_id=target_track_id,
                reason=f"initial conflict sep={sep:.2f}m",
                reserve_capacity=False,
            )

    def _push_from_walls(
        self,
        wps: List[Tuple[float, float, float]],
        min_dist: float = 2.0,
        iters: int = 5,
    ) -> List[Tuple[float, float, float]]:
        """Iteratively push interior waypoints away from walls along wall_dist gradient."""
        if len(wps) < 3:
            return wps
        walkable = self._walkable_mask()
        wall_dist_map = self._get_wall_dist()
        H, W = walkable.shape
        pts = list(wps)

        for _ in range(iters):
            new_pts = [pts[0]]
            for i in range(1, len(pts) - 1):
                wx, wy, wz = pts[i]
                gx = int(np.floor((wx - self.roi_map.world_min[0]) / self.roi_map.resolution))
                gy = int(np.floor((wy - self.roi_map.world_min[1]) / self.roi_map.resolution))
                gx = max(1, min(gx, W - 2))
                gy = max(1, min(gy, H - 2))
                d = float(wall_dist_map[gy, gx])
                if d < min_dist:
                    # Gradient of wall_dist (central difference)
                    dx = float(wall_dist_map[gy, gx + 1] - wall_dist_map[gy, gx - 1]) / 2.0
                    dy = float(wall_dist_map[gy + 1, gx] - wall_dist_map[gy - 1, gx]) / 2.0
                    g = float(np.hypot(dx, dy))
                    if g > 0.01:
                        push = (min_dist - d) * self.roi_map.resolution * 0.6
                        nx = wx + (dx / g) * push
                        ny = wy + (dy / g) * push
                        ngx, ngy = self.roi_map.world_to_cell(nx, ny)
                        if 0 <= ngx < W and 0 <= ngy < H and walkable[ngy, ngx]:
                            wx, wy = nx, ny
                new_pts.append((wx, wy, wz))
            new_pts.append(pts[-1])
            pts = new_pts
        return pts

    def _plan_segment(self, fi: int, ti: int) -> List[Tuple[float, float, float]]:
        fp_f = self.flow_points[fi]
        fp_t = self.flow_points[ti]
        gx_s, gy_s = self.roi_map.world_to_cell(fp_f.x, fp_f.y)
        gx_g, gy_g = self.roi_map.world_to_cell(fp_t.x, fp_t.y)
        cell_path = self._astar_cells((gx_s, gy_s), (gx_g, gy_g), wall_w=0.5)
        if cell_path is None or len(cell_path) < 2:
            print(f"[Sched] P{fi}→P{ti}: A* failed, using direct line")
            return [(fp_f.x, fp_f.y, fp_f.z), (fp_t.x, fp_t.y, fp_t.z)]
        # margin=3 → LoS shortcuts only allowed when all cells are ≥3 cells (1.5m) from walls
        cell_path = self._los_smooth_path(cell_path, margin=3)
        wps = []
        for cx, cy in cell_path:
            wp = self.roi_map.cell_to_world(cx, cy, z=fp_f.z)
            wps.append((float(wp.x), float(wp.y), float(wp.z)))
        # Push waypoints further from walls to create arc-like corners
        wps = self._push_from_walls(wps, min_dist=3.0, iters=8)
        print(f"[Sched] P{fi}→P{ti}: {len(wps)} waypoints")
        return wps

    def _load_trajectory_cache(self) -> Optional[List[List[Tuple[float, float, float]]]]:
        if not os.path.exists(self._trajectory_cache_path):
            return None
        try:
            with open(self._trajectory_cache_path) as f:
                data = json.load(f)
            segs = data["segments"]
            result = [[(float(p[0]), float(p[1]), float(p[2])) for p in seg]
                      for seg in segs]
            print(f"[Sched] loaded trajectory cache: {self._trajectory_cache_path}")
            for i, s in enumerate(result):
                print(f"[Sched]   seg{i}: {len(s)} waypoints")
            return result
        except Exception as e:
            print(f"[Sched] cache load failed ({e}), re-planning")
            return None

    def _save_trajectory_cache(self, seg_wps: List[List[Tuple[float, float, float]]]) -> None:
        data = {"segments": [[[p[0], p[1], p[2]] for p in seg] for seg in seg_wps]}
        tmp = self._trajectory_cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._trajectory_cache_path)
        print(f"[Sched] trajectory cache saved: {self._trajectory_cache_path}")

    def setup_scheduled_flow(
        self,
        dwell_s: float = 0.0,
        stagger_scale: float = 1.0,
        target_track_id: str = "",
    ) -> None:
        self._pending_spawns.clear()
        self._hidden_pending_spawns.clear()
        # Use cached trajectory if available, otherwise plan and save
        seg_wps = self._load_trajectory_cache()
        if seg_wps is None:
            seg_wps = [
                self._plan_segment(0, 1),
                self._plan_segment(1, 2),
                self._plan_segment(2, 0),
            ]
            self._save_trajectory_cache(seg_wps)
        # Store for corner-flow scheduling
        self._seg_wps = seg_wps
        self._seg_boundaries = [0, len(seg_wps[0]), len(seg_wps[0]) + len(seg_wps[1])]

        # Segment start offsets for staggered spawn
        seg_offsets = [0,
                       len(seg_wps[0]),
                       len(seg_wps[0]) + len(seg_wps[1])]
        full_loop = seg_wps[0] + seg_wps[1] + seg_wps[2]

        for i, ag in enumerate(self.agents):
            is_target_ag = (ag.track_id == target_track_id)
            # Stagger NPCs across the three starting positions
            # The target always runs the evaluation leg P0→P1. Distractors
            # remain staggered across the full three-segment loop.
            start_seg = 0 if is_target_ag else i % 3
            fp_start = self.flow_points[start_seg]
            home = carla.Location(x=fp_start.x, y=fp_start.y, z=fp_start.z + 0.5)
            for _ in range(16):
                jx = self.rng.uniform(-1.5, 1.5)
                jy = self.rng.uniform(-1.5, 1.5)
                c = carla.Location(x=fp_start.x + jx, y=fp_start.y + jy,
                                   z=fp_start.z + 0.5)
                if self.roi_map.is_world_walkable(c.x, c.y):
                    home = c
                    break
            ag.walker.set_location(home)

            # Rotate loop so NPC starts at its assigned flow point
            offset = seg_offsets[start_seg]
            rotated = (
                list(seg_wps[0])
                if is_target_ag
                else full_loop[offset:] + full_loop[:offset]
            )

            ag.wps_fwd    = list(rotated)
            ag.wps_rev    = []
            ag.wps        = list(rotated)
            ag.wp_idx     = 0
            ag.going_forward = True
            ag.seg_offset  = offset  # for seg detection in step()
            # Target walks P0→P1 once; distractors loop forever.
            ag.loop_mode  = not is_target_ag
            ag.dest_loc   = home
            ag.origin_loc = home
            ag.target     = home
            ag.speed      = float(self.rng.uniform(
                self.params["min_speed"], self.params["max_speed"]))
            if is_target_ag:
                ag.agent_type = "target"

        # Set up distractor routes for all non-target NPCs (completely independent)
        self._setup_distractors(target_track_id)
        # Load user-painted trunk routes
        self._load_extra_routes()
        # Load bottleneck zones
        self._load_zones()
        self._episode_done = False

    # ── Motion on/off ────────────────────────────────────────────────────────

    def set_motion_active(self, active: bool) -> None:
        self._motion_active = active
        if not active:
            for ag in self.agents:
                ctrl = carla.WalkerControl()
                ctrl.speed = 0.0
                ag.walker.apply_control(ctrl)

    def is_episode_done(self) -> bool:
        return self._episode_done

    # ── Corner-flow scheduling ─────────────────────────────────────────────

    def _which_seg(self, ag: "CrowdAgent") -> int:
        """Return 0/1/2 for which global seg this agent's current wp_idx falls in."""
        if not self._seg_boundaries or not self._seg_wps:
            return -1
        n_total = sum(len(s) for s in self._seg_wps)
        if n_total == 0:
            return -1
        global_idx = (ag.wp_idx + ag.seg_offset) % n_total
        b = self._seg_boundaries
        if global_idx < b[1]:
            return 0
        elif global_idx < b[2]:
            return 1
        return 2

    @staticmethod
    def _point_polyline_distance(
        x: float,
        y: float,
        wps: List[Tuple[float, float, float]],
    ) -> float:
        """Shortest planar distance from a point to a waypoint polyline."""
        if not wps:
            return float("inf")
        if len(wps) == 1:
            return float(np.hypot(x - wps[0][0], y - wps[0][1]))
        best = float("inf")
        for a, b in zip(wps, wps[1:]):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            vx, vy = bx - ax, by - ay
            denom = vx * vx + vy * vy
            if denom <= 1e-12:
                px, py = ax, ay
            else:
                t = max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / denom))
                px, py = ax + t * vx, ay + t * vy
            best = min(best, float(np.hypot(x - px, y - py)))
        return best

    def _physical_segment_for_xy(self, x: float, y: float) -> int:
        """Return the nearest connector segment, or -1 when outside all of them."""
        if not self._seg_wps:
            return -1
        distances = [self._point_polyline_distance(x, y, wps) for wps in self._seg_wps]
        nearest = int(np.argmin(distances))
        radius = float(self.params.get("segment_count_radius_m", 1.25))
        return nearest if distances[nearest] <= radius else -1

    def _unreserved_physical_occupancy(self, seg_i: int) -> int:
        """Count all visible pedestrians not already represented by a reservation."""
        count = 0
        for ag in self.agents:
            if ag.agent_type == "hidden_pending" or ag.seg_idx >= 0:
                continue
            loc = ag.walker.get_location()
            if self._physical_segment_for_xy(float(loc.x), float(loc.y)) == seg_i:
                count += 1
        return count

    def _segment_total_occupancy(self, seg_i: int) -> int:
        """Crowd count: reservations plus every other pedestrian on the segment."""
        return (self._seg_occupancy.get(seg_i, 0)
                + self._unreserved_physical_occupancy(seg_i))

    def _can_enter_seg(self, seg_i: int, now_t: float) -> bool:
        occ = self._segment_total_occupancy(seg_i)
        gap = now_t - self._seg_last_entry.get(seg_i, 0.0)
        return (occ < int(self.params.get("max_per_seg", 4))
                and gap >= float(self.params.get("min_gap_sec", 4.0)))

    def _route_start_tf(self, wps: List[Tuple[float, float, float]]) -> carla.Transform:
        start = wps[0]
        yaw = 0.0
        for wx, wy, _wz in wps[1:]:
            dx = float(wx) - float(start[0])
            dy = float(wy) - float(start[1])
            if float(np.hypot(dx, dy)) > 0.20:
                yaw = float(np.degrees(np.arctan2(dy, dx)))
                break
        return carla.Transform(
            carla.Location(x=float(start[0]), y=float(start[1]), z=float(start[2]) + 0.3),
            carla.Rotation(yaw=yaw),
        )

    def _hidden_holding_location(self, reserved_locs: List[carla.Location]) -> carla.Location:
        reserved_xy = [(float(loc.x), float(loc.y)) for loc in reserved_locs if loc is not None]
        if not reserved_xy:
            return self.roi_map.sample_world_location(self.rng, z=self.spawn_z + 0.5, jitter=True)
        best_score = -1.0
        best_cell = None
        for gy, gx in self.roi_map.walkable_cells:
            loc = self.roi_map.cell_to_world(int(gx), int(gy), z=self.spawn_z + 0.5)
            score = min(float(np.hypot(loc.x - rx, loc.y - ry)) for rx, ry in reserved_xy)
            if score > best_score:
                best_score = score
                best_cell = (int(gx), int(gy))
        if best_cell is None:
            return self.roi_map.sample_world_location(self.rng, z=self.spawn_z + 0.5, jitter=True)
        return self.roi_map.cell_to_world(best_cell[0], best_cell[1], z=self.spawn_z + 0.5)

    def _set_actor_hidden(self, ag: "CrowdAgent", hidden: bool, loc: Optional[carla.Location] = None) -> None:
        ctrl = carla.WalkerControl()
        ctrl.speed = 0.0
        try:
            ag.walker.apply_control(ctrl)
        except Exception:
            pass
        if loc is not None:
            try:
                ag.walker.set_location(loc)
            except Exception:
                pass
        for method_name, value in (
            ("set_actor_enable_collision", not hidden),
            ("set_hidden_in_game", hidden),
        ):
            method = getattr(ag.walker, method_name, None)
            if method is None:
                continue
            try:
                method(bool(value))
            except Exception:
                pass

    def _hidden_clear_to_release(
        self,
        wps: List[Tuple[float, float, float]],
        target_track_id: str,
        robot_actor=None,
    ) -> bool:
        if not wps:
            return False
        start = wps[0]
        reserved: List[carla.Location] = []
        target_actor = self.get_actor_by_track_id(target_track_id)
        if target_actor is not None:
            reserved.append(target_actor.get_location())
        if robot_actor is not None:
            reserved.append(robot_actor.get_location())
        if not reserved:
            reserved = list(self._hidden_reserved_locs)
        min_sep = max(float(self._hidden_release_clearance_m), 0.0)
        for loc in reserved:
            if float(np.hypot(float(start[0]) - loc.x, float(start[1]) - loc.y)) < min_sep:
                return False
        return True

    def _queue_hidden_pending(
        self,
        ag: "CrowdAgent",
        wps: List[Tuple[float, float, float]],
        pending_type: str,
        start_at: float,
        seg_i: int = -1,
        trunk_route_idx: int = -1,
        target_track_id: str = "",
        reason: str = "",
        reserve_capacity: bool = True,
    ) -> None:
        if not wps:
            return
        if ag.agent_type == "hidden_pending":
            return
        self._zone_unregister(ag)
        ag.agent_type = "hidden_pending"
        ag.seg_idx = int(seg_i)
        ag.trunk_route_idx = int(trunk_route_idx)
        ag.loop_mode = False
        ag.wps = []
        ag.wp_idx = 0
        ag._last_pos = None
        ag._stuck_since = 0.0
        ag.is_currently_stuck = False
        if reserve_capacity and pending_type in ("corner_flow", "shadow_follower") and seg_i >= 0:
            self._seg_occupancy[seg_i] = self._seg_occupancy.get(seg_i, 0) + 1
            self._seg_last_entry[seg_i] = start_at
        elif reserve_capacity and pending_type == "trunk_walk" and trunk_route_idx >= 0:
            self._trunk_occupancy[trunk_route_idx] = self._trunk_occupancy.get(trunk_route_idx, 0) + 1
            self._trunk_last_dispatch[trunk_route_idx] = start_at
        holding = self._hidden_holding_location(self._hidden_reserved_locs)
        self._set_actor_hidden(ag, True, holding)
        self._hidden_pending_spawns.append({
            "ag": ag,
            "wps": list(wps),
            "pending_type": str(pending_type),
            "seg_i": int(seg_i),
            "trunk_route_idx": int(trunk_route_idx),
            "start_at": float(start_at),
            "reserve_capacity": bool(reserve_capacity),
        })
        suffix = f" ({reason})" if reason else ""
        print(f"[HiddenPending] {ag.track_id} hidden as {pending_type}{suffix}")

    def _release_hidden_pending_spawns(
        self,
        target_track_id: str,
        robot_actor,
        now_t: float,
    ) -> None:
        remaining: List[Dict] = []
        for sp in self._hidden_pending_spawns:
            ag: CrowdAgent = sp["ag"]
            if ag.agent_type != "hidden_pending":
                continue
            wps = sp["wps"]
            pending_type = str(sp["pending_type"])
            seg_i = int(sp.get("seg_i", -1))
            trunk_route_idx = int(sp.get("trunk_route_idx", -1))
            if now_t < float(sp["start_at"]):
                remaining.append(sp)
                continue
            if not self._hidden_clear_to_release(wps, target_track_id, robot_actor):
                remaining.append(sp)
                continue
            if pending_type in ("corner_flow", "shadow_follower") and seg_i >= 0:
                if not sp.get("reserve_capacity", False) and not self._can_enter_seg(seg_i, now_t):
                    remaining.append(sp)
                    continue
                if not self._zone_route_ok(wps):
                    remaining.append(sp)
                    continue
                if not sp.get("reserve_capacity", False):
                    self._seg_occupancy[seg_i] = self._seg_occupancy.get(seg_i, 0) + 1
                    self._seg_last_entry[seg_i] = now_t
                ag.seg_idx = seg_i
                ag.trunk_route_idx = -1
                self._zone_register(ag, wps)
            elif pending_type == "trunk_walk" and trunk_route_idx >= 0:
                if not self._zone_route_ok(wps):
                    remaining.append(sp)
                    continue
                if not sp.get("reserve_capacity", False):
                    self._trunk_occupancy[trunk_route_idx] = self._trunk_occupancy.get(trunk_route_idx, 0) + 1
                    self._trunk_last_dispatch[trunk_route_idx] = now_t
                ag.seg_idx = -1
                ag.trunk_route_idx = trunk_route_idx
                self._zone_register(ag, wps)
            else:
                ag.seg_idx = -1
                ag.trunk_route_idx = -1
                ag.loop_mode = True

            tf = self._route_start_tf(wps)
            ag.walker.set_transform(tf)
            self._set_actor_hidden(ag, False)
            ag.agent_type = pending_type
            ag.wps = list(wps)
            ag.wps_fwd = list(wps)
            ag.wps_rev = []
            ag.wp_idx = 0
            ag.loop_mode = pending_type == "distractor_orbit"
            ag.dest_loc = tf.location
            ag.origin_loc = tf.location
            ag.target = tf.location
            ag.speed = float(self.rng.uniform(self.params["min_speed"], self.params["max_speed"]))
            ag._rescue_check_t = 0.0
            ag._rescue_accum_dist = 0.0
            print(f"[HiddenPending] {ag.track_id} released as {pending_type}")
        self._hidden_pending_spawns = remaining

    def _enter_seg(self, ag: "CrowdAgent", seg_i: int, now_t: float) -> bool:
        """Assign NPC to seg_i. Returns False if zone capacity blocks dispatch."""
        wps = self._seg_wps[seg_i]
        if not self._zone_route_ok(wps):
            return False
        start = wps[0]
        ag.walker.set_location(
            carla.Location(x=start[0], y=start[1], z=start[2] + 0.3))
        ag.wps      = list(wps)
        ag.wp_idx   = 0
        ag.loop_mode = False
        ag.seg_idx  = seg_i
        ag.speed    = float(self.rng.uniform(
            self.params["min_speed"], self.params["max_speed"]))
        self._seg_occupancy[seg_i] = self._seg_occupancy.get(seg_i, 0) + 1
        self._seg_last_entry[seg_i] = now_t
        self._zone_register(ag, wps)
        ag._rescue_check_t    = 0.0   # reset so dispatch teleport doesn't count as stuck
        ag._rescue_accum_dist = 0.0
        print(f"[CrowdFlow] {ag.track_id} enters seg{seg_i} as {ag.agent_type}")
        return True

    def _pick_orbit_npc(self, near_seg_i: int, target_track_id: str) -> Optional["CrowdAgent"]:
        """Pick the orbit NPC closest to the start of near_seg_i."""
        candidates = [ag for ag in self.agents
                      if ag.track_id != target_track_id
                      and ag.agent_type == "distractor_orbit"]
        if not candidates or not self._seg_wps:
            return None
        sx, sy = self._seg_wps[near_seg_i][0][0], self._seg_wps[near_seg_i][0][1]
        return min(candidates, key=lambda a: float(np.hypot(
            a.walker.get_location().x - sx,
            a.walker.get_location().y - sy)))

    def _capture_to_orbit(self, ag: "CrowdAgent", orbit_idx: int) -> None:
        """Re-assign NPC to the distractor orbit associated with orbit_idx."""
        if ag.seg_idx >= 0:
            self._seg_occupancy[ag.seg_idx] = max(
                0, self._seg_occupancy.get(ag.seg_idx, 0) - 1)
        ag.seg_idx = -1
        ag.agent_type = "distractor_orbit"
        # Release all zone tokens so future dispatches can use these zones
        self._zone_unregister(ag)
        if not self._distractor_routes or orbit_idx >= len(self._distractor_routes):
            ag.loop_mode = True
            return
        route = self._distractor_routes[orbit_idx]
        loc = ag.walker.get_location()
        best_i, best_d = 0, float("inf")
        for i, (rx, ry, _rz) in enumerate(route):
            d = float(np.hypot(loc.x - rx, loc.y - ry))
            if d < best_d:
                best_d, best_i = d, i
        ag.wps      = route[best_i:] + route[:best_i]
        ag.wp_idx   = 0
        ag.loop_mode = True
        print(f"[CrowdFlow] {ag.track_id} captured into orbit{orbit_idx}")

    def _handle_seg_transition(
        self, new_seg: int, target_track_id: str, now_t: float
    ) -> None:
        """Called when target enters a new seg; dispatches corner-flow and shadow NPCs."""
        p_flow   = float(self.params.get("p_corner_flow",   1.0))
        p_shadow = float(self.params.get("p_shadow_follower", 1.0))
        delay    = float(self.params.get("shadow_delay_sec",  3.0))
        other_segs = [s for s in range(3) if s != new_seg]

        # Corner flow: one NPC per other seg
        if self.rng.random() < p_flow:
            for seg_i in other_segs:
                if not self._can_enter_seg(seg_i, now_t):
                    continue
                ag = self._pick_orbit_npc(seg_i, target_track_id)
                if ag is None:
                    continue
                self._queue_hidden_pending(
                    ag,
                    self._seg_wps[seg_i],
                    pending_type="corner_flow",
                    start_at=now_t,
                    seg_i=seg_i,
                    target_track_id=target_track_id,
                    reason=f"corner seg{seg_i}",
                )

        # Shadow follower: delayed entry on the same seg as target
        if self.rng.random() < p_shadow:
            # Reserve an NPC now; it will enter the seg after the delay
            ag = self._pick_orbit_npc(new_seg, target_track_id)
            if ag is not None and self._can_enter_seg(new_seg, now_t):
                self._queue_hidden_pending(
                    ag,
                    self._seg_wps[new_seg],
                    pending_type="shadow_follower",
                    start_at=now_t + delay,
                    seg_i=new_seg,
                    target_track_id=target_track_id,
                    reason=f"shadow seg{new_seg}",
                )
                print(f"[CrowdFlow] {ag.track_id} scheduled shadow on seg{new_seg} "
                      f"in {delay:.1f}s")

    def _process_pending_spawns(self, target_track_id: str, now_t: float) -> None:
        remaining = []
        for sp in self._pending_spawns:
            ag: CrowdAgent = sp["ag"]
            seg_i: int     = sp["seg_i"]
            if now_t >= sp["start_at"]:
                if ag.agent_type == "pending" and self._can_enter_seg(seg_i, now_t):
                    ag.agent_type = "shadow_follower"
                    if not self._enter_seg(ag, seg_i, now_t):
                        # Zone full — return to orbit
                        ag.agent_type = "distractor_orbit"
                        ag.seg_idx    = -1
                        ag.loop_mode  = True
                else:
                    # Cancelled — return to orbit
                    ag.agent_type = "distractor_orbit"
                    ag.seg_idx    = -1
                    ag.loop_mode  = True
            else:
                remaining.append(sp)
        self._pending_spawns = remaining

    def _check_captures(self, now_t: float) -> None:
        """Capture corner_flow / shadow_follower / trunk_walk / rescue NPCs that finished their path."""
        capture_r = float(self.params.get("capture_radius", 5.0))
        for ag in self.agents:
            if ag.agent_type in ("corner_flow", "shadow_follower"):
                if ag.seg_idx < 0:
                    continue
                dest_fp_idx = (ag.seg_idx + 1) % 3
                dest_fp     = self.flow_points[dest_fp_idx]
                loc         = ag.walker.get_location()
                reached = (ag.wp_idx >= len(ag.wps)
                           or float(np.hypot(loc.x - dest_fp.x,
                                             loc.y - dest_fp.y)) < capture_r)
                if reached:
                    self._capture_to_orbit(ag, dest_fp_idx)

            elif ag.agent_type == "trunk_walk":
                if ag.wp_idx < len(ag.wps):
                    continue
                ri = ag.trunk_route_idx
                self._trunk_occupancy[ri] = max(0, self._trunk_occupancy.get(ri, 0) - 1)
                ag.trunk_route_idx = -1
                # Return to nearest distractor orbit
                if self._distractor_routes:
                    loc = ag.walker.get_location()
                    best_i = min(range(len(self._distractor_routes)),
                                 key=lambda i: float(np.hypot(
                                     loc.x - self._distractor_routes[i][0][0],
                                     loc.y - self._distractor_routes[i][0][1])))
                    self._capture_to_orbit(ag, best_i)   # also resets zone state
                    print(f"[Trunk] {ag.track_id} finished trunk{ri} → orbit{best_i}")
                else:
                    self._zone_unregister(ag)
                    ag.agent_type = "distractor_orbit"
                    ag.loop_mode  = True

            elif ag.agent_type == "rescue":
                if ag.wp_idx < len(ag.wps):
                    continue
                # Rescue NPC arrived at open point — return to orbit
                loc = ag.walker.get_location()
                if self._distractor_routes:
                    best_i = min(range(len(self._distractor_routes)),
                                 key=lambda i: float(np.hypot(
                                     loc.x - self._distractor_routes[i][0][0],
                                     loc.y - self._distractor_routes[i][0][1])))
                    self._capture_to_orbit(ag, best_i)
                    print(f"[Rescue] {ag.track_id} rescued → orbit{best_i}")
                else:
                    ag.agent_type = "distractor_orbit"
                    ag.loop_mode  = True

    def _dispatch_trunk_walkers(self, target_track_id: str, now_t: float) -> None:
        """Periodically send orbit NPCs along each trunk route."""
        max_per  = int(self.params.get("trunk_max_per_route", 1))
        interval = float(self.params.get("trunk_dispatch_interval_sec", 12.0))
        if max_per <= 0 or not self._trunk_routes:
            return
        for ri, route in enumerate(self._trunk_routes):
            if not route:
                continue
            if self._trunk_occupancy.get(ri, 0) >= max_per:
                continue
            if now_t - self._trunk_last_dispatch.get(ri, 0.0) < interval:
                continue
            candidates = [ag for ag in self.agents
                          if ag.track_id != target_track_id
                          and ag.agent_type == "distractor_orbit"]
            if not candidates:
                continue
            # Pick orbit NPC closest to route start (or end, if going reversed)
            sx, sy = route[0][0], route[0][1]
            ag = min(candidates, key=lambda a: float(np.hypot(
                a.walker.get_location().x - sx,
                a.walker.get_location().y - sy)))
            # Randomise direction for variety
            wps = list(route) if self.rng.random() < 0.5 else list(reversed(route))
            if not self._zone_route_ok(wps):
                continue  # zone(s) at capacity — skip this route this tick
            self._queue_hidden_pending(
                ag,
                wps,
                pending_type="trunk_walk",
                start_at=now_t,
                trunk_route_idx=ri,
                target_track_id=target_track_id,
                reason=f"trunk{ri}",
            )
            print(f"[Trunk] {ag.track_id} queued hidden on trunk{ri} "
                  f"({len(wps)} wps, fwd={'yes' if wps[0] == route[0] else 'rev'})")

    # ── Bottleneck zone — simple token system ────────────────────────────────
    # Only *dispatched* NPCs (corner_flow, trunk_walk, shadow_follower) hold
    # zone tokens.  This is separate from crowd segment capacity, which is
    # enforced for every pedestrian by _segment_total_occupancy().

    def _load_zones(self) -> None:
        if not os.path.exists(self._zones_path):
            print("[Zone] no bottleneck_zones.json found, zone tokens disabled")
            return
        with open(self._zones_path) as f:
            d = json.load(f)
        self._zones = d.get("zones", [])
        for z in self._zones:
            self._zone_active[z["id"]] = 0
        print(f"[Zone] loaded {len(self._zones)} zone(s) "
              f"(narrow cap={_ZONE_CAP_NARROW}, junction cap={_ZONE_CAP_JUNCTION})")

    def _point_in_zone(self, x: float, y: float, zone: dict) -> bool:
        if zone["shape"] == "circle":
            return float(np.hypot(x - zone["x"], y - zone["y"])) <= zone["radius"]
        verts = zone["vertices"]
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]; xj, yj = verts[j]
            if (yi > y) != (yj > y):
                if x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-9) + xi:
                    inside = not inside
            j = i
        return inside

    def _zones_on_wps(self, wps: List[Tuple]) -> List[str]:
        """Return unique zone IDs whose area any waypoint in wps falls inside."""
        seen: List[str] = []
        for pt in wps:
            wx, wy = float(pt[0]), float(pt[1])
            for zone in self._zones:
                zid = zone["id"]
                if zid not in seen and self._point_in_zone(wx, wy, zone):
                    seen.append(zid)
        return seen

    def _zone_cap(self, zone_id: str) -> int:
        zone = next((z for z in self._zones if z["id"] == zone_id), None)
        if zone is None:
            return _ZONE_CAP_JUNCTION
        return _ZONE_CAP_NARROW if zone["type"] == "narrow" else _ZONE_CAP_JUNCTION

    def _zone_route_ok(self, wps: List[Tuple]) -> bool:
        """True only if every zone on this route has a free token slot."""
        if not self._zones:
            return True
        for zid in self._zones_on_wps(wps):
            if self._zone_active.get(zid, 0) >= self._zone_cap(zid):
                return False
        return True

    def _zone_register(self, ag: "CrowdAgent", wps: List[Tuple]) -> None:
        """Claim tokens for all zones this route passes through."""
        for zid in self._zones_on_wps(wps):
            self._zone_active[zid] = self._zone_active.get(zid, 0) + 1
            if zid not in ag.registered_zones:
                ag.registered_zones.append(zid)

    def _zone_unregister(self, ag: "CrowdAgent") -> None:
        """Release all zone tokens this NPC holds."""
        for zid in ag.registered_zones:
            self._zone_active[zid] = max(0, self._zone_active.get(zid, 0) - 1)
        ag.registered_zones = []

    # ── Global rescue ─────────────────────────────────────────────────────────

    def _los_clear_rescue(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Bresenham LoS check: True if every cell along the line is walkable."""
        walkable = self._walkable_mask()
        H, W = walkable.shape
        res = self.roi_map.resolution
        gx0 = int(np.floor((x0 - self.roi_map.world_min[0]) / res))
        gy0 = int(np.floor((y0 - self.roi_map.world_min[1]) / res))
        gx1 = int(np.floor((x1 - self.roi_map.world_min[0]) / res))
        gy1 = int(np.floor((y1 - self.roi_map.world_min[1]) / res))
        dx, dy = abs(gx1 - gx0), abs(gy1 - gy0)
        sx = 1 if gx1 > gx0 else -1
        sy = 1 if gy1 > gy0 else -1
        cx, cy = gx0, gy0
        err = dx - dy
        for _ in range(dx + dy + 1):
            if not (0 <= cx < W and 0 <= cy < H) or not walkable[cy, cx]:
                return False
            if cx == gx1 and cy == gy1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; cx += sx
            if e2 < dx:
                err += dx; cy += sy
        return True

    def _find_rescue_point(self, x: float, y: float, track_id: str = "") -> Optional[Tuple[float, float]]:
        """Scan outward for a nearby open cell with clear LoS from (x,y).
        Start angle is randomised per track_id so multiple stuck NPCs spread out."""
        open_mask = self._open_mask(min_dist=_RESCUE_OPEN_DIST)
        H, W = open_mask.shape
        # Deterministic but per-NPC angle offset so each NPC looks in a different direction
        angle_offset = (hash(track_id) % 16) * (2 * np.pi / 16)
        for r in [4, 7, 11, 16]:
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False) + angle_offset
            for ang in angles:
                wx = x + r * float(np.cos(ang))
                wy = y + r * float(np.sin(ang))
                gx = int(np.floor((wx - self.roi_map.world_min[0]) / self.roi_map.resolution))
                gy = int(np.floor((wy - self.roi_map.world_min[1]) / self.roi_map.resolution))
                if 0 <= gx < W and 0 <= gy < H and open_mask[gy, gx]:
                    if self._los_clear_rescue(x, y, wx, wy):
                        return float(wx), float(wy)
        return None

    def _trigger_rescue(self, ag: "CrowdAgent") -> None:
        """Navigate stuck NPC to a nearby open cell, then return it to orbit."""
        loc = ag.walker.get_location()
        pt = self._find_rescue_point(loc.x, loc.y, track_id=ag.track_id)
        if pt is None:
            return
        # Release zone tokens and seg occupancy — NPC is leaving its route
        self._zone_unregister(ag)
        if ag.seg_idx >= 0:
            self._seg_occupancy[ag.seg_idx] = max(
                0, self._seg_occupancy.get(ag.seg_idx, 0) - 1)
        if ag.trunk_route_idx >= 0:
            self._trunk_occupancy[ag.trunk_route_idx] = max(
                0, self._trunk_occupancy.get(ag.trunk_route_idx, 0) - 1)
        # Save type for logging
        ag._pre_rescue_agent_type = ag.agent_type
        # Switch to rescue mode
        ag.agent_type      = "rescue"
        ag.seg_idx         = -1
        ag.trunk_route_idx = -1
        ag.loop_mode       = False
        ag.wps             = [(pt[0], pt[1], float(loc.z))]
        ag.wp_idx          = 0
        ag._stuck_since       = 0.0
        ag._rescue_check_t    = 0.0   # reset interval so we don't re-trigger immediately
        ag._rescue_accum_dist = 0.0
        print(f"[Rescue] {ag.track_id} ({ag._pre_rescue_agent_type}) stuck "
              f"→ open pt ({pt[0]:.1f},{pt[1]:.1f})")

    # ── SFM repulsion ─────────────────────────────────────────────────────────

    def _sfm_repulsion(
        self,
        ax: float, ay: float,
        positions: Dict[str, Tuple[float, float]],
        self_id: str,
        target_track_id: str,
    ) -> Tuple[float, float]:
        """
        Repulsion vector for one distractor NPC.
        Other distractors: normal weight.
        Target: 3× weight so distractors always yield.
        """
        sfm_radius  = float(self.params.get("sfm_radius",  2.5))
        sfm_A       = float(self.params.get("sfm_A",       2.0))
        sfm_B       = float(self.params.get("sfm_B",       0.8))
        target_mult = float(self.params.get("sfm_target_mult", 3.0))
        robot_mult  = float(self.params.get("sfm_robot_mult", 1.0))

        rep_x = rep_y = 0.0
        for tid, (bx, by) in positions.items():
            if tid == self_id:
                continue
            dx = ax - bx
            dy = ay - by
            d  = float(np.hypot(dx, dy))
            if d < 0.01 or d > sfm_radius:
                continue
            if tid == target_track_id:
                mult = target_mult
            elif tid == _ROBOT_SFM_KEY:
                mult = robot_mult
            else:
                mult = 1.0
            mag  = mult * sfm_A * float(np.exp(-d / sfm_B))
            rep_x += (dx / d) * mag
            rep_y += (dy / d) * mag
        return rep_x, rep_y

    # ── Step (core loop) ─────────────────────────────────────────────────────

    def step(
        self,
        debug_draw_ids: bool = False,
        label_life: float = 0.1,
        robot_actor=None,
        target_track_id: str = "",
        manual_track_id: str = None,
        **kwargs,
    ) -> None:
        if not self._motion_active:
            return

        SARR     = float(self.params.get("waypoint_arrive_dist", 0.6))
        stuck_to = float(self.params.get("stuck_timeout_sec", 4.0))
        self._sim_t += float(self._sim_dt)
        now_t    = self._sim_t

        # Corner-flow housekeeping (once per step)
        self._process_pending_spawns(target_track_id, now_t)
        self._release_hidden_pending_spawns(target_track_id, robot_actor, now_t)
        self._check_captures(now_t)
        self._dispatch_trunk_walkers(target_track_id, now_t)

        # Snapshot positions for SFM (avoids using stale data mid-loop)
        positions: Dict[str, Tuple[float, float]] = {
            ag.track_id: (ag.walker.get_location().x, ag.walker.get_location().y)
            for ag in self.agents
            if ag.agent_type != "hidden_pending"
        }
        if robot_actor is not None:
            rloc = robot_actor.get_location()
            positions[_ROBOT_SFM_KEY] = (rloc.x, rloc.y)

        # Prevent multiple unreserved walkers from claiming the last free place
        # on the same segment during one simulation tick.
        segment_entry_claims: Dict[int, int] = {0: 0, 1: 0, 2: 0}

        for ag in self.agents:
            if ag.agent_type == "hidden_pending":
                continue
            loc = ag.walker.get_location()

            # ── Stuck detection (per-tick) ────────────────────────────────────
            moved = 0.0
            if ag._last_pos is not None:
                moved = float(np.hypot(loc.x - ag._last_pos[0],
                                       loc.y - ag._last_pos[1]))
                if moved < 0.05:
                    if ag._stuck_since == 0.0:
                        ag._stuck_since = now_t
                    elif now_t - ag._stuck_since > stuck_to:
                        if ag.wp_idx < len(ag.wps) - 1:
                            ag.wp_idx += 1
                        ag._stuck_since = 0.0
                        ag.is_currently_stuck = True
                else:
                    ag._stuck_since = 0.0
                    ag.is_currently_stuck = False
            ag._last_pos = (loc.x, loc.y)

            # ── Interval accumulated-distance check → rescue ──────────────────
            is_target = (ag.track_id == target_track_id)
            if not is_target and ag.agent_type not in ("rescue", "pending", "hidden_pending"):
                # Accumulate per-tick travel (reuse `moved` from stuck detection)
                ag._rescue_accum_dist += moved
                if ag._rescue_check_t == 0.0:
                    ag._rescue_check_t = now_t
                elif now_t - ag._rescue_check_t >= _RESCUE_INTERVAL:
                    if ag._rescue_accum_dist < _RESCUE_MIN_MOVE:
                        self._trigger_rescue(ag)
                    ag._rescue_check_t    = now_t
                    ag._rescue_accum_dist = 0.0

            # ── Advance waypoint index ────────────────────────────────────────
            while ag.wp_idx < len(ag.wps):
                wx, wy, _ = ag.wps[ag.wp_idx]
                if float(np.hypot(loc.x - wx, loc.y - wy)) <= SARR:
                    ag.wp_idx += 1
                else:
                    break

            # ── Detect target seg transition ──────────────────────────────────
            if ag.track_id == target_track_id and ag.agent_type == "target":
                cur_seg = self._which_seg(ag)
                if cur_seg != self._target_seg and cur_seg >= 0:
                    self._target_seg = cur_seg
                    self._handle_seg_transition(cur_seg, target_track_id, now_t)

            # ── Loop / bounce on end-of-path ──────────────────────────────────
            if ag.wp_idx >= len(ag.wps):
                # Target reaching P1 after the P0→P1 evaluation leg ends the episode.
                if ag.track_id == target_track_id and ag.agent_type == "target":
                    if not self._episode_done:
                        self._episode_done = True
                        print(f"[Episode] target {ag.track_id} completed P0→P1 "
                              f"evaluation leg → motion stopped")
                    self.set_motion_active(False)
                    return
                if ag.loop_mode:
                    ag.wp_idx = 0
                    ag.speed  = float(self.rng.uniform(
                        self.params["min_speed"], self.params["max_speed"]))
                else:
                    next_wps = ag.wps_fwd if not ag.going_forward else ag.wps_rev
                    if not next_wps:
                        pass  # capture_to_orbit will handle this next tick
                    else:
                        ag.going_forward = not ag.going_forward
                        if ag.going_forward:
                            ag.wps    = list(ag.wps_fwd)
                            ag.target = ag.dest_loc
                        else:
                            ag.wps    = list(ag.wps_rev)
                            ag.target = ag.origin_loc
                        ag.wp_idx = 0
                        ag.speed  = float(self.rng.uniform(
                            self.params["min_speed"], self.params["max_speed"]))

            # ── Compute goal direction ────────────────────────────────────────
            if ag.wp_idx < len(ag.wps):
                goal_x, goal_y = ag.wps[ag.wp_idx][0], ag.wps[ag.wp_idx][1]
            else:
                goal_x, goal_y = ag.target.x, ag.target.y

            dx = goal_x - loc.x
            dy = goal_y - loc.y
            d  = float(np.hypot(dx, dy))

            ctrl = carla.WalkerControl()
            if d > 0.01:
                dir_x = dx / d
                dir_y = dy / d

                # SFM: only for distractor NPCs, never for target
                if not is_target:
                    rx, ry = self._sfm_repulsion(
                        loc.x, loc.y, positions, ag.track_id, target_track_id)
                    dir_x += rx
                    dir_y += ry
                    dir_len = float(np.hypot(dir_x, dir_y))
                    if dir_len > 0.01:
                        dir_x /= dir_len
                        dir_y /= dir_len

                    # Orbit, trunk and rescue walkers do not hold corner
                    # reservations.  They are nevertheless pedestrians for
                    # the crowd count, so stop them at the boundary when the predicted
                    # segment is already at max_per_seg.  The target is never
                    # blocked by this crowd-generation rule.
                    if ag.seg_idx < 0:
                        current_seg = self._physical_segment_for_xy(loc.x, loc.y)
                        lookahead = max(self._sim_dt * float(ag.speed), 0.05)
                        next_seg = self._physical_segment_for_xy(
                            loc.x + dir_x * lookahead,
                            loc.y + dir_y * lookahead,
                        )
                        if next_seg >= 0 and next_seg != current_seg:
                            occupancy = (self._segment_total_occupancy(next_seg)
                                         + segment_entry_claims.get(next_seg, 0))
                            if occupancy >= int(self.params.get("max_per_seg", 4)):
                                ctrl.speed = 0.0
                                ag.walker.apply_control(ctrl)
                                continue
                            segment_entry_claims[next_seg] = (
                                segment_entry_claims.get(next_seg, 0) + 1)

                ctrl.direction = carla.Vector3D(dir_x, dir_y, 0.0)
                ctrl.speed     = float(ag.speed)
            else:
                ctrl.speed = 0.0
            ag.walker.apply_control(ctrl)

    # ── State queries ────────────────────────────────────────────────────────

    def get_states(self) -> List[NpcState]:
        out = []
        for ag in self.agents:
            if ag.agent_type == "hidden_pending":
                continue
            loc = ag.walker.get_location()
            vel = ag.walker.get_velocity()
            tf  = ag.walker.get_transform()
            spd = float(np.hypot(vel.x, vel.y))
            wp_x = wp_y = None
            if ag.wps and ag.wp_idx < len(ag.wps):
                wp_x, wp_y = ag.wps[ag.wp_idx][0], ag.wps[ag.wp_idx][1]
            out.append(NpcState(
                track_id=ag.track_id,
                actor_id=ag.walker.id,
                x=float(loc.x), y=float(loc.y), z=float(loc.z),
                vx=float(vel.x), vy=float(vel.y), vz=float(vel.z),
                yaw_deg=float(tf.rotation.yaw),
                speed=spd,
                npc_state="stuck" if ag.is_currently_stuck else "transit",
                wp_x=wp_x,
                wp_y=wp_y,
            ))
        return out

    def get_target_by_track_id(self, track_id: str) -> Optional[NpcState]:
        """Returns NpcState (not Location) so callers can access vx/vy/yaw_deg."""
        for ag in self.agents:
            if ag.track_id == track_id:
                loc = ag.walker.get_location()
                vel = ag.walker.get_velocity()
                tf  = ag.walker.get_transform()
                wp_x = wp_y = None
                if ag.wps and ag.wp_idx < len(ag.wps):
                    wp_x, wp_y = ag.wps[ag.wp_idx][0], ag.wps[ag.wp_idx][1]
                return NpcState(
                    track_id=ag.track_id,
                    actor_id=ag.walker.id,
                    x=float(loc.x), y=float(loc.y), z=float(loc.z),
                    vx=float(vel.x), vy=float(vel.y), vz=float(vel.z),
                    yaw_deg=float(tf.rotation.yaw),
                    speed=float(np.hypot(vel.x, vel.y)),
                    npc_state="stuck" if ag.is_currently_stuck else "transit",
                    wp_x=wp_x, wp_y=wp_y,
                )
        return None

    def get_actor_by_track_id(self, track_id: str) -> Optional[carla.Actor]:
        for ag in self.agents:
            if ag.track_id == track_id:
                return ag.walker
        return None

    def get_npc_debug_info(self, target_track_id: str = "") -> dict:
        npc_paths = []
        for ag in self.agents:
            if ag.agent_type == "hidden_pending":
                continue
            if ag.wps and ag.wp_idx < len(ag.wps):
                remaining = ag.wps[ag.wp_idx: ag.wp_idx + 30]
                npc_paths.append({
                    "track_id": ag.track_id,
                    "wps": [[w[0], w[1]] for w in remaining],
                })
        flow_pts = [
            [float(fp.x), float(fp.y), f"P{i}"]
            for i, fp in enumerate(self.flow_points)
        ]
        return {
            "npc_paths":          npc_paths,
            "npc_states_ext":     [],
            "connector_cells_world": [],
            "flow_points_world":  flow_pts,
        }

    def write_debug_state(
        self,
        tick: int,
        robot_state,
        target_track_id: str = "",
        path: str = "/tmp/npc_debug_state.json",
    ) -> None:
        info   = self.get_npc_debug_info(target_track_id=target_track_id)
        states = self.get_states()
        npc_list = []
        for s in states:
            is_tgt = (s.track_id == target_track_id)
            npc_list.append({
                "track_id": s.track_id,
                "x": s.x, "y": s.y,
                "is_target": is_tgt,
                "state": ("target" if is_tgt
                          else ("stuck" if s.npc_state == "stuck" else "transit")),
                "wp_x": s.wp_x,
                "wp_y": s.wp_y,
            })
        rx = ry = ryaw = 0.0
        if robot_state is not None:
            rx, ry, ryaw = float(robot_state.x), float(robot_state.y), float(robot_state.yaw_rad)
        doc = {
            "tick": tick,
            "sim_t": round(tick * self._sim_dt, 2),
            "robot": {"x": rx, "y": ry, "yaw_rad": ryaw},
            "npcs": npc_list,
            "npc_paths": info["npc_paths"],
            "connector_cells_world": [],
            "flow_points_world": info["flow_points_world"],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        os.replace(tmp, path)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def destroy(self) -> None:
        for ag in self.agents:
            try:
                ag.walker.destroy()
            except Exception:
                pass
        self.agents.clear()
        self._pending_spawns.clear()
        self._hidden_pending_spawns.clear()
        self._hidden_reserved_locs.clear()
        self._target_seg = -1
        self._seg_occupancy = {0: 0, 1: 0, 2: 0}
        self._seg_last_entry = {0: 0.0, 1: 0.0, 2: 0.0}
        self._trunk_occupancy = {i: 0 for i in range(len(self._trunk_routes))}
        self._trunk_last_dispatch.clear()
        self._zone_active = {str(z["id"]): 0 for z in self._zones}
        self._motion_active = False
        self._spawned = False
        self._episode_done = False
