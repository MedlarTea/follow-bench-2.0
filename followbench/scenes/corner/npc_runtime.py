from __future__ import annotations

import json
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from followbench.sim.core_types import NpcState
from followbench.sim.pedestrian_sfm import (
    AgentKinematics,
    DynamicObstacle,
    SFMPlanner,
    SFMPlannerConfig,
    WallQueryResult,
)


_NON_HUMAN_WALKER_BLUEPRINTS = {
    "walker.pedestrian.robot",
    "walker.pedestrian.g1_unitree",
    "walker.pedestrian.dog",
}


def _parse_meta_scalar(meta_raw) -> Dict:
    if meta_raw is None:
        return {}
    if isinstance(meta_raw, np.ndarray):
        if meta_raw.shape == ():
            meta_raw = meta_raw.item()
        else:
            return {}
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode("utf-8", errors="ignore")
    if not isinstance(meta_raw, str):
        return {}
    txt = meta_raw.strip()
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        return {}


@dataclass
class ROIMap:
    world_min: np.ndarray
    world_max: np.ndarray
    resolution: float
    free_grid: np.ndarray
    roi_mask: np.ndarray
    clearance_m: np.ndarray = field(repr=False)
    clearance_grad_x: np.ndarray = field(repr=False)
    clearance_grad_y: np.ndarray = field(repr=False)
    walkable_cells: np.ndarray = field(repr=False)

    @classmethod
    def load(cls, npz_path: str, resolution_fallback: float = 0.5) -> "ROIMap":
        data = np.load(npz_path, allow_pickle=True)
        world_min = np.array(data["world_min"], dtype=np.float64)
        world_max = np.array(data["world_max"], dtype=np.float64)
        free_grid = np.array(data["free_grid"], dtype=np.uint8)
        roi_mask = (
            np.array(data["roi_mask"], dtype=np.uint8)
            if "roi_mask" in data
            else np.ones_like(free_grid, dtype=np.uint8)
        )
        meta = _parse_meta_scalar(data["__meta__"]) if "__meta__" in data else {}
        resolution = float(meta.get("resolution", resolution_fallback))
        walkable = (free_grid > 0) & (roi_mask > 0)
        clearance_m = distance_transform_edt(walkable.astype(np.uint8)).astype(np.float32) * float(resolution)
        clearance_smooth = gaussian_filter(clearance_m, sigma=1.1, mode="nearest").astype(np.float32)
        clearance_grad_y, clearance_grad_x = np.gradient(clearance_smooth, float(resolution), float(resolution))
        clearance_grad_x = clearance_grad_x.astype(np.float32)
        clearance_grad_y = clearance_grad_y.astype(np.float32)
        cells = np.argwhere(walkable)
        if len(cells) == 0:
            raise RuntimeError("No walkable cells found.")
        return cls(
            world_min,
            world_max,
            resolution,
            free_grid,
            roi_mask,
            clearance_m,
            clearance_grad_x,
            clearance_grad_y,
            cells,
        )

    def cell_to_world(self, gx: int, gy: int, z: float) -> carla.Location:
        x = self.world_min[0] + (float(gx) + 0.5) * self.resolution
        y = self.world_min[1] + (float(gy) + 0.5) * self.resolution
        return carla.Location(x=float(x), y=float(y), z=float(z))

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

    def sample_world_location(self, rng: random.Random, z: float, jitter: bool = True) -> carla.Location:
        gy, gx = self.walkable_cells[rng.randrange(len(self.walkable_cells))]
        x = self.world_min[0] + (float(gx) + 0.5) * self.resolution
        y = self.world_min[1] + (float(gy) + 0.5) * self.resolution
        if jitter:
            half = 0.45 * self.resolution
            x += rng.uniform(-half, half)
            y += rng.uniform(-half, half)
        return carla.Location(x=float(x), y=float(y), z=float(z))

    def nearest_walkable_cell(self, gx: int, gy: int) -> Tuple[int, int]:
        if len(self.walkable_cells) == 0:
            return gx, gy
        dif = self.walkable_cells - np.array([gy, gx], dtype=np.int32)
        d2 = np.sum(dif.astype(np.int64) ** 2, axis=1)
        idx = int(np.argmin(d2))
        gy2, gx2 = self.walkable_cells[idx]
        return int(gx2), int(gy2)


def _load_flow_points(json_path: str, roi_map: ROIMap, spawn_z: float, min_points: int = 4) -> List[carla.Location]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    points_world = payload.get("points_world", None)
    if isinstance(points_world, list) and len(points_world) >= min_points:
        out: List[carla.Location] = []
        for p in points_world[:max(4, len(points_world))]:
            out.append(carla.Location(x=float(p["x"]), y=float(p["y"]), z=float(p.get("z", spawn_z))))
        return out
    points_grid = payload.get("points_grid", None)
    if isinstance(points_grid, list) and len(points_grid) >= min_points:
        return [roi_map.cell_to_world(int(gx), int(gy), z=spawn_z) for gx, gy in points_grid]
    raise RuntimeError(f"flow points json must contain at least {min_points} points.")


@dataclass
class CrowdAgent:
    walker: carla.Actor
    track_id: str
    speed: float
    target: carla.Location
    is_target: bool = False
    base_speed: float = 0.0
    flow_side: int = 0
    target_band: int = 0
    pair_lane: int = -1
    pair_endpoint: int = -1  # 0 or 1 in pair_xx mode
    initial_pair_endpoint: int = -1
    preferred_lateral_offset_m: float = 0.0
    route_lateral_offset_m: float = 0.0
    route_spacing_m: float = 1.8
    route_lookahead_m: float = 1.4
    route_longitudinal_phase_m: float = 0.0
    route_turn_radius_offset_m: float = 0.0
    pass_side_pref: int = 0
    pass_side_until_t: float = 0.0
    route_world: List[Tuple[float, float, float]] = field(default_factory=list)
    route_idx: int = 0
    route_phase: str = "transit"
    pending_pair_endpoint: int = -1
    completed_route_legs: int = 0
    route_task_done: bool = False
    last_replan_t: float = 0.0
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    last_retarget_t: float = 0.0
    last_curb_t: float = 0.0
    block_state_t: float = 0.0
    block_state_x: float = 0.0
    block_state_y: float = 0.0
    wall_normal_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32), repr=False)
    wall_normal_valid: bool = False
    route_is_lane_aware: bool = False


@dataclass
class SpawnProfile:
    track_id: str
    is_target: bool
    spawn_loc: carla.Location
    target_loc: carla.Location
    base_speed: float = 0.0
    pair_lane: int = -1
    pair_endpoint: int = -1
    initial_pair_endpoint: int = -1
    preferred_lateral_offset_m: float = 0.0
    route_lateral_offset_m: float = 0.0
    route_spacing_m: float = 1.8
    route_lookahead_m: float = 1.4
    route_longitudinal_phase_m: float = 0.0
    route_turn_radius_offset_m: float = 0.0
    initial_yaw_deg: float = 0.0
    flow_side: int = 0
    target_band: int = 0
    route_world: List[Tuple[float, float, float]] = field(default_factory=list)
    route_is_lane_aware: bool = False


class NpcRuntime:
    def __init__(
        self,
        world: carla.World,
        grid_npz: str,
        flow_points_json: str,
        flow_mode: str = "ab_cd",
        rng_seed: Optional[int] = None,
        spawn_z: float = 0.5,
    ) -> None:
        self.world = world
        self.roi_map = ROIMap.load(grid_npz)
        if rng_seed is None:
            rng_seed = int(time.time_ns() % 2_147_483_647)
        self.rng_seed = int(rng_seed)
        print(f"[NPC] rng_seed={self.rng_seed}")
        self.rng = random.Random(self.rng_seed)
        # Starts at an offset so 0.0 keeps meaning "unset" for timestamp sentinels.
        self._sim_t: float = 1000.0
        self.lookalike_track_ids: List[str] = []
        self.spawn_z = spawn_z
        self.flow_mode = flow_mode
        min_pts = 3 if flow_mode == "free_3" else 4
        self.flow_points = _load_flow_points(flow_points_json, self.roi_map, spawn_z=spawn_z, min_points=min_pts)
        self.side_a = self.flow_points[:2]
        self.side_b = self.flow_points[2:4]
        # pair lanes are configurable by flow mode:
        # - pair_12_34: (1<->2), (3<->4)
        # - pair_13_24: (1<->3), (2<->4)
        # - free_3:     3 points, each NPC picks a random different point as target
        if self.flow_mode == "pair_13_24":
            self.pairs = [
                [self.flow_points[0], self.flow_points[2]],
                [self.flow_points[1], self.flow_points[3]],
            ]
        elif self.flow_mode == "free_3":
            self.pairs = []  # unused in free_3; targets are chosen dynamically
        else:
            self.pairs = [
                [self.flow_points[0], self.flow_points[1]],
                [self.flow_points[2], self.flow_points[3]],
            ]
        self.agents: List[CrowdAgent] = []
        self._active_move = False
        self._spawned = False
        self._sfm_planner: Optional[SFMPlanner] = None
        self.params = {
            "retarget_sec": 6.0,
            "arrive_threshold": 1.8,
            "stuck_window_sec": 4.0,
            "stuck_dist_threshold": 0.45,
            "curb_min_z": 0.05,
            "curb_max_z": 0.30,
            "curb_step_up": 0.05,
            "curb_forward_nudge": 0.10,
            "curb_cooldown_sec": 0.8,
            "vertical_gain": 1.0,
            "flow_anchor_radius": 9.0,
            "min_speed": 1.0,
            "max_speed": 2.0,
            "npc_speed_mean": 0.85,
            "npc_speed_std": 0.18,
            "npc_speed_min": 0.55,
            "npc_speed_max": 1.15,
            "target_speed_mean": 0.82,
            "target_speed_std": 0.10,
            "target_speed_min": 0.70,
            "target_speed_max": 0.95,
            "speed_retarget_jitter_mps": 0.08,
            "navigation_mode": "direct",  # direct | astar
            "pair_spawn_mode": "roi_random",          # roi_random | paired_endpoint
            "target_spawn_policy": "legacy",          # legacy | endpoint_center
            "crowd_spawn_policy": "legacy",           # legacy | roi_random
            "npc_spawn_min_sep_m": 1.4,
            "target_centerline_half_width_m": 0.8,
            "target_wall_clearance_min_m": 1.2,
            "target_endpoint_sample_radius_m": 5.0,
            "target_spawn_retry_attempts": 12,
            "pair_goal_policy": "fixed_until_arrival",  # fixed_until_arrival | retarget_on_block
            "pair_target_band_half_width_m": 2.5,
            "corner_route_mode": "l_shape",          # astar | l_shape
            "corner_route_waypoint_spacing_m": 1.8,
            "corner_route_start_lookahead_m": 1.4,
            "corner_turn_radius_m": 4.0,
            "corner_turn_samples": 8,
            "crowd_turn_sublane_mode": "off",          # off | preserve_sublane
            "crowd_turn_radius_jitter_m": 0.0,
            "crowd_turn_min_offset_keep_ratio": 0.65,
            "crowd_turn_samples": 8,
            "corner_route_spacing_jitter_m": 0.40,
            "corner_route_lookahead_jitter_m": 0.50,
            "corner_route_longitudinal_phase_min_m": -1.0,
            "corner_route_longitudinal_phase_max_m": 1.2,
            "right_hand_lane_sep_m": 0.72,
            "min_lane_offset_from_center_m": 0.45,
            "max_lane_offset_from_center_m": 1.35,
            "target_lane_bias_scale": 0.45,
            "target_lane_min_offset_m": 0.25,
            "target_lane_max_offset_m": 0.75,
            "lane_wall_clearance_m": 0.80,
            "npc_lateral_personal_span_m": 0.65,
            "npc_lateral_jitter_m": 0.20,
            "target_lateral_jitter_m": 0.15,
            "follow_position": "back",
            "desired_distance": 1.5,
            "target_lane_bias_mode": "right_hand",
            "robot_spawn_clearance_m": 2.2,
            "target_required_legs": 2,
            "target_turnaround_forward_m": 2.2,
            "target_turnaround_merge_m": 4.5,
            "target_turnaround_side_m": 1.6,
            "target_turnaround_samples": 20,
            "target_turnaround_speed_mps": 0.40,
            "astar_clearance_margin_m": 0.9,
            "astar_clearance_weight": 1.8,
            "astar_clearance_power": 2.0,
            "astar_replan_cooldown_sec": 0.8,
            "astar_waypoint_arrive_dist": 0.45,
            "replan_on_block": True,
            "block_replan_window_sec": 2.0,
            "block_replan_progress_thresh": 0.25,
            "block_replan_risk_thresh": 0.5,
            "sfm_dt": 0.1,
            "sfm_tau": 0.55,
            "sfm_desired_speed_scale": 1.0,
            "sfm_ped_A": 2.6,
            "sfm_ped_B": 0.55,
            "sfm_ped_radius": 0.34,
            "sfm_robot_A": 1.4,
            "sfm_robot_B": 0.70,
            "sfm_robot_radius": 0.65,
            "sfm_wall_A": 4.6,
            "sfm_wall_B": 0.28,
            "sfm_wall_influence_dist": 1.25,
            "sfm_wall_normal_min_grad": 0.08,
            "sfm_wall_normal_smooth_alpha": 0.35,
            "sfm_tangential_weight": 0.65,
            "sfm_robot_tangential_weight": 0.35,
            "sfm_robot_max_avoidance_angle_deg": 40.0,
            "sfm_anisotropy_lambda": 0.3,
            "sfm_prediction_enabled": True,
            "sfm_prediction_horizon": 1.2,
            "sfm_close_risk_margin": 0.65,
            "sfm_headon_cos_threshold": -0.65,
            "sfm_headon_front_dot_threshold": 0.2,
            "sfm_headon_lateral_trigger_m": 1.2,
            "sfm_headon_bias_weight": 1.4,
            "sfm_headon_min_risk": 0.35,
            "sfm_pass_side_trigger_risk": 0.22,
            "sfm_pass_side_hold_sec": 3.2,
            "sfm_neighbor_radius": 3.5,
            "sfm_max_neighbors": 8,
            "sfm_max_force": 5.0,
            "sfm_yield_trigger_risk": 0.45,
            "sfm_yield_min_speed_scale": 0.60,
            "sfm_close_surface_slowdown_m": 0.45,
        }

    def _spawn_one_at_endpoint(
        self,
        bp: carla.ActorBlueprint,
        endpoint: carla.Location,
        track_id: str,
        target: carla.Location,
    ) -> Optional[CrowdAgent]:
        walker = None
        base_xy = np.array([endpoint.x, endpoint.y], dtype=np.float64)
        for _ in range(40):
            jitter = self.rng.uniform(-0.35, 0.35), self.rng.uniform(-0.35, 0.35)
            x = float(base_xy[0] + jitter[0])
            y = float(base_xy[1] + jitter[1])
            if not self.roi_map.is_world_walkable(x, y):
                x = float(base_xy[0])
                y = float(base_xy[1])
            loc = carla.Location(x=x, y=y, z=float(endpoint.z))
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                loc2 = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.25)
                walker = self.world.try_spawn_actor(bp, carla.Transform(loc2))
            if walker is not None:
                break
        if walker is None:
            return None
        speed = float(self.rng.uniform(self.params["min_speed"], self.params["max_speed"]))
        return CrowdAgent(
            walker=walker,
            track_id=track_id,
            speed=speed,
            target=carla.Location(x=float(target.x), y=float(target.y), z=float(target.z)),
            base_speed=speed,
            route_world=[],
            route_idx=0,
            last_replan_t=0.0,
            last_retarget_t=self._sim_t,
        )

    def _spawn_one_random(
        self,
        bp: carla.ActorBlueprint,
        track_id: str,
        target: carla.Location,
    ) -> Optional[CrowdAgent]:
        walker = None
        loc = None
        for _ in range(40):
            loc = self._sample_nav(None)
            if loc is None:
                loc = self.roi_map.sample_world_location(self.rng, z=self.spawn_z, jitter=True)
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                loc2 = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.25)
                walker = self.world.try_spawn_actor(bp, carla.Transform(loc2))
            if walker is not None:
                break
        if walker is None:
            return None
        speed = float(self.rng.uniform(self.params["min_speed"], self.params["max_speed"]))
        return CrowdAgent(
            walker=walker,
            track_id=track_id,
            speed=speed,
            target=carla.Location(x=float(target.x), y=float(target.y), z=float(target.z)),
            base_speed=speed,
            route_world=[],
            route_idx=0,
            last_replan_t=0.0,
            last_retarget_t=self._sim_t,
        )

    def _location_clearance_m(self, loc: carla.Location) -> float:
        gx, gy = self.roi_map.world_to_cell(loc.x, loc.y)
        h, w = self.roi_map.clearance_m.shape
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            return 0.0
        return float(self.roi_map.clearance_m[gy, gx])

    def _clipped_gauss(self, mean: float, std: float, lo: float, hi: float) -> float:
        lo_f, hi_f = float(lo), float(hi)
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        std_f = max(float(std), 1e-3)
        for _ in range(24):
            value = float(self.rng.gauss(float(mean), std_f))
            if lo_f <= value <= hi_f:
                return value
        return float(np.clip(float(mean), lo_f, hi_f))

    def _crowd_speed_range(self) -> Tuple[float, float]:
        """Crowd speed interval = the difficulty preset [min_speed, max_speed]."""
        lo = float(self.params["min_speed"])
        hi = float(self.params["max_speed"])
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _sample_personal_speed(self, is_target: bool) -> float:
        if is_target:
            return self._clipped_gauss(
                self.params["target_speed_mean"],
                self.params["target_speed_std"],
                self.params["target_speed_min"],
                self.params["target_speed_max"],
            )
        # Personal speed: truncated Gaussian centred in the preset interval.
        lo, hi = self._crowd_speed_range()
        return self._clipped_gauss(0.5 * (lo + hi), max(0.25 * (hi - lo), 1e-3), lo, hi)

    def _retarget_speed(self, ag: CrowdAgent) -> float:
        base = float(ag.base_speed if ag.base_speed > 0.0 else ag.speed)
        jitter = float(self.params["speed_retarget_jitter_mps"])
        if ag.is_target:
            lo = float(self.params["target_speed_min"])
            hi = float(self.params["target_speed_max"])
        else:
            lo, hi = self._crowd_speed_range()
        return float(np.clip(base + self.rng.uniform(-jitter, jitter), lo, hi))

    def _far_enough_from_taken(
        self,
        loc: carla.Location,
        taken_locs: List[carla.Location],
        min_sep_m: float,
    ) -> bool:
        min_sep = max(0.0, float(min_sep_m))
        if min_sep <= 1e-6:
            return True
        return all(float(loc.distance(other)) >= min_sep for other in taken_locs)

    def _plan_route_world_between(
        self,
        start_loc: carla.Location,
        target_loc: carla.Location,
    ) -> List[Tuple[float, float, float]]:
        s_gx, s_gy = self.roi_map.world_to_cell(start_loc.x, start_loc.y)
        g_gx, g_gy = self.roi_map.world_to_cell(target_loc.x, target_loc.y)
        cell_path = self._astar_cells((s_gx, s_gy), (g_gx, g_gy))
        if not cell_path:
            return []
        route: List[Tuple[float, float, float]] = []
        z = max(float(self.spawn_z), float(target_loc.z), float(start_loc.z))
        for gx, gy in cell_path:
            wp = self.roi_map.cell_to_world(gx, gy, z=z)
            route.append((float(wp.x), float(wp.y), float(wp.z)))
        return route

    def _yaw_from_route_or_target(
        self,
        spawn_loc: carla.Location,
        target_loc: carla.Location,
        route_world: List[Tuple[float, float, float]],
    ) -> float:
        next_x = float(target_loc.x)
        next_y = float(target_loc.y)
        for wx, wy, _wz in route_world:
            if float(np.hypot(wx - spawn_loc.x, wy - spawn_loc.y)) >= 0.5:
                next_x, next_y = float(wx), float(wy)
                break
        dx = float(next_x - spawn_loc.x)
        dy = float(next_y - spawn_loc.y)
        if float(np.hypot(dx, dy)) <= 1e-6:
            return 0.0
        return float(math.degrees(math.atan2(dy, dx)))

    def _sample_roi_location_with_spacing(
        self,
        taken_locs: List[carla.Location],
        min_sep_m: float,
        reserved_locs: Optional[List[carla.Location]] = None,
        reserved_min_sep_m: Optional[float] = None,
    ) -> carla.Location:
        reserved_locs = reserved_locs or []
        reserved_sep = float(reserved_min_sep_m if reserved_min_sep_m is not None else min_sep_m)
        best: Optional[carla.Location] = None
        best_score = 1e18
        for attempt in range(240):
            loc = self._sample_nav(None)
            if loc is None:
                loc = self.roi_map.sample_world_location(self.rng, z=self.spawn_z, jitter=True)
            sep = float(min_sep_m) if attempt < 180 else float(min_sep_m) * 0.7
            nearest_reserved = min(
                (float(loc.distance(other)) for other in reserved_locs),
                default=1e18,
            )
            score = max(0.0, sep - min((float(loc.distance(other)) for other in taken_locs), default=1e18))
            score += max(0.0, reserved_sep - nearest_reserved) * 2.0
            if score < best_score:
                best = loc
                best_score = score
            if (
                self._far_enough_from_taken(loc, taken_locs, sep)
                and self._far_enough_from_taken(loc, reserved_locs, reserved_sep)
            ):
                return loc
        if best is not None:
            return best
        return self.roi_map.sample_world_location(self.rng, z=self.spawn_z, jitter=True)

    def _endpoint_lateral_offset(self, loc: carla.Location, lane: int) -> float:
        p0 = self._pair_anchor(lane, 0)
        _forward, lateral = self._pair_lane_frame(lane)
        rel = np.array([float(loc.x - p0.x), float(loc.y - p0.y)], dtype=np.float32)
        return float(np.dot(rel, lateral))

    def _target_follow_clearance_sign(self) -> float:
        if str(self.params.get("target_lane_bias_mode", "right_hand")) != "leave_follow_side_clear":
            return -1.0
        follow_position = str(self.params.get("follow_position", "back"))
        # Route offsets use the mathematical normal of the route. In CARLA's
        # target body frame, positive route offset is visual-right and negative
        # route offset is visual-left. Put the target on the side opposite the
        # requested robot side so the robot's desired side has room.
        if follow_position == "left_side":
            return 1.0
        return -1.0

    def _route_start_reference_location(
        self,
        lane: int,
        target_ep: int,
        route_lateral_offset_m: float,
    ) -> Optional[carla.Location]:
        spacing = max(float(self.params["corner_route_waypoint_spacing_m"]), 0.8)
        centerline = self._build_rounded_l_shape_centerline(lane, target_ep, spacing=spacing)
        if not centerline:
            return None
        z = float(self.spawn_z)
        offset_polyline = self._offset_centerline_samples(centerline, float(route_lateral_offset_m), z=z)
        if not offset_polyline:
            return None
        return offset_polyline[0]

    def _side_follow_spawn_point(
        self,
        target_loc: carla.Location,
        lane: int,
        target_ep: int,
    ) -> Optional[carla.Location]:
        follow_position = str(self.params.get("follow_position", "back"))
        if follow_position not in ("left_side", "right_side"):
            return None
        _anchor, forward, _lateral = self._pair_endpoint_frame(lane, target_ep)
        yaw = math.atan2(float(forward[1]), float(forward[0]))
        offset = -math.pi / 2.0 if follow_position == "left_side" else math.pi / 2.0
        distance = max(float(self.params.get("desired_distance", 1.5)), 0.0)
        return carla.Location(
            x=float(target_loc.x + math.cos(yaw + offset) * distance),
            y=float(target_loc.y + math.sin(yaw + offset) * distance),
            z=float(target_loc.z),
        )

    def _robot_spawn_reservation_for_target(self, profile: SpawnProfile) -> Optional[carla.Location]:
        follow_position = str(self.params.get("follow_position", "back"))
        offset_by_mode = {
            "back": math.pi,
            "left_side": -math.pi / 2.0,
            "right_side": math.pi / 2.0,
        }
        offset = offset_by_mode.get(follow_position)
        if offset is None:
            return None
        distance = max(float(self.params.get("desired_distance", 1.5)), 0.0)
        yaw = math.radians(float(profile.initial_yaw_deg))
        ideal_x = float(profile.spawn_loc.x + math.cos(yaw + offset) * distance)
        ideal_y = float(profile.spawn_loc.y + math.sin(yaw + offset) * distance)
        z = float(profile.spawn_loc.z)
        if self.roi_map.is_world_walkable(ideal_x, ideal_y):
            return carla.Location(x=ideal_x, y=ideal_y, z=z)
        for radius in np.arange(0.2, 1.6, 0.2):
            for ang in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
                x = float(ideal_x + math.cos(float(ang)) * float(radius))
                y = float(ideal_y + math.sin(float(ang)) * float(radius))
                if self.roi_map.is_world_walkable(x, y):
                    return carla.Location(x=x, y=y, z=z)
        return carla.Location(x=ideal_x, y=ideal_y, z=z)

    def _sample_centered_endpoint_location(
        self,
        lane: int,
        endpoint_idx: int,
        taken_locs: List[carla.Location],
        min_sep_m: float,
        target_ep: Optional[int] = None,
        preferred_route_offset_m: Optional[float] = None,
    ) -> carla.Location:
        anchor = self._pair_anchor(lane, endpoint_idx)
        radius = max(float(self.params["target_endpoint_sample_radius_m"]), 0.5)
        half_width = max(float(self.params["target_centerline_half_width_m"]), 0.05)
        min_clearance = max(float(self.params["target_wall_clearance_min_m"]), 0.0)
        desired_loc = None
        if target_ep is not None and preferred_route_offset_m is not None:
            desired_loc = self._route_start_reference_location(lane, target_ep, preferred_route_offset_m)
        best: Optional[carla.Location] = None
        best_score = 1e18

        for attempt in range(360):
            loc = self._sample_nav([anchor], anchor_radius_m=radius)
            if loc is None:
                angle = self.rng.uniform(-math.pi, math.pi)
                dist = radius * math.sqrt(self.rng.random())
                loc = carla.Location(
                    x=float(anchor.x + math.cos(angle) * dist),
                    y=float(anchor.y + math.sin(angle) * dist),
                    z=float(anchor.z),
                )
            if not self.roi_map.is_world_walkable(loc.x, loc.y):
                continue
            lateral_abs = abs(self._endpoint_lateral_offset(loc, lane))
            desired_dist = (
                float(np.hypot(loc.x - desired_loc.x, loc.y - desired_loc.y))
                if desired_loc is not None
                else lateral_abs
            )
            clearance = self._location_clearance_m(loc)
            side_loc = self._side_follow_spawn_point(loc, lane, target_ep) if target_ep is not None else None
            side_ok = True
            side_clearance = min_clearance
            if side_loc is not None:
                side_ok = bool(self.roi_map.is_world_walkable(side_loc.x, side_loc.y))
                side_clearance = self._location_clearance_m(side_loc) if side_ok else 0.0
            score = (
                desired_dist
                + max(0.0, min_clearance - clearance) * 3.0
                + (0.0 if side_ok else 50.0)
                + max(0.0, min_clearance * 0.65 - side_clearance) * 2.0
            )
            if score < best_score:
                best = carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z))
                best_score = score
            width_scale = 1.0 if attempt < 180 else 1.5
            clearance_scale = 1.0 if attempt < 240 else 0.75
            sep_scale = 1.0 if attempt < 240 else 0.7
            if desired_dist > half_width * width_scale:
                continue
            if clearance < min_clearance * clearance_scale:
                continue
            if side_loc is not None and attempt < 300:
                if not side_ok:
                    continue
                if side_clearance < min_clearance * 0.65 * clearance_scale:
                    continue
            if not self._far_enough_from_taken(loc, taken_locs, float(min_sep_m) * sep_scale):
                continue
            return carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z))

        if best is not None:
            return best
        return carla.Location(x=float(anchor.x), y=float(anchor.y), z=float(anchor.z))

    def _walkable_mask(self) -> np.ndarray:
        return (self.roi_map.free_grid > 0) & (self.roi_map.roi_mask > 0)

    def _astar_cell_penalty(self, gx: int, gy: int) -> float:
        clearance = float(self.roi_map.clearance_m[gy, gx])
        margin = float(self.params["astar_clearance_margin_m"])
        if margin <= 1e-6 or clearance >= margin:
            return 0.0
        weight = float(self.params["astar_clearance_weight"])
        power = float(self.params["astar_clearance_power"])
        deficit = (margin - clearance) / margin
        return weight * (deficit ** power)

    def _astar_cells(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        max_iters: int = 20000,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        8-neighbor A* on ROI walkable grid.
        cells are (gx, gy).
        """
        walkable = self._walkable_mask()
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
            return float(np.hypot(a[0] - b[0], a[1] - b[1]) * self.roi_map.resolution)

        open_heap: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, (sx, sy)))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        gscore: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
        iters = 0
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

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
                nx, ny = cx + dx, cy + dy
                if not valid(nx, ny):
                    continue
                if dx != 0 and dy != 0:
                    if not valid(cx + dx, cy) or not valid(cx, cy + dy):
                        continue
                step_cost = float(self.roi_map.resolution * (1.41421356237 if (dx != 0 and dy != 0) else 1.0))
                penalty = self._astar_cell_penalty(nx, ny)
                ng = gscore[cur] + step_cost + penalty
                nxt = (nx, ny)
                if ng < gscore.get(nxt, 1e18):
                    came_from[nxt] = cur
                    gscore[nxt] = ng
                    ff = ng + heur(nxt, (gx, gy))
                    heapq.heappush(open_heap, (ff, nxt))
        return None

    def _replan_route(self, ag: CrowdAgent, now_t: float) -> None:
        if ag.route_is_lane_aware and self._is_pair_flow_mode() and ag.pair_lane in (0, 1) and ag.pair_endpoint in (0, 1):
            loc = ag.walker.get_location()
            if ag.is_target:
                ag.route_lateral_offset_m = self._target_route_offset_for_endpoint(
                    ag.preferred_lateral_offset_m,
                    ag.pair_endpoint,
                )
            else:
                ag.route_lateral_offset_m = self._right_hand_route_offset(
                    ag.preferred_lateral_offset_m,
                    is_target=False,
                )
            ag.route_world = self._build_l_shape_route_world(
                loc,
                ag.target,
                lane=ag.pair_lane,
                target_ep=ag.pair_endpoint,
                route_lateral_offset_m=ag.route_lateral_offset_m,
                route_spacing_m=ag.route_spacing_m,
                route_lookahead_m=ag.route_lookahead_m,
                route_longitudinal_phase_m=ag.route_longitudinal_phase_m,
                route_turn_radius_offset_m=ag.route_turn_radius_offset_m,
                is_target=bool(ag.is_target),
            )
            if ag.route_world:
                x, y, z = ag.route_world[-1]
                ag.target = carla.Location(x=float(x), y=float(y), z=float(z))
            ag.route_idx = 0
            ag.last_replan_t = now_t
            return
        if self.params["navigation_mode"] != "astar":
            ag.route_world = []
            ag.route_idx = 0
            ag.last_replan_t = now_t
            return
        if now_t - ag.last_replan_t < self.params["astar_replan_cooldown_sec"]:
            return
        loc = ag.walker.get_location()
        s_gx, s_gy = self.roi_map.world_to_cell(loc.x, loc.y)
        g_gx, g_gy = self.roi_map.world_to_cell(ag.target.x, ag.target.y)
        cell_path = self._astar_cells((s_gx, s_gy), (g_gx, g_gy))
        if not cell_path or len(cell_path) <= 1:
            ag.route_world = []
            ag.route_idx = 0
            ag.last_replan_t = now_t
            return
        route: List[Tuple[float, float, float]] = []
        for gx, gy in cell_path:
            wp = self.roi_map.cell_to_world(gx, gy, z=max(self.spawn_z, ag.target.z))
            route.append((float(wp.x), float(wp.y), float(wp.z)))
        ag.route_world = route
        ag.route_idx = 0
        ag.last_replan_t = now_t

    def _sample_nav(
        self,
        anchor_points: Optional[List[carla.Location]],
        anchor_radius_m: Optional[float] = None,
    ) -> Optional[carla.Location]:
        # Seeded sampling only (CARLA's navmesh random point is not controlled by
        # the episode seed, which broke paired replay).
        if not anchor_points:
            return self.roi_map.sample_world_location(self.rng, z=self.spawn_z, jitter=True)
        radius = float(anchor_radius_m) if anchor_radius_m is not None else float(self.params["flow_anchor_radius"])
        for _ in range(120):
            a = anchor_points[self.rng.randrange(len(anchor_points))]
            angle = self.rng.uniform(-math.pi, math.pi)
            dist = radius * math.sqrt(self.rng.random())
            x = float(a.x + math.cos(angle) * dist)
            y = float(a.y + math.sin(angle) * dist)
            if not self.roi_map.is_world_walkable(x, y):
                continue
            return carla.Location(x=x, y=y, z=float(a.z))
        return None

    def _sample_target(
        self,
        anchor_points: Optional[List[carla.Location]],
        anchor_radius_m: Optional[float] = None,
    ) -> carla.Location:
        tgt = self._sample_nav(anchor_points, anchor_radius_m=anchor_radius_m)
        if tgt is None:
            tgt = self._sample_nav(None)
        if tgt is None:
            tgt = self.roi_map.sample_world_location(self.rng, z=self.spawn_z, jitter=True)
        return tgt

    def _pair_anchor(self, lane: int, endpoint_idx: int) -> carla.Location:
        p = self.pairs[lane][endpoint_idx]
        return carla.Location(x=float(p.x), y=float(p.y), z=float(p.z))

    def _pair_lane_frame(
        self,
        lane: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        p0 = self._pair_anchor(lane, 0)
        p1 = self._pair_anchor(lane, 1)
        dir_x = float(p1.x - p0.x)
        dir_y = float(p1.y - p0.y)
        norm = float(np.hypot(dir_x, dir_y))
        if norm > 1e-6:
            forward = np.array([dir_x / norm, dir_y / norm], dtype=np.float32)
        else:
            forward = np.array([1.0, 0.0], dtype=np.float32)
        lateral = np.array([-forward[1], forward[0]], dtype=np.float32)
        return forward, lateral

    def _walkable_location_near_xy(self, x: float, y: float, z: float) -> carla.Location:
        if self.roi_map.is_world_walkable(x, y):
            return carla.Location(x=float(x), y=float(y), z=float(z))
        gx, gy = self.roi_map.world_to_cell(x, y)
        gx, gy = self.roi_map.nearest_walkable_cell(gx, gy)
        return self.roi_map.cell_to_world(gx, gy, z=float(z))

    def _pair_elbow_anchor(self, lane: int) -> carla.Location:
        p0 = self._pair_anchor(lane, 0)
        p1 = self._pair_anchor(lane, 1)
        candidates = [
            carla.Location(x=float(p1.x), y=float(p0.y), z=float(max(p0.z, p1.z))),
            carla.Location(x=float(p0.x), y=float(p1.y), z=float(max(p0.z, p1.z))),
        ]
        best = candidates[0]
        best_score = -1.0
        for cand in candidates:
            loc = self._walkable_location_near_xy(cand.x, cand.y, cand.z)
            clearance = self._location_clearance_m(loc)
            displacement = float(np.hypot(loc.x - cand.x, loc.y - cand.y))
            score = clearance - 0.5 * displacement
            if score > best_score:
                best = loc
                best_score = score
        return best

    def _directed_l_shape_polyline(self, lane: int, target_ep: int) -> List[carla.Location]:
        p0 = self._pair_anchor(lane, 0)
        p1 = self._pair_anchor(lane, 1)
        elbow = self._pair_elbow_anchor(lane)
        if int(target_ep) == 1:
            return [p0, elbow, p1]
        return [p1, elbow, p0]

    def _right_hand_route_offset(
        self,
        preferred_lateral_offset_m: float,
        is_target: bool,
    ) -> float:
        lane_sep = max(float(self.params["right_hand_lane_sep_m"]), 0.0)
        half_width = max(float(self.params["pair_target_band_half_width_m"]), 0.1)
        if is_target:
            base = self._target_follow_clearance_sign() * lane_sep * float(self.params["target_lane_bias_scale"])
            min_abs = min(float(self.params["target_lane_min_offset_m"]), half_width)
            max_abs = min(float(self.params["target_lane_max_offset_m"]), half_width)
            offset = base + float(preferred_lateral_offset_m)
        else:
            min_abs = min(float(self.params["min_lane_offset_from_center_m"]), half_width)
            max_abs = min(float(self.params["max_lane_offset_from_center_m"]), half_width)
            offset = -lane_sep + float(preferred_lateral_offset_m)

        sign = -1.0 if offset < 0.0 else 1.0
        abs_off = float(np.clip(abs(offset), min_abs, max_abs))
        return float(sign * abs_off)

    def _target_route_offset_for_endpoint(
        self,
        preferred_lateral_offset_m: float,
        endpoint_idx: int,
    ) -> float:
        """Route offset for the target, expressed in the target's body frame.

        Positive route offset is the target person's visual-right side under
        CARLA's yaw convention. For side-follow tasks, place the target on the
        opposite body side so the requested robot side stays clear after both
        the outbound and return legs.
        """
        _ = int(endpoint_idx)
        lane_sep = max(float(self.params["right_hand_lane_sep_m"]), 0.0)
        half_width = max(float(self.params["pair_target_band_half_width_m"]), 0.1)
        min_abs = min(float(self.params["target_lane_min_offset_m"]), half_width)
        max_abs = min(float(self.params["target_lane_max_offset_m"]), half_width)
        follow_position = str(self.params.get("follow_position", "back"))
        use_follow_clearance = (
            str(self.params.get("target_lane_bias_mode", "right_hand")) == "leave_follow_side_clear"
            and follow_position in ("left_side", "right_side")
        )
        if use_follow_clearance:
            body_side_sign = 1.0 if follow_position == "left_side" else -1.0
        else:
            body_side_sign = self._target_follow_clearance_sign()
        offset = body_side_sign * lane_sep * float(self.params["target_lane_bias_scale"])
        offset += float(preferred_lateral_offset_m)
        sign = -1.0 if offset < 0.0 else 1.0
        abs_off = float(np.clip(abs(offset), min_abs, max_abs))
        return float(sign * abs_off)

    def _project_onto_polyline(
        self,
        loc: carla.Location,
        polyline: List[carla.Location],
    ) -> Tuple[float, float]:
        p = np.array([float(loc.x), float(loc.y)], dtype=np.float32)
        best_s = 0.0
        best_dist = 1e18
        accum = 0.0
        for i in range(len(polyline) - 1):
            a = np.array([float(polyline[i].x), float(polyline[i].y)], dtype=np.float32)
            b = np.array([float(polyline[i + 1].x), float(polyline[i + 1].y)], dtype=np.float32)
            ab = b - a
            seg_len = float(np.linalg.norm(ab))
            if seg_len <= 1e-6:
                continue
            t = float(np.clip(np.dot(p - a, ab) / (seg_len * seg_len), 0.0, 1.0))
            closest = a + ab * t
            dist = float(np.linalg.norm(p - closest))
            if dist < best_dist:
                best_dist = dist
                best_s = accum + t * seg_len
            accum += seg_len
        return best_s, best_dist

    def _line_samples_xy(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        spacing: float,
        include_start: bool,
        include_end: bool,
    ) -> List[np.ndarray]:
        seg = end_xy - start_xy
        length = float(np.linalg.norm(seg))
        if length <= 1e-6:
            return [start_xy.astype(np.float32)] if include_start else []
        n_steps = max(1, int(np.ceil(length / max(float(spacing), 0.2))))
        first = 0 if include_start else 1
        last = n_steps if include_end else n_steps - 1
        out: List[np.ndarray] = []
        for i in range(first, last + 1):
            t = float(np.clip(float(i) / float(n_steps), 0.0, 1.0))
            out.append((start_xy + seg * t).astype(np.float32))
        return out

    def _build_rounded_l_shape_centerline(
        self,
        lane: int,
        target_ep: int,
        spacing: float,
        turn_radius_offset_m: float = 0.0,
        turn_samples_override: Optional[int] = None,
    ) -> List[np.ndarray]:
        polyline = self._directed_l_shape_polyline(lane, target_ep)
        if len(polyline) != 3:
            return [np.array([float(p.x), float(p.y)], dtype=np.float32) for p in polyline]

        start = np.array([float(polyline[0].x), float(polyline[0].y)], dtype=np.float32)
        elbow = np.array([float(polyline[1].x), float(polyline[1].y)], dtype=np.float32)
        end = np.array([float(polyline[2].x), float(polyline[2].y)], dtype=np.float32)
        in_vec = elbow - start
        out_vec = end - elbow
        len_in = float(np.linalg.norm(in_vec))
        len_out = float(np.linalg.norm(out_vec))
        if len_in <= 1e-6 or len_out <= 1e-6:
            return [start, elbow, end]

        radius_cfg = max(float(self.params["corner_turn_radius_m"]) + float(turn_radius_offset_m), 0.0)
        radius = min(radius_cfg, len_in * 0.45, len_out * 0.45)
        if radius < 0.5:
            return [start, elbow, end]

        dir_in = in_vec / len_in
        dir_out = out_vec / len_out
        entry = elbow - dir_in * radius
        exit_pt = elbow + dir_out * radius
        turn_samples = max(
            3,
            int(turn_samples_override)
            if turn_samples_override is not None
            else int(self.params["corner_turn_samples"]),
        )

        samples: List[np.ndarray] = []
        samples.extend(self._line_samples_xy(start, entry, spacing, include_start=True, include_end=False))
        for i in range(turn_samples + 1):
            t = float(i) / float(turn_samples)
            q = ((1.0 - t) ** 2) * entry + 2.0 * (1.0 - t) * t * elbow + (t ** 2) * exit_pt
            if samples and float(np.linalg.norm(q - samples[-1])) < 0.2:
                continue
            samples.append(q.astype(np.float32))
        samples.extend(self._line_samples_xy(exit_pt, end, spacing, include_start=False, include_end=True))
        return samples

    def _offset_centerline_samples(
        self,
        samples_xy: List[np.ndarray],
        offset_m: float,
        z: float,
        min_offset_keep_ratio: float = 0.65,
    ) -> List[carla.Location]:
        if not samples_xy:
            return []
        out: List[carla.Location] = []
        for i, base in enumerate(samples_xy):
            if len(samples_xy) == 1:
                tangent = np.array([1.0, 0.0], dtype=np.float32)
            elif i == 0:
                tangent = samples_xy[1] - samples_xy[0]
            elif i == len(samples_xy) - 1:
                tangent = samples_xy[-1] - samples_xy[-2]
            else:
                tangent = samples_xy[i + 1] - samples_xy[i - 1]
            t_norm = float(np.linalg.norm(tangent))
            if t_norm <= 1e-6:
                tangent = np.array([1.0, 0.0], dtype=np.float32)
            else:
                tangent = tangent / t_norm
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)

            chosen: Optional[carla.Location] = None
            best: Optional[carla.Location] = None
            best_score = -1e18
            min_clearance = max(float(self.params["lane_wall_clearance_m"]), 0.0)
            keep_ratio = float(np.clip(float(min_offset_keep_ratio), 0.0, 1.0))
            min_center_offset = abs(float(offset_m)) * keep_ratio
            offset_weight = 0.25 if keep_ratio <= 0.65 else 0.65
            for scale in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4):
                xy = base + normal * float(offset_m) * float(scale)
                if not self.roi_map.is_world_walkable(float(xy[0]), float(xy[1])):
                    continue
                candidate = carla.Location(x=float(xy[0]), y=float(xy[1]), z=float(z))
                clearance = self._location_clearance_m(candidate)
                center_offset = abs(float(offset_m) * float(scale))
                offset_loss = max(0.0, min_center_offset - center_offset)
                score = clearance + offset_weight * center_offset - 1.5 * offset_loss
                if score > best_score:
                    best = candidate
                    best_score = score
                if clearance >= min_clearance and center_offset >= min_center_offset:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = best or self._walkable_location_near_xy(float(base[0]), float(base[1]), z)
            if out and float(np.hypot(chosen.x - out[-1].x, chosen.y - out[-1].y)) < 0.25:
                continue
            out.append(chosen)
        return out

    def _location_on_location_polyline(
        self,
        polyline: List[carla.Location],
        s: float,
    ) -> carla.Location:
        remaining = max(0.0, float(s))
        last = polyline[-1]
        for i in range(len(polyline) - 1):
            a = polyline[i]
            b = polyline[i + 1]
            seg_len = float(np.hypot(b.x - a.x, b.y - a.y))
            if seg_len <= 1e-6:
                continue
            if remaining <= seg_len:
                t = remaining / seg_len
                return carla.Location(
                    x=float(a.x + (b.x - a.x) * t),
                    y=float(a.y + (b.y - a.y) * t),
                    z=float(max(a.z, b.z)),
                )
            remaining -= seg_len
            last = b
        return carla.Location(x=float(last.x), y=float(last.y), z=float(last.z))

    def _polyline_length(self, polyline: List[carla.Location]) -> float:
        total = 0.0
        for i in range(len(polyline) - 1):
            total += float(np.hypot(polyline[i + 1].x - polyline[i].x, polyline[i + 1].y - polyline[i].y))
        return total

    def _build_l_shape_route_world(
        self,
        spawn_loc: carla.Location,
        target_loc: carla.Location,
        lane: int,
        target_ep: int,
        route_lateral_offset_m: float,
        route_spacing_m: float,
        route_lookahead_m: float,
        route_longitudinal_phase_m: float,
        route_turn_radius_offset_m: float = 0.0,
        is_target: bool = False,
    ) -> List[Tuple[float, float, float]]:
        spacing = max(0.8, float(route_spacing_m))
        preserve_sublane = (
            not bool(is_target)
            and str(self.params.get("crowd_turn_sublane_mode", "off")) == "preserve_sublane"
        )
        centerline = self._build_rounded_l_shape_centerline(
            lane,
            target_ep,
            spacing=spacing,
            turn_radius_offset_m=float(route_turn_radius_offset_m) if preserve_sublane else 0.0,
            turn_samples_override=int(self.params["crowd_turn_samples"]) if preserve_sublane else None,
        )
        if len(centerline) <= 1:
            return self._plan_route_world_between(spawn_loc, target_loc)

        z = max(float(self.spawn_z), float(target_loc.z), float(spawn_loc.z))
        keep_ratio = (
            float(self.params["crowd_turn_min_offset_keep_ratio"])
            if preserve_sublane
            else 0.65
        )
        offset_polyline = self._offset_centerline_samples(
            centerline,
            float(route_lateral_offset_m),
            z=z,
            min_offset_keep_ratio=keep_ratio,
        )
        if len(offset_polyline) <= 1:
            return self._plan_route_world_between(spawn_loc, target_loc)

        total_len = self._polyline_length(offset_polyline)
        if total_len <= 1e-6:
            return self._plan_route_world_between(spawn_loc, target_loc)

        start_s, _dist = self._project_onto_polyline(spawn_loc, offset_polyline)
        lookahead = max(0.5, float(route_lookahead_m))
        route: List[Tuple[float, float, float]] = []

        s = min(max(start_s + lookahead + float(route_longitudinal_phase_m), start_s + 0.5), total_len)
        while s < total_len - 0.4:
            wp = self._location_on_location_polyline(offset_polyline, s)
            if float(np.hypot(wp.x - spawn_loc.x, wp.y - spawn_loc.y)) > 0.35:
                route.append((float(wp.x), float(wp.y), float(wp.z)))
            s += spacing

        final_wp = offset_polyline[-1]
        route.append((float(final_wp.x), float(final_wp.y), float(final_wp.z)))
        deduped: List[Tuple[float, float, float]] = []
        for wp in route:
            if deduped and float(np.hypot(wp[0] - deduped[-1][0], wp[1] - deduped[-1][1])) < 0.35:
                continue
            deduped.append(wp)
        return deduped

    def _route_forward_at_end(self, ag: CrowdAgent) -> np.ndarray:
        vel = ag.walker.get_velocity()
        vel_xy = np.array([float(vel.x), float(vel.y)], dtype=np.float32)
        vel_norm = float(np.linalg.norm(vel_xy))
        if vel_norm > 0.05:
            return (vel_xy / vel_norm).astype(np.float32)
        if len(ag.route_world) >= 2:
            x0, y0, _ = ag.route_world[-2]
            x1, y1, _ = ag.route_world[-1]
            seg = np.array([float(x1 - x0), float(y1 - y0)], dtype=np.float32)
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm > 1e-6:
                return (seg / seg_norm).astype(np.float32)
        _anchor, forward, _lateral = self._pair_endpoint_frame(ag.pair_lane, ag.pair_endpoint)
        return forward.astype(np.float32)

    def _build_target_turnaround_route(
        self,
        ag: CrowdAgent,
        loc: carla.Location,
        next_endpoint: int,
    ) -> List[Tuple[float, float, float]]:
        start_offset = float(ag.route_lateral_offset_m)
        route_offset = self._target_route_offset_for_endpoint(
            ag.preferred_lateral_offset_m,
            next_endpoint,
        )
        ag.route_lateral_offset_m = route_offset
        spacing = max(float(ag.route_spacing_m), 0.8)
        centerline = self._build_rounded_l_shape_centerline(ag.pair_lane, next_endpoint, spacing=spacing)
        z = max(float(self.spawn_z), float(loc.z), float(ag.target.z))
        return_polyline = self._offset_centerline_samples(centerline, route_offset, z=z)
        if len(return_polyline) <= 1:
            return []

        total_len = self._polyline_length(return_polyline)
        start_s, _dist = self._project_onto_polyline(loc, return_polyline)
        merge_s = min(total_len, start_s + max(float(self.params["target_turnaround_merge_m"]), 0.8))
        merge_loc = self._location_on_location_polyline(return_polyline, merge_s)

        current_forward = self._route_forward_at_end(ag)
        next_s = min(total_len, merge_s + max(float(self.roi_map.resolution), 0.5))
        next_loc = self._location_on_location_polyline(return_polyline, next_s)
        return_dir = np.array([float(next_loc.x - merge_loc.x), float(next_loc.y - merge_loc.y)], dtype=np.float32)
        return_norm = float(np.linalg.norm(return_dir))
        if return_norm <= 1e-6:
            return_dir = -current_forward
        else:
            return_dir = return_dir / return_norm

        p0 = np.array([float(loc.x), float(loc.y)], dtype=np.float32)
        p3 = np.array([float(merge_loc.x), float(merge_loc.y)], dtype=np.float32)
        forward_m = max(float(self.params["target_turnaround_forward_m"]), 0.2)
        side_m = max(float(self.params["target_turnaround_side_m"]), 0.2)
        c1 = p0 + current_forward * forward_m
        c2 = p3 - return_dir * side_m
        samples = max(4, int(self.params["target_turnaround_samples"]))

        route: List[Tuple[float, float, float]] = []
        for i in range(1, samples + 1):
            t = float(i) / float(samples)
            smooth_t = t * t * (3.0 - 2.0 * t)
            xy = (
                ((1.0 - t) ** 3) * p0
                + 3.0 * ((1.0 - t) ** 2) * t * c1
                + 3.0 * (1.0 - t) * (t ** 2) * c2
                + (t ** 3) * p3
            )
            # Blend a small body-frame offset correction through the U-turn so
            # the target does not snap laterally when the return leg's side
            # clearance flips with the walking direction.
            offset_delta = (1.0 - smooth_t) * (start_offset - route_offset)
            if abs(offset_delta) > 1e-3:
                tangent = (
                    3.0 * ((1.0 - t) ** 2) * (c1 - p0)
                    + 6.0 * (1.0 - t) * t * (c2 - c1)
                    + 3.0 * (t ** 2) * (p3 - c2)
                )
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm > 1e-6:
                    tangent = tangent / tangent_norm
                    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
                    xy = xy + normal * float(offset_delta)
            wp = self._walkable_location_near_xy(float(xy[0]), float(xy[1]), z)
            if route and float(np.hypot(wp.x - route[-1][0], wp.y - route[-1][1])) < 0.25:
                continue
            route.append((float(wp.x), float(wp.y), float(wp.z)))
        if route:
            print(
                f"[NPC] target_turnaround track={ag.track_id} next_ep={next_endpoint} "
                f"route_offset={start_offset:.2f}->{route_offset:.2f} samples={len(route)} "
                f"merge=({merge_loc.x:.2f},{merge_loc.y:.2f})"
            )
        return route

    def _pair_endpoint_frame(
        self,
        lane: int,
        endpoint_idx: int,
    ) -> Tuple[carla.Location, np.ndarray, np.ndarray]:
        anchor = self._pair_anchor(lane, endpoint_idx)
        lane_forward, lane_lateral = self._pair_lane_frame(lane)
        if endpoint_idx == 1:
            forward = lane_forward
        else:
            forward = -lane_forward
        lateral = lane_lateral
        return anchor, forward, lateral

    def _sample_route_personality(
        self,
        is_target: bool,
        lane_spawn_index: int,
    ) -> Tuple[float, float, float, float, float]:
        spacing_base = max(float(self.params["corner_route_waypoint_spacing_m"]), 0.8)
        lookahead_base = max(float(self.params["corner_route_start_lookahead_m"]), 0.5)
        if is_target:
            lateral = self.rng.uniform(
                -float(self.params["target_lateral_jitter_m"]),
                float(self.params["target_lateral_jitter_m"]),
            )
            return lateral, spacing_base, lookahead_base, 0.0, 0.0

        spacing = max(
            0.8,
            spacing_base
            + self.rng.uniform(
                -float(self.params["corner_route_spacing_jitter_m"]),
                float(self.params["corner_route_spacing_jitter_m"]),
            ),
        )
        lookahead = max(
            0.5,
            lookahead_base
            + self.rng.uniform(
                -float(self.params["corner_route_lookahead_jitter_m"]),
                float(self.params["corner_route_lookahead_jitter_m"]),
            ),
        )
        phase = self.rng.uniform(
            float(self.params["corner_route_longitudinal_phase_min_m"]),
            float(self.params["corner_route_longitudinal_phase_max_m"]),
        )
        turn_radius_offset = self.rng.uniform(
            -float(self.params["crowd_turn_radius_jitter_m"]),
            float(self.params["crowd_turn_radius_jitter_m"]),
        )
        lateral = self._assign_pair_lateral_offset(lane_spawn_index)
        return lateral, spacing, lookahead, phase, turn_radius_offset

    def _assign_pair_lateral_offset(self, lane_spawn_index: int) -> float:
        span_cfg = float(self.params.get("npc_lateral_personal_span_m", 0.9))
        lane_sep = float(self.params["right_hand_lane_sep_m"])
        half_width = float(self.params["pair_target_band_half_width_m"])
        sublane_span = min(max(0.15, span_cfg), max(0.15, half_width - lane_sep - 0.15))
        golden = 0.61803398875
        u = (float(lane_spawn_index) * golden) % 1.0
        slot = (u * 2.0 - 1.0) * sublane_span
        jitter = float(self.rng.uniform(-1.0, 1.0)) * float(self.params.get("npc_lateral_jitter_m", 0.15))
        return float(np.clip(slot + jitter, -sublane_span, sublane_span))

    def _pair_direction_sign(self, endpoint_idx: int) -> float:
        return 1.0 if endpoint_idx == 1 else -1.0

    def _pair_directional_base_offset(self, ag: CrowdAgent) -> float:
        half_width = float(self.params["pair_target_band_half_width_m"])
        base_sep = min(float(self.params["right_hand_lane_sep_m"]), max(0.0, half_width - 0.35))
        direction_sign = self._pair_direction_sign(ag.pair_endpoint)
        return float(direction_sign * base_sep)

    def _pair_target_lateral_offset(self, ag: CrowdAgent) -> float:
        half_width = float(self.params["pair_target_band_half_width_m"])
        offset = self._pair_directional_base_offset(ag) + float(ag.preferred_lateral_offset_m)
        return float(np.clip(offset, -half_width, half_width))

    def _pair_midroute_lateral_offset(self, ag: CrowdAgent) -> float:
        half_width = float(self.params["pair_target_band_half_width_m"])
        directional = self._pair_directional_base_offset(ag) * 0.35
        mid_half_width = min(half_width * 0.6, max(0.45, half_width - 0.45))
        offset = directional + float(ag.preferred_lateral_offset_m)
        return float(np.clip(offset, -mid_half_width, mid_half_width))

    def _effective_pair_lateral_offset(self, ag: CrowdAgent) -> float:
        half_width = float(self.params["pair_target_band_half_width_m"])
        if ag.is_target:
            offset = self._target_route_offset_for_endpoint(
                ag.preferred_lateral_offset_m,
                ag.pair_endpoint,
            )
        else:
            offset = self._pair_target_lateral_offset(ag)
        return float(np.clip(offset, -half_width, half_width))

    def _set_pair_endpoint(self, ag: CrowdAgent, endpoint_idx: int) -> None:
        ag.pair_endpoint = int(endpoint_idx)
        ag.flow_side = 1 if ag.pair_endpoint == 1 else -1

    def _pair_target_band_points_for_agent(self, ag: CrowdAgent) -> List[carla.Location]:
        anchor = self._pair_anchor(ag.pair_lane, ag.pair_endpoint)
        _forward, lateral = self._pair_lane_frame(ag.pair_lane)
        half_width = float(self.params["pair_target_band_half_width_m"])
        center = self._effective_pair_lateral_offset(ag)
        local_half_width = min(0.6, max(0.35, half_width * 0.35))
        spacing = max(float(self.roi_map.resolution), 0.35)
        steps = max(1, int(np.ceil(local_half_width / spacing)))
        offsets = np.linspace(center - local_half_width, center + local_half_width, num=2 * steps + 1, dtype=np.float32)

        band_points: List[carla.Location] = []
        for off in offsets:
            off_clip = float(np.clip(float(off), -half_width, half_width))
            x = float(anchor.x + lateral[0] * off_clip)
            y = float(anchor.y + lateral[1] * off_clip)
            if self.roi_map.is_world_walkable(x, y):
                band_points.append(carla.Location(x=x, y=y, z=float(anchor.z)))
        if not band_points:
            band_points.append(anchor)
        return band_points

    def _sample_pair_target(self, ag: CrowdAgent) -> carla.Location:
        if ag.route_is_lane_aware and ag.pair_lane in (0, 1) and ag.pair_endpoint in (0, 1):
            if ag.is_target:
                route_lateral_offset_m = self._target_route_offset_for_endpoint(
                    ag.preferred_lateral_offset_m,
                    ag.pair_endpoint,
                )
            else:
                route_lateral_offset_m = self._right_hand_route_offset(
                    ag.preferred_lateral_offset_m,
                    is_target=False,
                )
            return self._route_terminal_location(
                lane=ag.pair_lane,
                target_ep=ag.pair_endpoint,
                route_lateral_offset_m=route_lateral_offset_m,
            )
        band_points = self._pair_target_band_points_for_agent(ag)
        anchor_radius = max(float(self.roi_map.resolution) * 0.75, 0.35)
        return self._sample_target(
            band_points,
            anchor_radius_m=anchor_radius,
        )

    def _route_terminal_location(
        self,
        lane: int,
        target_ep: int,
        route_lateral_offset_m: float,
    ) -> carla.Location:
        spacing = max(float(self.params["corner_route_waypoint_spacing_m"]), 0.8)
        centerline = self._build_rounded_l_shape_centerline(lane, target_ep, spacing=spacing)
        z = max(float(self.spawn_z), float(self._pair_anchor(lane, target_ep).z))
        offset_polyline = self._offset_centerline_samples(centerline, float(route_lateral_offset_m), z=z)
        if offset_polyline:
            return offset_polyline[-1]
        anchor = self._pair_anchor(lane, target_ep)
        return carla.Location(x=float(anchor.x), y=float(anchor.y), z=float(anchor.z))

    def _sample_pair_target_for_profile(
        self,
        lane: int,
        endpoint_idx: int,
        route_lateral_offset_m: float,
    ) -> carla.Location:
        return self._route_terminal_location(lane, endpoint_idx, route_lateral_offset_m)

    def _nearest_pair_lane(self, loc: carla.Location) -> int:
        best_lane = 0
        best_dist = 1e18
        for lane in range(len(self.pairs)):
            _s, dist = self._project_onto_polyline(loc, self._directed_l_shape_polyline(lane, 1))
            if dist < best_dist:
                best_dist = dist
                best_lane = lane
        return int(best_lane)

    def _choose_target_endpoint_for_crowd(
        self,
        loc: carla.Location,
        lane: int,
        flow_counts: Dict[Tuple[int, int], int],
    ) -> int:
        polyline = self._directed_l_shape_polyline(lane, 1)
        total = 0.0
        for i in range(len(polyline) - 1):
            total += float(np.hypot(polyline[i + 1].x - polyline[i].x, polyline[i + 1].y - polyline[i].y))
        s, _dist = self._project_onto_polyline(loc, polyline)
        t = 0.5 if total <= 1e-6 else float(np.clip(s / total, 0.0, 1.0))
        if t < 0.42:
            return 1
        if t > 0.58:
            return 0
        return 1 if flow_counts.get((lane, 1), 0) <= flow_counts.get((lane, 0), 0) else 0

    def _make_spawn_profile(
        self,
        track_id: str,
        is_target: bool,
        spawn_loc: carla.Location,
        pair_lane: int,
        pair_endpoint: int,
        initial_pair_endpoint: int,
        base_speed: float,
        preferred_lateral_offset_m: float,
        route_spacing_m: float,
        route_lookahead_m: float,
        route_longitudinal_phase_m: float,
        route_turn_radius_offset_m: float,
    ) -> SpawnProfile:
        route_is_lane_aware = (
            self._is_pair_flow_mode()
            and str(self.params.get("corner_route_mode", "astar")) == "l_shape"
            and pair_lane in (0, 1)
            and pair_endpoint in (0, 1)
        )
        if is_target:
            route_lateral_offset_m = self._target_route_offset_for_endpoint(
                preferred_lateral_offset_m,
                pair_endpoint,
            )
        else:
            route_lateral_offset_m = self._right_hand_route_offset(
                preferred_lateral_offset_m,
                is_target=False,
            )
        target_loc = self._sample_pair_target_for_profile(pair_lane, pair_endpoint, route_lateral_offset_m)
        if route_is_lane_aware:
            route = self._build_l_shape_route_world(
                spawn_loc,
                target_loc,
                lane=pair_lane,
                target_ep=pair_endpoint,
                route_lateral_offset_m=route_lateral_offset_m,
                route_spacing_m=route_spacing_m,
                route_lookahead_m=route_lookahead_m,
                route_longitudinal_phase_m=route_longitudinal_phase_m,
                route_turn_radius_offset_m=route_turn_radius_offset_m,
                is_target=bool(is_target),
            )
            if route:
                x, y, z = route[-1]
                target_loc = carla.Location(x=float(x), y=float(y), z=float(z))
        else:
            route = self._plan_route_world_between(spawn_loc, target_loc)
        yaw = self._yaw_from_route_or_target(spawn_loc, target_loc, route)
        flow_side = 1 if pair_endpoint == 1 else -1
        return SpawnProfile(
            track_id=track_id,
            is_target=is_target,
            spawn_loc=carla.Location(x=float(spawn_loc.x), y=float(spawn_loc.y), z=float(spawn_loc.z)),
            target_loc=carla.Location(x=float(target_loc.x), y=float(target_loc.y), z=float(target_loc.z)),
            base_speed=float(base_speed),
            pair_lane=int(pair_lane),
            pair_endpoint=int(pair_endpoint),
            initial_pair_endpoint=int(initial_pair_endpoint),
            preferred_lateral_offset_m=float(preferred_lateral_offset_m),
            route_lateral_offset_m=float(route_lateral_offset_m),
            route_spacing_m=float(route_spacing_m),
            route_lookahead_m=float(route_lookahead_m),
            route_longitudinal_phase_m=float(route_longitudinal_phase_m),
            route_turn_radius_offset_m=float(route_turn_radius_offset_m),
            initial_yaw_deg=float(yaw),
            flow_side=flow_side,
            target_band=flow_side,
            route_world=route,
            route_is_lane_aware=route_is_lane_aware,
        )

    def _build_target_spawn_profile(
        self,
        track_id: str,
        taken_locs: List[carla.Location],
    ) -> SpawnProfile:
        lane = self.rng.randrange(len(self.pairs))
        start_ep = self.rng.randrange(2)
        target_ep = self._flip_pair_endpoint(start_ep)
        min_sep = float(self.params["npc_spawn_min_sep_m"])
        preferred_offset, spacing, lookahead, phase, turn_radius_offset = self._sample_route_personality(
            is_target=True,
            lane_spawn_index=0,
        )
        route_offset = self._target_route_offset_for_endpoint(preferred_offset, target_ep)
        use_side_clear_spawn = (
            str(self.params.get("target_lane_bias_mode", "right_hand")) == "leave_follow_side_clear"
            and str(self.params.get("follow_position", "back")) in ("left_side", "right_side")
        )
        spawn_loc = self._sample_centered_endpoint_location(
            lane,
            start_ep,
            taken_locs,
            min_sep,
            target_ep=target_ep if use_side_clear_spawn else None,
            preferred_route_offset_m=route_offset if use_side_clear_spawn else None,
        )
        profile = self._make_spawn_profile(
            track_id=track_id,
            is_target=True,
            spawn_loc=spawn_loc,
            pair_lane=lane,
            pair_endpoint=target_ep,
            initial_pair_endpoint=start_ep,
            base_speed=self._sample_personal_speed(is_target=True),
            preferred_lateral_offset_m=preferred_offset,
            route_spacing_m=spacing,
            route_lookahead_m=lookahead,
            route_longitudinal_phase_m=phase,
            route_turn_radius_offset_m=turn_radius_offset,
        )
        print(
            f"[NPC] target_spawn track={profile.track_id} lane={lane} "
            f"start_ep={start_ep} goal_ep={target_ep} "
            f"follow_position={self.params.get('follow_position', 'back')} "
            f"lane_bias={self.params.get('target_lane_bias_mode', 'right_hand')} "
            f"route_offset={profile.route_lateral_offset_m:.2f} "
            f"loc=({profile.spawn_loc.x:.2f},{profile.spawn_loc.y:.2f}) "
            f"yaw={profile.initial_yaw_deg:.1f}"
        )
        return profile

    def _build_crowd_spawn_profile(
        self,
        track_id: str,
        lane_counts: List[int],
        taken_locs: List[carla.Location],
        reserved_locs: List[carla.Location],
        flow_counts: Dict[Tuple[int, int], int],
    ) -> SpawnProfile:
        spawn_loc = self._sample_roi_location_with_spacing(
            taken_locs,
            float(self.params["npc_spawn_min_sep_m"]),
            reserved_locs=reserved_locs,
            reserved_min_sep_m=float(self.params["robot_spawn_clearance_m"]),
        )
        lane = self._nearest_pair_lane(spawn_loc)
        target_ep = self._choose_target_endpoint_for_crowd(spawn_loc, lane, flow_counts)
        preferred_offset, spacing, lookahead, phase, turn_radius_offset = self._sample_route_personality(
            is_target=False,
            lane_spawn_index=lane_counts[lane],
        )
        lane_counts[lane] += 1
        flow_counts[(lane, target_ep)] = flow_counts.get((lane, target_ep), 0) + 1
        return self._make_spawn_profile(
            track_id=track_id,
            is_target=False,
            spawn_loc=spawn_loc,
            pair_lane=lane,
            pair_endpoint=target_ep,
            initial_pair_endpoint=self._flip_pair_endpoint(target_ep),
            base_speed=self._sample_personal_speed(is_target=False),
            preferred_lateral_offset_m=preferred_offset,
            route_spacing_m=spacing,
            route_lookahead_m=lookahead,
            route_longitudinal_phase_m=phase,
            route_turn_radius_offset_m=turn_radius_offset,
        )

    def _build_spawn_profiles(
        self,
        num_walkers: int,
        target_track_id: Optional[str],
    ) -> List[SpawnProfile]:
        profiles: List[SpawnProfile] = []
        taken_locs: List[carla.Location] = []
        reserved_locs: List[carla.Location] = []
        lane_counts = [0 for _ in self.pairs]
        flow_counts: Dict[Tuple[int, int], int] = {}
        target_id = target_track_id or "N01"

        for i in range(num_walkers):
            track_id = f"N{i+1:02d}"
            if track_id == target_id:
                profile = self._build_target_spawn_profile(track_id, taken_locs)
                robot_reservation = self._robot_spawn_reservation_for_target(profile)
                if robot_reservation is not None:
                    reserved_locs.append(robot_reservation)
                    print(
                        f"[NPC] robot_spawn_reserved "
                        f"loc=({robot_reservation.x:.2f},{robot_reservation.y:.2f}) "
                        f"clearance={float(self.params['robot_spawn_clearance_m']):.2f}"
                    )
            else:
                profile = self._build_crowd_spawn_profile(
                    track_id,
                    lane_counts,
                    taken_locs,
                    reserved_locs,
                    flow_counts,
                )
            profiles.append(profile)
            taken_locs.append(profile.spawn_loc)

        print(
            "[NPC] crowd_spawn policy=roi_random "
            f"min_sep={float(self.params['npc_spawn_min_sep_m']):.2f} "
            f"right_hand_lane_sep={float(self.params['right_hand_lane_sep_m']):.2f}"
        )
        for lane in range(len(self.pairs)):
            print(
                f"[NPC] flow_counts lane{lane} "
                f"ep0_goal={flow_counts.get((lane, 0), 0)} "
                f"ep1_goal={flow_counts.get((lane, 1), 0)}"
            )
        return profiles

    def _spawn_one_from_profile(
        self,
        bp: carla.ActorBlueprint,
        profile: SpawnProfile,
    ) -> Optional[CrowdAgent]:
        walker = None
        # Height retries first, then a fixed (RNG-free) ring of small lateral
        # offsets on walkable cells.  Deterministic so a transient CARLA spawn
        # collision neither drops the walker nor shifts the seeded RNG stream.
        offsets = [(0.0, 0.0)]
        for r in (0.4, 0.8):
            for k in range(8):
                ang = 2.0 * math.pi * k / 8.0
                offsets.append((r * math.cos(ang), r * math.sin(ang)))
        for ox, oy in offsets:
            x = float(profile.spawn_loc.x) + ox
            y = float(profile.spawn_loc.y) + oy
            if (ox or oy) and not self.roi_map.is_world_walkable(x, y):
                continue
            for dz in ((0.0, 0.25, 0.5) if not (ox or oy) else (0.25,)):
                loc = carla.Location(x=x, y=y, z=float(profile.spawn_loc.z + dz))
                spawn_tf = carla.Transform(
                    loc,
                    carla.Rotation(yaw=float(profile.initial_yaw_deg)),
                )
                walker = self.world.try_spawn_actor(bp, spawn_tf)
                if walker is not None:
                    break
            if walker is not None:
                break
        if walker is None:
            return None
        ag = CrowdAgent(
            walker=walker,
            track_id=profile.track_id,
            speed=float(profile.base_speed),
            target=carla.Location(
                x=float(profile.target_loc.x),
                y=float(profile.target_loc.y),
                z=float(profile.target_loc.z),
            ),
            is_target=bool(profile.is_target),
            base_speed=float(profile.base_speed),
            flow_side=int(profile.flow_side),
            target_band=int(profile.target_band),
            pair_lane=int(profile.pair_lane),
            pair_endpoint=int(profile.pair_endpoint),
            initial_pair_endpoint=int(profile.initial_pair_endpoint),
            preferred_lateral_offset_m=float(profile.preferred_lateral_offset_m),
            route_lateral_offset_m=float(profile.route_lateral_offset_m),
            route_spacing_m=float(profile.route_spacing_m),
            route_lookahead_m=float(profile.route_lookahead_m),
            route_longitudinal_phase_m=float(profile.route_longitudinal_phase_m),
            route_turn_radius_offset_m=float(profile.route_turn_radius_offset_m),
            route_world=list(profile.route_world),
            route_idx=0,
            last_replan_t=0.0,
            last_retarget_t=self._sim_t,
            route_is_lane_aware=bool(profile.route_is_lane_aware),
        )
        return ag

    def _flip_pair_endpoint(self, endpoint_idx: int) -> int:
        return 1 - endpoint_idx if endpoint_idx in (0, 1) else 0

    def _is_pair_flow_mode(self) -> bool:
        return self.flow_mode in ("pair_12_34", "pair_13_24")

    def _spawn_with_profiles(self, num_walkers: int, target_track_id: Optional[str]) -> None:
        self._sfm_planner = self._build_sfm_planner()
        bps = [bp for bp in self.world.get_blueprint_library().filter("walker.pedestrian.*")
               if bp.id not in _NON_HUMAN_WALKER_BLUEPRINTS]
        if num_walkers > len(bps):
            raise RuntimeError(
                f"[NPC] num_walkers={num_walkers} exceeds blueprint pool "
                f"({len(bps)}); each NPC needs a unique appearance.")
        bps = self.rng.sample(bps, num_walkers)
        failed = 0
        profiles = self._build_spawn_profiles(num_walkers, target_track_id)
        target_id = target_track_id or "N01"
        target_spawned = False
        # Legend: some distractors deliberately reuse the target's appearance.
        tgt_idx = next((k for k, p in enumerate(profiles) if p.track_id == target_id), 0)
        non_tgt_idx = [k for k in range(len(profiles)) if k != tgt_idx]
        n_look = min(int(self.params.get("num_lookalikes", 0)), len(non_tgt_idx))
        lookalike_idx = set(self.rng.sample(non_tgt_idx, n_look)) if n_look > 0 else set()
        for k in lookalike_idx:
            bps[k] = bps[tgt_idx]
        self.lookalike_track_ids = []
        for i, profile in enumerate(profiles):
            bp = bps[i]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")
            ag = self._spawn_one_from_profile(bp, profile)
            if ag is None and profile.track_id == target_id:
                retry_limit = max(int(self.params["target_spawn_retry_attempts"]), 1)
                taken_locs = [a.walker.get_location() for a in self.agents]
                taken_locs.extend(p.spawn_loc for p in profiles[i + 1:])
                for attempt in range(1, retry_limit + 1):
                    retry_profile = self._build_target_spawn_profile(profile.track_id, taken_locs)
                    ag = self._spawn_one_from_profile(bp, retry_profile)
                    if ag is not None:
                        print(
                            f"[NPC] target_spawn_retry track={profile.track_id} "
                            f"attempt={attempt}/{retry_limit} success"
                        )
                        break
                    taken_locs.append(retry_profile.spawn_loc)
                    print(
                        f"[NPC][WARN] target_spawn_retry track={profile.track_id} "
                        f"attempt={attempt}/{retry_limit} failed"
                    )
                if ag is None:
                    raise RuntimeError(
                        f"Target NPC {profile.track_id} failed to spawn after "
                        f"{retry_limit + 1} attempts. Check endpoint sampling, "
                        "ROI clearance, and CARLA walker collisions."
                    )
            if ag is None:
                failed += 1
                continue
            if ag.track_id == target_id:
                target_spawned = True
            if i in lookalike_idx:
                self.lookalike_track_ids.append(ag.track_id)
            if not ag.route_is_lane_aware:
                self._replan_route(ag, now_t=self._sim_t)
            self.agents.append(ag)
        self._spawned = True
        print(f"[NPC] spawned={len(self.agents)} failed={failed} lookalikes={self.lookalike_track_ids}")
        if len(self.agents) == 0:
            raise RuntimeError("NPC spawn failed: 0 actors created. Check ROI / flow points / nav mesh.")
        if target_track_id is not None and not target_spawned:
            raise RuntimeError(f"Target NPC {target_track_id} was not spawned.")

    def spawn(self, num_walkers: int, target_track_id: Optional[str] = None) -> None:
        if self._spawned:
            return
        use_profile_spawn = (
            self._is_pair_flow_mode()
            and str(self.params.get("target_spawn_policy", "legacy")) != "legacy"
            and str(self.params.get("crowd_spawn_policy", "legacy")) != "legacy"
        )
        if use_profile_spawn:
            self._spawn_with_profiles(num_walkers, target_track_id=target_track_id)
            return

        self._sfm_planner = self._build_sfm_planner()
        bps = [bp for bp in self.world.get_blueprint_library().filter("walker.pedestrian.*")
               if bp.id not in _NON_HUMAN_WALKER_BLUEPRINTS]
        if num_walkers > len(bps):
            raise RuntimeError(
                f"[NPC] num_walkers={num_walkers} exceeds blueprint pool "
                f"({len(bps)}); each NPC needs a unique appearance.")
        bps = self.rng.sample(bps, num_walkers)
        failed = 0
        pair_lane_counts = [0, 0]
        for i in range(num_walkers):
            bp = bps[i]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")
            if self.flow_mode == "free_3":
                # Distribute NPCs across 3 home points, target = random other point.
                n_pts = len(self.flow_points)
                home_idx = i % n_pts
                other_indices = [j for j in range(n_pts) if j != home_idx]
                tgt_idx = self.rng.choice(other_indices)
                ag = self._spawn_one_random(
                    bp,
                    track_id=f"N{i+1:02d}",
                    target=self._sample_target([self.flow_points[tgt_idx]]),
                )
                if ag is None:
                    failed += 1
                    continue
                ag.pair_lane = home_idx   # reuse pair_lane as home_idx
                ag.pair_endpoint = tgt_idx  # reuse pair_endpoint as current target_idx
                self._replan_route(ag, now_t=self._sim_t)
                self.agents.append(ag)
                continue
            if self._is_pair_flow_mode():
                lane = i % 2
                # alternate direction within each lane
                dir_sign = 1 if ((i // 2) % 2 == 0) else -1
                start_ep = 0 if dir_sign > 0 else 1
                target_ep = 1 - start_ep
                preferred_offset = self._assign_pair_lateral_offset(pair_lane_counts[lane])
                pair_lane_counts[lane] += 1
                target = self._pair_anchor(lane, target_ep)
                if self.params["pair_spawn_mode"] == "paired_endpoint":
                    ag = self._spawn_one_at_endpoint(
                        bp,
                        endpoint=self.pairs[lane][start_ep],
                        track_id=f"N{i+1:02d}",
                        target=target,
                    )
                else:
                    ag = self._spawn_one_random(bp, track_id=f"N{i+1:02d}", target=target)
            else:
                flow_side = 1 if (i % 2 == 0) else -1
                target_points = self.side_b if flow_side > 0 else self.side_a
                ag = self._spawn_one_random(
                    bp,
                    track_id=f"N{i+1:02d}",
                    target=self._sample_target(target_points),
                )
            if ag is None:
                failed += 1
                continue
            if self._is_pair_flow_mode():
                ag.pair_lane = lane
                self._set_pair_endpoint(ag, target_ep)
                ag.preferred_lateral_offset_m = preferred_offset
                ag.target = self._sample_pair_target(ag)
            else:
                ag.flow_side = flow_side
                ag.target_band = 1 if flow_side > 0 else -1
            self._replan_route(ag, now_t=self._sim_t)
            self.agents.append(ag)
        self._spawned = True
        print(f"[NPC] spawned={len(self.agents)} failed={failed}")
        if len(self.agents) == 0:
            raise RuntimeError("NPC spawn failed: 0 actors created. Check ROI / flow points / nav mesh.")

    def set_motion_active(self, active: bool) -> None:
        self._active_move = active

    def _is_stuck(self, ag: CrowdAgent, now_t: float) -> bool:
        loc = ag.walker.get_location()
        ag.history.append((now_t, loc.x, loc.y))
        while len(ag.history) > 2 and now_t - ag.history[0][0] > self.params["stuck_window_sec"]:
            ag.history.pop(0)
        if len(ag.history) < 2:
            return False
        x0, y0 = ag.history[0][1], ag.history[0][2]
        x1, y1 = ag.history[-1][1], ag.history[-1][2]
        moved = float(np.hypot(x1 - x0, y1 - y0))
        return moved < self.params["stuck_dist_threshold"]

    def _build_sfm_planner(self) -> SFMPlanner:
        cfg = SFMPlannerConfig(
            tau=float(self.params["sfm_tau"]),
            desired_speed_scale=float(self.params["sfm_desired_speed_scale"]),
            ped_A=float(self.params["sfm_ped_A"]),
            ped_B=float(self.params["sfm_ped_B"]),
            ped_radius=float(self.params["sfm_ped_radius"]),
            robot_A=float(self.params["sfm_robot_A"]),
            robot_B=float(self.params["sfm_robot_B"]),
            robot_radius=float(self.params["sfm_robot_radius"]),
            wall_A=float(self.params["sfm_wall_A"]),
            wall_B=float(self.params["sfm_wall_B"]),
            wall_influence_dist=float(self.params["sfm_wall_influence_dist"]),
            tangential_weight=float(self.params["sfm_tangential_weight"]),
            robot_tangential_weight=float(self.params["sfm_robot_tangential_weight"]),
            anisotropy_lambda=float(self.params["sfm_anisotropy_lambda"]),
            prediction_enabled=bool(self.params["sfm_prediction_enabled"]),
            prediction_horizon=float(self.params["sfm_prediction_horizon"]),
            close_risk_margin=float(self.params["sfm_close_risk_margin"]),
            headon_cos_threshold=float(self.params["sfm_headon_cos_threshold"]),
            headon_front_dot_threshold=float(self.params["sfm_headon_front_dot_threshold"]),
            headon_lateral_trigger_m=float(self.params["sfm_headon_lateral_trigger_m"]),
            headon_bias_weight=float(self.params["sfm_headon_bias_weight"]),
            headon_min_risk=float(self.params["sfm_headon_min_risk"]),
            max_force=float(self.params["sfm_max_force"]),
            max_speed=max(float(self.params["max_speed"]) * 1.1, float(self.params["min_speed"])),
            neighbor_radius=float(self.params["sfm_neighbor_radius"]),
            max_neighbors=int(self.params["sfm_max_neighbors"]),
        )
        return SFMPlanner(cfg)

    def _get_local_goal_xy(self, ag: CrowdAgent, loc: carla.Location, now_t: float) -> Tuple[float, float]:
        goal_x, goal_y = ag.target.x, ag.target.y
        if self.params["navigation_mode"] == "astar" and len(ag.route_world) > 0:
            while ag.route_idx < len(ag.route_world):
                wx, wy, _wz = ag.route_world[ag.route_idx]
                if np.hypot(loc.x - wx, loc.y - wy) <= self.params["astar_waypoint_arrive_dist"]:
                    ag.route_idx += 1
                else:
                    break
            if ag.route_idx < len(ag.route_world):
                goal_x, goal_y = ag.route_world[ag.route_idx][0], ag.route_world[ag.route_idx][1]
            elif np.hypot(loc.x - ag.target.x, loc.y - ag.target.y) > self.params["arrive_threshold"] * 1.2:
                self._replan_route(ag, now_t=now_t)
                if ag.route_idx < len(ag.route_world):
                    goal_x, goal_y = ag.route_world[ag.route_idx][0], ag.route_world[ag.route_idx][1]
        return float(goal_x), float(goal_y)

    def _build_agent_state(
        self,
        ag: CrowdAgent,
        goal_x: float,
        goal_y: float,
        reference_forward_xy: np.ndarray,
    ) -> AgentKinematics:
        loc = ag.walker.get_location()
        vel = ag.walker.get_velocity()
        return AgentKinematics(
            track_id=ag.track_id,
            position_xy=np.array([float(loc.x), float(loc.y)], dtype=np.float32),
            velocity_xy=np.array([float(vel.x), float(vel.y)], dtype=np.float32),
            radius=float(self.params["sfm_ped_radius"]),
            desired_speed=float(ag.speed),
            goal_xy=np.array([float(goal_x), float(goal_y)], dtype=np.float32),
            reference_forward_xy=reference_forward_xy.astype(np.float32),
            pass_side_pref=int(ag.pass_side_pref),
        )

    def _get_reference_forward_xy(
        self,
        ag: CrowdAgent,
        loc: carla.Location,
        goal_x: float,
        goal_y: float,
    ) -> np.ndarray:
        if ag.route_is_lane_aware and ag.route_idx + 1 < len(ag.route_world):
            x0, y0, _ = ag.route_world[ag.route_idx]
            x1, y1, _ = ag.route_world[ag.route_idx + 1]
            seg = np.array([float(x1 - x0), float(y1 - y0)], dtype=np.float32)
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm > 1e-6:
                return (seg / seg_norm).astype(np.float32)

        if self._is_pair_flow_mode() and ag.pair_lane in (0, 1) and ag.pair_endpoint in (0, 1):
            _anchor, forward, _lateral = self._pair_endpoint_frame(ag.pair_lane, ag.pair_endpoint)
            return forward.astype(np.float32)

        if self.params["navigation_mode"] == "astar" and ag.route_idx + 1 < len(ag.route_world):
            x0, y0, _ = ag.route_world[ag.route_idx]
            x1, y1, _ = ag.route_world[ag.route_idx + 1]
            seg = np.array([float(x1 - x0), float(y1 - y0)], dtype=np.float32)
            seg_norm = float(np.linalg.norm(seg))
            if seg_norm > 1e-6:
                return (seg / seg_norm).astype(np.float32)

        goal_vec = np.array([float(goal_x - loc.x), float(goal_y - loc.y)], dtype=np.float32)
        goal_norm = float(np.linalg.norm(goal_vec))
        if goal_norm > 1e-6:
            return (goal_vec / goal_norm).astype(np.float32)

        vel = ag.walker.get_velocity()
        vel_vec = np.array([float(vel.x), float(vel.y)], dtype=np.float32)
        vel_norm = float(np.linalg.norm(vel_vec))
        if vel_norm > 1e-6:
            return (vel_vec / vel_norm).astype(np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)

    def _apply_smoothed_pair_lateral_preference(
        self,
        ag: CrowdAgent,
        loc: carla.Location,
        goal_x: float,
        goal_y: float,
        reference_forward_xy: np.ndarray,
    ) -> Tuple[float, float]:
        if ag.route_is_lane_aware:
            return goal_x, goal_y
        if not (self._is_pair_flow_mode() and ag.pair_lane in (0, 1) and ag.pair_endpoint in (0, 1)):
            return goal_x, goal_y

        lateral_offset = self._pair_midroute_lateral_offset(ag)
        if abs(lateral_offset) <= 1e-3:
            return goal_x, goal_y

        _lane_forward, lateral = self._pair_lane_frame(ag.pair_lane)

        goal_vec = np.array([float(goal_x - loc.x), float(goal_y - loc.y)], dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist <= 1e-3:
            return goal_x, goal_y

        # Apply only part of the personal lane offset to the local reference point
        # so we get stable mid-route spreading without hard sideways turns.
        activation = float(np.clip(goal_dist / 1.2, 0.0, 1.0))
        offset_eff = lateral_offset * 0.55 * activation
        if abs(offset_eff) <= 1e-3:
            return goal_x, goal_y

        for scale in (1.0, 0.7, 0.4):
            x = float(goal_x + lateral[0] * offset_eff * scale)
            y = float(goal_y + lateral[1] * offset_eff * scale)
            if not self.roi_map.is_world_walkable(x, y):
                continue
            gx, gy = self.roi_map.world_to_cell(x, y)
            h, w = self.roi_map.free_grid.shape
            if 0 <= gx < w and 0 <= gy < h:
                clearance = float(self.roi_map.clearance_m[gy, gx])
                if clearance < max(0.35, float(self.params["sfm_ped_radius"]) * 1.2):
                    continue
            return x, y
        return goal_x, goal_y

    def _collect_dynamic_obstacles(
        self,
        agent_idx: int,
        loc_xy: np.ndarray,
        reference_forward_xy: np.ndarray,
        robot_actor: Optional[carla.Actor],
        target_track_id: Optional[str],
    ) -> List[DynamicObstacle]:
        scored_obstacles: List[Tuple[Tuple[float, float, float], DynamicObstacle]] = []
        for j, other in enumerate(self.agents):
            if j == agent_idx:
                continue
            o_loc = other.walker.get_location()
            o_vel = other.walker.get_velocity()
            pos_xy = np.array([float(o_loc.x), float(o_loc.y)], dtype=np.float32)
            rel = pos_xy - loc_xy
            dist = float(np.linalg.norm(rel))
            if dist <= 1e-6 or dist > float(self.params["sfm_neighbor_radius"]):
                continue
            ahead = float(np.dot(rel / dist, reference_forward_xy))
            scored_obstacles.append(
                (
                    (1.0, dist, -ahead),
                    DynamicObstacle(
                        kind="pedestrian",
                        position_xy=pos_xy,
                        velocity_xy=np.array([float(o_vel.x), float(o_vel.y)], dtype=np.float32),
                        radius=float(self.params["sfm_ped_radius"]),
                    ),
                )
            )

        is_target = (target_track_id is not None and self.agents[agent_idx].track_id == target_track_id)
        if robot_actor is not None and not is_target:
            r_loc = robot_actor.get_location()
            r_vel = robot_actor.get_velocity()
            pos_xy = np.array([float(r_loc.x), float(r_loc.y)], dtype=np.float32)
            rel = pos_xy - loc_xy
            dist = float(np.linalg.norm(rel))
            if 1e-6 < dist <= float(self.params["sfm_neighbor_radius"]):
                ahead = float(np.dot(rel / dist, reference_forward_xy))
                scored_obstacles.append(
                    (
                        (0.0, dist, -ahead),
                        DynamicObstacle(
                            kind="robot",
                            position_xy=pos_xy,
                            velocity_xy=np.array([float(r_vel.x), float(r_vel.y)], dtype=np.float32),
                            radius=float(self.params["sfm_robot_radius"]),
                        ),
                    )
                )
        scored_obstacles.sort(key=lambda item: item[0])
        max_neighbors = int(self.params["sfm_max_neighbors"])
        return [obs for _key, obs in scored_obstacles[:max_neighbors]]

    def _query_wall_state(self, ag: CrowdAgent, loc: carla.Location) -> WallQueryResult:
        gx, gy = self.roi_map.world_to_cell(loc.x, loc.y)
        h, w = self.roi_map.free_grid.shape
        if gx < 0 or gy < 0 or gx >= w or gy >= h:
            ag.wall_normal_valid = False
            return WallQueryResult(False, 0.0, np.zeros(2, dtype=np.float32))
        if not self.roi_map.is_world_walkable(loc.x, loc.y):
            ag.wall_normal_valid = False
            return WallQueryResult(False, 0.0, np.zeros(2, dtype=np.float32))

        fx = (float(loc.x) - float(self.roi_map.world_min[0])) / float(self.roi_map.resolution) - 0.5
        fy = (float(loc.y) - float(self.roi_map.world_min[1])) / float(self.roi_map.resolution) - 0.5
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        tx = float(np.clip(fx - x0, 0.0, 1.0))
        ty = float(np.clip(fy - y0, 0.0, 1.0))
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        def bilerp(arr: np.ndarray) -> float:
            v00 = float(arr[y0, x0])
            v10 = float(arr[y0, x1])
            v01 = float(arr[y1, x0])
            v11 = float(arr[y1, x1])
            return (
                (1.0 - tx) * (1.0 - ty) * v00
                + tx * (1.0 - ty) * v10
                + (1.0 - tx) * ty * v01
                + tx * ty * v11
            )

        clearance = bilerp(self.roi_map.clearance_m)
        grad_x = bilerp(self.roi_map.clearance_grad_x)
        grad_y = bilerp(self.roi_map.clearance_grad_y)
        grad = np.array([grad_x, grad_y], dtype=np.float32)
        norm = float(np.linalg.norm(grad))
        if norm <= float(self.params["sfm_wall_normal_min_grad"]):
            ag.wall_normal_valid = False
            return WallQueryResult(False, clearance, np.zeros(2, dtype=np.float32))
        raw_normal = grad / norm
        alpha = float(np.clip(self.params["sfm_wall_normal_smooth_alpha"], 0.0, 1.0))
        if ag.wall_normal_valid:
            smooth = (1.0 - alpha) * ag.wall_normal_xy + alpha * raw_normal
            smooth_norm = float(np.linalg.norm(smooth))
            if smooth_norm > 1e-6:
                raw_normal = smooth / smooth_norm
        ag.wall_normal_xy = raw_normal.astype(np.float32)
        ag.wall_normal_valid = True
        return WallQueryResult(True, clearance, ag.wall_normal_xy.copy())

    def _update_agent_target(
        self,
        ag: CrowdAgent,
        loc: carla.Location,
        now_t: float,
        stuck: bool,
    ) -> None:
        if ag.route_task_done:
            return
        arrived = (loc.distance(ag.target) < self.params["arrive_threshold"])
        timed_retarget = (now_t - ag.last_retarget_t > self.params["retarget_sec"])

        if self._is_pair_flow_mode() and ag.pair_lane in (0, 1):
            if ag.is_target and ag.route_phase == "turnaround":
                if arrived:
                    next_endpoint = ag.pending_pair_endpoint
                    ag.route_phase = "transit"
                    ag.pending_pair_endpoint = -1
                    self._set_pair_endpoint(ag, next_endpoint)
                    ag.target = self._sample_pair_target(ag)
                    ag.last_retarget_t = now_t
                    ag.speed = self._retarget_speed(ag)
                    self._replan_route(ag, now_t=now_t)
                return

            pair_goal_policy = str(self.params["pair_goal_policy"])
            if arrived:
                ag.completed_route_legs += 1
                if ag.is_target and ag.completed_route_legs >= int(self.params["target_required_legs"]):
                    ag.route_task_done = True
                    ag.speed = 0.0
                    ag.route_world = []
                    ag.route_idx = 0
                    return
                if ag.is_target:
                    next_endpoint = self._flip_pair_endpoint(ag.pair_endpoint)
                    route = self._build_target_turnaround_route(ag, loc, next_endpoint)
                    if route:
                        ag.route_phase = "turnaround"
                        ag.pending_pair_endpoint = next_endpoint
                        ag.route_world = route
                        ag.route_idx = 0
                        x, y, z = route[-1]
                        ag.target = carla.Location(x=float(x), y=float(y), z=float(z))
                        ag.last_retarget_t = now_t
                        ag.speed = min(self._retarget_speed(ag), float(self.params["target_turnaround_speed_mps"]))
                        return
                self._set_pair_endpoint(ag, self._flip_pair_endpoint(ag.pair_endpoint))
                ag.target = self._sample_pair_target(ag)
                ag.last_retarget_t = now_t
                ag.speed = self._retarget_speed(ag)
                self._replan_route(ag, now_t=now_t)
            elif pair_goal_policy == "retarget_on_block" and (stuck or timed_retarget):
                self._set_pair_endpoint(ag, self._flip_pair_endpoint(ag.pair_endpoint))
                ag.target = self._sample_pair_target(ag)
                ag.last_retarget_t = now_t
                ag.speed = self._retarget_speed(ag)
                self._replan_route(ag, now_t=now_t)
            elif stuck or timed_retarget:
                if timed_retarget:
                    ag.last_retarget_t = now_t
                self._replan_route(ag, now_t=now_t)
            return

        if not (arrived or timed_retarget or stuck):
            return
        if self.flow_mode == "free_3":
            n_pts = len(self.flow_points)
            other_indices = [j for j in range(n_pts) if j != ag.pair_endpoint]
            ag.pair_endpoint = self.rng.choice(other_indices)
            ag.target = self._sample_target([self.flow_points[ag.pair_endpoint]])
        else:
            ag.target_band = -ag.target_band if ag.target_band != 0 else (1 if ag.flow_side > 0 else -1)
            points = self.side_a if ag.target_band < 0 else self.side_b
            ag.target = self._sample_target(points)
        ag.last_retarget_t = now_t
        ag.speed = self._retarget_speed(ag)
        self._replan_route(ag, now_t=now_t)

    def _update_pass_side_commitment(
        self,
        ag: CrowdAgent,
        max_conflict_risk: float,
        suggested_pass_side: int,
        now_t: float,
    ) -> None:
        if now_t < ag.pass_side_until_t and ag.pass_side_pref in (-1, 1):
            return
        trigger = float(self.params["sfm_pass_side_trigger_risk"])
        if max_conflict_risk >= trigger and suggested_pass_side in (-1, 1):
            ag.pass_side_pref = int(suggested_pass_side)
            ag.pass_side_until_t = now_t + float(self.params["sfm_pass_side_hold_sec"])
        elif now_t >= ag.pass_side_until_t:
            ag.pass_side_pref = 0
            ag.pass_side_until_t = 0.0

    def _should_replan_for_block(
        self,
        ag: CrowdAgent,
        loc: carla.Location,
        desired_dir: np.ndarray,
        d_goal: float,
        conflict_risk: float,
        now_t: float,
    ) -> bool:
        if ag.route_phase == "turnaround":
            ag.block_state_t = 0.0
            return False
        if not bool(self.params["replan_on_block"]):
            ag.block_state_t = 0.0
            return False
        if self.params["navigation_mode"] != "astar":
            ag.block_state_t = 0.0
            return False
        if d_goal <= float(self.params["arrive_threshold"]) * 1.5:
            ag.block_state_t = 0.0
            return False

        risk_thresh = float(self.params["block_replan_risk_thresh"])
        progress_thresh = float(self.params["block_replan_progress_thresh"])
        window_sec = float(self.params["block_replan_window_sec"])

        if conflict_risk < risk_thresh:
            ag.block_state_t = 0.0
            ag.block_state_x = float(loc.x)
            ag.block_state_y = float(loc.y)
            return False

        if ag.block_state_t <= 0.0:
            ag.block_state_t = now_t
            ag.block_state_x = float(loc.x)
            ag.block_state_y = float(loc.y)
            return False

        progress_vec = np.array(
            [float(loc.x - ag.block_state_x), float(loc.y - ag.block_state_y)],
            dtype=np.float32,
        )
        forward_progress = float(np.dot(progress_vec, desired_dir))
        if forward_progress >= progress_thresh:
            ag.block_state_t = now_t
            ag.block_state_x = float(loc.x)
            ag.block_state_y = float(loc.y)
            return False

        if now_t - ag.block_state_t < window_sec:
            return False

        ag.block_state_t = now_t
        ag.block_state_x = float(loc.x)
        ag.block_state_y = float(loc.y)
        return True

    @staticmethod
    def _clamp_direction_to_cone(
        desired_dir: np.ndarray,
        move_dir: np.ndarray,
        max_angle_rad: float,
    ) -> np.ndarray:
        desired_norm = float(np.linalg.norm(desired_dir))
        move_norm = float(np.linalg.norm(move_dir))
        if desired_norm <= 1e-6 or move_norm <= 1e-6:
            return move_dir.astype(np.float32)
        desired = desired_dir / desired_norm
        move = move_dir / move_norm
        max_angle = float(np.clip(max_angle_rad, 0.0, math.pi))
        cos_angle = float(np.clip(np.dot(desired, move), -1.0, 1.0))
        angle = float(math.acos(cos_angle))
        if angle <= max_angle:
            return move.astype(np.float32)
        cross = float(desired[0] * move[1] - desired[1] * move[0])
        side = 1.0 if cross >= 0.0 else -1.0
        lateral = np.array([-desired[1], desired[0]], dtype=np.float32)
        clamped = desired * math.cos(max_angle) + lateral * side * math.sin(max_angle)
        norm = float(np.linalg.norm(clamped))
        if norm <= 1e-6:
            return desired.astype(np.float32)
        return (clamped / norm).astype(np.float32)

    def _compute_local_motion(
        self,
        agent_idx: int,
        ag: CrowdAgent,
        loc: carla.Location,
        now_t: float,
        robot_actor: Optional[carla.Actor],
        target_track_id: Optional[str],
    ) -> Tuple[np.ndarray, float]:
        goal_x, goal_y = self._get_local_goal_xy(ag, loc, now_t=now_t)
        reference_forward_xy = self._get_reference_forward_xy(ag, loc, goal_x=goal_x, goal_y=goal_y)
        goal_x, goal_y = self._apply_smoothed_pair_lateral_preference(
            ag,
            loc,
            goal_x,
            goal_y,
            reference_forward_xy,
        )
        d_goal = float(np.hypot(goal_x - loc.x, goal_y - loc.y))
        if d_goal <= 1e-3:
            return np.zeros(2, dtype=np.float32), 0.0

        # Target is exempt from all SFM forces (pedestrians, walls, robot): it
        # walks its scripted lane route directly, so the route is replayed
        # identically across policies.  Non-targets still avoid the target.
        if ag.is_target:
            direction = np.array([goal_x - loc.x, goal_y - loc.y], dtype=np.float32) / d_goal
            return direction, float(ag.speed)

        if self._sfm_planner is None:
            self._sfm_planner = self._build_sfm_planner()

        loc_xy = np.array([float(loc.x), float(loc.y)], dtype=np.float32)
        agent_state = self._build_agent_state(
            ag,
            goal_x=goal_x,
            goal_y=goal_y,
            reference_forward_xy=reference_forward_xy,
        )
        obstacles = self._collect_dynamic_obstacles(
            agent_idx,
            loc_xy=loc_xy,
            reference_forward_xy=reference_forward_xy,
            robot_actor=robot_actor,
            target_track_id=target_track_id,
        )
        wall_state = self._query_wall_state(ag, loc)
        sfm_result = self._sfm_planner.compute_velocity(
            agent=agent_state,
            neighbors=obstacles,
            wall=wall_state,
            dt=float(self.params["sfm_dt"]),
        )
        self._update_pass_side_commitment(
            ag,
            max_conflict_risk=float(sfm_result.max_conflict_risk),
            suggested_pass_side=int(sfm_result.suggested_pass_side),
            now_t=now_t,
        )

        desired_velocity = sfm_result.desired_velocity_xy
        desired_speed = float(np.linalg.norm(desired_velocity))
        if desired_speed > 1e-5:
            desired_dir = desired_velocity / desired_speed
        else:
            fallback = np.array([goal_x - loc.x, goal_y - loc.y], dtype=np.float32)
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm <= 1e-5:
                return np.zeros(2, dtype=np.float32), 0.0
            desired_dir = fallback / fallback_norm
            desired_speed = float(ag.speed)

        move_vec = sfm_result.velocity_xy
        move_speed = float(np.linalg.norm(move_vec))
        if move_speed > 1e-5:
            move_dir = move_vec / move_speed
        else:
            move_dir = desired_dir

        if str(getattr(sfm_result, "dominant_obstacle_kind", "")) == "robot":
            max_angle_rad = math.radians(float(self.params["sfm_robot_max_avoidance_angle_deg"]))
            move_dir = self._clamp_direction_to_cone(
                desired_dir.astype(np.float32),
                move_dir.astype(np.float32),
                max_angle_rad,
            )

        forward_speed = max(0.0, float(np.dot(move_vec, desired_dir)))
        conflict_risk = float(sfm_result.max_conflict_risk)
        min_surface_dist = float(sfm_result.min_surface_distance)
        close_surface_slowdown = float(self.params["sfm_close_surface_slowdown_m"])
        close_blocked = min_surface_dist < close_surface_slowdown
        if forward_speed < 0.15 * desired_speed and conflict_risk < 0.25 and not close_blocked:
            move_speed = max(forward_speed, 0.35 * desired_speed)
            move_dir = desired_dir
        else:
            move_speed = max(move_speed, forward_speed)

        min_speed_floor = desired_speed * (0.18 + 0.22 * max(0.0, 1.0 - conflict_risk))
        if close_blocked:
            clearance_scale = float(np.clip(min_surface_dist / max(close_surface_slowdown, 1e-3), 0.0, 1.0))
            min_speed_floor *= 0.25 * clearance_scale
        move_speed = max(move_speed, min_speed_floor if desired_speed > 1e-5 else 0.0)
        yield_trigger = float(self.params["sfm_yield_trigger_risk"])
        if desired_speed > 1e-5 and conflict_risk > yield_trigger:
            denom = max(1.0 - yield_trigger, 1e-3)
            risk_alpha = float(np.clip((conflict_risk - yield_trigger) / denom, 0.0, 1.0))
            min_scale = float(np.clip(self.params["sfm_yield_min_speed_scale"], 0.1, 1.0))
            speed_scale = 1.0 - (1.0 - min_scale) * risk_alpha
            move_speed = min(move_speed, desired_speed * speed_scale)

        if self._should_replan_for_block(
            ag,
            loc,
            desired_dir.astype(np.float32),
            d_goal,
            conflict_risk,
            now_t,
        ):
            self._replan_route(ag, now_t=now_t)

        return move_dir.astype(np.float32), float(move_speed)

    def step(
        self,
        debug_draw_ids: bool = True,
        label_life: float = 0.2,
        robot_actor: Optional[carla.Actor] = None,
        target_track_id: Optional[str] = None,
    ) -> None:
        if not self._spawned:
            return
        # Simulation clock: advances by one control tick per step() so retarget /
        # stuck / cooldown timers (and the RNG draws they trigger) are
        # reproducible under a fixed seed, independent of wall-clock speed.
        self._sim_t += float(self.params["sfm_dt"])
        now_t = self._sim_t
        for i, ag in enumerate(self.agents):
            loc = ag.walker.get_location()
            if debug_draw_ids:
                self.world.debug.draw_string(
                    loc + carla.Location(z=2.1),
                    ag.track_id,
                    draw_shadow=False,
                    color=carla.Color(0, 255, 255),
                    life_time=label_life,
                )

            if not self._active_move:
                ctrl = carla.WalkerControl()
                ctrl.direction = carla.Vector3D(0.0, 0.0, 0.0)
                ctrl.speed = 0.0
                ag.walker.apply_control(ctrl)
                continue

            if ag.route_task_done:
                ctrl = carla.WalkerControl()
                ctrl.direction = carla.Vector3D(0.0, 0.0, 0.0)
                ctrl.speed = 0.0
                ag.walker.apply_control(ctrl)
                continue

            stuck = self._is_stuck(ag, now_t)
            self._update_agent_target(ag, loc, now_t=now_t, stuck=stuck)
            if ag.route_task_done:
                ctrl = carla.WalkerControl()
                ctrl.direction = carla.Vector3D(0.0, 0.0, 0.0)
                ctrl.speed = 0.0
                ag.walker.apply_control(ctrl)
                continue
            move, move_speed = self._compute_local_motion(
                i,
                ag,
                loc,
                now_t=now_t,
                robot_actor=robot_actor,
                target_track_id=target_track_id,
            )
            z_dir = float(np.clip((ag.target.z - loc.z) * self.params["vertical_gain"], -0.35, 0.35))

            ctrl = carla.WalkerControl()
            ctrl.direction = carla.Vector3D(float(move[0]), float(move[1]), z_dir)
            ctrl.speed = float(move_speed)
            ag.walker.apply_control(ctrl)

            if (
                stuck
                and (ag.target.z - loc.z) > self.params["curb_min_z"]
                and (ag.target.z - loc.z) < self.params["curb_max_z"]
                and (now_t - ag.last_curb_t) > self.params["curb_cooldown_sec"]
            ):
                try:
                    tf = ag.walker.get_transform()
                    tf.location.x += float(move[0]) * float(self.params["curb_forward_nudge"])
                    tf.location.y += float(move[1]) * float(self.params["curb_forward_nudge"])
                    tf.location.z += float(self.params["curb_step_up"])
                    ag.walker.set_transform(tf)
                    ag.last_curb_t = now_t
                except Exception:
                    pass

    def get_states(self) -> List[NpcState]:
        out: List[NpcState] = []
        for ag in self.agents:
            loc = ag.walker.get_location()
            vel = ag.walker.get_velocity()
            tf = ag.walker.get_transform()
            out.append(
                NpcState(
                    track_id=ag.track_id,
                    actor_id=int(ag.walker.id),
                    x=float(loc.x),
                    y=float(loc.y),
                    z=float(loc.z),
                    vx=float(vel.x),
                    vy=float(vel.y),
                    vz=float(vel.z),
                    yaw_deg=float(tf.rotation.yaw),
                    speed=float(np.hypot(vel.x, vel.y)),
                )
            )
        return out

    def get_target_by_track_id(self, track_id: str) -> Optional[NpcState]:
        for s in self.get_states():
            if s.track_id == track_id:
                return s
        return None

    def get_actor_by_track_id(self, track_id: str) -> Optional[carla.Actor]:
        for ag in self.agents:
            if ag.track_id == track_id:
                return ag.walker
        return None

    def is_target_route_task_done(self, track_id: str) -> bool:
        for ag in self.agents:
            if ag.track_id == track_id:
                return bool(ag.route_task_done)
        return False

    def destroy(self) -> None:
        for ag in self.agents:
            try:
                ag.walker.destroy()
            except Exception:
                pass
        self.agents.clear()
        self._active_move = False
        self._spawned = False
        self._sfm_planner = None
