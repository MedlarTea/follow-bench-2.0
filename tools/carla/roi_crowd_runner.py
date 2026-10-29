"""Spawn and run random-walk NPC crowd inside ROI gridmap.

This script is designed for fast UE/CARLA acceptance:
- Load ROI occupancy from `gridmap_roi.npz` (or `gridmap.npz`)
- Convert grid <-> world coordinates
- Spawn many walker NPCs in ROI free cells
- Keep them moving for a fixed duration by repeatedly assigning random goals
- Anti-stuck re-targeting to reduce idle jams
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import carla
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import carla. Run this script with CARLA PythonAPI environment."
    ) from exc


def _parse_meta_scalar(meta_raw) -> dict:
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
    text = meta_raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


@dataclass
class ROIMap:
    world_min: np.ndarray
    world_max: np.ndarray
    resolution: float
    free_grid: np.ndarray
    roi_mask: np.ndarray
    walkable_cells: np.ndarray = field(repr=False)

    @classmethod
    def load(cls, npz_path: str, resolution_fallback: float) -> "ROIMap":
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
        cells = np.argwhere(walkable)  # (N,2) as [gy,gx]
        if len(cells) == 0:
            raise RuntimeError("No walkable cells found in ROI grid.")
        return cls(
            world_min=world_min,
            world_max=world_max,
            resolution=resolution,
            free_grid=free_grid,
            roi_mask=roi_mask,
            walkable_cells=cells,
        )

    def cell_to_world(self, gx: int, gy: int, ground_z: float = 0.0) -> carla.Location:
        x = self.world_min[0] + (float(gx) + 0.5) * self.resolution
        y = self.world_min[1] + (float(gy) + 0.5) * self.resolution
        return carla.Location(x=float(x), y=float(y), z=float(ground_z))

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

    def sample_world_location(
        self, rng: random.Random, ground_z: float, jitter: bool = True
    ) -> carla.Location:
        gy, gx = self.walkable_cells[rng.randrange(len(self.walkable_cells))]
        x = self.world_min[0] + (float(gx) + 0.5) * self.resolution
        y = self.world_min[1] + (float(gy) + 0.5) * self.resolution
        if jitter:
            half = 0.45 * self.resolution
            x += rng.uniform(-half, half)
            y += rng.uniform(-half, half)
        return carla.Location(x=float(x), y=float(y), z=float(ground_z))


@dataclass
class CrowdAgent:
    walker: carla.Actor
    speed: float
    target: carla.Location
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    last_retarget_t: float = 0.0
    last_jump_t: float = 0.0
    flow_side: int = 0           # +1 / -1 for bidirectional flow
    target_band: int = 0         # +1 high-end band, -1 low-end band


def _sfm_like_avoidance(
    agent_idx: int,
    agents: List[CrowdAgent],
    avoid_radius: float,
    avoid_gain: float,
    max_neighbors: int,
    lateral_bias: float,
) -> np.ndarray:
    """Compute SFM-like repulsive vector in XY plane.

    Includes:
    - distance-based exponential repulsion
    - stronger lateral sidestep when a neighbor is ahead and approaching
    """
    me = agents[agent_idx].walker.get_location()
    my_v = agents[agent_idx].walker.get_velocity()
    my_speed = float(np.hypot(my_v.x, my_v.y))
    if my_speed > 1e-3:
        facing = np.array([my_v.x / my_speed, my_v.y / my_speed], dtype=np.float32)
    else:
        facing = None
    rep = np.zeros(2, dtype=np.float32)
    neigh_count = 0
    for j, other in enumerate(agents):
        if j == agent_idx:
            continue
        o = other.walker.get_location()
        ov = other.walker.get_velocity()
        dx = me.x - o.x
        dy = me.y - o.y
        d = float(np.hypot(dx, dy))
        if d < 1e-5 or d > avoid_radius:
            continue
        u = np.array([dx / d, dy / d], dtype=np.float32)

        # Exponential distance repulsion (closer to sfm_demo behavior)
        d_eff = max(0.0, d - 0.7)  # approximate two-body radius clearance
        base = float(avoid_gain * np.exp(-d_eff / max(0.3, avoid_radius * 0.35)))
        force = u * base

        if facing is not None:
            # neighbor position relative to my facing
            to_other = np.array([o.x - me.x, o.y - me.y], dtype=np.float32) / d
            ahead = float(np.dot(to_other, facing)) > 0.25
            o_speed = float(np.hypot(ov.x, ov.y))
            approaching = False
            if o_speed > 0.1:
                ov_dir = np.array([ov.x / o_speed, ov.y / o_speed], dtype=np.float32)
                # if neighbor moving toward me
                approaching = float(np.dot(ov_dir, -to_other)) > 0.2

            if ahead and approaching:
                right = np.array([facing[1], -facing[0]], dtype=np.float32)
                force = force * (1.0 - 0.5 * lateral_bias) + right * (base * lateral_bias)

        rep += force
        neigh_count += 1
        if neigh_count >= max_neighbors:
            break
    return rep


def _pick_walker_blueprints(world: carla.World, n: int, rng: random.Random) -> List[carla.ActorBlueprint]:
    bps = list(world.get_blueprint_library().filter("walker.pedestrian.*"))
    if not bps:
        raise RuntimeError("No walker.pedestrian.* blueprint found.")
    rng.shuffle(bps)
    return [bps[i % len(bps)] for i in range(n)]


def _load_flow_points(
    json_path: str,
    roi_map: ROIMap,
    spawn_z: float,
) -> List[carla.Location]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload: Dict = json.load(f)

    points_world = payload.get("points_world", None)
    if isinstance(points_world, list) and len(points_world) >= 4:
        out: List[carla.Location] = []
        for p in points_world[:4]:
            x = float(p["x"])
            y = float(p["y"])
            z = float(p.get("z", spawn_z))
            out.append(carla.Location(x=x, y=y, z=z))
        return out

    points_grid = payload.get("points_grid", None)
    if isinstance(points_grid, list) and len(points_grid) >= 4:
        out = []
        for gx, gy in points_grid[:4]:
            out.append(roi_map.cell_to_world(int(gx), int(gy), ground_z=spawn_z))
        return out

    raise RuntimeError("Flow points JSON must contain at least 4 points in points_world or points_grid.")


def _load_roi_polygon_points(
    json_path: str,
    roi_map: ROIMap,
    spawn_z: float,
) -> List[carla.Location]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload: Dict = json.load(f)

    pts_grid = payload.get("points_grid", None)
    if isinstance(pts_grid, list) and len(pts_grid) >= 3:
        out: List[carla.Location] = []
        for p in pts_grid:
            gx = int(p[0])
            gy = int(p[1])
            out.append(roi_map.cell_to_world(gx=gx, gy=gy, ground_z=spawn_z))
        return out

    pts_px = payload.get("points_px", None)
    if isinstance(pts_px, list) and len(pts_px) >= 3:
        out = []
        for p in pts_px:
            gx = int(round(float(p[0])))
            gy = int(round(float(p[1])))
            out.append(roi_map.cell_to_world(gx=gx, gy=gy, ground_z=spawn_z))
        return out

    raise RuntimeError("ROI polygon JSON must contain points_grid or points_px with at least 3 points.")


def _draw_roi_and_anchors(
    world: carla.World,
    roi_polygon_world: List[carla.Location],
    flow_points_world: List[carla.Location],
    life_time: float,
) -> None:
    if len(roi_polygon_world) >= 3:
        for i in range(len(roi_polygon_world)):
            p1 = roi_polygon_world[i] + carla.Location(z=0.08)
            p2 = roi_polygon_world[(i + 1) % len(roi_polygon_world)] + carla.Location(z=0.08)
            world.debug.draw_line(
                p1,
                p2,
                thickness=0.08,
                color=carla.Color(0, 255, 0),
                life_time=life_time,
            )

    if len(flow_points_world) >= 4:
        labels = ["A", "B", "C", "D"]
        for i, p in enumerate(flow_points_world[:4]):
            color = carla.Color(30, 144, 255) if i < 2 else carla.Color(255, 64, 64)
            world.debug.draw_point(
                p + carla.Location(z=0.20),
                size=0.14,
                color=color,
                life_time=life_time,
            )
            world.debug.draw_string(
                p + carla.Location(z=0.35),
                labels[i],
                draw_shadow=False,
                color=color,
                life_time=life_time,
            )


def _spawn_one_agent(
    world: carla.World,
    roi_map: ROIMap,
    bp: carla.ActorBlueprint,
    speed: float,
    rng: random.Random,
    spawn_z: float,
    sample_nav_in_roi,
    max_spawn_attempts: int,
) -> Optional[CrowdAgent]:
    walker = None
    for _ in range(max_spawn_attempts):
        loc = sample_nav_in_roi()
        if loc is None:
            loc = roi_map.sample_world_location(rng, ground_z=spawn_z, jitter=True)
        walker = world.try_spawn_actor(bp, carla.Transform(loc))
        if walker is not None:
            break
    if walker is None:
        return None

    tgt = sample_nav_in_roi()
    if tgt is None:
        tgt = roi_map.sample_world_location(rng, ground_z=spawn_z, jitter=True)
    return CrowdAgent(
        walker=walker,
        speed=float(speed),
        target=tgt,
        last_retarget_t=time.time(),
    )


def _needs_retarget(agent: CrowdAgent, now_t: float, retarget_sec: float, arrive_threshold: float) -> bool:
    loc = agent.walker.get_location()
    dist = loc.distance(agent.target)
    if dist < arrive_threshold:
        return True
    if now_t - agent.last_retarget_t > retarget_sec:
        return True
    return False


def _is_stuck(
    agent: CrowdAgent,
    now_t: float,
    stuck_window_sec: float,
    stuck_dist_threshold: float,
) -> bool:
    loc = agent.walker.get_location()
    agent.history.append((now_t, loc.x, loc.y))
    while len(agent.history) > 2 and now_t - agent.history[0][0] > stuck_window_sec:
        agent.history.pop(0)
    if len(agent.history) < 2:
        return False
    x0, y0 = agent.history[0][1], agent.history[0][2]
    x1, y1 = agent.history[-1][1], agent.history[-1][2]
    moved = float(np.hypot(x1 - x0, y1 - y0))
    return moved < stuck_dist_threshold


def run(args):
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    roi_map = ROIMap.load(args.grid_npz, resolution_fallback=args.resolution)
    print(
        f"[ROI] cells={len(roi_map.walkable_cells)} res={roi_map.resolution:.3f} "
        f"world_min={roi_map.world_min[:2]} world_max={roi_map.world_max[:2]}"
    )

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    if args.load_map:
        print(f"[CARLA] loading map: {args.map_name}")
        world = client.load_world(args.map_name)
        time.sleep(1.0)
    else:
        world = client.get_world()

    bps = _pick_walker_blueprints(world, args.num_walkers, rng)
    walk_xy = np.array(
        [
            [
                roi_map.world_min[0] + (float(gx) + 0.5) * roi_map.resolution,
                roi_map.world_min[1] + (float(gy) + 0.5) * roi_map.resolution,
            ]
            for gy, gx in roi_map.walkable_cells
        ],
        dtype=np.float32,
    )
    x_min, y_min = walk_xy.min(axis=0)
    x_max, y_max = walk_xy.max(axis=0)
    span_x = float(x_max - x_min)
    span_y = float(y_max - y_min)
    flow_axis = "x" if span_x >= span_y else "y"
    if flow_axis == "x":
        low_cut = x_min + args.flow_band_ratio * span_x
        high_cut = x_max - args.flow_band_ratio * span_x
    else:
        low_cut = y_min + args.flow_band_ratio * span_y
        high_cut = y_max - args.flow_band_ratio * span_y
    flow_points_world: List[carla.Location] = []
    if args.flow_points_json:
        flow_points_world = _load_flow_points(args.flow_points_json, roi_map=roi_map, spawn_z=args.spawn_z)
        for i, p in enumerate(flow_points_world):
            if not roi_map.is_world_walkable(p.x, p.y):
                print(
                    f"[FLOW] warning: point {i + 1} is not in walkable ROI cell "
                    f"(x={p.x:.2f}, y={p.y:.2f})"
                )
        print(
            "[FLOW] loaded 4 anchors: "
            + ", ".join([f"({p.x:.1f},{p.y:.1f},{p.z:.1f})" for p in flow_points_world])
        )
    roi_polygon_world: List[carla.Location] = []
    if args.roi_polygon_json:
        roi_polygon_world = _load_roi_polygon_points(
            args.roi_polygon_json,
            roi_map=roi_map,
            spawn_z=args.spawn_z,
        )
        print(f"[ROI] loaded polygon vertices: {len(roi_polygon_world)}")

    use_flow_anchors = args.motion_mode == "bidirectional" and len(flow_points_world) >= 4
    side_a = flow_points_world[:2] if use_flow_anchors else []
    side_b = flow_points_world[2:4] if use_flow_anchors else []
    if args.motion_mode == "bidirectional":
        if use_flow_anchors:
            print("[FLOW] bidirectional mode: Side A (A,B) <-> Side B (C,D), alternating by retarget.")
        else:
            print("[FLOW] bidirectional mode: two ROI end-bands <-> alternating by retarget.")

    def sample_nav_in_roi(
        target_band: int = 0,
        anchor_points: Optional[List[carla.Location]] = None,
    ) -> Optional[carla.Location]:
        for _ in range(args.nav_sample_tries):
            nav_loc = world.get_random_location_from_navigation()
            if nav_loc is None:
                continue
            if roi_map.is_world_walkable(nav_loc.x, nav_loc.y):
                if anchor_points:
                    d_min = min([nav_loc.distance(a) for a in anchor_points])
                    if d_min > args.flow_anchor_radius:
                        continue
                elif target_band != 0:
                    coord = nav_loc.x if flow_axis == "x" else nav_loc.y
                    if target_band < 0 and coord > low_cut:
                        continue
                    if target_band > 0 and coord < high_cut:
                        continue
                # Keep destination on navigation surface to avoid unreachable goals.
                return carla.Location(x=nav_loc.x, y=nav_loc.y, z=nav_loc.z)
        return None
    agents: List[CrowdAgent] = []
    failed = 0

    def far_enough_from_existing(loc: carla.Location) -> bool:
        if not agents:
            return True
        for ex in agents:
            p = ex.walker.get_location()
            if p.distance(loc) < args.spawn_min_sep:
                return False
        return True

    for i in range(args.num_walkers):
        speed = rng.uniform(args.min_speed, args.max_speed)
        flow_side = 1 if (i % 2 == 0) else -1
        ag = None
        for _ in range(args.spawn_retries_per_agent):
            # bidirectional mode: spawn from one end, target the other end
            start_band = 0
            target_band = 0
            if args.motion_mode == "bidirectional":
                start_band = -1 if flow_side > 0 else 1
                target_band = 1 if flow_side > 0 else -1
                start_points = side_a if start_band < 0 else side_b
                target_points = side_a if target_band < 0 else side_b
            else:
                start_points = []
                target_points = []
            cand = _spawn_one_agent(
                world=world,
                roi_map=roi_map,
                bp=bps[i],
                speed=speed,
                rng=rng,
                spawn_z=args.spawn_z,
                sample_nav_in_roi=(lambda: sample_nav_in_roi(start_band, start_points)),
                max_spawn_attempts=args.max_spawn_attempts,
            )
            if cand is None:
                continue
            if args.motion_mode == "bidirectional":
                tgt = sample_nav_in_roi(target_band=target_band, anchor_points=target_points)
                if tgt is not None:
                    cand.target = tgt
                    cand.target_band = target_band
                cand.flow_side = flow_side
            if far_enough_from_existing(cand.walker.get_location()):
                ag = cand
                break
            try:
                cand.walker.destroy()
            except Exception:
                pass
        if ag is None:
            failed += 1
            continue
        agents.append(ag)

    print(f"[SPAWN] requested={args.num_walkers} spawned={len(agents)} failed={failed}")
    if not agents:
        raise RuntimeError("No walkers spawned. Check CARLA map/nav and ROI region.")

    start_t = time.time()
    next_debug_t = start_t
    next_roi_debug_t = start_t
    next_stat_t = start_t + args.stat_every_sec
    if args.debug_draw_roi:
        _draw_roi_and_anchors(
            world=world,
            roi_polygon_world=roi_polygon_world,
            flow_points_world=flow_points_world,
            life_time=max(args.debug_life, args.debug_roi_every_sec + 0.3),
        )
    try:
        while True:
            now_t = time.time()
            if now_t - start_t >= args.duration_sec:
                break
            for i, ag in enumerate(agents):
                stuck = _is_stuck(
                    ag,
                    now_t=now_t,
                    stuck_window_sec=args.stuck_window_sec,
                    stuck_dist_threshold=args.stuck_dist_threshold,
                )
                if stuck or _needs_retarget(
                    ag,
                    now_t=now_t,
                    retarget_sec=args.retarget_sec,
                    arrive_threshold=args.arrive_threshold,
                ):
                    if args.motion_mode == "bidirectional":
                        ag.target_band = -ag.target_band if ag.target_band != 0 else (1 if ag.flow_side > 0 else -1)
                        points = side_a if ag.target_band < 0 else side_b
                        new_tgt = sample_nav_in_roi(target_band=ag.target_band, anchor_points=points)
                        if new_tgt is None:
                            new_tgt = sample_nav_in_roi(target_band=0, anchor_points=points if use_flow_anchors else None)
                    else:
                        new_tgt = sample_nav_in_roi(target_band=0, anchor_points=None)
                    if new_tgt is None:
                        new_tgt = roi_map.sample_world_location(rng, ground_z=args.spawn_z, jitter=True)
                    ag.target = new_tgt
                    ag.last_retarget_t = now_t
                    ag.speed = float(rng.uniform(args.min_speed, args.max_speed))
                    if args.debug_draw_targets:
                        world.debug.draw_point(
                            ag.target + carla.Location(z=0.15),
                            size=0.10,
                            color=carla.Color(255, 120, 0),
                            life_time=args.debug_life,
                        )

                # Manual walker control + SFM-like local avoidance.
                loc = ag.walker.get_location()
                dx = ag.target.x - loc.x
                dy = ag.target.y - loc.y
                dist = float(np.hypot(dx, dy))
                goal_vec = np.array([dx, dy], dtype=np.float32)
                if dist > 1e-5:
                    goal_vec = goal_vec / dist
                else:
                    goal_vec = np.zeros(2, dtype=np.float32)

                rep_vec = _sfm_like_avoidance(
                    agent_idx=i,
                    agents=agents,
                    avoid_radius=args.avoid_radius,
                    avoid_gain=args.avoid_gain,
                    max_neighbors=args.max_neighbors,
                    lateral_bias=args.lateral_bias,
                )
                move_vec = goal_vec + rep_vec
                move_norm = float(np.linalg.norm(move_vec))
                if move_norm > 1e-5:
                    move_vec = move_vec / move_norm
                else:
                    move_vec = goal_vec

                crowd_factor = min(1.0, float(np.linalg.norm(rep_vec)))
                speed_scale = max(0.35, 1.0 - args.crowd_slowdown_gain * crowd_factor)
                control = carla.WalkerControl()
                if dist > 1e-3:
                    z_dir = 0.0
                    if args.enable_vertical_guidance:
                        z_dir = float(np.clip((ag.target.z - loc.z) * args.vertical_gain, -0.35, 0.35))
                    control.direction = carla.Vector3D(
                        x=float(move_vec[0]),
                        y=float(move_vec[1]),
                        z=z_dir,
                    )
                    control.speed = float(ag.speed * speed_scale)

                    # Curb helper: small step-up + nudge when stuck at low curb-like height gap.
                    if (
                        args.enable_curb_assist
                        and stuck
                        and (ag.target.z - loc.z) > args.curb_min_z
                        and (ag.target.z - loc.z) < args.curb_max_z
                        and (now_t - ag.last_jump_t) > args.curb_cooldown_sec
                    ):
                        try:
                            nudge = float(args.curb_forward_nudge)
                            step_up = float(args.curb_step_up)
                            tf = ag.walker.get_transform()
                            tf.location.x += float(move_vec[0]) * nudge
                            tf.location.y += float(move_vec[1]) * nudge
                            tf.location.z += step_up
                            ag.walker.set_transform(tf)
                            ag.last_jump_t = now_t
                        except Exception:
                            pass
                else:
                    control.direction = carla.Vector3D(x=0.0, y=0.0, z=0.0)
                    control.speed = 0.0
                ag.walker.apply_control(control)

            if args.debug_draw_targets and now_t >= next_debug_t:
                next_debug_t = now_t + args.debug_every_sec
                for ag in agents:
                    loc = ag.walker.get_location()
                    world.debug.draw_point(
                        loc + carla.Location(z=0.20),
                        size=0.08,
                        color=carla.Color(0, 255, 255),
                        life_time=args.debug_life,
                    )

            if args.debug_draw_roi and now_t >= next_roi_debug_t:
                next_roi_debug_t = now_t + args.debug_roi_every_sec
                _draw_roi_and_anchors(
                    world=world,
                    roi_polygon_world=roi_polygon_world,
                    flow_points_world=flow_points_world,
                    life_time=max(args.debug_life, args.debug_roi_every_sec + 0.3),
                )

            if now_t >= next_stat_t:
                next_stat_t = now_t + args.stat_every_sec
                moving = 0
                for ag in agents:
                    v = ag.walker.get_velocity()
                    speed = float(np.hypot(v.x, v.y))
                    if speed >= args.moving_speed_threshold:
                        moving += 1
                print(
                    f"[STAT] t={now_t - start_t:6.1f}s moving={moving}/{len(agents)} "
                    f"(thr={args.moving_speed_threshold:.2f} m/s)"
                )

            time.sleep(args.loop_sleep)
    finally:
        print("[CLEANUP] destroying walker actors...")
        for ag in agents:
            try:
                ag.walker.destroy()
            except Exception:
                pass
        print("[DONE] crowd runner finished.")


def build_parser():
    p = argparse.ArgumentParser(description="ROI random crowd runner in CARLA")
    p.add_argument("--grid-npz", required=True, help="Path to gridmap_roi.npz or gridmap.npz")
    p.add_argument("--resolution", type=float, default=0.5, help="Fallback grid resolution (m)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--load-map", action="store_true", help="Load CARLA map before run")
    p.add_argument("--map-name", default="Town10HD_Opt")
    p.add_argument("--num-walkers", type=int, default=40)
    p.add_argument("--duration-sec", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-speed", type=float, default=1.0)
    p.add_argument("--max-speed", type=float, default=2.0)
    p.add_argument("--retarget-sec", type=float, default=6.0)
    p.add_argument("--arrive-threshold", type=float, default=1.6)
    p.add_argument("--stuck-window-sec", type=float, default=3.5)
    p.add_argument("--stuck-dist-threshold", type=float, default=0.4)
    p.add_argument("--spawn-z", type=float, default=0.5)
    p.add_argument("--max-spawn-attempts", type=int, default=40)
    p.add_argument("--nav-sample-tries", type=int, default=120)
    p.add_argument("--loop-sleep", type=float, default=0.05)
    p.add_argument("--debug-draw-targets", action="store_true")
    p.add_argument("--debug-draw-roi", action="store_true", help="Draw ROI polygon and A/B/C/D anchors in UE")
    p.add_argument("--roi-polygon-json", type=str, default="", help="ROI polygon json (points_grid/points_px)")
    p.add_argument("--debug-roi-every-sec", type=float, default=1.2, help="Refresh period for ROI/anchor debug draw")
    p.add_argument("--debug-every-sec", type=float, default=0.7)
    p.add_argument("--debug-life", type=float, default=1.2)
    p.add_argument("--stat-every-sec", type=float, default=3.0)
    p.add_argument("--moving-speed-threshold", type=float, default=0.15)
    p.add_argument("--motion-mode", choices=["random", "bidirectional"], default="random", help="NPC motion mode")
    p.add_argument("--flow-band-ratio", type=float, default=0.18, help="Band ratio for bidirectional endpoints")
    p.add_argument(
        "--flow-points-json",
        type=str,
        default="",
        help="Optional 4-point anchor json. In bidirectional mode: points[0,1]=side A, points[2,3]=side B.",
    )
    p.add_argument(
        "--flow-anchor-radius",
        type=float,
        default=12.0,
        help="Max distance (m) from anchor points when sampling nav targets/spawns.",
    )
    p.add_argument("--avoid-radius", type=float, default=1.8, help="Neighbor distance for avoidance (m)")
    p.add_argument("--avoid-gain", type=float, default=1.4, help="Repulsion gain in SFM-like avoidance")
    p.add_argument("--lateral-bias", type=float, default=0.55, help="Rightward sidestep bias in head-on avoidance")
    p.add_argument("--max-neighbors", type=int, default=10, help="Max neighbors considered per walker")
    p.add_argument("--crowd-slowdown-gain", type=float, default=0.45, help="Speed reduction under crowd pressure")
    p.add_argument("--enable-vertical-guidance", action="store_true", help="Add z-direction guidance for slopes/stairs")
    p.add_argument("--vertical-gain", type=float, default=1.0, help="Gain for z-direction guidance")
    p.add_argument("--enable-curb-assist", action="store_true", help="Enable subtle curb step-up nudge (no jump)")
    p.add_argument("--curb-min-z", type=float, default=0.05, help="Min z gap for curb assist trigger")
    p.add_argument("--curb-max-z", type=float, default=0.35, help="Max z gap for curb assist trigger")
    p.add_argument("--curb-step-up", type=float, default=0.06, help="Temporary step-up height for curb assist")
    p.add_argument("--curb-forward-nudge", type=float, default=0.10, help="Forward nudge distance for curb assist")
    p.add_argument("--curb-cooldown-sec", type=float, default=0.8, help="Cooldown between curb assists")
    p.add_argument("--spawn-min-sep", type=float, default=0.9, help="Minimum initial spawn separation (m)")
    p.add_argument("--spawn-retries-per-agent", type=int, default=5, help="Spawn retries per agent for spacing")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())

