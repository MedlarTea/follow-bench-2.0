"""Random scenario NPC runtime.

Three crowd-flow modes (one per episode, picked randomly when mode="random"):

* **parallel**     — NPCs walk along the same A↔B axis as target. Half spawn near A
                     and walk towards B (same direction as target); the other half spawn
                     near B and walk towards A (counter flow). Lateral jitter keeps
                     them from walking single-file.

* **perpendicular** — NPCs cross the target's path. Each NPC picks a random
                     progress t along the axis, spawns 3-5m off-axis on one side,
                     and walks across to 3-5m off-axis on the other side. Once
                     across they re-pick a fresh (t, side) and keep crossing.

* **chaotic**       — NPCs spawn anywhere walkable inside the axis bounding band
                     and pick fresh random walkable goals each time they arrive.
                     "All-walkers wandering" baseline.

Target N01:
  - walks straight from A to B at constant speed
  - never receives SFM force (other NPCs avoid it instead)
  - episode terminates `route_done=True` when target reaches B

Other NPCs:
  - apply SFM repulsion against other NPCs (1×), the robot (1×), and the target (3×)
  - re-target on arrival to keep traffic going

The runtime intentionally skips A* / zone tokens / rescue / shadow follower / trunk
features from park. The random scenario is a clean linear-flow benchmark.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np

from followbench.sim.core_types import NpcState  # noqa: F401  (re-exported elsewhere)


# ── Walker blueprint blacklist (non-human meshes) ────────────────────────────
_NON_HUMAN_WALKER_BLUEPRINTS = {
    "walker.pedestrian.robot",
    "walker.pedestrian.g1_unitree",
    "walker.pedestrian.dog",
}

# ── SFM parameters (mirror passage / park style) ──────────────────────────
_SFM_PED_A        = 2.0     # pedestrian repulsion strength
_SFM_PED_B        = 0.8     # repulsion decay distance (m)
_SFM_PED_R        = 1.50    # pedestrian search radius (m)
_SFM_TARGET_MULT  = 3.0     # target gets 3× weight (NPCs strongly avoid target)
_SFM_ROBOT_R      = 2.0     # robot search radius (m)
_SFM_ROBOT_MULT   = 1.0     # robot treated like another pedestrian
_BODY_RADIUS      = 0.25    # robot body radius for d_eff subtraction

# ── Difficulty presets ───────────────────────────────────────────────────────
# num_lookalikes: distractors that reuse the target's appearance (legend only).
DIFFICULTY_PRESETS: Dict[str, dict] = {
    "easy":   {"num_walkers": 10, "min_speed": 1.20, "max_speed": 1.80, "num_lookalikes": 0},
    "hard":   {"num_walkers": 15, "min_speed": 1.40, "max_speed": 2.10, "num_lookalikes": 0},
    "boss":   {"num_walkers": 20, "min_speed": 1.60, "max_speed": 2.40, "num_lookalikes": 0},
    "legend": {"num_walkers": 25, "min_speed": 1.80, "max_speed": 2.70, "num_lookalikes": 2},
}

_TARGET_MIN_SPEED        = 0.80   # target N01 episode speed sampled from this range
_TARGET_MAX_SPEED        = 1.40   #   (constant within one episode, varies across episodes)
_NPC_SPEED_DRIFT_HZ      = 0.5    # how often each NPC re-rolls its speed (Hz, simulation time)
_NPC_SPEED_DRIFT_FRAC    = 0.30   # ±30% jitter per resample around its base speed

# ── Geometry constants ───────────────────────────────────────────────────────
_PARALLEL_LATERAL_HALF_M = 1.5    # half-width of NPC band along axis (parallel mode)
_PERP_OFF_MIN_M          = 3.0    # NPC perpendicular spawn distance min
_PERP_OFF_MAX_M          = 5.0    # NPC perpendicular spawn distance max
_PERP_AXIS_MARGIN        = 0.10   # avoid spawning right at axis endpoints (10% margin)
_ARRIVE_M                = 0.6    # waypoint-reached threshold (m)


@dataclass
class _Agent:
    walker:        carla.Actor
    track_id:      str
    is_target:     bool   = False
    speed:         float  = 1.0          # current commanded speed (m/s)
    base_speed:    float  = 1.0          # speed centroid for the per-NPC random walk
    next_resample_t: float = 0.0          # simulation time when speed gets re-rolled
    goal_xy:       Tuple[float, float] = (0.0, 0.0)
    # NPC re-target state
    role:          str    = "target"    # "target" | "parallel_fwd" | "parallel_rev" | "perp"
    perp_axis_t:   float  = 0.5         # progress along axis (perp mode)
    perp_side:     int    = +1          # +1 = right side, -1 = left
    perp_off:      float  = 4.0         # off-axis distance
    is_lookalike:  bool   = False       # distractor sharing the target's appearance


def _load_meta(grid_npz: str) -> Tuple[np.ndarray, float, np.ndarray]:
    data = np.load(grid_npz, allow_pickle=True)
    walkable = data["free_grid"].astype(bool)
    if "roi_mask" in data.files:
        walkable &= data["roi_mask"].astype(bool)
    world_min = np.array(data["world_min"], dtype=np.float64)[:2]
    res = 0.5
    if "__meta__" in data.files:
        try:
            raw = data["__meta__"]
            item = raw.item() if isinstance(raw, np.ndarray) else raw
            if isinstance(item, bytes):
                item = item.decode()
            meta = json.loads(item) if isinstance(item, str) else {}
            res = float(meta.get("resolution", 0.5))
        except Exception:
            pass
    return walkable, res, world_min


class _ROIProxy:
    def __init__(self, walkable: np.ndarray, res: float, world_min: np.ndarray):
        self._walk = walkable
        self._res = res
        self._wmin = world_min

    def is_world_walkable(self, x: float, y: float) -> bool:
        gx = int(round((x - self._wmin[0]) / self._res))
        gy = int(round((y - self._wmin[1]) / self._res))
        if 0 <= gy < self._walk.shape[0] and 0 <= gx < self._walk.shape[1]:
            return bool(self._walk[gy, gx])
        return False


def _farthest_pair(points: List[Tuple[float, float]]) -> Tuple[int, int]:
    best, bestd = (0, 1), -1.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            if d > bestd:
                bestd, best = d, (i, j)
    return best


class RandomNpcRuntime:
    def __init__(
        self,
        world: carla.World,
        grid_npz: str,
        flow_points_json: str,
        target_track_id: str = "N01",
        mode: str = "random",                # "parallel" | "perpendicular" | "chaotic" | "random"
        spawn_z: float = 0.5,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.world           = world
        self.target_track_id = target_track_id
        self.spawn_z         = spawn_z

        if rng_seed is None:
            rng_seed = int(time.time_ns() % 2_147_483_647)
        self.rng_seed = int(rng_seed)
        print(f"[NPC] rng_seed={self.rng_seed}")
        self.rng = random.Random(self.rng_seed)

        # Decide flow mode
        if mode == "random":
            self.flow_mode = self.rng.choice(["parallel", "perpendicular", "chaotic"])
            print(f"[NPC] flow_mode → {self.flow_mode} (random)")
        else:
            assert mode in ("parallel", "perpendicular", "chaotic"), f"unknown mode {mode!r}"
            self.flow_mode = mode
            print(f"[NPC] flow_mode → {self.flow_mode} (forced)")

        # Walkable map
        walkable, res, world_min = _load_meta(grid_npz)
        self.roi_map = _ROIProxy(walkable, res, world_min)

        # Pick A/B endpoints — farthest two flow points
        with open(flow_points_json) as f:
            fp_data = json.load(f)
        pts = [(float(p["x"]), float(p["y"])) for p in fp_data["points_world"]]
        self._flow_points = pts
        i, j = _farthest_pair(pts)
        self.A = pts[i]
        self.B = pts[j]
        # axis: A → B
        ax = self.B[0] - self.A[0]
        ay = self.B[1] - self.A[1]
        L = math.hypot(ax, ay)
        self.axis_len = float(L)
        self.axis_u   = (ax / L, ay / L)
        self.axis_n   = (-self.axis_u[1], self.axis_u[0])   # +90° rotation = "right" of axis
        self.axis_mid = ((self.A[0] + self.B[0]) / 2.0, (self.A[1] + self.B[1]) / 2.0)
        print(f"[NPC] axis A={self.A} → B={self.B}  len={self.axis_len:.2f} m")

        # Difficulty knobs (set via set_difficulty())
        self.params = {
            "num_walkers": 10,
            "min_speed":   1.2,
            "max_speed":   1.8,
            "num_lookalikes": 0,
        }
        self._difficulty = "easy"
        # Simulation clock (advanced by step(dt)); drives speed re-rolls so the
        # RNG stream is reproducible under a fixed seed.
        self._sim_t: float = 0.0
        self.lookalike_track_ids: List[str] = []

        self.agents: List[_Agent] = []
        self._target: Optional[_Agent] = None
        self._motion_active = False
        self._spawned = False
        self._target_route_done = False
        # Sampled by spawn() for every episode; initialise for API safety only.
        self.target_speed: float = _TARGET_MIN_SPEED

        # Walker blueprint pool (lazy)
        self._walker_bps: Optional[list] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def set_difficulty(self, level: str) -> None:
        preset = DIFFICULTY_PRESETS.get(level.lower())
        if preset is None:
            print(f"[NPC] unknown difficulty {level!r}, choices={list(DIFFICULTY_PRESETS)}")
            return
        self._difficulty = level.lower()
        self.params.update(preset)
        print(f"[NPC] difficulty → {level.upper()} "
              f"(walkers={preset['num_walkers']} speed=[{preset['min_speed']:.2f},{preset['max_speed']:.2f}] "
              f"lookalikes={preset.get('num_lookalikes', 0)})")

    def spawn(self, num_walkers: Optional[int] = None) -> None:
        if self._spawned:
            return
        # A Manager may reuse this runtime for consecutive episodes.  Reset the
        # completion latch and sample a fresh per-episode target speed here,
        # rather than carrying either value over from the previous episode.
        self._target_route_done = False
        self._sim_t = 0.0
        self.target_speed = float(self.rng.uniform(_TARGET_MIN_SPEED, _TARGET_MAX_SPEED))
        print(f"[NPC] episode target_speed={self.target_speed:.2f} m/s "
              f"(range=[{_TARGET_MIN_SPEED:.2f},{_TARGET_MAX_SPEED:.2f}])")
        if num_walkers is not None:
            self.params["num_walkers"] = int(num_walkers)
        n = int(self.params["num_walkers"])

        if self._walker_bps is None:
            lib = self.world.get_blueprint_library()
            self._walker_bps = [bp for bp in lib.filter("walker.pedestrian.*")
                                if bp.id not in _NON_HUMAN_WALKER_BLUEPRINTS]
            for bp in self._walker_bps:
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "true")
        if n > len(self._walker_bps):
            raise RuntimeError(
                f"[NPC] num_walkers={n} exceeds blueprint pool ({len(self._walker_bps)}); "
                f"each NPC needs a unique appearance.")
        # rng_seed defaults to time.time_ns(), so a fresh sample (and thus a fresh
        # cast of walker appearances) is drawn every episode.
        chosen_bps = self.rng.sample(self._walker_bps, n)
        # Legend: some distractors deliberately reuse the target's appearance.
        n_look = min(int(self.params.get("num_lookalikes", 0)), max(0, n - 1))
        lookalike_idx = set(self.rng.sample(range(1, n), n_look)) if n_look > 0 else set()
        for li in lookalike_idx:
            chosen_bps[li] = chosen_bps[0]
        self.lookalike_track_ids = []
        print(f"[NPC] blueprint cast = {[bp.id for bp in chosen_bps[:3]]}...{[bp.id for bp in chosen_bps[-2:]]} "
              f"(unique={len(set(bp.id for bp in chosen_bps))}/{n}, lookalikes={n_look})")

        # Spawn target N01 at A, goal = B — target always walks at constant _TARGET_SPEED
        target_bp = chosen_bps[0]
        tgt_loc = carla.Location(x=self.A[0], y=self.A[1], z=self.spawn_z)
        actor = self._try_spawn_walker(target_bp, tgt_loc, jitter=False)
        if actor is None:
            raise RuntimeError("[NPC] target spawn failed")
        target = _Agent(
            walker=actor, track_id=self.target_track_id, is_target=True,
            speed=float(self.target_speed), base_speed=float(self.target_speed),
            goal_xy=self.B, role="target",
        )
        self.agents.append(target)
        self._target = target

        # Distractors. For parallel mode we enforce a strict 50/50 counter-flow split:
        # half walk A→B (same direction as target), half walk B→A.
        n_distractors = n - 1
        parallel_dirs: List[int] = []
        if self.flow_mode == "parallel":
            half = n_distractors // 2
            parallel_dirs = [+1] * half + [-1] * (n_distractors - half)
            self.rng.shuffle(parallel_dirs)
        for i in range(1, n):
            tid = f"N{i+1:02d}"
            bp  = chosen_bps[i]
            dir_hint = parallel_dirs[i - 1] if parallel_dirs else 0
            ag  = self._spawn_distractor(bp, tid, parallel_dir=dir_hint)
            if ag is not None:
                if i in lookalike_idx:
                    ag.is_lookalike = True
                    self.lookalike_track_ids.append(tid)
                self.agents.append(ag)

        self._spawned = True
        print(f"[NPC] spawned={len(self.agents)}/{n}  mode={self.flow_mode}  difficulty={self._difficulty}  "
              f"lookalikes={self.lookalike_track_ids}")

    def set_motion_active(self, active: bool) -> None:
        self._motion_active = bool(active)

    def step(self, dt: float = 0.1, robot_actor: Optional[carla.Actor] = None,
             target_track_id: Optional[str] = None, **_kwargs) -> None:
        if not self._motion_active:
            return
        if not self.agents:
            return

        # Snapshot positions (incl robot)
        pos: Dict[str, Tuple[float, float]] = {}
        for ag in self.agents:
            loc = ag.walker.get_location()
            pos[ag.track_id] = (loc.x, loc.y)
        if robot_actor is not None:
            try:
                rl = robot_actor.get_location()
                pos["__robot__"] = (rl.x, rl.y)
            except Exception:
                pass

        target_id = target_track_id or self.target_track_id
        self._sim_t += float(dt)
        now_t = self._sim_t
        configured_min = float(self.params["min_speed"])
        configured_max = float(self.params["max_speed"])
        # Keep every commanded crowd speed inside the configured interval.
        speed_floor = configured_min

        for ag in self.agents:
            ax, ay = pos[ag.track_id]
            gx, gy = ag.goal_xy

            # Arrival re-target
            if math.hypot(gx - ax, gy - ay) < _ARRIVE_M:
                if ag.is_target:
                    self._target_route_done = True
                    # stop in place
                    ctrl = carla.WalkerControl()
                    ctrl.speed = 0.0
                    ag.walker.apply_control(ctrl)
                    continue
                else:
                    self._reassign_distractor_goal(ag)
                    gx, gy = ag.goal_xy

            # Goal direction
            dx, dy = gx - ax, gy - ay
            d = math.hypot(dx, dy) or 1e-6
            ux, uy = dx / d, dy / d

            # Target moves straight at its per-episode constant speed, no SFM, no jitter.
            if ag.is_target:
                ctrl = carla.WalkerControl()
                ctrl.direction = carla.Vector3D(ux, uy, 0.0)
                ctrl.speed = float(self.target_speed)
                ag.walker.apply_control(ctrl)
                continue

            # Distractor: re-roll commanded speed every ~(1/_NPC_SPEED_DRIFT_HZ) sec
            # around its base speed, clamped to the configured crowd range.
            if now_t >= ag.next_resample_t:
                jitter = 1.0 + self.rng.uniform(-_NPC_SPEED_DRIFT_FRAC, _NPC_SPEED_DRIFT_FRAC)
                ag.speed = min(
                    configured_max,
                    max(speed_floor, ag.base_speed * jitter),
                )
                ag.next_resample_t = now_t + 1.0 / max(_NPC_SPEED_DRIFT_HZ, 1e-3)

            # Distractor: add SFM
            rx, ry = self._sfm_repulsion(ax, ay, pos, ag.track_id, target_id)
            dxn, dyn = ux + rx, uy + ry
            mag = math.hypot(dxn, dyn) or 1e-6
            dxn, dyn = dxn / mag, dyn / mag
            ctrl = carla.WalkerControl()
            ctrl.direction = carla.Vector3D(dxn, dyn, 0.0)
            ctrl.speed = float(ag.speed)
            ag.walker.apply_control(ctrl)

    def get_states(self) -> List[NpcState]:
        out: List[NpcState] = []
        for ag in self.agents:
            tf = ag.walker.get_transform()
            v  = ag.walker.get_velocity()
            speed = math.hypot(v.x, v.y)
            out.append(NpcState(
                track_id=ag.track_id, actor_id=int(ag.walker.id),
                x=float(tf.location.x), y=float(tf.location.y), z=float(tf.location.z),
                vx=float(v.x), vy=float(v.y), vz=float(v.z),
                yaw_deg=float(tf.rotation.yaw), speed=float(speed),
                npc_state="moving",
            ))
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

    def is_target_route_task_done(self, track_id: Optional[str] = None) -> bool:
        return bool(self._target_route_done)

    def is_episode_done(self) -> bool:
        return self.is_target_route_task_done()

    def destroy(self) -> None:
        for ag in self.agents:
            try:
                ag.walker.destroy()
            except Exception:
                pass
        self.agents.clear()
        self._target = None
        self._spawned = False
        self._target_route_done = False
        print("[NPC] all agents destroyed")

    # ── Distractor spawning helpers ─────────────────────────────────────────

    def _spawn_distractor(self, bp: carla.ActorBlueprint, track_id: str,
                          parallel_dir: int = 0) -> Optional[_Agent]:
        # Draw every distractor directly from the configured Open Plaza range.
        configured_min = float(self.params["min_speed"])
        configured_max = float(self.params["max_speed"])
        floor = configured_min
        ceil = configured_max
        base = float(self.rng.uniform(floor, ceil))
        for _ in range(20):
            spawn_xy, goal_xy, role, perp_t, perp_side, perp_off = self._random_distractor_endpoints(parallel_dir)
            if not self.roi_map.is_world_walkable(*spawn_xy):
                continue
            if not self.roi_map.is_world_walkable(*goal_xy):
                continue
            loc = carla.Location(x=spawn_xy[0], y=spawn_xy[1], z=self.spawn_z)
            actor = self._try_spawn_walker(bp, loc, jitter=True)
            if actor is None:
                continue
            # First resample staggered per agent so they don't all re-roll together.
            next_t = self._sim_t + self.rng.uniform(0.0, 1.0 / max(_NPC_SPEED_DRIFT_HZ, 1e-3))
            return _Agent(
                walker=actor, track_id=track_id, is_target=False,
                speed=base, base_speed=base, next_resample_t=next_t,
                goal_xy=goal_xy, role=role,
                perp_axis_t=perp_t, perp_side=perp_side, perp_off=perp_off,
            )
        print(f"[NPC] distractor {track_id} spawn failed (no walkable endpoints)")
        return None

    def _random_distractor_endpoints(self, parallel_dir: int = 0
                                     ) -> Tuple[Tuple[float, float], Tuple[float, float],
                                                str, float, int, float]:
        """Generate (spawn_xy, goal_xy, role, perp_t, perp_side, perp_off) for one NPC."""
        if self.flow_mode == "parallel":
            # parallel_dir = +1 → walk same direction as target (A→B); -1 → counter (B→A).
            # Falls back to random if caller didn't pass a hint (chaotic re-route, etc).
            if parallel_dir == 0:
                parallel_dir = +1 if self.rng.random() < 0.5 else -1
            lat = self.rng.uniform(-_PARALLEL_LATERAL_HALF_M, _PARALLEL_LATERAL_HALF_M)
            if parallel_dir > 0:
                spawn_xy = self._axis_to_world(0.0, lat)
                goal_xy  = self._axis_to_world(1.0, lat)
                role = "parallel_fwd"
            else:
                spawn_xy = self._axis_to_world(1.0, lat)
                goal_xy  = self._axis_to_world(0.0, lat)
                role = "parallel_rev"
            return spawn_xy, goal_xy, role, 0.0, 0, 0.0

        if self.flow_mode == "perpendicular":
            t = self.rng.uniform(_PERP_AXIS_MARGIN, 1.0 - _PERP_AXIS_MARGIN)
            side_in = +1 if self.rng.random() < 0.5 else -1
            off_in  = self.rng.uniform(_PERP_OFF_MIN_M, _PERP_OFF_MAX_M)
            off_out = self.rng.uniform(_PERP_OFF_MIN_M, _PERP_OFF_MAX_M)
            spawn_xy = self._axis_to_world(t, side_in * off_in)
            goal_xy  = self._axis_to_world(t, -side_in * off_out)
            return spawn_xy, goal_xy, "perp", t, side_in, off_in

        # chaotic: spawn anywhere walkable inside the axis bounding band; pick a fresh
        # random walkable goal anywhere in the same band. NPCs stay roughly in the
        # arena around the target route but otherwise wander.
        spawn_xy = self._random_chaotic_point()
        goal_xy  = self._random_chaotic_point()
        return spawn_xy, goal_xy, "chaotic", 0.0, 0, 0.0

    def _reassign_distractor_goal(self, ag: _Agent) -> None:
        """When a distractor reaches its goal, give it a fresh route to keep flowing."""
        if self.flow_mode == "parallel":
            # Reverse direction along same lateral offset
            cur = ag.walker.get_location()
            # Project current position onto axis to recover lateral offset
            t, lat = self._world_to_axis(cur.x, cur.y)
            if ag.role == "parallel_fwd":
                ag.goal_xy = self._axis_to_world(0.0, lat)
                ag.role = "parallel_rev"
            else:
                ag.goal_xy = self._axis_to_world(1.0, lat)
                ag.role = "parallel_fwd"
            return

        if self.flow_mode == "perpendicular":
            # pick a fresh crossing
            for _ in range(10):
                t = self.rng.uniform(_PERP_AXIS_MARGIN, 1.0 - _PERP_AXIS_MARGIN)
                side_in = +1 if self.rng.random() < 0.5 else -1
                off_out = self.rng.uniform(_PERP_OFF_MIN_M, _PERP_OFF_MAX_M)
                new_goal = self._axis_to_world(t, -side_in * off_out)
                if self.roi_map.is_world_walkable(*new_goal):
                    ag.goal_xy = new_goal
                    ag.perp_axis_t = t
                    ag.perp_side = -side_in
                    ag.perp_off  = off_out
                    return
            ag.goal_xy = self.A
            return

        # chaotic: pick a new random walkable point.
        ag.goal_xy = self._random_chaotic_point()

    # ── Geometry utils ──────────────────────────────────────────────────────

    def _random_chaotic_point(self) -> Tuple[float, float]:
        """Random walkable point inside the axis bounding band.

        Spans the full axis length (with a small margin) and a wider lateral band
        than parallel mode so NPCs feel scattered rather than queuing on a line.
        """
        lat_half = max(_PERP_OFF_MAX_M, 5.0)
        for _ in range(40):
            t   = self.rng.uniform(-0.05, 1.05)
            lat = self.rng.uniform(-lat_half, lat_half)
            xy  = self._axis_to_world(t, lat)
            if self.roi_map.is_world_walkable(*xy):
                return xy
        # Fallback: axis midpoint.
        return self.axis_mid

    def _axis_to_world(self, t: float, lat: float) -> Tuple[float, float]:
        """t in [0,1] along A→B; lat = lateral offset (positive = right side of axis)."""
        ax, ay = self.A
        bx, by = self.B
        x = ax + (bx - ax) * t + self.axis_n[0] * lat
        y = ay + (by - ay) * t + self.axis_n[1] * lat
        return (x, y)

    def _world_to_axis(self, x: float, y: float) -> Tuple[float, float]:
        """Inverse of _axis_to_world: return (t, lat)."""
        ax, ay = self.A
        bx, by = self.B
        # axis vector
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        # parameter along axis
        t = ((x - ax) * vx + (y - ay) * vy) / L2 if L2 > 0 else 0.0
        # foot on axis
        fx = ax + vx * t
        fy = ay + vy * t
        # signed lateral via cross product
        lat = (x - fx) * self.axis_n[0] + (y - fy) * self.axis_n[1]
        return (float(t), float(lat))

    # ── CARLA spawn helper ──────────────────────────────────────────────────

    def _try_spawn_walker(self, bp: carla.ActorBlueprint, loc: carla.Location,
                          jitter: bool = True) -> Optional[carla.Actor]:
        cand = carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.3)
        actor = self.world.try_spawn_actor(bp, carla.Transform(cand))
        if actor is not None:
            return actor
        if not jitter:
            return None
        for _ in range(10):
            jx = self.rng.uniform(-1.0, 1.0)
            jy = self.rng.uniform(-1.0, 1.0)
            cand = carla.Location(x=loc.x + jx, y=loc.y + jy, z=loc.z + 0.3)
            actor = self.world.try_spawn_actor(bp, carla.Transform(cand))
            if actor is not None:
                return actor
        return None

    # ── SFM ─────────────────────────────────────────────────────────────────

    def _sfm_repulsion(
        self, ax: float, ay: float,
        positions: Dict[str, Tuple[float, float]],
        self_id: str,
        target_track_id: str,
    ) -> Tuple[float, float]:
        rep_x = rep_y = 0.0
        for tid, (bx, by) in positions.items():
            if tid == self_id:
                continue
            dx = ax - bx
            dy = ay - by
            d  = math.hypot(dx, dy)
            if d < 0.01:
                continue
            if tid == "__robot__":
                if d > _SFM_ROBOT_R:
                    continue
                d_eff = max(d - _BODY_RADIUS, 0.05)
                mult = _SFM_ROBOT_MULT
                mag = mult * _SFM_PED_A * math.exp(-d_eff / _SFM_PED_B)
            else:
                if d > _SFM_PED_R:
                    continue
                mult = _SFM_TARGET_MULT if tid == target_track_id else 1.0
                mag = mult * _SFM_PED_A * math.exp(-d / _SFM_PED_B)
            rep_x += (dx / d) * mag
            rep_y += (dy / d) * mag
        return rep_x, rep_y
