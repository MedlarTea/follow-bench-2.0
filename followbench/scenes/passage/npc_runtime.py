#!/usr/bin/env python3
"""
Passage NPC runtime — ordered-queue state machine.

Scene flow
----------
1. Spawn: agents spawn near flow-point areas (far from door).
   Half near side-A flow points → assigned Queue A.
   Half near side-B flow points → assigned Queue B.
2. APPROACH: every agent walks toward its assigned queue position immediately.
3. QUEUE: holds position with SFM. As soon as pos=0 is occupied and no one
   is currently passing from that queue, the agent departs — no gate, no
   waiting for the other queue.
4. On depart: queue shifts forward; _passing lock clears at route midpoint
   (agent has cleared the passage), allowing the next person to depart sooner.
5. After passing: non-target agents join the back of the opposite queue.
   The target agent ends the episode once it completes the A->B passage.

Advance rule
------------
Only QUEUE-state agents advance. When a QUEUE agent's pos-1 becomes vacant
(checked each tick), it re-enters APPROACH toward the closer waypoint.
APPROACH agents keep their current target; their queue_pos is updated by
_depart_front when the front agent crosses.

SFM rules
---------
- Velocity-integration SFM (matches sfm_demo.py architecture).
- APPROACH/QUEUE: tight search radius (< queue spacing) — no twitching.
- PASS: same radius; head-on encounters use rightward lateral bias.
- Target person: SFM forces not applied to target.
- Robot: repulsion obstacle for all NPCs.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np


from followbench.sim.core_types import NpcState  # noqa: E402

# ── Route ID constants ────────────────────────────────────────────────────────
_R_QUEUE_A  = 0
_R_QUEUE_B  = 1
_R_PASS_A2B = 2
_R_PASS_B2A = 3

_ARRIVE_M      = 0.55   # waypoint-reached threshold (m)
_QUEUE_SPACING = 1.5    # uniform gap between queue positions (m)

# ── Difficulty presets ────────────────────────────────────────────────────────
# max_concurrent: how many agents may be in the passage simultaneously.
# num_walkers:    total walkers incl. target.  Kept well above max_concurrent so
#                 the concurrency limit actually binds and queues form.
# num_lookalikes: distractors that reuse the target's appearance (legend only).
DIFFICULTY_PRESETS: Dict[str, dict] = {
    "easy":   {"max_concurrent": 2, "min_speed": 0.80, "max_speed": 1.04, "num_walkers": 6,  "num_lookalikes": 0},
    "hard":   {"max_concurrent": 4, "min_speed": 0.88, "max_speed": 1.20, "num_walkers": 10, "num_lookalikes": 0},
    "boss":   {"max_concurrent": 6, "min_speed": 0.96, "max_speed": 1.36, "num_walkers": 14, "num_lookalikes": 0},
    "legend": {"max_concurrent": 8, "min_speed": 1.12, "max_speed": 1.52, "num_walkers": 18, "num_lookalikes": 2},
}

# ── Long-term rescue (anti-deadlock) ─────────────────────────────────────────
# Every _RESCUE_CHECK_TICKS sim ticks, a non-target agent in APPROACH/PASS whose
# displacement since the last check is below _RESCUE_MIN_DISP_M accumulates one
# stuck window; after _RESCUE_WINDOWS consecutive windows it is re-routed.
# Sim-tick based (not wall clock) so it is deterministic under a fixed seed.
_RESCUE_CHECK_TICKS = 50      # 5 s at dt=0.1
_RESCUE_MIN_DISP_M  = 0.5
_RESCUE_WINDOWS     = 2       # → re-route after ~10 s without progress
_RESCUE_SKIP_WPS    = 2       # PASS: skip this many route waypoints

# ── SFM parameters ───────────────────────────────────────────────────────────
_SFM_PED_A        = 2.0    # pedestrian repulsion strength
_SFM_PED_B        = 0.8    # repulsion decay distance (m)
_SFM_PED_R        = 1.25   # pedestrian search radius (m)
_SFM_ROBOT_A      = 5.0    # robot repulsion strength
_SFM_ROBOT_B      = 0.8    # robot repulsion decay (m)
_SFM_ROBOT_R      = 2.0    # robot search radius (m)
_BODY_RADIUS      = 0.25   # body radius for robot d_eff subtraction
_SFM_LATERAL_BIAS = 0.6    # rightward-bias weight for head-on encounters

# ── Spawn spacing ─────────────────────────────────────────────────────────────
_SPAWN_MIN_DIST = 1.4    # minimum distance between any two spawn points (m)

# ── Walker blueprint blacklist (non-human meshes) ────────────────────────────
_NON_HUMAN_WALKER_BLUEPRINTS = {
    "walker.pedestrian.robot",
    "walker.pedestrian.g1_unitree",
    "walker.pedestrian.dog",
}


class _Phase(Enum):
    IDLE     = auto()   # waiting for release tick
    APPROACH = auto()   # walking to assigned queue position
    QUEUE    = auto()   # standing at position, waiting for gate / turn
    PASS     = auto()   # walking passage route


@dataclass
class _Agent:
    walker:        carla.Actor
    track_id:      str
    is_target:     bool
    phase:         _Phase  = _Phase.IDLE
    release_tick:  int     = 0
    # Queue info
    queue_id:      int     = _R_QUEUE_A
    queue_pos:     int     = 0
    queue_wp_idx:  int     = 0
    # Passage info
    pass_id:       int     = _R_PASS_A2B
    pass_wp_idx:   int     = 0
    speed:         float   = 1.0
    pass_count:    int     = 0      # number of completed passages
    is_lookalike:  bool    = False  # distractor sharing the target's appearance
    # Long-term rescue bookkeeping
    rescue_anchor: Optional[Tuple[float, float]] = None
    stuck_windows: int     = 0


class _ROIProxy:
    def __init__(self, walkable_pts: np.ndarray, res: float):
        self._pts = walkable_pts
        self._res = res

    def is_world_walkable(self, x: float, y: float) -> bool:
        return float(np.min(np.hypot(self._pts[:, 0] - x, self._pts[:, 1] - y))) < self._res * 1.5

    def sample_world_location(self, rng, z: float = 0.5, jitter: bool = False) -> carla.Location:
        idx = int(rng.integers(0, len(self._pts)))
        wx, wy = float(self._pts[idx, 0]), float(self._pts[idx, 1])
        if jitter:
            wx += float(rng.uniform(-self._res * 0.4, self._res * 0.4))
            wy += float(rng.uniform(-self._res * 0.4, self._res * 0.4))
        return carla.Location(x=wx, y=wy, z=z)


class PassageNpcRuntime:
    def __init__(
        self,
        world: carla.World,
        routes_json: str,
        grid_npz: str,
        flow_points_json: str,
        target_track_id: str = "N01",
        npc_min_speed: float = 0.7,
        npc_max_speed: float = 1.2,
        spawn_z: float = 0.5,
        seed: Optional[int] = None,
    ):
        self.world           = world
        self.target_track_id = target_track_id
        self._min_speed      = npc_min_speed
        self._max_speed      = npc_max_speed
        self.spawn_z         = spawn_z
        if seed is None:
            import time as _time
            seed = int(_time.time_ns() % 2_147_483_647)
        self._seed           = int(seed)
        print(f"[NPC] rng_seed={self._seed}")
        self.rng             = np.random.default_rng(self._seed)
        self._rand           = random.Random(self._seed)

        with open(routes_json) as f:
            rdata = json.load(f)
        self._routes: Dict[int, List[Tuple[float, float]]] = {
            int(k): [(float(p[0]), float(p[1])) for p in v]
            for k, v in rdata["routes"].items()
        }
        for r in range(4):   # routes 0-3 required; 4 (distract) no longer used
            assert r in self._routes, f"routes.json missing route {r}"

        with open(flow_points_json) as f:
            fp_data = json.load(f)
        # points_world: list of {x, y, z}; [0,1]=side A (Queue A spawns), [2,3]=side B
        self._flow_points: List[dict] = fp_data["points_world"]
        assert len(self._flow_points) >= 4, "flow_points_json needs at least 4 points"

        self._rebuild_queue_routes(_QUEUE_SPACING)

        self._cap = {
            _R_QUEUE_A: len(self._routes[_R_QUEUE_A]),
            _R_QUEUE_B: len(self._routes[_R_QUEUE_B]),
        }

        data     = np.load(grid_npz, allow_pickle=True)
        free_grid = np.array(data["free_grid"], dtype=np.uint8)
        roi_mask  = (np.array(data["roi_mask"], dtype=np.uint8)
                     if "roi_mask" in data else np.ones_like(free_grid))
        world_min = np.array(data["world_min"], dtype=np.float64)[:2]
        meta: dict = {}
        if "__meta__" in data:
            try:
                raw  = data["__meta__"]
                item = raw.item() if isinstance(raw, np.ndarray) else raw
                if isinstance(item, bytes):
                    item = item.decode()
                meta = json.loads(item) if isinstance(item, str) else {}
            except Exception:
                pass
        res      = float(meta.get("resolution", 0.5))
        walkable = (free_grid > 0) & (roi_mask > 0)
        gy_arr, gx_arr = np.where(walkable)
        self._walkable_pts = np.stack([
            world_min[0] + (gx_arr + 0.5) * res,
            world_min[1] + (gy_arr + 0.5) * res,
        ], axis=1).astype(np.float64)
        self.roi_map = _ROIProxy(self._walkable_pts, res)

        # Blueprint library is fetched lazily in spawn() to avoid an early RPC
        # racing with CARLA's post-load_world settling window.
        self._walker_bps: Optional[list] = None
        self.agents: List[_Agent]            = []
        self._queue: Dict[int, List[_Agent]] = {_R_QUEUE_A: [], _R_QUEUE_B: []}
        self._passing_count: int             = 0   # agents currently in passage
        self._max_concurrent: int            = 1   # set by set_difficulty()
        self._num_lookalikes: int            = 0   # set by set_difficulty()
        self._difficulty: str                = "easy"
        self.rescue_count: int               = 0
        self.lookalike_track_ids: List[str]  = []
        self._spawned                        = False
        self._motion_active                   = False
        self._tick                            = 0
        self._tick_robot_xy: Optional[Tuple[float, float]] = None
        self.episode_done: bool              = False

    # ── Queue route rebuild ───────────────────────────────────────────────────

    def _rebuild_queue_routes(self, spacing: float) -> None:
        for queue_id in (_R_QUEUE_A, _R_QUEUE_B):
            orig = self._routes[queue_id]
            if len(orig) < 2:
                continue
            if queue_id == _R_QUEUE_A:
                anchor = orig[0]
                dir_x  = orig[-1][0] - orig[0][0]
                dir_y  = orig[-1][1] - orig[0][1]
            else:
                anchor = orig[-1]
                dir_x  = orig[0][0] - orig[-1][0]
                dir_y  = orig[0][1] - orig[-1][1]
            mag = math.hypot(dir_x, dir_y)
            if mag < 1e-3:
                continue
            ux, uy = dir_x / mag, dir_y / mag
            n    = len(orig)
            wps  = [(anchor[0] + ux * i * spacing,
                     anchor[1] + uy * i * spacing)
                    for i in range(n)]
            # Queue B: _wp_for_pos returns cap-1-pos, so index cap-1 must be
            # the FRONT (closest to door).  The wps list above has index 0 =
            # front, so reverse it for Queue B.
            if queue_id == _R_QUEUE_B:
                wps = list(reversed(wps))
            self._routes[queue_id] = wps

    # ── Queue helpers ─────────────────────────────────────────────────────────

    def _wp_for_pos(self, queue_id: int, pos: int) -> int:
        if queue_id == _R_QUEUE_A:
            return min(pos, self._cap[_R_QUEUE_A] - 1)
        else:
            return max(self._cap[_R_QUEUE_B] - 1 - pos, 0)

    def _assign_queue_back(self, queue_id: int, ag: _Agent) -> None:
        q   = self._queue[queue_id]
        pos = min(len(q), self._cap[queue_id] - 1)
        ag.queue_id     = queue_id
        ag.queue_pos    = pos
        ag.queue_wp_idx = self._wp_for_pos(queue_id, pos)
        ag.pass_id      = _R_PASS_A2B if queue_id == _R_QUEUE_A else _R_PASS_B2A
        ag.pass_wp_idx  = 0
        ag.phase        = _Phase.APPROACH
        q.append(ag)

    def _depart_front(self, queue_id: int, ag: _Agent) -> None:
        """Remove departing agent, shift rest forward, increment passing counter."""
        q = self._queue[queue_id]
        if q and q[0] is ag:
            q.pop(0)
        for new_pos, other in enumerate(q):
            other.queue_pos    = new_pos
            other.queue_wp_idx = self._wp_for_pos(queue_id, new_pos)
            if other.phase == _Phase.QUEUE:
                other.phase = _Phase.APPROACH
        self._passing_count += 1

    # ── Difficulty ───────────────────────────────────────────────────────────

    def set_difficulty(self, level: str) -> None:
        level = level.lower()
        if level not in DIFFICULTY_PRESETS:
            print(f"[passage_npc] unknown difficulty: {level}")
            return
        preset = DIFFICULTY_PRESETS[level]
        self._difficulty      = level
        self._max_concurrent  = preset["max_concurrent"]
        self._min_speed       = preset["min_speed"]
        self._max_speed       = preset["max_speed"]
        self._num_lookalikes  = int(preset.get("num_lookalikes", 0))
        for ag in self.agents:
            ag.speed = float(self.rng.uniform(self._min_speed, self._max_speed))
        print(f"[passage_npc] difficulty={level}  "
              f"max_concurrent={self._max_concurrent}  "
              f"speed=[{self._min_speed},{self._max_speed}]  "
              f"num_walkers={preset['num_walkers']}  lookalikes={self._num_lookalikes}")

    def preset_num_walkers(self) -> int:
        return int(DIFFICULTY_PRESETS[self._difficulty]["num_walkers"])

    # ── Spawn ─────────────────────────────────────────────────────────────────

    def _spawn_loc_for_side(
        self,
        queue_id: int,
        taken: List[Tuple[float, float]],
    ) -> carla.Location:
        """Spawn near the flow-point area for this queue side, far from the door."""
        # Queue A north side → FP[1,2] (both north, y≈164-165)
        # Queue B south/west side → FP[0,3] (both south, y≈146)
        fp_indices = [1, 2] if queue_id == _R_QUEUE_A else [0, 3]
        for attempt in range(90):
            fp = self._flow_points[self._rand.choice(fp_indices)]
            # Widen the sampling box after repeated failures (dense presets).
            half = 3.0 if attempt < 45 else 4.5
            ox = float(self.rng.uniform(-half, half))
            oy = float(self.rng.uniform(-half, half))
            wx, wy = fp["x"] + ox, fp["y"] + oy
            if not self.roi_map.is_world_walkable(wx, wy):
                continue
            if all(math.hypot(wx - tx, wy - ty) >= _SPAWN_MIN_DIST for tx, ty in taken):
                return carla.Location(x=wx, y=wy, z=self.spawn_z)
        # Fallback: flow point itself with tiny jitter
        fp = self._flow_points[fp_indices[0]]
        return carla.Location(
            x=fp["x"] + float(self.rng.uniform(-0.3, 0.3)),
            y=fp["y"] + float(self.rng.uniform(-0.3, 0.3)),
            z=self.spawn_z,
        )

    def _initial_yaw_for_spawn(self, queue_id: int, q_pos: int, loc: carla.Location) -> float:
        wp_idx = self._wp_for_pos(queue_id, q_pos)
        wp_x, wp_y = self._routes[queue_id][wp_idx]
        return float(math.degrees(math.atan2(float(wp_y) - float(loc.y), float(wp_x) - float(loc.x))))

    def spawn(
        self,
        num_walkers: Optional[int] = None,
        min_speed: Optional[float] = None,
        max_speed: Optional[float] = None,
    ) -> None:
        if self._spawned:
            return
        if num_walkers is None:
            num_walkers = self.preset_num_walkers()
        num_walkers = int(num_walkers)
        if min_speed is not None:
            self._min_speed = min_speed
        if max_speed is not None:
            self._max_speed = max_speed

        if self._walker_bps is None:
            self._walker_bps = [bp for bp in self.world.get_blueprint_library().filter("walker.pedestrian.*")
                                if bp.id not in _NON_HUMAN_WALKER_BLUEPRINTS]

        # Each walker gets a unique blueprint, except the legend-level
        # target-look-alike distractors, which deliberately reuse the target's
        # appearance to stress perception / ReID.
        if num_walkers > len(self._walker_bps):
            raise RuntimeError(
                f"[NPC] num_walkers={num_walkers} exceeds blueprint pool "
                f"({len(self._walker_bps)}); each NPC needs a unique appearance.")
        chosen_bps = self._rand.sample(self._walker_bps, num_walkers)
        n_look = min(self._num_lookalikes, max(0, num_walkers - 1))
        lookalike_idx = set(self._rand.sample(range(1, num_walkers), n_look)) if n_look > 0 else set()
        for i in lookalike_idx:
            chosen_bps[i] = chosen_bps[0]

        # Split walkers between the two queues proportionally to queue capacity
        # (A=target side), so dense presets do not overflow the shorter queue.
        cap_a, cap_b = self._cap[_R_QUEUE_A], self._cap[_R_QUEUE_B]
        n_a = int(round(num_walkers * cap_a / float(cap_a + cap_b)))
        n_a = max(1, min(num_walkers - (1 if num_walkers > 1 else 0), n_a))
        n_b = num_walkers - n_a
        assignments = [_R_QUEUE_A] * n_a + [_R_QUEUE_B] * n_b
        non_tgt = assignments[1:]
        self._rand.shuffle(non_tgt)
        assignments = [assignments[0]] + non_tgt

        taken_locs: List[Tuple[float, float]] = []
        failed = 0
        self.lookalike_track_ids = []
        self.rescue_count = 0
        for i in range(num_walkers):
            track_id  = f"N{i+1:02d}"
            is_target = (track_id == self.target_track_id)
            queue_id  = _R_QUEUE_A if is_target else assignments[i]
            q_pos     = min(len(self._queue[queue_id]), self._cap[queue_id] - 1)

            bp = chosen_bps[i]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "true")

            actor = None
            for _ in range(15):
                loc   = self._spawn_loc_for_side(queue_id, taken_locs)
                yaw   = self._initial_yaw_for_spawn(queue_id, q_pos, loc)
                actor = self.world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=yaw)))
                if actor is not None:
                    taken_locs.append((loc.x, loc.y))
                    break
            if actor is None:
                failed += 1
                continue

            speed = float(self.rng.uniform(self._min_speed, self._max_speed))
            ag    = _Agent(
                walker=actor, track_id=track_id, is_target=is_target,
                speed=speed,
                phase=_Phase.IDLE,
                release_tick=i * 3,   # small stagger to avoid SFM overload at t=0
                is_lookalike=(i in lookalike_idx and not is_target),
            )
            if ag.is_lookalike:
                self.lookalike_track_ids.append(track_id)
            self.agents.append(ag)
            self._assign_queue_back(queue_id, ag)
            # Restore to IDLE after assign (assign sets APPROACH)
            ag.phase = _Phase.IDLE

        self._spawned        = True
        self._passing_count  = 0
        self._tick           = 0
        print(f"[passage_npc] spawned={len(self.agents)}  failed={failed}")
        print(f"[passage_npc] queue_A={len(self._queue[_R_QUEUE_A])}/{cap_a}  "
              f"queue_B={len(self._queue[_R_QUEUE_B])}/{cap_b}  "
              f"lookalikes={self.lookalike_track_ids}")
        if not self.agents:
            raise RuntimeError("[passage_npc] 0 agents spawned")

    # ── Motion control ────────────────────────────────────────────────────────

    def set_motion_active(self, active: bool) -> None:
        self._motion_active = active
        if not active:
            stop = carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0)
            for ag in self.agents:
                try:
                    ag.walker.apply_control(stop)
                except Exception:
                    pass

    def step(self, dt: float = 0.1, robot_actor: Optional[carla.Actor] = None, **_kwargs) -> None:
        if not self._motion_active or not self._spawned:
            return

        self._tick += 1

        # Cache robot position
        self._tick_robot_xy = None
        if robot_actor is not None:
            try:
                rl = robot_actor.get_location()
                self._tick_robot_xy = (float(rl.x), float(rl.y))
            except Exception:
                pass

        # Snapshot positions once per tick so SFM sees a consistent world
        pos_snap: Dict[str, Tuple[float, float]] = {}
        for ag in self.agents:
            try:
                loc = ag.walker.get_location()
                pos_snap[ag.track_id] = (float(loc.x), float(loc.y))
            except Exception:
                pass

        for ag in self.agents:
            try:
                self._step_one(ag, dt, pos_snap)
            except Exception as exc:
                print(f"[passage_npc] {ag.track_id}: {exc}")

        if self._tick % _RESCUE_CHECK_TICKS == 0:
            self._rescue_check(pos_snap)

    # ── Long-term rescue ─────────────────────────────────────────────────────

    def _rescue_check(self, pos_snap: Dict[str, Tuple[float, float]]) -> None:
        """Re-route non-target agents whose displacement stays too small over time.

        APPROACH: the agent cannot reach its queue slot (blocked by the crowd);
                  accept its current spot as the slot so the queue can proceed.
        PASS:     the agent is wedged in the passage; skip ahead along its route.
        QUEUE/IDLE agents are waiting by design and are never rescued.
        """
        for ag in self.agents:
            xy = pos_snap.get(ag.track_id)
            if xy is None:
                continue
            if ag.is_target or ag.phase not in (_Phase.APPROACH, _Phase.PASS):
                ag.rescue_anchor = xy
                ag.stuck_windows = 0
                continue
            if ag.rescue_anchor is None:
                ag.rescue_anchor = xy
                continue
            disp = math.hypot(xy[0] - ag.rescue_anchor[0], xy[1] - ag.rescue_anchor[1])
            ag.rescue_anchor = xy
            if disp >= _RESCUE_MIN_DISP_M:
                ag.stuck_windows = 0
                continue
            ag.stuck_windows += 1
            if ag.stuck_windows < _RESCUE_WINDOWS:
                continue
            ag.stuck_windows = 0
            self.rescue_count += 1
            if ag.phase == _Phase.APPROACH:
                ag.phase = _Phase.QUEUE
                self._stop(ag)
                print(f"[passage_npc] rescue: {ag.track_id} APPROACH→QUEUE "
                      f"(q{ag.queue_id} pos={ag.queue_pos})")
            else:
                route = self._routes[ag.pass_id]
                ag.pass_wp_idx = min(len(route), ag.pass_wp_idx + _RESCUE_SKIP_WPS)
                print(f"[passage_npc] rescue: {ag.track_id} PASS skip→wp{ag.pass_wp_idx}/{len(route)}")

    # ── SFM (velocity-integration, matches sfm_demo.py architecture) ─────────

    def _sfm_step(
        self,
        ag: _Agent,
        goal_x: float,
        goal_y: float,
        target_speed: float,
        pos_snap: Dict[str, Tuple[float, float]],
    ) -> float:
        """Direction-addition SFM (like park).
        Adds repulsion vector to goal direction, renormalises, keeps speed constant.
        Returns distance to goal."""
        px, py = pos_snap.get(ag.track_id, (0.0, 0.0))
        d_goal = math.hypot(goal_x - px, goal_y - py)

        if ag.is_target:
            if d_goal > 1e-3:
                ag.walker.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(
                        (goal_x - px) / d_goal, (goal_y - py) / d_goal, 0.0),
                    speed=target_speed))
            return d_goal

        # Goal direction — dead zone: no pull when already close to waypoint.
        # Prevents oscillation at queue positions; goal force resumes if pushed away.
        if d_goal > 0.2:
            gx, gy = (goal_x - px) / d_goal, (goal_y - py) / d_goal
        else:
            gx, gy = 0.0, 0.0

        right_x, right_y = gy, -gx   # keep-right lateral direction

        # ── Pedestrian repulsion ──────────────────────────────────────────────
        rep_x, rep_y = 0.0, 0.0
        for tid, (bx, by) in pos_snap.items():
            if tid == ag.track_id:
                continue
            dx   = px - bx
            dy   = py - by
            dist = math.hypot(dx, dy)
            if dist < 0.05 or dist >= _SFM_PED_R:
                continue
            n_x, n_y = dx / dist, dy / dist
            f_mag    = _SFM_PED_A * math.exp(-dist / _SFM_PED_B)
            # Head-on: other is directly in front → add rightward lateral push
            cos_front = -n_x * gx - n_y * gy
            if cos_front > 0.3:
                rep_x += (f_mag * _SFM_LATERAL_BIAS * right_x +
                          f_mag * (1.0 - _SFM_LATERAL_BIAS * 0.5) * n_x)
                rep_y += (f_mag * _SFM_LATERAL_BIAS * right_y +
                          f_mag * (1.0 - _SFM_LATERAL_BIAS * 0.5) * n_y)
            else:
                rep_x += f_mag * n_x
                rep_y += f_mag * n_y

        # ── Robot repulsion ───────────────────────────────────────────────────
        if self._tick_robot_xy is not None:
            rx, ry = self._tick_robot_xy
            dx     = px - rx
            dy     = py - ry
            dist   = math.hypot(dx, dy)
            if 0.05 < dist < _SFM_ROBOT_R:
                d_eff = max(0.01, dist - _BODY_RADIUS)
                f_mag = _SFM_ROBOT_A * math.exp(-d_eff / _SFM_ROBOT_B)
                rep_x += f_mag * dx / dist
                rep_y += f_mag * dy / dist

        # ── Direction addition + renormalise ──────────────────────────────────
        dir_x = gx + rep_x
        dir_y = gy + rep_y
        dir_len = math.hypot(dir_x, dir_y)
        if dir_len > 0.01:
            dir_x /= dir_len
            dir_y /= dir_len
            ag.walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(dir_x, dir_y, 0.0),
                speed=target_speed))
        else:
            # No goal force and no repulsion → naturally stop
            ag.walker.apply_control(carla.WalkerControl(
                direction=carla.Vector3D(0.0, 0.0, 0.0), speed=0.0))

        return d_goal

    # ── Movement helpers ──────────────────────────────────────────────────────

    def _stop(self, ag: _Agent) -> None:
        ag.walker.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(0, 0, 0), speed=0.0))

    # ── Per-agent step ────────────────────────────────────────────────────────

    def _step_one(self, ag: _Agent, dt: float,
                  pos_snap: Dict[str, Tuple[float, float]]) -> None:

        # ── IDLE ──────────────────────────────────────────────────────────────
        if ag.phase == _Phase.IDLE:
            if self._tick >= ag.release_tick:
                ag.phase = _Phase.APPROACH
            else:
                self._stop(ag)
            return

        # ── APPROACH ──────────────────────────────────────────────────────────
        if ag.phase == _Phase.APPROACH:
            wp = self._routes[ag.queue_id][ag.queue_wp_idx]
            if self._sfm_step(ag, wp[0], wp[1], ag.speed, pos_snap) < _ARRIVE_M:
                ag.phase = _Phase.QUEUE
                self._stop(ag)

        # ── QUEUE ─────────────────────────────────────────────────────────────
        elif ag.phase == _Phase.QUEUE:
            # Depart when at front and under concurrency limit
            if ag.queue_pos == 0 and self._passing_count < self._max_concurrent:
                q = self._queue[ag.queue_id]
                if q and q[0] is ag:
                    ag.phase       = _Phase.PASS
                    ag.pass_wp_idx = 0
                    self._depart_front(ag.queue_id, ag)
                    return

            # Advance: if the slot directly ahead is vacant, move up one position.
            # Only QUEUE-state agents advance; APPROACH agents keep their target.
            # This prevents two agents racing for the same slot.
            if ag.queue_pos > 0:
                ahead = ag.queue_pos - 1
                slot_taken = any(
                    other.queue_pos == ahead
                    for other in self._queue[ag.queue_id]
                    if other is not ag
                )
                if not slot_taken:
                    ag.queue_pos    = ahead
                    ag.queue_wp_idx = self._wp_for_pos(ag.queue_id, ahead)
                    ag.phase        = _Phase.APPROACH
                    return

            wp = self._routes[ag.queue_id][ag.queue_wp_idx]
            self._sfm_step(ag, wp[0], wp[1], ag.speed, pos_snap)

        # ── PASS ──────────────────────────────────────────────────────────────
        elif ag.phase == _Phase.PASS:
            route = self._routes[ag.pass_id]
            if ag.pass_wp_idx >= len(route):
                self._on_passage_done(ag)
                return
            wp = route[ag.pass_wp_idx]
            d  = self._sfm_step(ag, wp[0], wp[1], ag.speed, pos_snap)
            if d < _ARRIVE_M:
                ag.pass_wp_idx += 1
            else:
                # If SFM deflected agent past current waypoint, skip it
                vel  = ag.walker.get_velocity()
                v_sp = math.hypot(vel.x, vel.y)
                if v_sp > 0.1:
                    loc   = ag.walker.get_location()
                    to_wp = ((wp[0] - loc.x) * vel.x / v_sp +
                             (wp[1] - loc.y) * vel.y / v_sp)
                    if to_wp < -0.2:
                        ag.pass_wp_idx += 1

    def _on_passage_done(self, ag: _Agent) -> None:
        # The crowd cap means all walkers physically inside the complete
        # passage.  Release the slot only after the agent leaves the passage;
        # the previous half-route release made the observed count exceed the
        # configured limit.
        self._passing_count = max(0, self._passing_count - 1)
        ag.pass_count   += 1
        if ag.is_target and ag.pass_id == _R_PASS_A2B:
            self.episode_done = True
            self._stop(ag)
            print(f"[passage_npc] episode_done: target {ag.track_id} completed A->B passage")
            return
        dest = _R_QUEUE_B if ag.pass_id == _R_PASS_A2B else _R_QUEUE_A
        self._assign_queue_back(dest, ag)

    # ── Queries ───────────────────────────────────────────────────────────────

    def _make_state(self, ag: _Agent) -> NpcState:
        loc = ag.walker.get_location()
        vel = ag.walker.get_velocity()
        tf  = ag.walker.get_transform()
        return NpcState(
            track_id=ag.track_id, actor_id=ag.walker.id,
            x=float(loc.x), y=float(loc.y), z=float(loc.z),
            vx=float(vel.x), vy=float(vel.y), vz=float(vel.z),
            yaw_deg=float(tf.rotation.yaw),
            speed=float(math.hypot(vel.x, vel.y)),
        )

    def get_states(self) -> List[NpcState]:
        out = []
        for ag in self.agents:
            try:
                out.append(self._make_state(ag))
            except Exception:
                pass
        return out

    def get_target_by_track_id(self, track_id: str) -> Optional[NpcState]:
        for ag in self.agents:
            if ag.track_id == track_id:
                try:
                    return self._make_state(ag)
                except Exception:
                    return None
        return None

    def get_actor_by_track_id(self, track_id: str) -> Optional[carla.Actor]:
        for ag in self.agents:
            if ag.track_id == track_id:
                return ag.walker
        return None

    def destroy(self) -> None:
        for ag in self.agents:
            try:
                ag.walker.destroy()
            except Exception:
                pass
        self.agents.clear()
        self._queue         = {_R_QUEUE_A: [], _R_QUEUE_B: []}
        self._passing_count = 0
        self._spawned       = False
        self.episode_done   = False
        print("[passage_npc] all agents destroyed")

    # ── Debug state file ──────────────────────────────────────────────────────

    _PHASE_COLOR = {
        _Phase.IDLE:     "idle",
        _Phase.APPROACH: "transit",
        _Phase.QUEUE:    "waiting",
        _Phase.PASS:     "staged",
    }

    def write_debug_state(
        self,
        tick: int,
        robot_state,
        target_track_id: str = "",
        state_file: str = "/tmp/npc_debug_state.json",
        sim_t: float = 0.0,
    ) -> None:
        npcs, paths = [], []
        for ag in self.agents:
            try:
                loc    = ag.walker.get_location()
                is_tgt = ag.track_id == target_track_id

                wp_x = wp_y = None
                if ag.phase in (_Phase.IDLE, _Phase.APPROACH, _Phase.QUEUE):
                    wp = self._routes[ag.queue_id][ag.queue_wp_idx]
                    wp_x, wp_y = wp[0], wp[1]
                elif ag.phase == _Phase.PASS:
                    route = self._routes[ag.pass_id]
                    if ag.pass_wp_idx < len(route):
                        wp_x, wp_y = route[ag.pass_wp_idx]

                npcs.append({
                    "track_id": ag.track_id,
                    "x": float(loc.x), "y": float(loc.y),
                    "is_target": is_tgt,
                    "state":     "target" if is_tgt else self._PHASE_COLOR[ag.phase],
                    "phase":     ag.phase.name,
                    "queue_pos": ag.queue_pos,
                    "wp_x": wp_x, "wp_y": wp_y,
                })

                if ag.phase == _Phase.PASS:
                    remaining = self._routes[ag.pass_id][ag.pass_wp_idx: ag.pass_wp_idx + 20]
                    if remaining:
                        paths.append({"track_id": ag.track_id,
                                      "wps": [list(p) for p in remaining]})
            except Exception:
                pass

        robot_dict: dict = {}
        if robot_state is not None:
            robot_dict = {"x": float(robot_state.x), "y": float(robot_state.y),
                          "yaw_rad": float(robot_state.yaw_rad)}

        flow_pts = [[fp["x"], fp["y"], f"FP{i}"]
                    for i, fp in enumerate(self._flow_points)]

        import json as _json
        tmp = state_file + ".tmp"
        try:
            with open(tmp, "w") as f:
                _json.dump({
                    "tick": tick, "sim_t": sim_t,
                    "robot": robot_dict,
                    "npcs":  npcs,
                    "npc_paths": paths,
                    "flow_points_world":    flow_pts,
                    "connector_cells_world": [],
                    "passing_count":  self._passing_count,
                    "max_concurrent": self._max_concurrent,
                    "difficulty":     self._difficulty,
                }, f)
            os.replace(tmp, state_file)
        except Exception:
            pass
