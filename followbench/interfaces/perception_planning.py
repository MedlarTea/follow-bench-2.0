"""Perception -> planning interface for modular follower methods.

A modular follower is split into two parts that talk through
:class:`PerceptionOutput`:

* :class:`PerceptionModule` turns raw sensors (RGB / depth / LiDAR, robot
  odometry, calibration) into a world-frame estimate of the target and the
  surrounding people.
* :class:`Planner` turns that estimate (plus the raw sensors, e.g. LiDAR for
  obstacles) into a velocity command :class:`FollowAction`.

:class:`PerceptionPlanningPolicy` glues the two together and exposes the
standard :class:`FollowerPolicyAdapter` API that every scene runner calls
(``reset()`` / ``act(obs)`` / ``get_debug_info()``). It also

* removes privileged ground truth from the observation before it reaches the
  perception module and the planner (see :func:`sensor_only_observation`);
* hands the perception module a one-off :class:`TargetHint` (GT position and
  front-camera bbox of the target) so it knows *whom* to follow, by default
  only until the first lock-on;
* builds the debug dict consumed by the evaluation logger, the identity
  monitor, the HUD and the bbox overlays.

End-to-end methods (e.g. TrackVLA) implement :class:`FollowerPolicyAdapter`
directly and do not use this module.
"""
from __future__ import annotations

import dataclasses
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter

XY = Tuple[float, float]

# Keys of ``FollowObservation.extras`` that carry ground truth.
PRIVILEGED_EXTRAS_KEYS = ("target_bbox_xyxy", "target_yaw_rad")

HINT_POLICIES = ("until_first_lock", "always", "never")


# ── Data passed from perception to planning ────────────────────────────────────


@dataclass
class TrackedPerson:
    """One tracked person in the world frame (metres, m/s)."""

    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.35
    is_target: bool = False
    confidence: Optional[float] = None


@dataclass
class BBox2D:
    """One detection / track box in image pixels.

    ``view`` is the camera it belongs to: ``"front"``, ``"left"`` or ``"right"``.
    """

    view: str
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int = -1
    is_target: bool = False
    confidence: Optional[float] = None


@dataclass
class PerceptionOutput:
    """What a perception module reports each tick.

    ``has_target`` must be ``False`` whenever the module is not confident that
    ``target_xy`` belongs to the target (lost, occluded, re-identifying ...).
    Planners are expected to handle that case (brake, coast, search ...).
    """

    has_target: bool
    target_xy: Optional[XY] = None
    target_vel: Optional[XY] = None
    target_yaw_rad: Optional[float] = None
    target_id: Optional[int] = None
    # Free-form state label shown in logs, e.g. "tracking" / "lost" / "reid".
    state: str = "tracking"
    people: List[TrackedPerson] = field(default_factory=list)
    bboxes: List[BBox2D] = field(default_factory=list)
    timing_ms: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TargetHint:
    """Designates the person to follow (ground truth, initialisation only)."""

    xy: XY
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None


# ── Module interfaces ──────────────────────────────────────────────────────────


class PerceptionModule(ABC):
    """Sensors -> :class:`PerceptionOutput`."""

    # Set to True only for oracle / debugging modules that need the
    # unfiltered observation (GT target and NPC states).
    requires_privileged_obs: bool = False

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of every episode."""

    @abstractmethod
    def perceive(self, obs: FollowObservation, hint: Optional[TargetHint]) -> PerceptionOutput:
        """Process one tick.

        ``obs`` has GT fields removed (``target=None``, ``npcs=[]``,
        ``target_visible=False``) unless ``requires_privileged_obs`` is True.
        ``hint`` is non-None only while the composite policy grants it.
        """

    def get_debug_info(self) -> Dict[str, Any]:
        return {}


class Planner(ABC):
    """(:class:`PerceptionOutput`, sensors) -> :class:`FollowAction`."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of every episode."""

    @abstractmethod
    def plan(self, obs: FollowObservation, perception: PerceptionOutput) -> FollowAction:
        """Return the body-frame velocity command for this tick.

        ``obs`` never contains GT target / NPC states; use ``obs.robot``,
        ``obs.lidar_points``, ``obs.depth_image`` ... for ego state and obstacles.
        """

    def get_debug_info(self) -> Dict[str, Any]:
        """Optional planner debug, merged into the policy debug dict.

        Keys recognised by the evaluation logger include ``goal_point``,
        ``traj_points``, ``obstacles``, ``search_mode`` and ``timing``
        (``{"planner_core_ms": ...}``).
        """
        return {}


# ── Helpers ────────────────────────────────────────────────────────────────────


def sensor_only_observation(obs: FollowObservation) -> FollowObservation:
    """Copy of ``obs`` without privileged ground truth.

    Kept: tick/dt, ego robot state, all images, depth, LiDAR, calibration.
    Removed: target / NPC states, GT visibility, GT extras.
    """
    extras = {k: v for k, v in (obs.extras or {}).items() if k not in PRIVILEGED_EXTRAS_KEYS}
    return dataclasses.replace(
        obs,
        target=None,  # type: ignore[arg-type]
        npcs=[],
        target_visible=False,
        target_pixel_count=0,
        extras=extras,
    )


def target_hint_from_obs(obs: FollowObservation) -> Optional[TargetHint]:
    if obs.target is None:
        return None
    bbox = (obs.extras or {}).get("target_bbox_xyxy")
    bbox_t = tuple(float(v) for v in bbox) if bbox is not None else None
    return TargetHint(xy=(float(obs.target.x), float(obs.target.y)), bbox_xyxy=bbox_t)  # type: ignore[arg-type]


def _bboxes_by_view(bboxes: List[BBox2D]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for b in bboxes:
        out.setdefault(str(b.view), []).append({
            "track_id": int(b.track_id),
            "x1": float(b.x1), "y1": float(b.y1), "x2": float(b.x2), "y2": float(b.y2),
            "conf": None if b.confidence is None else float(b.confidence),
            "is_target": bool(b.is_target),
        })
    return out


# ── Composite policy ───────────────────────────────────────────────────────────


class PerceptionPlanningPolicy(FollowerPolicyAdapter):
    """Runs ``perception`` then ``planner`` every tick.

    Args:
        perception: the perception module.
        planner: the planner.
        hint_policy: when the GT :class:`TargetHint` is passed to perception:
            ``"until_first_lock"`` (default; until ``has_target`` is first True),
            ``"always"`` (debugging only) or ``"never"``.
        brake_when_lost: if True, emit a zero command whenever
            ``has_target`` is False instead of calling the planner.
    """

    def __init__(
        self,
        perception: PerceptionModule,
        planner: Planner,
        *,
        hint_policy: str = "until_first_lock",
        brake_when_lost: bool = False,
    ) -> None:
        if hint_policy not in HINT_POLICIES:
            raise ValueError(f"hint_policy must be one of {HINT_POLICIES}, got {hint_policy!r}")
        self.perception = perception
        self.planner = planner
        self.hint_policy = hint_policy
        self.brake_when_lost = bool(brake_when_lost)
        self._locked_once = False
        self._last_debug: Dict[str, Any] = {}

    def reset(self) -> None:
        self.perception.reset()
        self.planner.reset()
        self._locked_once = False
        self._last_debug = {}

    def _hint_for(self, obs: FollowObservation) -> Optional[TargetHint]:
        if self.hint_policy == "never":
            return None
        if self.hint_policy == "until_first_lock" and self._locked_once:
            return None
        return target_hint_from_obs(obs)

    def act(self, obs: FollowObservation) -> FollowAction:
        sensor_obs = sensor_only_observation(obs)
        hint = self._hint_for(obs)

        t0 = time.perf_counter()
        p_obs = obs if self.perception.requires_privileged_obs else sensor_obs
        percep = self.perception.perceive(p_obs, hint)
        t1 = time.perf_counter()
        if percep.has_target:
            self._locked_once = True

        if self.brake_when_lost and not percep.has_target:
            action = FollowAction(0.0, 0.0)
        else:
            action = self.planner.plan(sensor_obs, percep)
        t2 = time.perf_counter()

        self._last_debug = self._build_debug(obs, percep, (t1 - t0) * 1e3, (t2 - t1) * 1e3)
        return action

    def get_debug_info(self) -> Dict[str, Any]:
        return self._last_debug

    # Debug dict layout matches what the scene runners, the evaluation logger
    # (build_step_perception_summary / build_step_timing) and the target
    # identity monitor read.
    def _build_debug(
        self,
        obs: FollowObservation,
        percep: PerceptionOutput,
        percep_ms: float,
        plan_ms: float,
    ) -> Dict[str, Any]:
        tracker_xy = list(percep.target_xy) if (percep.has_target and percep.target_xy is not None) else None
        gt_xy = [float(obs.target.x), float(obs.target.y)] if obs.target is not None else None
        err_m = None
        if tracker_xy is not None and gt_xy is not None:
            err_m = math.hypot(tracker_xy[0] - gt_xy[0], tracker_xy[1] - gt_xy[1])

        bboxes_by_view = _bboxes_by_view(percep.bboxes)
        tracked = [
            {"track_id": int(p.track_id), "x": float(p.x), "y": float(p.y),
             "vx": float(p.vx), "vy": float(p.vy), "is_target": bool(p.is_target)}
            for p in percep.people
        ]
        percep_debug = dict(self.perception.get_debug_info() or {})
        percep_debug.setdefault("state", percep.state)
        percep_debug.setdefault("target_id", percep.target_id)
        percep_debug.setdefault("target_xy", tracker_xy)
        percep_debug.setdefault("tracked_peds", tracked)
        percep_debug.setdefault("track_bboxes_by_view", bboxes_by_view)
        percep_debug.setdefault("timing", dict(percep.timing_ms))

        planner_debug = dict(self.planner.get_debug_info() or {})
        planner_timing = planner_debug.pop("timing", {}) or {}

        debug: Dict[str, Any] = dict(planner_debug)
        debug.update({
            "tracked_peds": tracked,
            "track_bboxes_by_view": bboxes_by_view,
            "track_bboxes_age": 0,
            "target_track_id": percep.target_id,
            "active_track_id": percep.target_id,
            "target_lost": not percep.has_target,
            "perception_has_target": bool(percep.has_target),
            "perception_state": percep.state,
            "reid_match": bool(percep.has_target and percep.target_id is not None),
            "target_pos": {"tracker": tracker_xy, "gt": gt_xy, "err_m": err_m},
            "planner_target": {
                "xy": tracker_xy,
                "vel": list(percep.target_vel) if percep.target_vel is not None else None,
                "yaw_rad": percep.target_yaw_rad,
            },
            "perception": percep_debug,
            "timing": {
                "perception_total_ms": float(percep_ms),
                "perception": {"total_ms": float(percep_ms), "raw": dict(percep.timing_ms)},
                "planner_core_ms": float(planner_timing.get("planner_core_ms", plan_ms)),
                "policy_total_ms": float(percep_ms + plan_ms),
            },
        })
        return debug
