from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

from followbench.sim.core_types import NpcState
from baselines.perception.gt.gt_visibility import gt_distance_xy, gt_target_xy


@dataclass
class PeoplePipelineResult:
    tracked_peds: list
    target_ped: object | None
    target_tracker_id: int | None
    modified_obs: object
    using_tracker_target: bool
    target_error_m: float | None
    timing: dict
    track_bboxes_by_view: dict[str, list[dict]]


class PeopleTrackerPipeline:
    def __init__(
        self,
        tracker,
        tracker_update_fn: Callable[[object, object], list],
        synthetic_target_track_id: str,
        synthetic_npc_track_id_fn: Callable[[int], str],
        lock_max_dist_m: float = 2.5,
        relock_interval: int = 5,
        default_speed: float = 0.5,
    ) -> None:
        self._tracker = tracker
        self._tracker_update_fn = tracker_update_fn
        self.synthetic_target_track_id = str(synthetic_target_track_id)
        self.synthetic_npc_track_id_fn = synthetic_npc_track_id_fn
        self.lock_max_dist_m = float(lock_max_dist_m)
        self.relock_interval = max(1, int(relock_interval))
        self.default_speed = float(default_speed)
        self._target_tracker_id: Optional[int] = None
        self._target_last_xy: Optional[tuple[float, float]] = None
        self._ticks_since_lock: int = 0

    @property
    def target_tracker_id(self) -> Optional[int]:
        return self._target_tracker_id

    def reset(self) -> None:
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
        self._target_tracker_id = None
        self._target_last_xy = None
        self._ticks_since_lock = 0

    def update(self, obs) -> PeoplePipelineResult:
        t_track_start = time.perf_counter()
        tracked_peds = list(self._tracker_update_fn(self._tracker, obs) or [])
        t_track_ms = (time.perf_counter() - t_track_start) * 1000.0

        t_id_start = time.perf_counter()
        target_ped = self._identify_target(tracked_peds, obs)
        t_id_ms = (time.perf_counter() - t_id_start) * 1000.0
        tracker_timing = getattr(self._tracker, "last_timing", {}) or {}

        modified_obs = obs
        using_tracker_target = target_ped is not None
        if target_ped is not None:
            modified_obs = dataclasses.replace(
                obs,
                target=_synthetic_target(target_ped, obs, self.synthetic_target_track_id, self.default_speed),
                npcs=_synthetic_npcs(tracked_peds, self._target_tracker_id, self.synthetic_npc_track_id_fn, self.default_speed),
            )

        target_error_m = None
        if target_ped is not None:
            target_error_m = gt_distance_xy(obs, (float(target_ped.x), float(target_ped.y)))

        timing = {
            "track_ms": float(t_track_ms),
            "id_ms": float(t_id_ms),
            "percep_ms": float(t_track_ms + t_id_ms),
        }
        timing.update({str(k): float(v) for k, v in tracker_timing.items() if _is_number(v)})

        return PeoplePipelineResult(
            tracked_peds=tracked_peds,
            target_ped=target_ped,
            target_tracker_id=self._target_tracker_id,
            modified_obs=modified_obs,
            using_tracker_target=using_tracker_target,
            target_error_m=target_error_m,
            timing=timing,
            track_bboxes_by_view=_build_track_bboxes_by_view(
                self._tracker, self._target_tracker_id
            ),
        )

    def _identify_target(self, tracked_peds: list, obs) -> object | None:
        if not tracked_peds:
            self._ticks_since_lock += 1
            return None

        if self._target_tracker_id is not None:
            for ped in tracked_peds:
                if getattr(ped, "track_id", None) == self._target_tracker_id:
                    self._target_last_xy = (float(ped.x), float(ped.y))
                    self._ticks_since_lock = 0
                    return ped
            self._ticks_since_lock += 1
            if self._ticks_since_lock < self.relock_interval:
                return None
            self._target_tracker_id = None

        gt_xy = gt_target_xy(obs)
        if gt_xy is None:
            return None

        best_ped = None
        best_d = float("inf")
        for ped in tracked_peds:
            ped_id = int(getattr(ped, "track_id", -1))
            if ped_id < 0:
                continue
            d = math.hypot(float(ped.x) - gt_xy[0], float(ped.y) - gt_xy[1])
            if d < best_d:
                best_d = d
                best_ped = ped

        if best_ped is None or best_d >= self.lock_max_dist_m:
            return None

        self._target_tracker_id = int(getattr(best_ped, "track_id"))
        self._target_last_xy = (float(best_ped.x), float(best_ped.y))
        self._ticks_since_lock = 0
        return best_ped


def _synthetic_target(target_ped, obs, track_id: str, default_speed: float) -> NpcState:
    gt = getattr(obs, "target", None)
    speed = float(getattr(target_ped, "speed", 0.0))
    if speed <= 0.0:
        speed = float(default_speed)
    return NpcState(
        track_id=track_id,
        actor_id=-1,
        x=float(target_ped.x),
        y=float(target_ped.y),
        z=float(gt.z) if gt is not None else 0.0,
        vx=float(getattr(target_ped, "vx", 0.0)),
        vy=float(getattr(target_ped, "vy", 0.0)),
        vz=0.0,
        yaw_deg=float(math.degrees(float(getattr(target_ped, "yaw_rad", 0.0)))),
        speed=speed,
    )


def _synthetic_npcs(tracked_peds: list, target_tracker_id: Optional[int], track_id_fn: Callable[[int], str], default_speed: float) -> list[NpcState]:
    out: list[NpcState] = []
    for ped in tracked_peds:
        ped_id = int(getattr(ped, "track_id", -1))
        if target_tracker_id is not None and ped_id == target_tracker_id:
            continue
        out.append(
            NpcState(
                track_id=track_id_fn(ped_id),
                actor_id=-1,
                x=float(ped.x),
                y=float(ped.y),
                z=0.0,
                vx=float(getattr(ped, "vx", 0.0)),
                vy=float(getattr(ped, "vy", 0.0)),
                vz=0.0,
                yaw_deg=float(math.degrees(float(getattr(ped, "yaw_rad", 0.0)))),
                speed=float(getattr(ped, "speed", 0.0)),
            )
        )
    return out


def _build_track_bboxes_by_view(tracker, target_tracker_id: Optional[int]) -> dict[str, list[dict]]:
    """Convert the tracker's per-view raw bboxes into per-view dict form.

    Falls back to an empty dict when the tracker exposes no per-view output.
    """
    dets_by_view = getattr(tracker, "last_yolo_dets_by_view", {}) or {}
    depth_per_id = getattr(tracker, "last_depth_per_global_id", {}) or {}
    out: dict[str, list[dict]] = {}
    for v_name, yolo_dets in dets_by_view.items():
        view_out: list[dict] = []
        for tid, x1, y1, x2, y2, cf in yolo_dets:
            item = {
                "track_id": int(tid),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(cf),
                "is_target": int(tid) == target_tracker_id,
            }
            if int(tid) in depth_per_id:
                item["depth"] = float(depth_per_id[int(tid)])
            view_out.append(item)
        out[v_name] = view_out
    return out


def _is_number(value) -> bool:
    return isinstance(value, (int, float))


__all__ = ["PeoplePipelineResult", "PeopleTrackerPipeline"]
