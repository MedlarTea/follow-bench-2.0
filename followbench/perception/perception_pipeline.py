"""PerceptionPipeline — unified tracking + appearance ReID + state machine.

Wraps three building blocks behind one ``update()`` call so any downstream
follower planner can plug perception in without re-implementing the wiring:

  PedTrackerDepth  →  TargetReID  →  states FSM (Initial → Training → Tracking
                                                 ⇄ Reid)

Inputs  (per tick): RGB image, depth image, robot pose (x, y, z, yaw), camera
                    intrinsics + extrinsics, an optional GT XY hint used **only**
                    to bootstrap the initial lock.
Outputs (per tick): a ``PerceptionResult`` with the target's pixel bbox, world
                    XY position, the full set of all-track world XYs, the
                    state-machine name, and per-stage timing.

This file deliberately knows nothing about the planner / MPC layer — the
returned ``PerceptionResult`` carries everything a planner needs to choose its
goal point and maintain a list of obstacles. Mirrors the workflow established
by ``follow/follow_server/src/following_node.py``.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from followbench.perception.ped_tracker_depth import PedTrackerDepth, TrackedPed, ViewInputs
from followbench.perception.states import (
    InitialState,
    InitialTrainingState,
    ReidState,
    State,
    StateContext,
    TrackingState,
)
from followbench.perception.states.base import StateConfig
from followbench.perception.target_reid import TargetReID


@dataclass
class PerceptionResult:
    state_name: str
    target_id: int                                  # -1 if no target
    target_bbox: Optional[Tuple[float, float, float, float]]   # (x1,y1,x2,y2)
    target_view: Optional[str]                      # 'front' | 'left' | 'right' | None
    target_xy_world: Optional[Tuple[float, float]]
    all_tracks: List[TrackedPed] = field(default_factory=list)
    yolo_dets: List[Tuple[int, float, float, float, float, float]] = field(default_factory=list)
    yolo_dets_by_view: Dict[str, List[Tuple[int, float, float, float, float, float]]] = (
        field(default_factory=dict)
    )
    depth_per_id: Dict[int, float] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerceptionConfig:
    yolo_model: str = "yolo11s.pt"
    tracker_cfg: str = "botsort.yaml"
    tracker_conf: float = 0.25
    tracker_iou: float = 0.7
    tracker_imgsz: int = 640
    tracker_device: str = "cuda"
    tracker_yolo_stride: int = 1
    tracker_max_range_m: float = 15.0
    dt: float = 0.1                   # tick period; used for tracker kinematics
    kinematics_mode: str = "kf"      # 'ema' | 'kf'
    kf_pos_sigma: float = 0.20        # KF measurement noise σ (m); only used when kinematics_mode='kf'
    kf_vel_sigma_q: float = 0.05      # KF process noise σ on velocity (m/s)

    reid_mode: str = "basic"          # 'basic' | 'kpr'
    reid_kpr_config: str = "kpr_occ_duke_test"
    reid_device: str = "auto"
    reid_consecutive_required: int = 3
    # Recovery lock thresholds. Original (master branch) values are
    # ridge=0.35 / cosine=0.55; the tuned "better" freeze is 0.85 / 0.85.
    reid_ridge_threshold: float = 0.35
    reid_cosine_threshold: float = 0.55
    reid_min_bbox_area_px: int = 1500

    fsm_initial_training_samples: int = 10
    fsm_id_switch_thresh: float = 0.05
    fsm_initial_lateral_max: float = 0.8
    fsm_initial_max_dist_m: float = 6.0
    fsm_initial_gt_match_radius_m: float = 1.5


class PerceptionPipeline:
    def __init__(self, config: Optional[PerceptionConfig] = None) -> None:
        self.config = config or PerceptionConfig()

        self._tracker = PedTrackerDepth(
            model_name=self.config.yolo_model,
            tracker=self.config.tracker_cfg,
            conf=self.config.tracker_conf,
            iou=self.config.tracker_iou,
            imgsz=self.config.tracker_imgsz,
            device=self.config.tracker_device,
            yolo_stride=self.config.tracker_yolo_stride,
            max_range_m=self.config.tracker_max_range_m,
            dt=self.config.dt,
            kinematics_mode=self.config.kinematics_mode,
            kf_pos_sigma=self.config.kf_pos_sigma,
            kf_vel_sigma_q=self.config.kf_vel_sigma_q,
        )
        self._reid = TargetReID(
            mode=self.config.reid_mode,
            kpr_config=self.config.reid_kpr_config,
            device=self.config.reid_device,
            min_bbox_area_px=self.config.reid_min_bbox_area_px,
            consecutive_required=self.config.reid_consecutive_required,
            ridge_lock_threshold=self.config.reid_ridge_threshold,
            cosine_lock_threshold=self.config.reid_cosine_threshold,
        )
        self._state_cfg = StateConfig(
            initial_training_num_samples=self.config.fsm_initial_training_samples,
            id_switch_thresh=self.config.fsm_id_switch_thresh,
            initial_select_lateral_max=self.config.fsm_initial_lateral_max,
            initial_select_max_dist_m=self.config.fsm_initial_max_dist_m,
            initial_gt_match_radius_m=self.config.fsm_initial_gt_match_radius_m,
        )
        self._state: State = InitialState(self._state_cfg)
        self._last_target_xy: Optional[Tuple[float, float]] = None
        self._last_result: Optional[PerceptionResult] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._tracker.reset()
        self._reid.reset()
        self._state = InitialState(self._state_cfg)
        self._last_target_xy = None
        self._last_result = None

    @property
    def state_name(self) -> str:
        return self._state.state_name()

    @property
    def target_id(self) -> int:
        return self._state.target()

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        rgb_image: Optional[np.ndarray],
        depth_image: Optional[np.ndarray],
        robot_x: float,
        robot_y: float,
        robot_z: float,
        robot_yaw: float,
        rgb_intrinsics: Optional[Dict[str, Any]],
        rgb_extrinsics: Optional[Dict[str, Any]],
        gt_target_xy: Optional[Tuple[float, float]] = None,
        # Multi-view inputs (left / right). Any missing view is skipped.
        rgb_image_left: Optional[np.ndarray] = None,
        depth_image_left: Optional[np.ndarray] = None,
        rgb_intrinsics_left: Optional[Dict[str, Any]] = None,
        rgb_extrinsics_left: Optional[Dict[str, Any]] = None,
        rgb_image_right: Optional[np.ndarray] = None,
        depth_image_right: Optional[np.ndarray] = None,
        rgb_intrinsics_right: Optional[Dict[str, Any]] = None,
        rgb_extrinsics_right: Optional[Dict[str, Any]] = None,
    ) -> PerceptionResult:
        # ── Build the views dict (front + optional left/right) ───────────────
        views: Dict[str, ViewInputs] = {}
        if rgb_image is not None:
            views["front"] = ViewInputs(
                rgb=rgb_image, depth=depth_image,
                intrinsics=rgb_intrinsics, extrinsics=rgb_extrinsics,
            )
        if rgb_image_left is not None:
            views["left"] = ViewInputs(
                rgb=rgb_image_left, depth=depth_image_left,
                intrinsics=rgb_intrinsics_left, extrinsics=rgb_extrinsics_left,
            )
        if rgb_image_right is not None:
            views["right"] = ViewInputs(
                rgb=rgb_image_right, depth=depth_image_right,
                intrinsics=rgb_intrinsics_right, extrinsics=rgb_extrinsics_right,
            )

        # 1. Detection + 3D positioning across all views + cross-view merge.
        t0 = time.perf_counter()
        tracked: List[TrackedPed] = self._tracker.update(
            views=views,
            robot_x=robot_x, robot_y=robot_y,
            robot_z=robot_z, robot_yaw=robot_yaw,
        )
        t_track = (time.perf_counter() - t0) * 1000.0

        # Per-view bboxes with merged global ids (populated by the tracker).
        yolo_dets_by_view: Dict[
            str, List[Tuple[int, float, float, float, float, float]]
        ] = dict(getattr(self._tracker, "last_yolo_dets_by_view", {}) or {})
        # FSM-facing flat list: iterate views in priority order so that any
        # downstream ``next(d for d in yolo_dets if d[0] == target_id)`` picks
        # the front-view bbox first.
        yolo_dets: List[Tuple[int, float, float, float, float, float]] = []
        for v in ("front", "left", "right"):
            yolo_dets.extend(yolo_dets_by_view.get(v, []))

        # 2. Appearance features per view (ReID). Aggregate per-global-id by
        # view priority — no averaging, front wins.
        t1 = time.perf_counter()
        per_view_feats: Dict[int, List[Tuple[str, np.ndarray]]] = {}
        _VIEW_PRIORITY_LOCAL = {"front": 0, "left": 1, "right": 2}
        for v_name, img in (("front", rgb_image),
                            ("left",  rgb_image_left),
                            ("right", rgb_image_right)):
            if img is None:
                continue
            bboxes_v = yolo_dets_by_view.get(v_name, [])
            if not bboxes_v:
                continue
            feats_v = self._reid.extract(img, bboxes_v)
            for gid, f in feats_v.items():
                per_view_feats.setdefault(gid, []).append((v_name, f))

        features: Dict[int, np.ndarray] = {}
        for gid, pairs in per_view_feats.items():
            pairs.sort(key=lambda p: _VIEW_PRIORITY_LOCAL.get(p[0], 99))
            features[gid] = pairs[0][1]
        t_reid_extract = (time.perf_counter() - t1) * 1000.0

        # 3. Drive the FSM with this tick's data.
        tracks_world = {tp.track_id: (tp.x, tp.y) for tp in tracked
                        if tp.track_id >= 0}
        ctx = StateContext(
            features=features,
            bboxes=yolo_dets,
            tracks_world=tracks_world,
            robot_x=robot_x, robot_y=robot_y, robot_yaw=robot_yaw,
            gt_target_xy=gt_target_xy,
        )

        t2 = time.perf_counter()
        next_state = self._state.update(self._reid, ctx)
        if next_state is not self._state:
            print(f"[Perception] {self._state.state_name()} → "
                  f"{next_state.state_name()}  target_id={next_state.target()}",
                  flush=True)
            self._state = next_state
        t_fsm = (time.perf_counter() - t2) * 1000.0

        target_id = self._state.target()
        target_bbox = None
        target_xy = None
        target_view: Optional[str] = None
        if target_id >= 0:
            for tp in tracked:
                if tp.track_id == target_id:
                    target_xy = (tp.x, tp.y)
                    self._last_target_xy = target_xy
                    break
            # Find target bbox in the highest-priority view that has it.
            for v in ("front", "left", "right"):
                for det in yolo_dets_by_view.get(v, []):
                    if det[0] == target_id:
                        target_bbox = (det[1], det[2], det[3], det[4])
                        target_view = v
                        break
                if target_bbox is not None:
                    break

        sub_t = getattr(self._tracker, "last_timing", {}) or {}
        sub_r = getattr(self._reid, "last_timing", {}) or {}
        result = PerceptionResult(
            state_name=self._state.state_name(),
            target_id=target_id,
            target_bbox=target_bbox,
            target_view=target_view,
            target_xy_world=target_xy,
            all_tracks=tracked,
            yolo_dets=yolo_dets,
            yolo_dets_by_view=yolo_dets_by_view,
            depth_per_id=dict(
                getattr(self._tracker, "last_depth_per_global_id", {}) or {}
            ),
            timing={
                "track_ms": t_track,
                "reid_extract_ms": t_reid_extract,
                "fsm_ms": t_fsm,
                "yolo_ms": float(sub_t.get("yolo_ms", 0.0)),
                "yolo_front_ms": float(sub_t.get("yolo_front_ms", 0.0)),
                "yolo_left_ms":  float(sub_t.get("yolo_left_ms",  0.0)),
                "yolo_right_ms": float(sub_t.get("yolo_right_ms", 0.0)),
                "depth_ms": float(sub_t.get("depth_ms", 0.0)),
                "project_ms": float(sub_t.get("project_ms", 0.0)),
                "merge_ms": float(sub_t.get("merge_ms", 0.0)),
                "ran_yolo": bool(sub_t.get("ran_yolo", 0.0)),
                "reid_crop_ms": float(sub_r.get("crop_ms", 0.0)),
                "reid_infer_ms": float(sub_r.get("infer_ms", 0.0)),
                "reid_fit_ms": float(sub_r.get("fit_ms", 0.0)),
            },
        )
        self._last_result = result
        return result

    # ── Debug introspection ──────────────────────────────────────────────────

    def get_debug_info(self) -> Dict[str, Any]:
        if self._last_result is None:
            return {}
        r = self._last_result
        return {
            "state": r.state_name,
            "target_id": r.target_id,
            "target_bbox": r.target_bbox,
            "target_view": r.target_view,
            "target_xy": r.target_xy_world,
            "track_bboxes_by_view": {
                v: [{"track_id": int(tid),
                     "x1": float(x1), "y1": float(y1),
                     "x2": float(x2), "y2": float(y2),
                     "conf": float(cf),
                     "depth": float(r.depth_per_id.get(int(tid), float("nan"))),
                     "is_target": (int(tid) == r.target_id)}
                    for (tid, x1, y1, x2, y2, cf) in bboxes]
                for v, bboxes in r.yolo_dets_by_view.items()
            },
            "tracked_peds": [
                {"track_id": tp.track_id, "x": tp.x, "y": tp.y,
                 "is_target": (tp.track_id == r.target_id)}
                for tp in r.all_tracks
            ],
            "timing": dict(r.timing),
            "reid": self._reid.get_debug_snapshot(),
        }
