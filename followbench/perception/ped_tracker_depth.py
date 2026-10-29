"""
PedTrackerDepth — multi-view RGB+depth pedestrian tracker with cross-view merge.

Per tick, for each available view (front / left / right):
  1. YOLO person detection + ByteTrack/BotSort tracking on the RGB image
     → bounding boxes with stable per-episode integer IDs (scoped to the view).
  2. For each bbox, sample a central sub-patch of the *depth* image and take the
     median of valid depth values  →  robust per-detection depth (metres).
  3. Back-project (u_center, v_center, depth_median) using camera intrinsics +
     camera yaw (``extrinsics_robot_to_sensor['yaw_deg']``) → world BEV frame.

Then across views:
  4. Resolve or allocate a global integer track id for every (view, local_id).
  5. Union-find merge of different-view detections whose world positions are
     within ``_MERGE_DIST_M``. Winner is the oldest global id (smallest
     birth_tick, tied by view priority front>left>right, then smallest gid).
  6. Emit one ``TrackedPed`` per surviving global id, with the world XY taken
     from the highest-priority view that has a valid depth for this tick.

Result is ``List[TrackedPed(track_id=global_id, x, y, conf, yaw_rad, vx, vy,
speed)]`` in world BEV frame; per-view bboxes (keyed by the same global ids)
are exposed on ``self.last_yolo_dets_by_view`` for the ReID layer and UI.

Coordinate frames used throughout:
  image  : (u col, v row); u increases rightward, v increases downward
  camera : X right, Y down, Z forward (depth) — CARLA/OpenCV pinhole convention
  body   : X forward, Y right, Z up — CARLA vehicle body frame
  world  : CARLA world; X East, Y South, Z up; yaw=0 → facing +X
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO as _YOLO_CLS
    _HAS_ULTRALYTICS = True
except ImportError:
    _YOLO_CLS = None  # type: ignore
    _HAS_ULTRALYTICS = False

from followbench.perception.track_kinematics import TrackKinematicsEstimator, TrackKinematicsKF

# Resolve the central YOLO weight directory via ``followbench.weights`` so
# ultralytics always loads from ``<repo>/data/weights/yolo/`` regardless of
# the cwd the episode runner is launched from.
from followbench.weights import yolo_weight  # noqa: E402


# ── Public dataclasses ────────────────────────────────────────────────────────

@dataclass
class TrackedPed:
    """Single tracked pedestrian in world BEV frame, keyed by a *global* id."""
    track_id: int   # global integer id (stable across views after merging)
    x: float        # world frame metres
    y: float        # world frame metres
    conf: float     # YOLO detection confidence (0..1) of the chosen view
    # Frame-to-frame motion estimates (filled by the tracker, zero on first sight).
    yaw_rad: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0


@dataclass
class ViewInputs:
    """Bundle of RGB + depth + calibration for a single camera view."""
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    intrinsics: Optional[Dict]
    extrinsics: Optional[Dict]


# ── Internal dataclasses ──────────────────────────────────────────────────────

@dataclass
class _LocalDet:
    """One bounding box detected in one view, resolved to a global id."""
    view: str
    local_id: int          # BoT-SORT-assigned id, scoped to its own view
    x1: float; y1: float; x2: float; y2: float
    conf: float
    depth_m: Optional[float]   # robust depth; None if too few valid pixels
    world_x: Optional[float]   # None if depth was None
    world_y: Optional[float]
    global_id: int = -1    # filled in after resolver pass


@dataclass
class _GlobalTrack:
    """Per-global-id ledger: tells us who is 'oldest' for merge tie-breaks."""
    global_id: int
    view_origin: str       # view where this id was first born
    birth_tick: int
    last_seen_tick: int
    world_x: float
    world_y: float
    last_seen_view_local: Dict[Tuple[str, int], int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.last_seen_view_local is None:
            self.last_seen_view_local = {}


# ── Depth sampling parameters ────────────────────────────────────────────────
# Use the central PATCH_FRAC × PATCH_FRAC sub-region of each bbox to compute the
# median depth, which avoids background pixels bleeding in at the edges.
_PATCH_FRAC      = 0.5
_DEPTH_MIN_M     = 0.3
_DEPTH_MAX_M     = 15.0
_MIN_SAMPLES     = 5
# Reject the brightest depth tail (likely background through gaps) before the median.
_DEPTH_PCTL_HI   = 75.0

# ── Cross-view merge parameters ──────────────────────────────────────────────
_MERGE_DIST_M   = 0.5   # world-BEV Euclidean merge threshold
# Priority for choosing winners: smaller == preferred.
_VIEW_PRIORITY: Dict[str, int] = {"front": 0, "left": 1, "right": 2}
# Resolver entries not seen for this many ticks get evicted (bounded memory).
_RESOLVER_TTL_TICKS = 60


def _back_project_pixel_to_world(
    u: float, v: float, depth_m: float,
    intrinsics: Dict,
    cam_extr: Optional[Dict],
    robot_x: float, robot_y: float, robot_z: float, robot_yaw: float,
) -> Tuple[float, float, float]:
    """(u, v, depth_m) → world frame (wx, wy, wz).

    Four successive rigid-body transforms: image → cam → body(mount-rotated) →
    world. The camera's mount yaw (``cam_extr['yaw_deg']``) is applied between
    cam→body axis permutation and the body-space translation — without this,
    the left/right side cameras would have mirrored world coords.
    """
    fx = float(intrinsics["fx"]); fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"]); cy_i = float(intrinsics["cy"])

    # ── Step 1: image → camera frame ─────────────────────────────────────────
    # Inverse perspective projection: P_cam = K^{-1} · [u·d, v·d, d]^T
    P_cam = np.array([
        (u  - cx)  * depth_m / fx,
        (v  - cy_i) * depth_m / fy,
        depth_m,
    ])

    # ── Step 2: camera axes → robot body axes (fixed permutation) ────────────
    # Body: X forward, Y right, Z up  (CARLA vehicle convention)
    #   body X =  cam Z (depth → forward)
    #   body Y =  cam X (right → right)
    #   body Z = -cam Y (up   → -down)
    R_cam_to_body = np.array([
        [0,  0,  1],
        [1,  0,  0],
        [0, -1,  0],
    ], dtype=np.float64)

    # ── Step 3: apply the camera's mount yaw about body-Z ────────────────────
    # ``extrinsics_robot_to_sensor.yaw_deg`` is the sensor's rotation relative
    # to the robot body (see robot_runtime.py:107-139). Front camera yaw=0°
    # (identity), left=-90° (rotates body +X → -Y), right=+90° (body +X → +Y).
    cam_yaw_rad = math.radians(float(cam_extr.get("yaw_deg", 0.0))) if cam_extr else 0.0
    cc, ss = math.cos(cam_yaw_rad), math.sin(cam_yaw_rad)
    R_z_cam = np.array([
        [cc, -ss, 0.0],
        [ss,  cc, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Camera origin expressed in robot body frame (mount translation).
    Tx = float(cam_extr.get("x", 0.30)) if cam_extr else 0.30
    Ty = float(cam_extr.get("y", 0.0))  if cam_extr else 0.0
    Tz = float(cam_extr.get("z", 1.05)) if cam_extr else 1.05
    t_cam_in_body = np.array([Tx, Ty, Tz])

    P_body = R_z_cam @ R_cam_to_body @ P_cam + t_cam_in_body  # → robot body frame

    # ── Step 4: robot body frame → world frame ────────────────────────────────
    # Flat-world yaw-only rotation; yaw=0 → body X aligns with world +X (East).
    cy_, sy_ = math.cos(robot_yaw), math.sin(robot_yaw)
    R_body_to_world = np.array([
        [cy_, -sy_, 0.0],
        [sy_,  cy_, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)
    t_robot = np.array([robot_x, robot_y, robot_z])

    P_world = R_body_to_world @ P_body + t_robot   # → world frame
    return float(P_world[0]), float(P_world[1]), float(P_world[2])


def _robust_depth_in_bbox(
    depth: np.ndarray,           # (H, W) float32 metres
    x1: float, y1: float, x2: float, y2: float,
    max_range_m: float,
) -> Tuple[Optional[float], int]:
    """Return (median_depth, n_valid) sampled from a central patch of the bbox.

    None depth = not enough valid samples to trust the estimate.
    """
    H, W = depth.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx_b = 0.5 * (x1 + x2)
    cy_b = 0.5 * (y1 + y2)
    half_w = 0.5 * _PATCH_FRAC * bw
    half_h = 0.5 * _PATCH_FRAC * bh

    u_lo = max(0, int(round(cx_b - half_w)))
    u_hi = min(W, int(round(cx_b + half_w)))
    v_lo = max(0, int(round(cy_b - half_h)))
    v_hi = min(H, int(round(cy_b + half_h)))
    if u_hi - u_lo < 2 or v_hi - v_lo < 2:
        return None, 0

    patch = depth[v_lo:v_hi, u_lo:u_hi]
    valid = patch[(patch > _DEPTH_MIN_M) & (patch < max_range_m)]
    n = int(valid.size)
    if n < _MIN_SAMPLES:
        return None, n

    # Drop the far-tail (background showing through gaps in the silhouette),
    # then take the median for robustness against the remaining outliers.
    cutoff = float(np.percentile(valid, _DEPTH_PCTL_HI))
    kept = valid[valid <= cutoff]
    if kept.size < _MIN_SAMPLES:
        kept = valid
    return float(np.median(kept)), n


class PedTrackerDepth:
    """Multi-view RGB+depth pedestrian tracker.

    Each view gets its own YOLO instance (BoT-SORT state is model-scoped in
    ultralytics, so a single YOLO across views would corrupt IDs).

    Parameters
    ----------
    kinematics_mode : ``"ema"`` | ``"kf"``
        Selects the per-track velocity estimator:
        - ``"ema"`` (default) — finite-difference + exponential moving average.
          Fast, no extra deps, but velocity inherits depth-reprojection noise.
        - ``"kf"``  — constant-velocity Kalman filter (position-only measurement,
          requires ``filterpy``). The KF's measurement noise matrix R decouples
          velocity from raw position noise, producing much smoother vx/vy
          estimates at the cost of a few ticks of latency on direction changes.
          Recommended when downstream planners use target velocity for trajectory
          prediction (e.g. ``rda_traj + --use-perception``).
    kf_pos_sigma : float
        Measurement noise std-dev (metres) for the KF estimator.  Set to the
        typical one-sigma depth reprojection error at your operating distance.
        Larger → smoother but more latent velocity.  Default 0.20 m.
    kf_vel_sigma_q : float
        Process noise std-dev on velocity (m/s).  Reflects how quickly
        pedestrian velocity can change.  Default 0.80 m/s.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        tracker: str = "botsort.yaml",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str = "cpu",
        yolo_stride: int = 1,
        max_range_m: float = _DEPTH_MAX_M,
        dt: float = 0.1,
        kinematics_mode: str = "ema",
        kf_pos_sigma: float = 0.20,
        kf_vel_sigma_q: float = 0.80,
    ) -> None:
        if not _HAS_ULTRALYTICS:
            raise RuntimeError(
                "PedTrackerDepth requires 'ultralytics'. Install with: pip install ultralytics"
            )
        kinematics_mode = str(kinematics_mode).lower()
        if kinematics_mode not in {"ema", "kf"}:
            raise ValueError(
                f"kinematics_mode must be 'ema' or 'kf'; got {kinematics_mode!r}"
            )
        self._model_name  = model_name
        self._tracker_cfg = tracker
        self._conf        = conf
        self._iou         = iou
        self._imgsz       = imgsz
        self._device      = device
        self._yolo_stride = max(1, int(yolo_stride))
        self._max_range   = float(max_range_m)
        self._tick: int   = 0

        # One YOLO per view, lazily created when the view first appears.
        self._models: Dict[str, object] = {}

        # (view, local_id) → global_id resolver + per-global-id ledger.
        self._view_local_to_global: Dict[Tuple[str, int], int] = {}
        self._global_tracks: Dict[int, _GlobalTrack] = {}
        self._next_global_id: int = 0
        if kinematics_mode == "kf":
            self._kin: object = TrackKinematicsKF(
                dt=dt,
                pos_sigma=kf_pos_sigma,
                vel_sigma_q=kf_vel_sigma_q,
            )
        else:
            self._kin = TrackKinematicsEstimator(dt=dt)

        # Per-view cached raw detections (pre-resolve), for non-YOLO ticks.
        # Each entry: List[(local_id, x1,y1,x2,y2, conf)]
        self._last_yolo_raw_by_view: Dict[
            str, List[Tuple[int, float, float, float, float, float]]
        ] = {}
        # Age in ticks since each view's last YOLO run.
        self._yolo_age_by_view: Dict[str, int] = {}

        # Public outputs.  Each value is a list of
        #   (global_id, x1, y1, x2, y2, conf)
        # with merged global ids (so one pedestrian seen in multiple views has
        # the same global id across all lists).
        self.last_yolo_dets_by_view: Dict[
            str, List[Tuple[int, float, float, float, float, float]]
        ] = {}
        # Last measured median depth per global id (debug only).
        self.last_depth_per_global_id: Dict[int, float] = {}
        # Ticks since each view ran YOLO (0 on YOLO tick).
        self.last_yolo_age_by_view: Dict[str, int] = {}

        self.last_timing: Dict[str, float] = {
            "yolo_front_ms": 0.0, "yolo_left_ms": 0.0, "yolo_right_ms": 0.0,
            "yolo_ms": 0.0,      # legacy alias = sum of per-view YOLO ms
            "depth_ms": 0.0, "project_ms": 0.0, "merge_ms": 0.0,
            "total_ms": 0.0, "ran_yolo": 0.0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _load_model_for_view(self, view: str):
        if view not in self._models:
            # Resolve bare filenames (e.g. "yolo11s.pt") against the central
            # data/weights/yolo/ directory so ultralytics doesn't dump auto-
            # downloaded copies into whatever cwd the episode manager ran in.
            self._models[view] = _YOLO_CLS(yolo_weight(self._model_name))
        return self._models[view]

    def _run_yolo_on_view(
        self, view: str, rgb: np.ndarray,
    ) -> List[Tuple[int, float, float, float, float, float]]:
        """Run YOLO+tracker on one view; return list of (local_id, x1..x4, conf)."""
        m = self._load_model_for_view(view)
        dets: List[Tuple[int, float, float, float, float, float]] = []
        try:
            res = m.track(                                  # type: ignore[union-attr]
                source=rgb, persist=True,
                conf=self._conf, iou=self._iou,
                classes=[0], verbose=False,
                imgsz=self._imgsz,
                tracker=self._tracker_cfg,
                device=self._device,
            )
            if res and res[0].boxes is not None:
                boxes_obj = res[0].boxes
                if getattr(boxes_obj, "id", None) is not None:
                    xyxy  = boxes_obj.xyxy.cpu().numpy()
                    ids   = boxes_obj.id.int().cpu().numpy()
                    confs = boxes_obj.conf.cpu().numpy()
                    for tid, box, cf in zip(ids, xyxy, confs):
                        dets.append((
                            int(tid),
                            float(box[0]), float(box[1]),
                            float(box[2]), float(box[3]),
                            float(cf),
                        ))
        except RuntimeError as e:
            # CUDA kernel mismatch / OOM → skip this view this tick.
            print(f"[PedTrackerDepth] YOLO inference error on view={view} (skipping): {e}",
                  flush=True)
        return dets

    def reset(self) -> None:
        for m in self._models.values():
            try:
                m.predictor = None  # type: ignore[attr-defined]
            except Exception:
                pass
        self._tick = 0
        self._kin.reset()
        self._view_local_to_global.clear()
        self._global_tracks.clear()
        self._next_global_id = 0
        self._last_yolo_raw_by_view.clear()
        self._yolo_age_by_view.clear()

        self.last_yolo_dets_by_view = {}
        self.last_depth_per_global_id = {}
        self.last_yolo_age_by_view = {}
        self.last_timing = {
            "yolo_front_ms": 0.0, "yolo_left_ms": 0.0, "yolo_right_ms": 0.0,
            "yolo_ms": 0.0,
            "depth_ms": 0.0, "project_ms": 0.0, "merge_ms": 0.0,
            "total_ms": 0.0, "ran_yolo": 0.0,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _alloc_global_id(self, det: _LocalDet) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        self._global_tracks[gid] = _GlobalTrack(
            global_id=gid,
            view_origin=det.view,
            birth_tick=self._tick,
            last_seen_tick=self._tick,
            world_x=float(det.world_x) if det.world_x is not None else 0.0,
            world_y=float(det.world_y) if det.world_y is not None else 0.0,
        )
        return gid

    def _evict_stale_resolver_entries(self) -> None:
        """Drop (view, local_id) → gid mappings whose gid hasn't been seen
        recently. Bounds memory in long episodes when BoT-SORT churns ids."""
        stale_keys = []
        for (view, loc_id), gid in self._view_local_to_global.items():
            gt = self._global_tracks.get(gid)
            if gt is None:
                stale_keys.append((view, loc_id))
            elif self._tick - gt.last_seen_tick >= _RESOLVER_TTL_TICKS:
                stale_keys.append((view, loc_id))
        for k in stale_keys:
            self._view_local_to_global.pop(k, None)
        # Also drop _global_tracks entries with no live resolver references.
        live_gids = set(self._view_local_to_global.values())
        for gid in list(self._global_tracks.keys()):
            if gid not in live_gids:
                self._global_tracks.pop(gid, None)

    # ── Per-tick update ───────────────────────────────────────────────────────

    def update(
        self,
        views: Optional[Dict[str, ViewInputs]] = None,
        *,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_z: float = 0.0,
        robot_yaw: float = 0.0,
        # Legacy single-view shim (treated as "front" when views=None):
        rgb_image: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        rgb_intrinsics: Optional[Dict] = None,
        rgb_extrinsics: Optional[Dict] = None,
    ) -> List[TrackedPed]:
        # ── Normalize inputs to a views dict ─────────────────────────────────
        if views is None:
            views = {}
            if (rgb_image is not None and depth_image is not None
                    and rgb_intrinsics is not None):
                views["front"] = ViewInputs(
                    rgb=rgb_image, depth=depth_image,
                    intrinsics=rgb_intrinsics, extrinsics=rgb_extrinsics,
                )
        # Filter views that actually have usable RGB+depth+intrinsics.
        usable: Dict[str, ViewInputs] = {}
        for vname in ("front", "left", "right"):
            vi = views.get(vname)
            if vi is None:
                continue
            if vi.rgb is None or vi.depth is None or vi.intrinsics is None:
                continue
            usable[vname] = vi

        if not usable:
            self.last_timing = {
                "yolo_front_ms": 0.0, "yolo_left_ms": 0.0, "yolo_right_ms": 0.0,
                "yolo_ms": 0.0,
                "depth_ms": 0.0, "project_ms": 0.0, "merge_ms": 0.0,
                "total_ms": 0.0, "ran_yolo": 0.0,
            }
            self.last_yolo_dets_by_view = {}
            return []

        t_total_start = time.perf_counter()
        self._tick += 1
        run_yolo_tick = (self._tick % self._yolo_stride == 0)

        # ── Phase 1: per-view YOLO + depth + back-projection ─────────────────
        yolo_ms_by_view: Dict[str, float] = {"front": 0.0, "left": 0.0, "right": 0.0}
        depth_ms_total = 0.0
        project_ms_total = 0.0
        all_dets: List[_LocalDet] = []
        any_view_ran_yolo = False

        for vname, vi in usable.items():
            # YOLO (or reuse previous raw dets).
            t_yolo_start = time.perf_counter()
            if run_yolo_tick:
                raw = self._run_yolo_on_view(vname, vi.rgb)
                self._last_yolo_raw_by_view[vname] = raw
                self._yolo_age_by_view[vname] = 0
                any_view_ran_yolo = True
            else:
                raw = list(self._last_yolo_raw_by_view.get(vname, []))
                self._yolo_age_by_view[vname] = (
                    self._yolo_age_by_view.get(vname, 0) + 1
                )
            yolo_ms_by_view[vname] = (time.perf_counter() - t_yolo_start) * 1000.0

            # Depth sampling per bbox.
            t_depth_start = time.perf_counter()
            per_det_depth: List[Optional[float]] = []
            for (_lid, x1, y1, x2, y2, _cf) in raw:
                d_med, _n = _robust_depth_in_bbox(
                    vi.depth, x1, y1, x2, y2, self._max_range,
                )
                per_det_depth.append(d_med)
            depth_ms_total += (time.perf_counter() - t_depth_start) * 1000.0

            # Back-projection.
            t_proj_start = time.perf_counter()
            for (lid, x1, y1, x2, y2, cf), d_med in zip(raw, per_det_depth):
                if d_med is None:
                    wx = wy = None
                else:
                    u_c = 0.5 * (x1 + x2)
                    v_c = 0.5 * (y1 + y2)
                    wx, wy, _ = _back_project_pixel_to_world(
                        u_c, v_c, d_med,
                        vi.intrinsics, vi.extrinsics,
                        robot_x, robot_y, robot_z, robot_yaw,
                    )
                all_dets.append(_LocalDet(
                    view=vname, local_id=int(lid),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    conf=float(cf),
                    depth_m=d_med, world_x=wx, world_y=wy,
                ))
            project_ms_total += (time.perf_counter() - t_proj_start) * 1000.0

        # ── Phase 2: resolve-or-allocate global ids ──────────────────────────
        for d in all_dets:
            key = (d.view, d.local_id)
            gid = self._view_local_to_global.get(key)
            if gid is None or gid not in self._global_tracks:
                gid = self._alloc_global_id(d)
                self._view_local_to_global[key] = gid
            d.global_id = gid

        # ── Phase 3: union-find cross-view merge ─────────────────────────────
        t_merge_start = time.perf_counter()
        parent: Dict[int, int] = {d.global_id: d.global_id for d in all_dets}

        def _find(g: int) -> int:
            # Path-compressing.
            while parent[g] != g:
                parent[g] = parent[parent[g]]
                g = parent[g]
            return g

        def _union(ga: int, gb: int) -> None:
            ra, rb = _find(ga), _find(gb)
            if ra == rb:
                return
            ta = self._global_tracks.get(ra)
            tb = self._global_tracks.get(rb)
            if ta is None and tb is None:
                return
            if ta is None:
                parent[ra] = rb; return
            if tb is None:
                parent[rb] = ra; return
            # Winner = oldest birth_tick; tie by view priority; tie by smaller gid.
            key_a = (ta.birth_tick, _VIEW_PRIORITY.get(ta.view_origin, 99), ra)
            key_b = (tb.birth_tick, _VIEW_PRIORITY.get(tb.view_origin, 99), rb)
            if key_a <= key_b:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i, a in enumerate(all_dets):
            if a.world_x is None:
                continue
            for b in all_dets[i + 1:]:
                if b.world_x is None or b.view == a.view:
                    continue
                if (math.hypot(a.world_x - b.world_x, a.world_y - b.world_y)
                        < _MERGE_DIST_M):
                    _union(a.global_id, b.global_id)

        # Apply merge: rewrite resolver, drop absorbed globals.
        for d in all_dets:
            winner = _find(d.global_id)
            if winner != d.global_id:
                loser = d.global_id
                self._view_local_to_global[(d.view, d.local_id)] = winner
                self._global_tracks.pop(loser, None)
                d.global_id = winner
        merge_ms = (time.perf_counter() - t_merge_start) * 1000.0

        # ── Phase 4: build one TrackedPed per merged global id ───────────────
        # Group by global id; pick world xy by view priority (need depth).
        grouped: Dict[int, List[_LocalDet]] = {}
        for d in all_dets:
            grouped.setdefault(d.global_id, []).append(d)

        result: List[TrackedPed] = []
        self.last_depth_per_global_id = {}
        for gid, group in grouped.items():
            # Prefer a det from the highest-priority view that has world_xy.
            group_with_world = [d for d in group if d.world_x is not None]
            if not group_with_world:
                continue
            chosen = min(
                group_with_world,
                key=lambda d: _VIEW_PRIORITY.get(d.view, 99),
            )
            vx, vy, yaw, sp = self._kin.update(
                gid, float(chosen.world_x), float(chosen.world_y),
            )
            result.append(TrackedPed(
                track_id=gid,
                x=float(chosen.world_x), y=float(chosen.world_y),
                conf=float(chosen.conf),
                yaw_rad=float(yaw), vx=float(vx), vy=float(vy),
                speed=float(sp),
            ))
            gt = self._global_tracks.get(gid)
            if gt is not None:
                gt.last_seen_tick = self._tick
                gt.world_x = float(chosen.world_x)
                gt.world_y = float(chosen.world_y)
            if chosen.depth_m is not None:
                self.last_depth_per_global_id[gid] = float(chosen.depth_m)

        # ── Phase 5: housekeeping ────────────────────────────────────────────
        self._kin.prune({tp.track_id for tp in result})
        self._evict_stale_resolver_entries()

        # ── Phase 6: per-view exports (merged global ids) ────────────────────
        self.last_yolo_dets_by_view = {
            v: [(d.global_id, d.x1, d.y1, d.x2, d.y2, d.conf)
                for d in all_dets if d.view == v]
            for v in usable.keys()
        }
        self.last_yolo_age_by_view = dict(self._yolo_age_by_view)

        yolo_ms_total = sum(yolo_ms_by_view.values())
        self.last_timing = {
            "yolo_front_ms": yolo_ms_by_view.get("front", 0.0),
            "yolo_left_ms":  yolo_ms_by_view.get("left",  0.0),
            "yolo_right_ms": yolo_ms_by_view.get("right", 0.0),
            "yolo_ms":       yolo_ms_total,
            "depth_ms":      depth_ms_total,
            "project_ms":    project_ms_total,
            "merge_ms":      merge_ms,
            "total_ms":      (time.perf_counter() - t_total_start) * 1000.0,
            "ran_yolo":      1.0 if any_view_ran_yolo else 0.0,
        }
        return result
