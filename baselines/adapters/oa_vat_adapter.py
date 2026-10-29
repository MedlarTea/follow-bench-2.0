"""End-to-end OA-VAT follower adapter.

Vision-only baseline: YOLOe (text-prompted) for first-frame init and re-acq
candidates → ORTrack (deit_tiny) for short-term tracking → DINOv3 (vitb16) for
appearance matching → confaware Kalman filter for occlusion gap-fill → P
controller on bbox center/area for (v, w).

First frame uses the GT target world position to pick the right person from
YOLOe candidates. Every subsequent tick is purely vision-based.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

_THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
_OAVAT_DIR = os.path.join(_THIRD_PARTY_DIR, "oa-vat")
if _OAVAT_DIR not in sys.path:
    sys.path.insert(0, _OAVAT_DIR)

from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter

try:
    import torch
    from constants import (
        DATA_ROOT,
        DEFAULT_DINOV3_VITB16,
        DEFAULT_DINOV3_VITS16,
        MOBILECLIP_PATH,
    )
    # Pin ULTRALYTICS_CACHE_DIR so YOLOe's text encoder (mobileclip_blt.ts) is
    # loaded from / downloaded into data/oa-vat/ rather than the process cwd.
    import os as _os_env
    if _os_env.path.exists(MOBILECLIP_PATH):
        _os_env.environ.setdefault("ULTRALYTICS_CACHE_DIR", DATA_ROOT)
    del _os_env
    from detectors import (
        initialize_yolo_model,
        initialize_ortrack_tracker,
        perform_yolo_detection_for_candidates,
    )
    from dinov3_feature_extractor import DINOv3FeatureExtractor
    from tracking import ReferenceFeatureStore, match_candidates_with_reference
    from tracking.workers import OnlineEnhancerWorker

    _HAS_OAVAT = True
    _OAVAT_IMPORT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_OAVAT = False
    _OAVAT_IMPORT_ERR = str(_e)


def _project_world_to_image(
    target_xyz: np.ndarray,
    robot_xyz: np.ndarray,
    robot_yaw_rad: float,
    intrinsics: dict,
    extr_robot_to_sensor: dict,
) -> Optional[Tuple[float, float]]:
    # World → robot frame (CARLA: +x forward, +y left, +z up; yaw is CW-positive).
    dx = target_xyz[0] - robot_xyz[0]
    dy = target_xyz[1] - robot_xyz[1]
    dz = target_xyz[2] - robot_xyz[2]
    cy_, sy_ = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
    rx =  dx * cy_ + dy * sy_
    ry = -dx * sy_ + dy * cy_
    rz =  dz

    # Robot → camera frame (subtract sensor mount; assume yaw=0 mount as in calibration).
    sx = float(extr_robot_to_sensor.get("x", 0.0))
    sy = float(extr_robot_to_sensor.get("y", 0.0))
    sz = float(extr_robot_to_sensor.get("z", 0.0))
    cx_off = rx - sx
    cy_off = ry - sy
    cz_off = rz - sz

    if cx_off <= 0.05:  # behind / on top of camera
        return None

    fx = float(intrinsics["fx"]); fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"]); cy = float(intrinsics["cy"])
    # CARLA pinhole: u = cx - fx * (y_cam / x_cam), v = cy - fy * (z_cam / x_cam)
    u = cx - fx * (cy_off / cx_off)
    v = cy - fy * (cz_off / cx_off)
    return float(u), float(v)


class OaVatFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        text_query: str = "person",
        kf_mode: str = "confaware",          # 'none' | 'standard' | 'confaware'
        online_enhance: bool = True,
        dinov3_model: str = "dinov3_vitb16",
        dinov3_ckpt: Optional[str] = None,
        match_thresh: float = 0.4,
        reacq_match_thresh: float = 0.5,
        yolo_conf_thresh: float = 0.1,
        ortrack_lost_thresh: float = 0.5,
        ortrack_max_lost_frames: int = 3,
        target_area_ratio: float = 0.12,
        v_max: float = 1.0,
        w_max: float = 1.0,
        yaw_kp: float = 1.0,
        v_kp: float = 1.0,
        first_frame_max_pixel_dist: float = 200.0,
        device: Optional[str] = None,
    ) -> None:
        if not _HAS_OAVAT:
            raise ImportError(
                f"OA-VAT imports failed: {_OAVAT_IMPORT_ERR}. Ensure "
                "baselines/third_party/oa-vat/ is present and "
                "ultralytics/timm/jpeg4py/ftfy are installed."
            )

        self._dt = float(dt)
        self._text_query = text_query
        self._kf_mode = kf_mode
        self._online_enhance = bool(online_enhance)
        self._match_thresh = float(match_thresh)
        self._reacq_match_thresh = float(reacq_match_thresh)
        self._yolo_conf_thresh = float(yolo_conf_thresh)
        self._target_area_ratio = float(target_area_ratio)
        self._v_max = float(v_max)
        self._w_max = float(w_max)
        self._yaw_kp = float(yaw_kp)
        self._v_kp = float(v_kp)
        self._first_frame_max_pixel_dist = float(first_frame_max_pixel_dist)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if dinov3_ckpt is None:
            dinov3_ckpt = DEFAULT_DINOV3_VITS16 if dinov3_model == "dinov3_vits16" else DEFAULT_DINOV3_VITB16

        if not os.path.isfile(dinov3_ckpt):
            raise FileNotFoundError(
                f"DINOv3 ckpt missing: {dinov3_ckpt}. rsync to {DATA_ROOT}/dinov3/?"
            )

        # YOLOe text encoder (mobileclip_blt.ts) is resolved against the cwd
        # when ultralytics calls model.get_text_pe(). chdir to DATA_ROOT so it
        # finds the pre-downloaded file instead of re-downloading to the cwd
        # every time. The directory change is restored in the finally block.
        _saved_cwd = os.getcwd()
        try:
            os.chdir(DATA_ROOT)
            print(f"[OAVAT] loading YOLOe (text={text_query})", flush=True)
            t0 = time.perf_counter()
            self._yolo = initialize_yolo_model(target_classes=[text_query])
            print(f"[OAVAT] YOLOe ready ({time.perf_counter() - t0:.1f}s)", flush=True)
        finally:
            os.chdir(_saved_cwd)

        print(f"[OAVAT] loading ORTrack", flush=True)
        t0 = time.perf_counter()
        self._tracker = initialize_ortrack_tracker(
            model_name="deit_tiny_patch16_224",
            lost_threshold=ortrack_lost_thresh,
            max_lost_frames=ortrack_max_lost_frames,
            template_size=128,
            search_size=256,
        )
        print(f"[OAVAT] ORTrack ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        print(f"[OAVAT] loading DINOv3 ({dinov3_model})", flush=True)
        t0 = time.perf_counter()
        self._extractor = DINOv3FeatureExtractor(
            model_name=dinov3_model, checkpoint_path=dinov3_ckpt, device=str(self._device),
        )
        print(f"[OAVAT] DINOv3 ready ({time.perf_counter() - t0:.1f}s)", flush=True)

        self._cfg = {
            "yolo_confidence_thresh": self._yolo_conf_thresh,
            "match_similarity_thresh": self._match_thresh,
            "reacq_match_thresh": self._reacq_match_thresh,
            "enhance_alpha": 0.2,
            "enhance_iou_thresh": 0.3,
            "enhance_crop_expand": 0.2,
            "enhance_interval": 20,
        }

        self._ref_store = ReferenceFeatureStore()
        self._enhancer: Optional[OnlineEnhancerWorker] = None
        # Constant-velocity KF state: [cx, cy, vx, vy].
        self._kf_x: Optional[np.ndarray] = None
        self._kf_P: Optional[np.ndarray] = None

        self._first_frame = True
        self._tick = 0
        self._last_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_conf: float = 0.0
        self._consecutive_lost = 0
        self._max_predict_frames = 15
        self._predict_count = 0
        self._frame_h: int = 0
        self._frame_w: int = 0
        self._last_action = (0.0, 0.0)
        self._last_render: Optional[np.ndarray] = None

    def reset(self) -> None:
        if self._enhancer is not None:
            try:
                self._enhancer.stop()
            except Exception:
                pass
        self._enhancer = None
        self._ref_store = ReferenceFeatureStore()
        self._kf_x = None
        self._kf_P = None
        self._first_frame = True
        self._tick = 0
        self._last_bbox = None
        self._last_conf = 0.0
        self._consecutive_lost = 0
        self._predict_count = 0
        self._last_action = (0.0, 0.0)
        self._last_render = None

    def get_debug_info(self) -> dict:
        info = {
            "obstacles": [],
            "traj_points": [],
            "oa_vat_bbox": list(self._last_bbox) if self._last_bbox else None,
            "oa_vat_conf": float(self._last_conf),
            "oa_vat_action": list(self._last_action),
        }
        if self._last_render is not None:
            info["rendered_front"] = self._last_render
        return info

    @torch.no_grad()
    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.rgb_image is None:
            return FollowAction(v_mps=0.0, w_radps=0.0)

        # OA-VAT modules expect BGR; obs.rgb_image is RGB per core_types contract.
        frame_rgb = obs.rgb_image
        frame_bgr = frame_rgb[..., ::-1].copy() if frame_rgb.shape[2] == 3 else frame_rgb
        H, W = frame_bgr.shape[:2]
        self._frame_h, self._frame_w = H, W

        if self._first_frame:
            bbox = self._first_frame_init(frame_bgr, obs)
            if bbox is None:
                return FollowAction(v_mps=0.0, w_radps=0.0)
            self._last_bbox = bbox
            self._last_conf = 1.0
            self._first_frame = False
            if self._online_enhance:
                self._enhancer = OnlineEnhancerWorker(
                    self._yolo, self._extractor, self._ref_store, self._cfg,
                )
                self._enhancer.start()
        else:
            bbox = self._track_frame(frame_bgr)
            self._last_bbox = bbox

        action = self._compute_control(bbox)
        self._last_action = action
        self._last_render = self._render_overlay(frame_rgb, bbox)
        self._tick += 1
        return FollowAction(v_mps=action[0], w_radps=action[1])

    def _first_frame_init(self, frame_bgr: np.ndarray, obs: FollowObservation) -> Optional[Tuple[int, int, int, int]]:
        candidates, masks = perform_yolo_detection_for_candidates(
            frame_bgr, self._yolo, confidence_threshold=self._yolo_conf_thresh,
        )
        if not candidates:
            print("[OAVAT] first frame: YOLOe found no person", flush=True)
            return None

        bbox = self._select_target_with_gt(candidates, obs)
        if bbox is None:
            print("[OAVAT] first frame: GT-projected target outside any YOLOe bbox", flush=True)
            return None

        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        feat = self._extractor.extract_features(frame_bgr, np.array([[x1, y1, x2, y2]], dtype=np.float32))
        feat_cpu = feat[0].detach().cpu()
        self._ref_store.initialize_from_list([feat_cpu])

        frame_rgb = frame_bgr[..., ::-1].copy()
        self._tracker.initialize(frame_rgb, {"init_bbox": list(bbox)})
        print(f"[OAVAT] first frame init bbox={bbox}", flush=True)
        return bbox

    def _select_target_with_gt(self, candidates, obs: FollowObservation) -> Optional[Tuple[int, int, int, int]]:
        if (obs.rgb_intrinsics is None or
                obs.rgb_extrinsics_robot_to_sensor is None or
                obs.target is None):
            return candidates[0] if candidates else None

        target_xyz = np.array([obs.target.x, obs.target.y, obs.target.z], dtype=np.float32)
        robot_xyz = np.array([obs.robot.x, obs.robot.y, obs.robot.z], dtype=np.float32)
        proj = _project_world_to_image(
            target_xyz, robot_xyz, obs.robot.yaw_rad,
            obs.rgb_intrinsics, obs.rgb_extrinsics_robot_to_sensor,
        )
        if proj is None:
            return candidates[0] if candidates else None

        u, v = proj
        best, best_d = None, float("inf")
        for x, y, w, h in candidates:
            cx, cy = x + w / 2.0, y + h / 2.0
            d = math.hypot(cx - u, cy - v)
            if d < best_d:
                best_d, best = d, (x, y, w, h)

        if best_d > self._first_frame_max_pixel_dist:
            print(f"[OAVAT] first-frame nearest YOLOe bbox is {best_d:.0f}px from GT proj — no candidate close enough", flush=True)
            return None
        return best

    def _track_frame(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        frame_rgb = frame_bgr[..., ::-1].copy()
        out = self._tracker.track(frame_rgb)
        bbox = out.get("target_bbox")
        conf = float(out.get("confidence", 0.0))
        is_lost = bool(out.get("is_lost", False))
        self._last_conf = conf

        if bbox is not None and not is_lost:
            self._consecutive_lost = 0
            self._predict_count = 0
            x, y, w, h = (int(v) for v in bbox)
            x = max(0, x); y = max(0, y)
            w = max(1, min(w, self._frame_w - x)); h = max(1, min(h, self._frame_h - y))
            bbox_t = (x, y, w, h)

            if self._enhancer is not None and self._tick % self._cfg["enhance_interval"] == 0:
                try:
                    self._enhancer.enqueue(frame_bgr, bbox_t, frame_idx=self._tick)
                except Exception:
                    pass
            if self._kf_mode != "none":
                self._update_kf(bbox_t, conf)
            return bbox_t

        # Lost path.
        self._consecutive_lost += 1
        # Try re-acquisition every few frames.
        reacq = self._reacquire(frame_bgr)
        if reacq is not None:
            x, y, w, h = reacq
            self._tracker.initialize(frame_rgb, {"init_bbox": [x, y, w, h]})
            self._consecutive_lost = 0
            self._predict_count = 0
            print(f"[OAVAT] re-acquired at bbox={reacq}", flush=True)
            return reacq

        # Predict via KF if enabled.
        if self._kf_mode != "none" and self._last_bbox is not None and self._predict_count < self._max_predict_frames:
            pred = self._predict_kf()
            if pred is not None:
                self._predict_count += 1
                return pred
        return None

    def _reacquire(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if not self._ref_store.is_valid():
            return None
        feats = self._ref_store.get_features()
        if not feats:
            return None
        candidates, masks = perform_yolo_detection_for_candidates(
            frame_bgr, self._yolo, confidence_threshold=self._yolo_conf_thresh,
        )
        if not candidates:
            return None
        try:
            best_idx, best_sim, _, best_bbox = match_candidates_with_reference(
                candidates, masks, frame_bgr, self._extractor, feats, self._cfg,
            )
        except Exception as e:
            print(f"[OAVAT] match failed: {e}", flush=True)
            return None
        if best_sim < self._reacq_match_thresh:
            return None
        return tuple(int(v) for v in best_bbox)

    # CV-KF on bbox center: state = [cx, cy, vx, vy], measurement = [cx, cy].
    _KF_F = staticmethod(lambda dt: np.array([
        [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float))
    _KF_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    _KF_Q = np.diag([1.0, 1.0, 4.0, 4.0])

    def _update_kf(self, bbox: Tuple[int, int, int, int], conf: float) -> None:
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        z = np.array([cx, cy], dtype=float)
        if self._kf_x is None:
            self._kf_x = np.array([cx, cy, 0.0, 0.0])
            self._kf_P = np.eye(4) * 100.0
            return
        F = self._KF_F(self._dt)
        self._kf_x = F @ self._kf_x
        self._kf_P = F @ self._kf_P @ F.T + self._KF_Q
        # Confidence-aware: lower conf → larger R (less trust in observation).
        r_scale = max(0.05, float(conf)) if self._kf_mode == "confaware" else 1.0
        R = np.eye(2) * (5.0 / r_scale)
        S = self._KF_H @ self._kf_P @ self._KF_H.T + R
        K = self._kf_P @ self._KF_H.T @ np.linalg.inv(S)
        y = z - self._KF_H @ self._kf_x
        self._kf_x = self._kf_x + K @ y
        self._kf_P = (np.eye(4) - K @ self._KF_H) @ self._kf_P

    def _predict_kf(self) -> Optional[Tuple[int, int, int, int]]:
        if self._kf_x is None or self._last_bbox is None:
            return None
        F = self._KF_F(self._dt)
        self._kf_x = F @ self._kf_x
        self._kf_P = F @ self._kf_P @ F.T + self._KF_Q
        cx, cy = float(self._kf_x[0]), float(self._kf_x[1])
        w, h = self._last_bbox[2], self._last_bbox[3]
        x = int(round(cx - w / 2.0)); y = int(round(cy - h / 2.0))
        x = max(0, min(self._frame_w - 1, x))
        y = max(0, min(self._frame_h - 1, y))
        w = max(1, min(w, self._frame_w - x))
        h = max(1, min(h, self._frame_h - y))
        return (x, y, w, h)

    def _compute_control(self, bbox: Optional[Tuple[int, int, int, int]]) -> Tuple[float, float]:
        if bbox is None:
            return 0.0, 0.0
        x, y, w, h = bbox
        cx = x + w / 2.0
        frame_cx = self._frame_w / 2.0
        # Yaw: positive error_x means target right of center; CARLA yaw is CW-positive
        # looking down (left-handed), so a right-of-center target needs +w_radps.
        err_x_norm = (cx - frame_cx) / max(1.0, frame_cx)  # [-1, 1]
        w_cmd = self._yaw_kp * err_x_norm * self._w_max
        w_cmd = float(max(-self._w_max, min(self._w_max, w_cmd)))

        # Forward: drive bbox area toward target_area.
        cur_area = float(w * h)
        target_area = self._target_area_ratio * (self._frame_w * self._frame_h)
        # Normalize: 1.0 means 'too small by exactly target' (max forward).
        area_err_norm = (target_area - cur_area) / max(1.0, target_area)
        v_cmd = self._v_kp * area_err_norm * self._v_max
        v_cmd = float(max(-self._v_max, min(self._v_max, v_cmd)))
        return v_cmd, w_cmd

    def _render_overlay(self, frame_rgb: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        out = frame_rgb.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"conf={self._last_conf:.2f}"
            cv2.putText(out, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        action_label = f"v={self._last_action[0]:+.2f} w={self._last_action[1]:+.2f}"
        cv2.putText(out, action_label, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out
