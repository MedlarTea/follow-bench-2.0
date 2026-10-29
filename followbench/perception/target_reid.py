"""TargetReID — appearance-based target person recovery for follower planners.

Pipeline per perception tick:
  1. Crop YOLO bboxes from the RGB image, resize to 64x128.
  2. Run the deep ReID extractor → 512-d L2-normalised embedding per track.
  3. While the planner still has a confident lock on the target track ID:
       update(track_features, target_id) → enroll target as positive,
       all other detected tracks as negative, refit Ridge classifier.
  4. When the planner reports the target tracker_id is gone:
       find_target(track_features) → return the track_id whose features
       match the gallery best (Ridge confidence, falls back to cosine
       similarity to the mean positive sample if the classifier is
       under-trained). Requires N consecutive positive frames before
       returning a re-lock candidate.

Replaces the previous "fall back to GT closest-by-distance" strategy.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from followbench.perception.reid_classifier import RidgeReIDClassifier
from followbench.perception.reid_model import PersonReIDExtractor


# Bbox tuple convention used by ped_tracker* (track_id, x1, y1, x2, y2, conf).
BBoxRecord = Tuple[int, float, float, float, float, float]


@dataclass
class ReIDMatch:
    track_id: int
    confidence: float
    cosine: float


class TargetReID:
    """Stateful target re-identification helper for the depth/lidar-TPT planners."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        # Backbone selection. 'basic' = ResNet (lightweight, ~5MB),
        # 'kpr' = SOLIDER-Swin part-based (heavy, ~370MB, more robust to occlusion).
        mode: str = "basic",
        kpr_config: str = "kpr_occ_duke_test",
        # Geometry / sampling — only honoured by the basic extractor; KPR
        # always uses the patch size declared in its YAML.
        patch_w: int = 64,
        patch_h: int = 128,
        # Online classifier
        ridge_alpha: float = 1.0,
        max_pos_samples: int = 64,
        max_neg_samples: int = 128,
        min_train_pairs: int = 4,
        # Recovery thresholds
        ridge_lock_threshold: float = 0.35,
        cosine_lock_threshold: float = 0.55,
        consecutive_required: int = 3,
        # Stop training when the locked detection drifts off-screen.
        min_bbox_area_px: int = 400,
    ) -> None:
        self._mode = str(mode).lower()
        if self._mode == "kpr":
            from followbench.perception.reid_kpr import KPRReIDExtractor
            self._extractor = KPRReIDExtractor(
                config_name=kpr_config,
                device=device,
            )
            self._patch_w, self._patch_h = self._extractor.patch_size
        elif self._mode == "basic":
            self._extractor = PersonReIDExtractor(
                model_path=model_path,
                device=device,
                patch_size=(patch_w, patch_h),
            )
            self._patch_w = int(patch_w)
            self._patch_h = int(patch_h)
        else:
            raise ValueError(
                f"TargetReID mode must be 'basic' or 'kpr', got {mode!r}"
            )
        self._classifier = RidgeReIDClassifier(
            alpha=ridge_alpha,
            max_pos_samples=max_pos_samples,
            max_neg_samples=max_neg_samples,
            min_train_pairs=min_train_pairs,
        )
        self._ridge_thr = float(ridge_lock_threshold)
        self._cos_thr = float(cosine_lock_threshold)
        self._consecutive_required = max(1, int(consecutive_required))
        self._min_area = int(min_bbox_area_px)

        # Per-track positive-vote streaks while running ReID recovery.
        self._consec: Dict[int, int] = {}
        # Most recent extracted features per track id.
        self._last_features: Dict[int, np.ndarray] = {}
        self.last_timing: Dict[str, float] = {
            "crop_ms": 0.0, "infer_ms": 0.0, "fit_ms": 0.0,
            "n_dets": 0.0, "n_pos": 0.0, "n_neg": 0.0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._classifier.reset()
        self._consec.clear()
        self._last_features.clear()
        self.last_timing = {
            "crop_ms": 0.0, "infer_ms": 0.0, "fit_ms": 0.0,
            "n_dets": 0.0, "n_pos": 0.0, "n_neg": 0.0,
        }

    @property
    def num_positive(self) -> int:
        return self._classifier.num_positive

    @property
    def num_negative(self) -> int:
        return self._classifier.num_negative

    @property
    def trained(self) -> bool:
        return self._classifier.trained

    # ── Feature extraction ────────────────────────────────────────────────────

    def extract(
        self,
        rgb_image: Optional[np.ndarray],
        bboxes: Sequence[BBoxRecord],
    ) -> Dict[int, np.ndarray]:
        """Run the CNN on each bbox in ``bboxes``. Returns {track_id: feature}."""
        feats: Dict[int, np.ndarray] = {}
        if rgb_image is None or not bboxes:
            self.last_timing.update({"crop_ms": 0.0, "infer_ms": 0.0, "n_dets": 0.0})
            return feats

        H, W = rgb_image.shape[:2]
        t0 = time.perf_counter()
        crops: List[np.ndarray] = []
        ids: List[int] = []
        for tid, x1, y1, x2, y2, _cf in bboxes:
            if int(tid) < 0:
                continue
            ix1 = max(0, int(round(x1))); iy1 = max(0, int(round(y1)))
            ix2 = min(W, int(round(x2))); iy2 = min(H, int(round(y2)))
            if ix2 - ix1 < 4 or iy2 - iy1 < 8:
                continue
            patch = rgb_image[iy1:iy2, ix1:ix2]
            if patch.size == 0:
                continue
            patch_r = cv2.resize(patch, (self._patch_w, self._patch_h),
                                  interpolation=cv2.INTER_LINEAR)
            crops.append(patch_r)
            ids.append(int(tid))
        t_crop = (time.perf_counter() - t0) * 1000.0

        if not crops:
            self.last_timing.update({"crop_ms": t_crop, "infer_ms": 0.0,
                                      "n_dets": 0.0})
            return feats

        t1 = time.perf_counter()
        embeds = self._extractor(crops)  # (N, 512)
        t_infer = (time.perf_counter() - t1) * 1000.0
        for i, tid in enumerate(ids):
            feats[tid] = embeds[i]
        self._last_features = dict(feats)

        self.last_timing.update({
            "crop_ms": t_crop, "infer_ms": t_infer, "n_dets": float(len(ids)),
        })
        return feats

    # ── Online enrollment / training ──────────────────────────────────────────

    def update(
        self,
        features: Dict[int, np.ndarray],
        target_id: int,
        bboxes: Optional[Sequence[BBoxRecord]] = None,
    ) -> bool:
        """Update positive/negative banks then refit Ridge.

        Returns True if a fresh classifier exists after the update.
        """
        if target_id < 0 or target_id not in features:
            return self._classifier.maybe_refit()

        # Skip training samples that are too small to be reliable.
        if bboxes is not None and self._min_area > 0:
            for tid, x1, y1, x2, y2, _cf in bboxes:
                if int(tid) == int(target_id):
                    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    if area < self._min_area:
                        return self._classifier.maybe_refit()
                    break

        self._classifier.add_positive([features[target_id]])
        for tid, feat in features.items():
            if int(tid) == int(target_id):
                continue
            self._classifier.add_negative([feat])

        t0 = time.perf_counter()
        ok = self._classifier.maybe_refit()
        self.last_timing["fit_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_timing["n_pos"] = float(self._classifier.num_positive)
        self.last_timing["n_neg"] = float(self._classifier.num_negative)
        return ok

    # ── Confidence query ──────────────────────────────────────────────────────

    def predict_one(self, feature: np.ndarray) -> float:
        """Ridge confidence for one feature; cosine-to-mean if untrained."""
        if self._classifier.trained:
            return float(self._classifier.predict(feature))
        return float(self._classifier.cosine_to_positive_mean(feature))

    def predict_for_tracks(
        self, features: Dict[int, np.ndarray]
    ) -> Dict[int, float]:
        return {int(tid): self.predict_one(f) for tid, f in features.items()}

    # ── Recovery ──────────────────────────────────────────────────────────────

    def find_target(
        self,
        features: Dict[int, np.ndarray],
        excluded_ids: Optional[Sequence[int]] = None,
    ) -> Optional[ReIDMatch]:
        """Return the best matching track id, or None until a streak builds up."""
        if not features:
            self._consec.clear()
            return None

        excluded = set(int(t) for t in (excluded_ids or []))

        # Score every candidate. Use Ridge if trained; otherwise cosine similarity.
        scores: Dict[int, Tuple[float, float]] = {}  # tid -> (ridge_score, cosine)
        for tid, feat in features.items():
            if int(tid) < 0 or int(tid) in excluded:
                continue
            cos = self._classifier.cosine_to_positive_mean(feat)
            ridge = self._classifier.predict(feat) if self._classifier.trained else 0.0
            scores[int(tid)] = (ridge, cos)

        if not scores:
            self._consec.clear()
            return None

        # Pick the top-scoring candidate.
        if self._classifier.trained:
            best_tid, (best_ridge, best_cos) = max(
                scores.items(), key=lambda kv: kv[1][0]
            )
        else:
            best_tid, (best_ridge, best_cos) = max(
                scores.items(), key=lambda kv: kv[1][1]
            )

        passed = (
            (self._classifier.trained and best_ridge >= self._ridge_thr)
            or best_cos >= self._cos_thr
        )

        if passed:
            self._consec[best_tid] = self._consec.get(best_tid, 0) + 1
            # Decay all other streaks so a flickering match doesn't accidentally lock.
            for tid in list(self._consec.keys()):
                if tid != best_tid:
                    self._consec[tid] = max(0, self._consec[tid] - 1)
                    if self._consec[tid] == 0:
                        self._consec.pop(tid, None)
            if self._consec[best_tid] >= self._consecutive_required:
                # Reset all streaks; caller has now adopted this id.
                self._consec.clear()
                return ReIDMatch(track_id=int(best_tid),
                                 confidence=float(best_ridge),
                                 cosine=float(best_cos))
        else:
            self._consec.clear()
        return None

    # ── Debug introspection ───────────────────────────────────────────────────

    def get_debug_snapshot(self) -> dict:
        return {
            "trained": self.trained,
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
            "consecutive": dict(self._consec),
            "timing": dict(self.last_timing),
        }
