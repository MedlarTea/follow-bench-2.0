"""Smoke test: load YOLOe + ORTrack + DINOv3 in multiRobots and run a synthetic frame."""
import os
import sys
import time

import numpy as np
import torch

# Make sibling modules importable when running this script directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OAVAT_ROOT = os.path.dirname(_THIS_DIR)
if _OAVAT_ROOT not in sys.path:
    sys.path.insert(0, _OAVAT_ROOT)

# Pin mobileclip cache so YOLOe text mode finds it offline.
from constants import (
    DATA_ROOT,
    DEFAULT_DINOV3_CHECKPOINT,
    DEFAULT_ORTRACK_CHECKPOINT,
    DEFAULT_YOLO_MODEL,
    MOBILECLIP_PATH,
)

if os.path.exists(MOBILECLIP_PATH):
    os.environ.setdefault("ULTRALYTICS_CACHE_DIR", DATA_ROOT)


def main():
    print(f"DATA_ROOT = {DATA_ROOT}")
    for label, path in (
        ("YOLOe", DEFAULT_YOLO_MODEL),
        ("ORTrack", DEFAULT_ORTRACK_CHECKPOINT),
        ("DINOv3", DEFAULT_DINOV3_CHECKPOINT),
        ("mobileclip", MOBILECLIP_PATH),
    ):
        ok = os.path.exists(path)
        size_mb = os.path.getsize(path) / 1e6 if ok else 0
        print(f"  [{label:10s}] {'OK' if ok else 'MISSING':8s} {size_mb:7.1f} MB  {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Synthetic 800x600 BGR frame (matches FollowBench front camera).
    H, W = 600, 800
    rng = np.random.default_rng(0)
    frame_bgr = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
    print(f"Frame: {frame_bgr.shape} {frame_bgr.dtype}")

    print("\n[1/3] Loading YOLOe (text-prompted)...")
    t0 = time.time()
    from detectors import initialize_yolo_model
    yolo = initialize_yolo_model(target_classes=["person"])
    print(f"      ready in {time.time() - t0:.1f}s")
    t0 = time.time()
    res = yolo.detect_single(frame_bgr, confidence_threshold=0.05)
    print(f"      first detect: {res}  ({time.time() - t0:.2f}s)")

    print("\n[2/3] Loading ORTrack...")
    t0 = time.time()
    from detectors import initialize_ortrack_tracker
    tracker = initialize_ortrack_tracker(
        model_name="deit_tiny_patch16_224",
        lost_threshold=0.5,
        max_lost_frames=3,
        template_size=128,
        search_size=256,
    )
    print(f"      ready in {time.time() - t0:.1f}s")
    init_bbox = [W // 2 - 40, H // 2 - 80, 80, 160]  # x,y,w,h
    frame_rgb = frame_bgr[..., ::-1].copy()
    tracker.initialize(frame_rgb, {"init_bbox": init_bbox})
    out = tracker.track(frame_rgb)
    print(f"      track output keys: {list(out.keys())[:5]}")

    print("\n[3/3] Loading DINOv3 (vitb16)...")
    t0 = time.time()
    from dinov3_feature_extractor import DINOv3FeatureExtractor
    extractor = DINOv3FeatureExtractor(model_name="dinov3_vitb16", device=device)
    print(f"      ready in {time.time() - t0:.1f}s")
    bboxes = np.array([[init_bbox[0], init_bbox[1],
                        init_bbox[0] + init_bbox[2],
                        init_bbox[1] + init_bbox[3]]], dtype=np.float32)
    t0 = time.time()
    feats = extractor.extract_features(frame_bgr, bboxes)
    print(f"      feature shape: {tuple(feats.shape)}  ({time.time() - t0:.2f}s)")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\nPeak GPU mem: {peak_gb:.2f} GB")
    print("\n=== OA-VAT smoke test PASSED ===")


if __name__ == "__main__":
    main()
