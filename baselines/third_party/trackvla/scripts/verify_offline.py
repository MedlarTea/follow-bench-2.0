"""Smoke test: run TrackVLA offline inference on a short video clip.

Confirms the self-contained code+ckpt copy works under the multiRobots conda env
without depending on the original upstream TrackVLA tree.

Usage (from any cwd):
    conda run -n multiRobots python -u verify_offline.py \\
        --video /path/to/clip.mp4 \\
        --max-frames 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKVLA_DIR = os.path.dirname(THIS_DIR)
if TRACKVLA_DIR not in sys.path:
    sys.path.insert(0, TRACKVLA_DIR)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from socialRPF.baseline.offline_agent_unified import OfflineUnifiedAgent
from socialRPF.constants import DATA_ROOT, IMAGE_SIZE


def _resize_with_padding(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(side / w, side / h)
    rw, rh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = side - rh, side - rw
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=0,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--instruction", default="Follow the person")
    p.add_argument(
        "--ckpt",
        default=os.path.join(
            DATA_ROOT,
            "ckpts",
            "260417-qwen3-4b-unified-nav-qa-view-fwd-ws16-bs16-acc1-glb256-qabs16-ep5-lrb2e-5-lrh2e-4-bn10.0-bq1.0",
            "model_epoch04_step024000.pt",
        ),
    )
    p.add_argument("--max-frames", type=int, default=10)
    args = p.parse_args()

    print(f"[verify_offline] DATA_ROOT = {DATA_ROOT}")
    print(f"[verify_offline] CKPT      = {args.ckpt}")
    if not os.path.isfile(args.ckpt):
        print(f"  ckpt missing — rsync did not finish? exiting.", flush=True)
        return 2

    agent = OfflineUnifiedAgent(ckpt_path=args.ckpt)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"  cannot open video {args.video}")
        return 3

    n = 0
    while n < args.max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _resize_with_padding(rgb, IMAGE_SIZE)
        t0 = time.perf_counter()
        action, traj, _ = agent.predict_frame(rgb, args.instruction, render=False)
        dt = (time.perf_counter() - t0) * 1000.0
        traj_shape = None if traj is None else traj.shape
        print(
            f"  [frame {n:03d}] action=[{action[0]:+.4f}, {action[1]:+.4f}, {action[2]:+.4f}]"
            f"  traj={traj_shape}  infer={dt:.1f}ms",
            flush=True,
        )
        n += 1
    cap.release()
    print(f"[verify_offline] OK — ran {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
