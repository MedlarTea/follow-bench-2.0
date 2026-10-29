"""Deterministic mock test for TargetReID.

No CARLA / no ultralytics required. We synthesise two appearance-distinct
"persons" by drawing solid-coloured rectangles into a fake RGB frame, then:

  1. enrol the red person (``track_id=1``, "blue" person ``id=2`` is a
     negative);
  2. drop tracker ids 1/2 and re-issue them as 5/6 — TargetReID should
     re-identify the red person as id=5 within ``consecutive_required`` frames;
  3. show only the wrong person (blue) — TargetReID must NOT re-identify;
  4. show the red person under another fresh id (9) — must re-identify.

Run:  python tests/test_target_reid.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from followbench.perception.target_reid import TargetReID  # noqa: E402


def _frame_with_two_people() -> np.ndarray:
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[100:500, 200:300] = (180, 30, 30)   # red guy
    img[100:500, 400:500] = (30, 30, 180)   # blue guy
    return img


def _frame_only_blue() -> np.ndarray:
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[100:500, 400:500] = (30, 30, 180)
    return img


def _bb(tid: int, x1: int, y1: int, x2: int, y2: int) -> tuple:
    return (tid, float(x1), float(y1), float(x2), float(y2), 0.9)


def main() -> int:
    np.random.seed(0)
    reid = TargetReID(device="cpu", consecutive_required=3,
                       ridge_lock_threshold=0.30, cosine_lock_threshold=0.55)

    # 1. Enrol target as id=1 (red), id=2 (blue) is negative. 20 frames.
    frame = _frame_with_two_people()
    bb = [_bb(1, 200, 100, 300, 500), _bb(2, 400, 100, 500, 500)]
    for _ in range(20):
        reid.update(reid.extract(frame, bb), target_id=1, bboxes=bb)
    assert reid.trained, "Ridge classifier should be trained after 20 frames"
    print(f"[1] enrolled pos={reid.num_positive} neg={reid.num_negative} trained={reid.trained}")

    # 2. Tracker assigns new ids 5 & 6 to the same two people.
    bb_new = [_bb(5, 200, 100, 300, 500), _bb(6, 400, 100, 500, 500)]
    match = None
    for i in range(8):
        match = reid.find_target(reid.extract(frame, bb_new))
        if match is not None:
            print(f"[2] recovered after {i+1} frames: {match}")
            break
    assert match is not None and match.track_id == 5, \
        f"Expected re-identification as id 5 (red), got {match}"

    # 3. Distractor-only frames (only blue id=7, no red person).
    bb_dist = [_bb(7, 400, 100, 500, 500)]
    for _ in range(15):
        m = reid.find_target(reid.extract(_frame_only_blue(), bb_dist))
        assert m is None, f"Distractor should NOT match as target; got {m}"
    print("[3] distractor frames correctly rejected (no false positives)")

    # 4. Same target re-appears under id=9.
    bb_red9 = [_bb(9, 200, 100, 300, 500)]
    match = None
    for i in range(8):
        match = reid.find_target(reid.extract(frame, bb_red9))
        if match is not None:
            print(f"[4] re-locked as id 9 after {i+1} frames: {match}")
            break
    assert match is not None and match.track_id == 9, \
        f"Expected re-identification as id 9 (red), got {match}"

    print("\nPASS: TargetReID enrol → re-identify → reject distractor → re-identify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
