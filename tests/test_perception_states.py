"""Deterministic mock test for the perception state machine.

Drives Initial → InitialTraining → Tracking → (Reid → Tracking) using a real
``TargetReID`` instance and a synthetic frame with two coloured rectangles
(same approach as ``tests/test_target_reid.py``). No CARLA / no YOLO required.

Run:
    python tests/test_perception_states.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from followbench.perception.states import (  # noqa: E402
    InitialState, InitialTrainingState, ReidState, StateContext, TrackingState,
)
from followbench.perception.states.base import StateConfig  # noqa: E402
from followbench.perception.target_reid import TargetReID  # noqa: E402


def _frame_two_people() -> np.ndarray:
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[100:500, 200:300] = (180, 30, 30)   # red
    img[100:500, 400:500] = (30, 30, 180)   # blue
    return img


def _frame_only_blue() -> np.ndarray:
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[100:500, 400:500] = (30, 30, 180)
    return img


def _bb(tid, x1, y1, x2, y2):
    return (int(tid), float(x1), float(y1), float(x2), float(y2), 0.9)


def _ctx(reid, frame, bboxes, tracks_world, gt=None):
    feats = reid.extract(frame, bboxes)
    return StateContext(
        features=feats,
        bboxes=bboxes,
        tracks_world=tracks_world,
        robot_x=0.0, robot_y=0.0, robot_yaw=0.0,
        gt_target_xy=gt,
    )


def main() -> int:
    np.random.seed(0)
    reid = TargetReID(device="cpu", consecutive_required=3,
                      ridge_lock_threshold=0.30, cosine_lock_threshold=0.55)
    cfg = StateConfig(initial_training_num_samples=8, id_switch_thresh=0.05)
    state = InitialState(cfg)

    # Step 1 — InitialState picks the closest in-front track via GT hint.
    frame = _frame_two_people()
    bb = [_bb(1, 200, 100, 300, 500), _bb(2, 400, 100, 500, 500)]
    tracks_world = {1: (3.0, 0.0), 2: (3.0, 1.5)}
    state = state.update(reid, _ctx(reid, frame, bb, tracks_world,
                                     gt=(3.0, 0.0)))
    assert isinstance(state, InitialTrainingState), \
        f"Expected InitialTrainingState, got {type(state).__name__}"
    assert state.target() == 1
    print(f"[1] {state.state_name()}  target={state.target()}")

    # Step 2 — Run InitialTraining until it transitions to Tracking.
    for i in range(20):
        state = state.update(reid, _ctx(reid, frame, bb, tracks_world))
        if isinstance(state, TrackingState):
            print(f"[2] transitioned to {state.state_name()} after {i+1} updates")
            break
    assert isinstance(state, TrackingState), \
        f"Expected TrackingState, got {type(state).__name__}"
    assert state.target() == 1

    # Step 3 — Drop ids 1/2 → 5/6, TrackingState should switch to Reid.
    bb_new = [_bb(5, 200, 100, 300, 500), _bb(6, 400, 100, 500, 500)]
    tracks_world = {5: (3.0, 0.0), 6: (3.0, 1.5)}
    state = state.update(reid, _ctx(reid, frame, bb_new, tracks_world))
    assert isinstance(state, ReidState), \
        f"Expected ReidState after id drop, got {type(state).__name__}"
    print(f"[3] target lost → {state.state_name()}")

    # Step 4 — ReidState should re-lock onto id 5 after a few frames.
    for i in range(8):
        state = state.update(reid, _ctx(reid, frame, bb_new, tracks_world))
        if isinstance(state, TrackingState):
            print(f"[4] re-locked onto target={state.target()} after {i+1} updates")
            break
    assert isinstance(state, TrackingState), \
        f"Expected TrackingState recovery, got {type(state).__name__}"
    assert state.target() == 5

    # Step 5 — Distractor only (blue) should NOT trigger a re-lock.
    bb_dist = [_bb(7, 400, 100, 500, 500)]
    tracks_world = {7: (3.0, 1.5)}
    # Force the lock-loss first by removing id 5.
    state = state.update(reid, _ctx(reid, _frame_only_blue(), bb_dist,
                                     tracks_world))
    assert isinstance(state, ReidState)
    for _ in range(15):
        nxt = state.update(reid, _ctx(reid, _frame_only_blue(), bb_dist,
                                       tracks_world))
        assert isinstance(nxt, ReidState), \
            f"Distractor must not trigger re-lock; got {type(nxt).__name__}"
        state = nxt
    print("[5] distractor frames correctly stayed in reid state")

    print("\nPASS: state machine Initial → Training → Tracking ⇄ Reid works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
