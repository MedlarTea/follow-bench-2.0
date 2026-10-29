"""State machine base for the perception pipeline.

Mirrors the lightweight FSM in ``follow/follow_server/src/states_standalone``:
Initial → InitialTraining → Tracking → (Reid → Tracking)*. Each transition
returns a (possibly new) State; the orchestrator simply calls
``state = state.update(reid, ctx)``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BBoxRecord = Tuple[int, float, float, float, float, float]


@dataclass
class StateContext:
    # Per-tick perception inputs supplied by the orchestrator (PerceptionPipeline).
    features: Dict[int, np.ndarray] = field(default_factory=dict)
    bboxes: List[BBoxRecord] = field(default_factory=list)
    tracks_world: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_yaw: float = 0.0
    # Optional GT hint, used only to bootstrap the initial lock when the user
    # has chosen a "GT-seeded" startup (matches the original `initial GT lock`).
    gt_target_xy: Optional[Tuple[float, float]] = None


@dataclass
class StateConfig:
    initial_training_num_samples: int = 10
    id_switch_thresh: float = 0.05    # Ridge confidence below this → reid.
    initial_select_lateral_max: float = 0.8   # body-Y absolute (m).
    initial_select_max_dist_m: float = 6.0
    initial_gt_match_radius_m: float = 1.5


class State(ABC):
    def __init__(self, config: StateConfig):
        self.config = config

    @abstractmethod
    def state_name(self) -> str: ...

    def target(self) -> int:
        return -1

    @abstractmethod
    def update(self, reid, ctx: StateContext) -> "State": ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.target()})"
