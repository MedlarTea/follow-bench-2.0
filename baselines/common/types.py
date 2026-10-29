from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class RobotState2D:
    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class AgentState2D:
    track_id: str
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    speed: float
    radius: float = 0.35
    is_target: bool = False

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity_xy(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)


@dataclass
class PredictionBundle:
    target_traj: Optional[np.ndarray] = None
    npc_trajs_by_id: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class SearchResult:
    goal_pose: np.ndarray
    mode: str
    detail_mode: Optional[str] = None
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=float))
    occluder_id: Optional[str] = None
    cost: float = float("inf")
    anchor_xy: Optional[np.ndarray] = None
    hidden_steps: int = 0
