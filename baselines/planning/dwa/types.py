from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KinematicState:
    x: float
    y: float
    yaw: float
    vx: float = 0.0
    vy: float = 0.0
    v: float = 0.0
    w: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class AgentPrediction:
    positions: np.ndarray
    velocities: np.ndarray
    yaws: np.ndarray

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 2)
        self.velocities = np.asarray(self.velocities, dtype=float).reshape(-1, 2)
        self.yaws = np.asarray(self.yaws, dtype=float).reshape(-1)
