"""Trajectory history storage and predictor-input formatting helpers."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from baselines.common.types import AgentState2D, RobotState2D


class TrajectoryHistory:
    def __init__(self, history_length: int) -> None:
        self.history_length = int(history_length)
        self.robot = deque(maxlen=self.history_length)
        self.target = deque(maxlen=self.history_length)
        self.npcs: Dict[str, deque] = {}
        self.agent_order: List[str] = []

    def reset(self) -> None:
        self.robot.clear()
        self.target.clear()
        self.npcs.clear()
        self.agent_order = []

    def update(
        self,
        robot: RobotState2D,
        target: Optional[AgentState2D],
        npcs: List[AgentState2D],
        target_visible: bool,
    ) -> None:
        self.robot.append(_robot_record(robot))
        if target is not None and (target_visible or len(self.target) == 0):
            self.target.append(_agent_record(target))

        for agent in npcs:
            if agent.is_target:
                continue
            if agent.track_id not in self.npcs:
                self.npcs[agent.track_id] = deque(maxlen=self.history_length)
            self.npcs[agent.track_id].append(_agent_record(agent))

        self.agent_order = sorted([aid for aid in self.npcs.keys() if len(self.npcs[aid]) > 0])

    def ready(self) -> bool:
        return len(self.robot) > 0 and len(self.target) > 0

    def last_target_xy(self) -> Optional[np.ndarray]:
        if not self.target:
            return None
        item = self.target[-1]
        return np.array([item[0], item[1]], dtype=float)

    def as_predictor_array(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("trajectory history is not ready")

        agent_ids = self.agent_order
        arr = np.zeros((self.history_length, len(agent_ids) + 2, 4), dtype=float)
        arr[:, 0, :] = _padded(self.robot, self.history_length)
        arr[:, 1, :] = _padded(self.target, self.history_length)
        for idx, agent_id in enumerate(agent_ids):
            arr[:, idx + 2, :] = _padded(self.npcs[agent_id], self.history_length)
        return arr


def _robot_record(robot: RobotState2D) -> tuple:
    return (float(robot.x), float(robot.y), float(robot.vx), float(robot.vy))


def _agent_record(agent: AgentState2D) -> tuple:
    return (float(agent.x), float(agent.y), float(agent.vx), float(agent.vy))


def _padded(items: deque, length: int) -> np.ndarray:
    if not items:
        return np.zeros((length, 4), dtype=float)
    values = list(items)
    if len(values) < length:
        values = [values[0]] * (length - len(values)) + values
    return np.asarray(values[-length:], dtype=float)
