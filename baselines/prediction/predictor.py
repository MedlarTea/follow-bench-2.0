"""Thin predictor facade that hides backend-specific predictor wiring details."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from baselines.common.types import PredictionBundle
from baselines.prediction.backends import get_predictor
from baselines.prediction.predictor_configs import PREDICTOR_CONFIGS
from baselines.prediction.trajectory_buffer import TrajectoryHistory


class TrajectoryPredictionService:
    def __init__(self, predictor_name: str, dt: float) -> None:
        if predictor_name not in PREDICTOR_CONFIGS:
            raise ValueError(f"Unsupported predictor: {predictor_name}")
        self.predictor_name = predictor_name
        self.params = deepcopy(PREDICTOR_CONFIGS[predictor_name])
        self.params["dt"] = float(dt)
        self.predictor = get_predictor(self.params["name"], self.params)
        self._last_agent_order = None

    @property
    def history_length(self) -> int:
        return int(self.params["history_length"])

    def reset(self) -> None:
        reset = getattr(self.predictor, "reset", None)
        if callable(reset):
            reset()
        self._last_agent_order = None

    def predict(self, history: TrajectoryHistory) -> PredictionBundle:
        if not history.ready():
            return PredictionBundle()

        if self._last_agent_order != history.agent_order:
            reset = getattr(self.predictor, "reset", None)
            if callable(reset):
                reset()
            self._last_agent_order = list(history.agent_order)

        raw = self.predictor.predict(history.as_predictor_array())
        pred = np.squeeze(raw, axis=(0, 1))
        if pred.ndim != 3 or pred.shape[1] == 0:
            return PredictionBundle()

        target_traj = pred[:, 0, :]
        npc_trajs = {}
        if pred.shape[1] > 1:
            for idx, agent_id in enumerate(history.agent_order):
                if idx + 1 < pred.shape[1]:
                    npc_trajs[agent_id] = pred[:, idx + 1, :]
        return PredictionBundle(target_traj=target_traj, npc_trajs_by_id=npc_trajs)
