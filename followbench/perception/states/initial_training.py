"""Initial-training state — bootstrap the appearance classifier."""
from __future__ import annotations

from .base import State, StateContext


class InitialTrainingState(State):
    def __init__(self, target_id: int, config):
        super().__init__(config)
        self.target_id = int(target_id)
        self.num_pos_samples = 0

    def state_name(self) -> str:
        return "initial_training"

    def target(self) -> int:
        return self.target_id

    def update(self, reid, ctx: StateContext) -> State:
        from .initial import InitialState
        from .tracking import TrackingState

        if self.target_id not in ctx.features:
            return InitialState(self.config)

        if reid.update(ctx.features, target_id=self.target_id, bboxes=ctx.bboxes):
            self.num_pos_samples += 1
        else:
            # Even if the classifier was not yet refit, count the positive
            # sample once it is recorded — the bank size grows monotonically.
            if reid.num_positive > self.num_pos_samples:
                self.num_pos_samples = reid.num_positive

        if self.num_pos_samples >= self.config.initial_training_num_samples:
            return TrackingState(self.target_id, self.config)
        return self
