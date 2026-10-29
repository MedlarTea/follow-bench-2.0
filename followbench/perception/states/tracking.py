"""Tracking state — actively follow the target, detect ID-switches."""
from __future__ import annotations

from .base import State, StateContext


class TrackingState(State):
    def __init__(self, target_id: int, config):
        super().__init__(config)
        self.target_id = int(target_id)

    def state_name(self) -> str:
        return "tracking"

    def target(self) -> int:
        return self.target_id

    def update(self, reid, ctx: StateContext) -> State:
        from .reid import ReidState

        if self.target_id not in ctx.features:
            return ReidState(self.config)

        target_feat = ctx.features[self.target_id]
        score = reid.predict_one(target_feat)
        if score < self.config.id_switch_thresh:
            return ReidState(self.config)

        reid.update(ctx.features, target_id=self.target_id, bboxes=ctx.bboxes)
        return self
