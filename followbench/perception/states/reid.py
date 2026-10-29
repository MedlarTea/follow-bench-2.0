"""Re-identification state — search for the lost target."""
from __future__ import annotations

from .base import State, StateContext


class ReidState(State):
    def state_name(self) -> str:
        return "reid"

    def update(self, reid, ctx: StateContext) -> State:
        from .tracking import TrackingState

        if not ctx.features:
            return self

        match = reid.find_target(ctx.features)
        if match is None:
            return self
        return TrackingState(match.track_id, self.config)
