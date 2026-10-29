from __future__ import annotations

from dataclasses import dataclass


class SearchMode:
    FOLLOW = "follow"
    REACQUIRE_TRANSITION = "reacquire_transition"
    OVERTAKE = "overtake"
    OVERTAKE_FAILED = "overtake_failed"
    FLUID = "fluid"
    FLUID_FALLBACK = "fluid_fallback"


@dataclass
class SearchState:
    hidden_steps: int = 0
    last_mode: str = SearchMode.FOLLOW


class SearchModeManager:
    def __init__(self, reacquire_transition_ticks: int = 1) -> None:
        self.reacquire_transition_ticks = max(0, int(reacquire_transition_ticks))
        self._state = SearchState()

    @property
    def state(self) -> SearchState:
        return SearchState(
            hidden_steps=int(self._state.hidden_steps),
            last_mode=str(self._state.last_mode),
        )

    def reset(self) -> None:
        self._state = SearchState()

    def on_target_visible(self) -> None:
        self.reset()

    def on_search_result(self, base_mode: str) -> tuple[str, int]:
        self._state.hidden_steps += 1
        if self._state.hidden_steps <= self.reacquire_transition_ticks:
            resolved_mode = SearchMode.REACQUIRE_TRANSITION
        else:
            resolved_mode = str(base_mode)
        self._state.last_mode = resolved_mode
        return resolved_mode, int(self._state.hidden_steps)


__all__ = ["SearchMode", "SearchModeManager", "SearchState"]
