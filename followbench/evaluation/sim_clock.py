from __future__ import annotations

import time


class SimulationPacer:
    """Wall-clock pacing for a synchronous fixed-step CARLA loop."""

    def __init__(self, dt: float, speed: float, *, enabled: bool) -> None:
        if speed <= 0.0:
            raise ValueError("simulation speed must be greater than zero")
        self.enabled = bool(enabled)
        self.frame_wall_seconds = float(dt) / float(speed)

    @staticmethod
    def begin_frame() -> float:
        return time.perf_counter()

    def end_frame(self, frame_started_at: float) -> None:
        if not self.enabled:
            return
        remaining = self.frame_wall_seconds - (time.perf_counter() - frame_started_at)
        if remaining > 0.0:
            time.sleep(remaining)
