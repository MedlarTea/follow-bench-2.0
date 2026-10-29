from __future__ import annotations

import math
import queue
from typing import Callable, Iterable, Optional


def _impulse_norm(vec) -> float:
    return float(math.sqrt(float(vec.x) ** 2 + float(vec.y) ** 2 + float(vec.z) ** 2))


class HumanCollisionMonitor:
    """CARLA collision sensor wrapper for robot-pedestrian events."""

    def __init__(self, world, robot_actor, enabled: bool = True) -> None:
        self.world = world
        self.robot_actor = robot_actor
        self.enabled = bool(enabled)
        self.sensor = None
        self._events: "queue.Queue[dict]" = queue.Queue()
        if self.enabled and self.robot_actor is not None:
            self._spawn_sensor()

    def _spawn_sensor(self) -> None:
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, self.robot_actor.get_transform(), attach_to=self.robot_actor)

        def _callback(event):
            other = getattr(event, "other_actor", None)
            other_type = "" if other is None else str(getattr(other, "type_id", ""))
            if not other_type.startswith("walker.pedestrian"):
                return
            self._events.put(
                {
                    "source": "collision_sensor",
                    "collision_type": "human",
                    "robot_actor_id": int(getattr(event.actor, "id", -1)),
                    "other_actor_id": int(getattr(other, "id", -1)) if other is not None else -1,
                    "other_type_id": other_type,
                    "impulse_norm": _impulse_norm(event.normal_impulse),
                }
            )

        self.sensor.listen(_callback)

    def drain(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def destroy(self) -> None:
        if self.sensor is None:
            return
        try:
            self.sensor.stop()
        except Exception:
            pass
        try:
            self.sensor.destroy()
        except Exception:
            pass
        self.sensor = None


def actor_track_lookup(npc_states: Iterable) -> Callable[[int], Optional[str]]:
    lookup = {}
    for state in npc_states or []:
        actor_id = getattr(state, "actor_id", None)
        track_id = getattr(state, "track_id", None)
        if actor_id is not None and track_id is not None:
            lookup[int(actor_id)] = str(track_id)

    def _find(actor_id: int) -> Optional[str]:
        return lookup.get(int(actor_id))

    return _find

