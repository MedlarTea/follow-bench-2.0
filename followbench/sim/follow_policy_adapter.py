from __future__ import annotations

from abc import ABC, abstractmethod

from followbench.sim.core_types import FollowAction, FollowObservation


class FollowerPolicyAdapter(ABC):
    """Algorithm plugin interface for robot follower policies."""

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def act(self, obs: FollowObservation) -> FollowAction:
        pass
