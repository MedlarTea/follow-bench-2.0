"""Minimal ``--planner custom`` example: GT oracle perception + pure pursuit.

This only demonstrates the interface; it is NOT a baseline. The oracle reads
ground truth (``requires_privileged_obs = True``) and the planner ignores
obstacles.

Run:
    bash scripts/run_episode.sh --scene corner --planner custom \
        --policy-factory examples/custom_policy/gt_pursuit.py:build_policy \
        --policy-arg max_speed=1.2
"""
from __future__ import annotations

import math
from typing import Optional

from followbench.interfaces import (
    PerceptionModule,
    PerceptionOutput,
    PerceptionPlanningPolicy,
    Planner,
    PolicyBuildContext,
    TargetHint,
    TrackedPerson,
)
from followbench.sim.core_types import FollowAction, FollowObservation


class GtOraclePerception(PerceptionModule):
    requires_privileged_obs = True

    def reset(self) -> None:
        pass

    def perceive(self, obs: FollowObservation, hint: Optional[TargetHint]) -> PerceptionOutput:
        t = obs.target
        people = [TrackedPerson(track_id=0, x=t.x, y=t.y, vx=t.vx, vy=t.vy, is_target=True)]
        people += [TrackedPerson(track_id=i + 1, x=n.x, y=n.y, vx=n.vx, vy=n.vy) for i, n in enumerate(obs.npcs)]
        return PerceptionOutput(
            has_target=True,
            target_xy=(t.x, t.y),
            target_vel=(t.vx, t.vy),
            target_yaw_rad=math.radians(t.yaw_deg),
            target_id=0,
            state="oracle",
            people=people,
        )


class PurePursuitPlanner(Planner):
    def __init__(self, desired_distance: float, max_speed: float, max_yaw_rate: float) -> None:
        self.desired_distance = float(desired_distance)
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self._dbg: dict = {}

    def reset(self) -> None:
        self._dbg = {}

    def plan(self, obs: FollowObservation, perception: PerceptionOutput) -> FollowAction:
        if not perception.has_target or perception.target_xy is None:
            return FollowAction(0.0, 0.0)
        r = obs.robot
        dx, dy = perception.target_xy[0] - r.x, perception.target_xy[1] - r.y
        dist = math.hypot(dx, dy)
        heading_err = math.atan2(math.sin(math.atan2(dy, dx) - r.yaw_rad), math.cos(math.atan2(dy, dx) - r.yaw_rad))
        v = max(0.0, min(self.max_speed, 1.0 * (dist - self.desired_distance))) * max(0.0, math.cos(heading_err))
        w = max(-self.max_yaw_rate, min(self.max_yaw_rate, 2.0 * heading_err))
        self._dbg = {"goal_point": list(perception.target_xy)}
        return FollowAction(v_mps=v, w_radps=w)

    def get_debug_info(self) -> dict:
        return self._dbg


def build_policy(ctx: PolicyBuildContext) -> PerceptionPlanningPolicy:
    planner = PurePursuitPlanner(
        desired_distance=ctx.desired_distance,
        max_speed=float(ctx.options.get("max_speed", 1.2)),
        max_yaw_rate=float(ctx.options.get("max_yaw_rate", 1.5)),
    )
    return PerceptionPlanningPolicy(GtOraclePerception(), planner)
