from __future__ import annotations

import math
from typing import List, Optional

from baselines.common.types import AgentState2D, RobotState2D


def robot_state_from_obs(obs) -> RobotState2D:
    yaw = float(obs.robot.yaw_rad)
    speed = float(obs.robot.speed)
    return RobotState2D(
        x=float(obs.robot.x),
        y=float(obs.robot.y),
        yaw=yaw,
        vx=speed * math.cos(yaw),
        vy=speed * math.sin(yaw),
        speed=speed,
    )


def target_state_from_obs(obs, radius: float) -> Optional[AgentState2D]:
    target = getattr(obs, "target", None)
    if target is None:
        return None
    yaw = target_yaw_from_obs(obs)
    speed = float(target.speed)
    return AgentState2D(
        track_id=str(target.track_id),
        x=float(target.x),
        y=float(target.y),
        vx=float(target.vx),
        vy=float(target.vy),
        yaw=yaw,
        speed=speed,
        radius=float(radius),
        is_target=True,
    )


def agent_states_from_obs(obs, radius: float) -> List[AgentState2D]:
    out = []
    target_track_id = str(obs.target.track_id) if getattr(obs, "target", None) is not None else None
    for npc in getattr(obs, "npcs", []) or []:
        if target_track_id is not None and str(npc.track_id) == target_track_id:
            continue
        speed = float(npc.speed)
        if speed > 0.05 or math.hypot(float(npc.vx), float(npc.vy)) > 0.05:
            yaw = math.atan2(float(npc.vy), float(npc.vx))
        else:
            yaw = math.radians(float(npc.yaw_deg))
        out.append(
            AgentState2D(
                track_id=str(npc.track_id),
                x=float(npc.x),
                y=float(npc.y),
                vx=float(npc.vx),
                vy=float(npc.vy),
                yaw=yaw,
                speed=speed,
                radius=float(radius),
                is_target=False,
            )
        )
    return out


def target_yaw_from_obs(obs) -> float:
    if getattr(obs, "extras", None):
        # Side-follow targets are body-frame goals, so prefer the CARLA actor
        # yaw when the run manager provides multiple yaw interpretations.
        for key in ("target_actor_yaw_rad", "target_yaw_rad", "target_velocity_yaw_rad"):
            extra_yaw = obs.extras.get(key)
            if extra_yaw is not None:
                return float(extra_yaw)
    return math.radians(float(obs.target.yaw_deg))


__all__ = [
    "agent_states_from_obs",
    "robot_state_from_obs",
    "target_state_from_obs",
    "target_yaw_from_obs",
]
