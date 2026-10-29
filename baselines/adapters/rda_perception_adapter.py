"""Basic RDA follower (no lidar) with a perception frontend."""
from __future__ import annotations

import os
import sys
import time


from baselines.adapters.rda_adapter import RdaFollowerPolicy
from baselines.adapters.rda_obstacles import NPC_OBS_HALF_SIZE_M, obstacles_debug, target_box_obstacle
from baselines.adapters._perception_frontend import (
    PerceptionFrontend,
    PerceptionFrontendConfig,
)
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter


class RdaPerceptionFollowerPolicy(FollowerPolicyAdapter):
    """Basic RDA (no lidar avoidance) driven by perception output.

    Note: naming collides with the old ``rda_perception_adapter`` which wrapped
    ``RdaLidarFollowerPolicy``. That adapter is now
    ``rda_lidar_perception_adapter.RdaLidarPerceptionFollowerPolicy``; this
    class is the thinner wrapping of the *basic* RDA.
    """

    def __init__(
        self,
        dt: float,
        yolo_model: str = "yolo11s.pt",
        tracker_device: str = "cuda",
        tracker_yolo_stride: int = 1,
        reid_mode: str = "basic",
        reid_kpr_config: str = "kpr_occ_duke_test",
        reid_device: str = "auto",
        lost_policy: str = "last_known",
        kinematics_mode: str = "kf",
        kf_pos_sigma: float = 0.20,
        kf_vel_sigma_q: float = 0.05,
        planner_target_state: str = "stable_route_heading",
        route_segments=None,
        **inner_kwargs,
    ) -> None:
        self._inner = RdaFollowerPolicy(dt=dt, **inner_kwargs)
        self._frontend = PerceptionFrontend(PerceptionFrontendConfig(
            dt=dt, yolo_model=yolo_model, tracker_device=tracker_device,
            tracker_yolo_stride=tracker_yolo_stride, reid_mode=reid_mode,
            reid_kpr_config=reid_kpr_config, reid_device=reid_device,
            lost_policy=lost_policy, log_prefix="RDA_PRC",
            kinematics_mode=kinematics_mode,
            kf_pos_sigma=kf_pos_sigma,
            kf_vel_sigma_q=kf_vel_sigma_q,
            planner_target_state=planner_target_state,
            route_segments=route_segments,
            planner_follow_position=inner_kwargs.get("follow_position"),
        ))
        self._last_debug: dict = {}

    def reset(self) -> None:
        self._inner.reset()
        self._frontend.reset()
        self._last_debug = {}

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        step = self._frontend.step(obs)
        t_planner = time.perf_counter()
        if step.brake:
            action = FollowAction(v_mps=0.0, w_radps=0.0)
        else:
            action = self._inner.act(step.modified_obs)
        planner_core_ms = 0.0 if step.brake else (time.perf_counter() - t_planner) * 1000.0
        inner_debug = getattr(self._inner, "get_debug_info", lambda: {})() or {}
        timing = dict(inner_debug.get("timing") or {})
        timing.update(step.debug.get("timing") or {})
        timing["planner_core_ms"] = planner_core_ms
        perception_obstacles = _perception_obstacles_debug(step.modified_obs)
        self._last_debug = {
            **inner_debug,
            **step.debug,
            "obstacles": list(inner_debug.get("obstacles") or []) + perception_obstacles,
            "perception_obstacles": perception_obstacles,
            "timing": timing,
        }
        return action


def _perception_obstacles_debug(obs: FollowObservation) -> list:
    """Debug-only obstacle boxes from camera perception tracks.

    In perception mode, ``obs.npcs`` has already been replaced by the
    perception frontend with synthetic detection tracks. Do not read simulator
    ground-truth NPC positions here; these boxes are only for debug_vis and are
    not passed into the basic RDA MPC.
    """
    obstacle_list = [
        target_box_obstacle(npc, half_size=NPC_OBS_HALF_SIZE_M)
        for npc in (obs.npcs or [])
    ]
    return obstacles_debug(obstacle_list)
