"""BSO-HFC follower with a perception frontend."""
from __future__ import annotations

import os
import sys
import time


from baselines.adapters.bso_hfc_adapter import BsoHfcFollowerPolicy
from baselines.adapters._perception_frontend import (
    PerceptionFrontend,
    PerceptionFrontendConfig,
)
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter


class BsoHfcPerceptionFollowerPolicy(FollowerPolicyAdapter):
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
        self._inner = BsoHfcFollowerPolicy(dt=dt, **inner_kwargs)
        self._frontend = PerceptionFrontend(PerceptionFrontendConfig(
            dt=dt, yolo_model=yolo_model, tracker_device=tracker_device,
            tracker_yolo_stride=tracker_yolo_stride, reid_mode=reid_mode,
            reid_kpr_config=reid_kpr_config, reid_device=reid_device,
            lost_policy=lost_policy, log_prefix="BSO_PRC",
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
        self._last_debug = {**inner_debug, **step.debug, "timing": timing}
        return action
