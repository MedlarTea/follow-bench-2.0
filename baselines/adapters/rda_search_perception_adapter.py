"""RDA-Search follower with a perception frontend.

Unlike the seven simple wrappers, this one has two extra responsibilities:

1. **Perception-driven target visibility**. The inner ``RdaSearchFollowerPolicy``
   consults ``obs.target_visible`` / ``obs.target_pixel_count`` (through
   ``is_target_visible``) to decide when to enter *search* mode. With the
   perception frontend in place, those GT-derived fields are meaningless —
   we override them with the FSM lock state (``step.target_locked``) before
   passing ``modified_obs`` to the inner policy.

2. **Force ``lost_policy='last_known'``**. ``brake`` would short-circuit the
   search (robot stops instead of hunting). ``gt_fallback`` defeats the whole
   point of perception-based search. Only ``last_known`` makes sense: on loss,
   the frontend hands the inner search planner a stationary target at the last
   known pose, and the inner policy's own prediction + search samplers take
   over from there.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time


from baselines.adapters.rda_search_adapter import RdaSearchFollowerPolicy
from baselines.adapters._perception_frontend import (
    PerceptionFrontend,
    PerceptionFrontendConfig,
)
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter


# Magic number: large enough to pass any ``min_pixel_count`` gate downstream
# when ``target_visible=True``. The inner rda_search only checks visibility as
# a boolean via ``is_target_visible(obs, min_pixel_count=0)``, so any value
# ≥ 1 works — we use 9999 so any accidental higher threshold still passes.
_VISIBLE_PIXEL_COUNT_SENTINEL = 9999


class RdaSearchPerceptionFollowerPolicy(FollowerPolicyAdapter):
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
        if str(lost_policy).lower() != "last_known":
            raise ValueError(
                "RdaSearchPerceptionFollowerPolicy only supports "
                "lost_policy='last_known' — brake would disable the inner "
                "search, gt_fallback defeats the purpose of perception. "
                f"Got {lost_policy!r}."
            )

        # The inner search planner must treat perception's visibility as
        # authoritative; GT-based shortcuts (``gt_target_always_visible``) are
        # incompatible with this wrapper.
        if inner_kwargs.pop("gt_target_always_visible", False):
            print(
                "[RDA_SEARCH_PRC] warning: gt_target_always_visible=True is "
                "ignored when running through the perception frontend.",
                flush=True,
            )
        # Search is always enabled for this wrapper — the whole purpose is to
        # drive the search logic from perception.
        inner_kwargs.setdefault("enable_search", True)

        self._inner = RdaSearchFollowerPolicy(dt=dt, **inner_kwargs)
        self._frontend = PerceptionFrontend(PerceptionFrontendConfig(
            dt=dt,
            yolo_model=yolo_model,
            tracker_device=tracker_device,
            tracker_yolo_stride=tracker_yolo_stride,
            reid_mode=reid_mode,
            reid_kpr_config=reid_kpr_config,
            reid_device=reid_device,
            lost_policy="last_known",
            log_prefix="RDA_SEARCH_PRC",
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

        # Bootstrap with GT while perception is still initializing, then hand
        # over to the perception-derived target once the FSM leaves initial.
        if self._frontend.state_name == "initial":
            mo = obs
        elif step.target_locked:
            # Inject perception-derived visibility so the inner search decides
            # based on the FSM lock, not on GT instance-mask pixel counts.
            mo = dataclasses.replace(
                step.modified_obs,
                target_visible=True,
                target_pixel_count=_VISIBLE_PIXEL_COUNT_SENTINEL,
            )
        else:
            # last_known policy already produced a synthetic target at the
            # cached xy (or passed obs through if we've never locked). Either
            # way the inner search planner gets an anchor to search around.
            mo = dataclasses.replace(
                step.modified_obs,
                target_visible=False,
                target_pixel_count=0,
            )

        t_planner = time.perf_counter()
        action = self._inner.act(mo)
        planner_core_ms = (time.perf_counter() - t_planner) * 1000.0

        inner_debug = getattr(self._inner, "get_debug_info", lambda: {})() or {}
        timing = dict(inner_debug.get("timing") or {})
        timing.update(step.debug.get("timing") or {})
        timing["planner_core_ms"] = planner_core_ms
        self._last_debug = {**inner_debug, **step.debug, "timing": timing}
        return action
