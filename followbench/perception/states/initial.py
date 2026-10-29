"""Initial state — select the target person."""
from __future__ import annotations

import math
from typing import Optional

from .base import State, StateContext


class InitialState(State):
    def state_name(self) -> str:
        return "initial"

    def update(self, reid, ctx: StateContext) -> State:
        from .initial_training import InitialTrainingState

        if not ctx.tracks_world:
            return self

        target_id = self._select_target(ctx)
        if target_id < 0:
            return self
        return InitialTrainingState(target_id, self.config)

    def _select_target(self, ctx: StateContext) -> int:
        # GT-seeded lock: pick the visible track closest to the GT XY hint.
        if ctx.gt_target_xy is not None:
            gx, gy = ctx.gt_target_xy
            best_id, best_d = -1, self.config.initial_gt_match_radius_m
            for tid, (x, y) in ctx.tracks_world.items():
                d = math.hypot(x - gx, y - gy)
                if d < best_d:
                    best_d = d
                    best_id = tid
            if best_id >= 0:
                return best_id
            # GT hint provided but nothing close — fall through to heuristic.

        # Heuristic: closest pedestrian directly in front of the robot.
        cy_, sy_ = math.cos(ctx.robot_yaw), math.sin(ctx.robot_yaw)
        best_id, best_d = -1, self.config.initial_select_max_dist_m
        multi = len(ctx.tracks_world) > 1
        for tid, (x, y) in ctx.tracks_world.items():
            dx_w, dy_w = x - ctx.robot_x, y - ctx.robot_y
            body_x = cy_ * dx_w + sy_ * dy_w     # forward
            body_y = -sy_ * dx_w + cy_ * dy_w    # right
            if body_x <= 0.2:
                continue
            if multi and abs(body_y) > self.config.initial_select_lateral_max:
                continue
            d = math.hypot(dx_w, dy_w)
            if d < best_d:
                best_d = d
                best_id = tid
        return best_id
