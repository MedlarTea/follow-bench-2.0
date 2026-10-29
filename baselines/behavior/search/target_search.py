"""High-level search orchestration that chooses overtake first, then fluid fallback."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from baselines.behavior.follow_goal import compute_follow_goal
from baselines.behavior.search.config import SearchConfig
from baselines.behavior.search.fluid import select_fluid_goal
from baselines.behavior.search.mode_manager import SearchMode, SearchModeManager
from baselines.behavior.search.overtake import select_overtake_goal
from baselines.maps.map_query import MapQuery
from baselines.common.types import AgentState2D, PredictionBundle, RobotState2D, SearchResult


class TargetSearchPlanner:
    def __init__(self, config: SearchConfig | None = None, target_radius: float = 0.35) -> None:
        self.config = config or SearchConfig()
        self.target_radius = float(target_radius)
        self._last_anchor_xy: np.ndarray | None = None
        self._mode_manager = SearchModeManager(self.config.reacquire_transition_ticks)

    def reset(self) -> None:
        self._last_anchor_xy = None
        self._mode_manager.reset()

    def on_target_visible(self) -> None:
        self._last_anchor_xy = None
        self._mode_manager.on_target_visible()

    def plan_goal(
        self,
        robot: RobotState2D,
        target: AgentState2D,
        target_visible: bool,
        agents: List[AgentState2D],
        prediction: PredictionBundle,
        map_query: MapQuery,
        search_direction: Sequence[float],
        follow_position: str,
        desired_distance: float,
    ) -> SearchResult:
        if target_visible:
            return self._plan_visible_follow(
                robot=robot,
                target=target,
                follow_position=follow_position,
                desired_distance=desired_distance,
            )
        return self.select_goal(
            robot=robot,
            agents=agents,
            prediction=prediction,
            map_query=map_query,
            search_direction=search_direction,
        )

    def select_goal(
        self,
        robot: RobotState2D,
        agents: List[AgentState2D],
        prediction: PredictionBundle,
        map_query: MapQuery,
        search_direction: Sequence[float],
    ) -> SearchResult:
        raw_direction = _coerce_direction(search_direction, robot.yaw)
        direction = raw_direction / np.linalg.norm(raw_direction)
        anchor_xy = self._resolve_anchor(robot, prediction, raw_direction)
        search_dir = anchor_xy - robot.xy
        if np.linalg.norm(search_dir) <= 1e-6:
            search_dir = direction

        overtake = select_overtake_goal(
            robot=robot,
            search_anchor_xy=anchor_xy,
            agents=agents,
            predicted_agents=prediction.npc_trajs_by_id,
            map_query=map_query,
            target_radius=self.target_radius,
            config=self.config,
        )
        if overtake is not None and overtake.mode == SearchMode.OVERTAKE:
            return self._resolve_result(overtake, anchor_xy)

        fluid = select_fluid_goal(
            robot=robot,
            agents=agents,
            search_direction=search_dir,
            map_query=map_query,
            config=self.config,
        )
        if overtake is not None and len(overtake.samples) > 0:
            fluid.samples = np.vstack([overtake.samples, fluid.samples])
            fluid.occluder_id = overtake.occluder_id
        return self._resolve_result(fluid, anchor_xy)

    def _resolve_anchor(
        self,
        robot: RobotState2D,
        prediction: PredictionBundle,
        fallback_direction: np.ndarray,
    ) -> np.ndarray:
        desired_anchor = self._predicted_anchor_xy(robot, prediction)
        if desired_anchor is None:
            desired_anchor = robot.xy + fallback_direction
        desired_anchor = _clamp_anchor_distance(
            robot.xy,
            desired_anchor,
            fallback_direction,
            self.config.anchor_min_distance,
            self.config.anchor_max_distance,
        )

        if self._last_anchor_xy is None:
            anchor_xy = desired_anchor
        else:
            alpha = float(np.clip(self.config.anchor_smoothing_alpha, 0.0, 1.0))
            anchor_xy = (1.0 - alpha) * self._last_anchor_xy + alpha * desired_anchor
            anchor_xy = _clamp_anchor_distance(
                robot.xy,
                anchor_xy,
                fallback_direction,
                self.config.anchor_min_distance,
                self.config.anchor_max_distance,
            )

        self._last_anchor_xy = np.asarray(anchor_xy, dtype=float)
        return self._last_anchor_xy.copy()

    def _predicted_anchor_xy(
        self,
        robot: RobotState2D,
        prediction: PredictionBundle,
    ) -> np.ndarray | None:
        target_traj = getattr(prediction, "target_traj", None)
        if target_traj is None:
            return None

        traj = np.asarray(target_traj, dtype=float)
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
            return None

        finite_mask = np.all(np.isfinite(traj[:, :2]), axis=1)
        if not np.any(finite_mask):
            return None
        traj_xy = traj[finite_mask, :2]
        if traj_xy.shape[0] == 0:
            return None

        pref_idx = int(np.clip(self.config.anchor_prediction_index, 0, traj_xy.shape[0] - 1))
        candidate = traj_xy[pref_idx]
        min_dist = max(float(self.config.anchor_min_distance), 1e-3) * 0.5
        if np.linalg.norm(candidate - robot.xy) >= min_dist:
            return candidate

        for sample in traj_xy[pref_idx + 1 :]:
            if np.linalg.norm(sample - robot.xy) >= min_dist:
                return sample
        return candidate

    def _resolve_result(self, result: SearchResult, anchor_xy: np.ndarray) -> SearchResult:
        detail_mode = result.detail_mode or result.mode
        resolved_mode, hidden_steps = self._mode_manager.on_search_result(detail_mode)
        result.anchor_xy = anchor_xy
        result.detail_mode = detail_mode
        result.hidden_steps = hidden_steps
        result.mode = resolved_mode
        return result

    def _plan_visible_follow(
        self,
        robot: RobotState2D,
        target: AgentState2D,
        follow_position: str,
        desired_distance: float,
    ) -> SearchResult:
        self.on_target_visible()
        goal = compute_follow_goal(robot, target, follow_position, desired_distance)
        return SearchResult(
            goal_pose=goal,
            mode=SearchMode.FOLLOW,
            detail_mode=SearchMode.FOLLOW,
            anchor_xy=target.xy,
            hidden_steps=0,
        )


def _coerce_direction(search_direction: Sequence[float], robot_yaw: float) -> np.ndarray:
    direction = np.asarray(search_direction, dtype=float)
    if np.linalg.norm(direction) <= 1e-6:
        direction = np.array([np.cos(robot_yaw), np.sin(robot_yaw)], dtype=float)
    return direction


def _clamp_anchor_distance(
    robot_xy: np.ndarray,
    anchor_xy: np.ndarray,
    fallback_direction: np.ndarray,
    min_distance: float,
    max_distance: float,
) -> np.ndarray:
    delta = np.asarray(anchor_xy, dtype=float) - np.asarray(robot_xy, dtype=float)
    dist = float(np.linalg.norm(delta))
    direction = np.asarray(fallback_direction, dtype=float)
    if np.linalg.norm(direction) <= 1e-6:
        direction = np.array([1.0, 0.0], dtype=float)
    else:
        direction = direction / np.linalg.norm(direction)

    if dist <= 1e-6:
        dist = max(float(min_distance), 1e-3)
        delta = direction * dist
    else:
        delta = delta / dist
        dist = float(np.clip(dist, max(float(min_distance), 1e-3), max(float(max_distance), float(min_distance))))
        delta = delta * dist
    return np.asarray(robot_xy, dtype=float) + delta
