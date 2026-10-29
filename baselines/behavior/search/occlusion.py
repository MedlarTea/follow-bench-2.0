"""Occluder selection helpers used by the overtake-style search branch."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from baselines.common.types import AgentState2D


def find_occluder(
    robot_xy: Sequence[float],
    target_anchor_xy: Sequence[float],
    candidates: Iterable[AgentState2D],
    max_line_distance: float = 1.0,
) -> Optional[str]:
    robot = np.asarray(robot_xy, dtype=float)
    anchor = np.asarray(target_anchor_xy, dtype=float)
    line = anchor - robot
    line_len = float(np.linalg.norm(line))
    if line_len <= 1e-6:
        return None

    line_dir = line / line_len
    candidates_list = list(candidates)
    if not candidates_list:
        return None

    centers = np.asarray([agent.xy for agent in candidates_list], dtype=float)
    rel = centers - robot[None, :]
    proj_len = rel @ line_dir
    valid = (proj_len >= 0.0) & (proj_len <= line_len)
    if not np.any(valid):
        return None

    proj_points = robot[None, :] + proj_len[:, None] * line_dir[None, :]
    dists = np.linalg.norm(centers - proj_points, axis=1)
    valid &= dists <= float(max_line_distance)
    if not np.any(valid):
        return None

    valid_idx = np.flatnonzero(valid)
    best_local_idx = int(np.argmin(dists[valid_idx]))
    best_idx = int(valid_idx[best_local_idx])
    return candidates_list[best_idx].track_id
