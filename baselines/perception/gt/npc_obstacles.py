from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

NPC_OBS_RANGE_M = 6.0
NPC_OBS_HALF_SIZE_M = 0.35


def npc_box_obstacles(
    npcs,
    target_track_id: Optional[str],
    robot_xy: Sequence[float],
    range_m: float,
    half_size: float,
) -> list[dict[str, np.ndarray | None | str]]:
    out: list[dict[str, np.ndarray | None | str]] = []
    rx = float(robot_xy[0])
    ry = float(robot_xy[1])
    range_sq = float(range_m) * float(range_m)
    for npc in npcs:
        if target_track_id is not None and getattr(npc, "track_id", None) == target_track_id:
            continue
        cx = float(npc.x)
        cy = float(npc.y)
        dx = cx - rx
        dy = cy - ry
        if dx * dx + dy * dy > range_sq:
            continue
        vertex = np.array(
            [
                [cx - half_size, cy - half_size],
                [cx + half_size, cy - half_size],
                [cx + half_size, cy + half_size],
                [cx - half_size, cy + half_size],
            ],
            dtype=np.float32,
        ).T
        center = np.array([[cx], [cy]], dtype=np.float32)
        velocity = np.array([[float(npc.vx)], [float(npc.vy)]], dtype=np.float32)
        out.append(
            {
                "center": center,
                "radius": None,
                "vertex": vertex,
                "cone_type": "Rpositive",
                "velocity": velocity,
            }
        )
    return out


__all__ = ["NPC_OBS_HALF_SIZE_M", "NPC_OBS_RANGE_M", "npc_box_obstacles"]
