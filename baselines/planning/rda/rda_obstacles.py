from __future__ import annotations

from collections import namedtuple
from typing import List

Obstacle = namedtuple("obstacle", "center radius vertex cone_type velocity")

ROBOT_RADIUS = 0.5


def obstacles_debug(obstacle_list: List[Obstacle]) -> list:
    out = []
    for obstacle in obstacle_list:
        if obstacle.cone_type == "Rpositive" and obstacle.vertex is not None:
            out.append({"vertices": obstacle.vertex.T.tolist()})
        elif obstacle.cone_type == "norm2" and obstacle.center is not None:
            cx = float(obstacle.center[0, 0])
            cy = float(obstacle.center[1, 0])
            r = float(obstacle.radius or ROBOT_RADIUS)
            out.append(
                {
                    "vertices": [
                        [cx - r, cy - r],
                        [cx + r, cy - r],
                        [cx + r, cy + r],
                        [cx - r, cy + r],
                    ]
                }
            )
    return out


def to_rda_obstacle(item) -> Obstacle:
    if isinstance(item, Obstacle):
        return item
    return Obstacle(
        item["center"],
        item["radius"],
        item["vertex"],
        item["cone_type"],
        item["velocity"],
    )


__all__ = [
    "Obstacle",
    "ROBOT_RADIUS",
    "obstacles_debug",
    "to_rda_obstacle",
]
