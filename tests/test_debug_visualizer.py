"""
Standalone test for DebugVisualizer — no CARLA required.

Simulates a robot moving in a circle chasing a target, with several bystanders.
Run with: python tests/test_debug_visualizer.py
"""
import math
import sys
import os
import time
from dataclasses import dataclass
from typing import List

# Make sure debug_vis package is importable

from followbench.paths import scene_asset
from followbench.visualization.debug_visualizer import DebugVisualizer


# ── Minimal stand-ins for RobotState / NpcState ──────────────────────────────

@dataclass
class FakeRobotState:
    x: float
    y: float
    z: float = 0.0
    yaw_rad: float = 0.0
    speed: float = 0.0


@dataclass
class FakeNpcState:
    track_id: str
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_deg: float = 0.0
    speed: float = 0.0


# ── Grid NPZ path (corner scenario) ────────────────────────────────────────
_NPZ = scene_asset("corner", "gridmap_roi.npz")


def main() -> None:
    print(f"[Test] Using occ map: {_NPZ}")
    print(f"[Test] occ map exists: {os.path.exists(_NPZ)}")

    vis = DebugVisualizer(half_size=20.0)

    # Robot starts near a recognizable corner centre
    # (picked from world_min/max: roughly centre of the map)
    robot_x0, robot_y0 = 60.0, 20.0

    vis.start(robot_x0, robot_y0, _NPZ)
    print("[Test] Visualizer process started, sending frames …")
    time.sleep(0.8)   # let the window open

    n_frames = 300
    dt = 0.05

    for i in range(n_frames):
        t = i * dt

        # Robot: small circle
        r_radius = 3.0
        rx = robot_x0 + r_radius * math.cos(t * 0.5)
        ry = robot_y0 + r_radius * math.sin(t * 0.5)
        ryaw = t * 0.5 + math.pi / 2.0
        robot = FakeRobotState(x=rx, y=ry, yaw_rad=ryaw)

        # Target: slightly larger circle, ahead of robot
        tgt_radius = 5.0
        tx = robot_x0 + tgt_radius * math.cos(t * 0.5 + 0.4)
        ty = robot_y0 + tgt_radius * math.sin(t * 0.5 + 0.4)
        target = FakeNpcState(track_id="N01", x=tx, y=ty)

        # Bystanders
        npcs: List[FakeNpcState] = [target]
        for k in range(6):
            angle = t * 0.3 + k * (2 * math.pi / 6)
            bx = robot_x0 + 8.0 * math.cos(angle)
            by = robot_y0 + 8.0 * math.sin(angle)
            npcs.append(FakeNpcState(track_id=f"N{k+2:02d}", x=bx, y=by))

        # Fake obstacle polygon (a small diamond near the robot)
        ox, oy = rx + 2.5 * math.cos(ryaw), ry + 2.5 * math.sin(ryaw)
        obstacles = [
            {
                "vertices": [
                    [ox + 0.4,  oy],
                    [ox,        oy + 0.4],
                    [ox - 0.4,  oy],
                    [ox,        oy - 0.4],
                ]
            }
        ]

        vis.update(
            tick=i,
            robot_state=robot,
            npc_states=npcs,
            active_target_id="N01",
            obstacles=obstacles,
        )

        time.sleep(dt)

    print("[Test] Done. Closing visualizer in 2 s …")
    time.sleep(2.0)
    vis.stop()
    print("[Test] OK")


if __name__ == "__main__":
    main()
