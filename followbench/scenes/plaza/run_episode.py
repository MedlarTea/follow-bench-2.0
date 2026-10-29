#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np

SCENE = "plaza"

from followbench.paths import module_path, scene_asset, scene_runs_dir  # noqa: E402

from followbench.sim.core_types import FollowObservation, RobotState  # noqa: E402


from followbench.visualization.lidar_vis import debug_lidar_world_points, lidar_to_bev as _lidar_to_bev  # noqa: E402

from followbench.scenes.plaza.npc_runtime import RandomNpcRuntime, DIFFICULTY_PRESETS  # noqa: E402
from followbench.sim.robot_runtime import ScoutRobotRuntime  # noqa: E402
from followbench.sim.humanoid_runtime import G1RobotRuntime  # noqa: E402
from followbench.sim.visibility_instance import InstanceVisibilityChecker  # noqa: E402
import followbench.sim.weather as weather_mod  # noqa: E402
from followbench.evaluation.collision import HumanCollisionMonitor, actor_track_lookup  # noqa: E402
from followbench.evaluation.logging import (  # noqa: E402
    EvaluationLogger,
    enrich_sensor_collision_events,
    make_geometry_collision_event,
)
from followbench.evaluation.schemas import add_evaluation_args  # noqa: E402
from followbench.evaluation.sim_clock import SimulationPacer  # noqa: E402
from followbench.sim.carla_world import connect_world  # noqa: E402

try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


_FOLLOW_OFFSET_RAD = {
    "back":       math.pi,
    "front":      0.0,
    "left_side":  math.pi / 2.0,
    "right_side": -math.pi / 2.0,
}

MANUAL_ROBOT_V_MPS = 2.0
MANUAL_ROBOT_W_RADPS = 2.2
MANUAL_ROBOT_SPRINT_SCALE = 1.8
MANUAL_CAM_DIST = 4.5
MANUAL_CAM_HEIGHT = 2.0
MANUAL_MOUSE_SENS = 0.25


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random scenario episode manager (parallel/perpendicular crowd flow)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--load-map", action="store_true")
    p.add_argument("--map-name", type=str, default="Town10HD_Opt")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--duration-sec", type=float, default=300.0)
    p.add_argument("--target-track-id", type=str, default="N01")
    p.add_argument("--visibility-threshold", type=int, default=400)
    p.add_argument("--grid-npz", type=str, default=scene_asset(SCENE, "gridmap_roi.npz"))
    p.add_argument("--flow-points-json", type=str, default=scene_asset(SCENE, "gridmap_roi_flow_points.json"))
    p.add_argument(
        "--flow-mode",
        choices=["random", "parallel", "perpendicular", "chaotic"],
        default="random",
        help="NPC crowd flow geometry. 'random' picks one of the three per episode.",
    )
    p.add_argument(
        "--difficulty",
        choices=list(DIFFICULTY_PRESETS.keys()),
        default="easy",
        help="Crowd difficulty preset. Cycle at runtime with D key.",
    )
    p.add_argument(
        "--num-walkers",
        type=int,
        default=None,
        help="Total pedestrians including the target. Defaults to the selected difficulty preset.",
    )
    p.add_argument(
        "--npc-rng-seed",
        type=int,
        default=None,
        help="NPC random seed. Omit for fresh random per run; set int to reproduce.",
    )
    p.add_argument("--robot-spawn-z", type=float, default=0.5)
    p.add_argument("--robot-spawn-yaw-deg", type=float, default=0.0,
                   help="Used only by --robot-spawn-mode=fixed.")
    p.add_argument(
        "--robot-spawn-mode",
        choices=["near_target"],
        default="near_target",
        help="Robot spawn strategy. near_target spawns next to the target along --follow-position offset.",
    )
    p.add_argument("--robot-type", choices=["scout", "g1"], default="scout")
    p.add_argument("--sensor-image-w", type=int, default=800)
    p.add_argument("--sensor-image-h", type=int, default=600)
    p.add_argument("--sensor-fov-deg", type=float, default=90.0)
    p.add_argument("--sensor-lidar-range", type=float, default=30.0)
    p.add_argument("--sensor-lidar-height-agl", type=float, default=0.50,
                   help="Single-line LiDAR scan plane height above ground (m).")
    p.add_argument("--robot-rescue-lift-z", type=float, default=0.20)
    p.add_argument(
        "--planner",
        choices=["pid", "dwa_traj", "sfm", "rda", "rda_lidar", "gt_planner", "rda_traj", "rda_search", "bso_hfc", "trackvla", "oa_vat", "custom"],
        default="pid",
    )
    from followbench.interfaces.policy_factory import add_policy_factory_args  # noqa: PLC0415
    add_policy_factory_args(p)
    p.add_argument("--follow-position", choices=["back", "left_side", "right_side"], default="back")
    p.add_argument("--desired-distance", type=float, default=1.5)
    p.add_argument(
        "--rda-search-use-camera-visibility",
        action="store_true",
    )
    p.add_argument(
        "--rda-search-disable-target-protection",
        action="store_true",
    )
    p.add_argument(
        "--rda-traj-disable-lidar-obstacles",
        action="store_true",
        help="Disable lidar/NPC obstacle injection for rda_traj (legacy behavior).",
    )
    p.add_argument("--use-perception", action="store_true")
    p.add_argument("--reid-mode", choices=["basic", "kpr"], default="basic")
    p.add_argument("--reid-kpr-config", type=str, default="kpr_occ_duke_test")
    p.add_argument("--reid-device", type=str, default="auto")
    p.add_argument("--lost-policy", choices=["gt_fallback", "brake", "last_known"], default="last_known")
    p.add_argument("--kinematics-mode", choices=["ema", "kf"], default="kf")
    p.add_argument("--kf-pos-sigma", type=float, default=0.20)
    p.add_argument("--kf-vel-sigma-q", type=float, default=0.05)
    p.add_argument("--trackvla-instruction", type=str, default="Follow the person")
    p.add_argument("--trackvla-ckpt", type=str, default=None)
    p.add_argument("--oa-vat-text-query", type=str, default="person")
    p.add_argument("--oa-vat-kf-mode", type=str, default="confaware",
                   choices=["none", "standard", "confaware"])
    p.add_argument("--oa-vat-online-enhance", action="store_true", default=True)
    p.add_argument("--oa-vat-no-online-enhance", dest="oa_vat_online_enhance", action="store_false")
    p.add_argument("--oa-vat-dinov3", type=str, default="dinov3_vitb16",
                   choices=["dinov3_vitb16", "dinov3_vits16"])
    p.add_argument("--oa-vat-match-thresh", type=float, default=0.4)
    p.add_argument("--oa-vat-reacq-thresh", type=float, default=0.5)
    p.add_argument("--oa-vat-yolo-conf", type=float, default=0.1)
    p.add_argument("--oa-vat-target-area-ratio", type=float, default=0.12)
    p.add_argument("--teleop", action="store_true",
                   help="Drive the robot with WASD and log the episode as planner=teleop.")
    p.add_argument("--operator-id", type=str, default="anonymous",
                   help="Anonymous operator identifier stored in episode metadata.")
    p.add_argument("--num-episodes", type=int, default=1,
                   help="Number of teleoperation episodes to run in one CARLA/UI session.")
    p.add_argument("--sim-speed", type=float, default=1.0,
                   help="Teleop simulation speed relative to wall clock (1.0=realtime).")
    p.add_argument("--no-ui", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--auto-spawn-sec", type=float, default=-1.0)
    p.add_argument("--auto-follow-sec", type=float, default=-1.0)
    p.add_argument("--output-dir", type=str, default=scene_runs_dir(SCENE))
    p.add_argument("--draw-debug", action="store_true")
    p.add_argument("--draw-laser", action="store_true")
    p.add_argument(
        "--weather",
        choices=["random"] + weather_mod.CHOICES,
        default="random",
    )
    add_evaluation_args(p)
    return p.parse_args()


def _update_manual_spectator(
    spectator: carla.Actor,
    actor: carla.Actor,
    cam_yaw: float,
    cam_pitch: float,
) -> None:
    """Position the CARLA spectator in third person behind ``actor``."""
    loc = actor.get_location()
    yaw_rad = math.radians(cam_yaw)
    pitch_rad = math.radians(cam_pitch)
    ox = -MANUAL_CAM_DIST * math.cos(yaw_rad) * math.cos(pitch_rad)
    oy = -MANUAL_CAM_DIST * math.sin(yaw_rad) * math.cos(pitch_rad)
    oz = MANUAL_CAM_HEIGHT - MANUAL_CAM_DIST * math.sin(pitch_rad)
    spectator.set_transform(carla.Transform(
        carla.Location(
            x=loc.x + ox,
            y=loc.y + oy,
            z=loc.z + max(oz, 0.5),
        ),
        carla.Rotation(pitch=cam_pitch, yaw=cam_yaw, roll=0.0),
    ))


def _apply_manual_robot_control(
    robot: ScoutRobotRuntime,
    key_state,
) -> Tuple[float, float]:
    if key_state is None:
        robot.hold_still()
        return 0.0, 0.0
    forward = float(key_state[pygame.K_w]) - float(key_state[pygame.K_s])
    turn = float(key_state[pygame.K_d]) - float(key_state[pygame.K_a])
    sprint = bool(key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT])
    scale = MANUAL_ROBOT_SPRINT_SCALE if sprint else 1.0
    v_cmd = MANUAL_ROBOT_V_MPS * scale * forward
    w_cmd = MANUAL_ROBOT_W_RADPS * scale * turn
    robot.apply_velocity_command(v_cmd, w_cmd)
    return v_cmd, w_cmd


def follow_position_spawn_xy(target_x, target_y, target_yaw_rad,
                              follow_position, desired_distance,
                              roi_map, forbidden_xys=None,
                              min_clearance_m: float = 0.7,
                              search_max_radius: float = 2.0):
    """Pick a spawn (x, y) at the requested follow offset, avoiding existing
    walkers. ``forbidden_xys`` is a list of (x, y) world coords (typically
    NPC positions) that the candidate must stay at least ``min_clearance_m``
    away from."""
    forbidden = list(forbidden_xys) if forbidden_xys else []
    clearance_sq = float(min_clearance_m) * float(min_clearance_m)

    def _ok(x: float, y: float) -> bool:
        if not roi_map.is_world_walkable(x, y):
            return False
        for fx, fy in forbidden:
            dx = x - float(fx)
            dy = y - float(fy)
            if dx * dx + dy * dy < clearance_sq:
                return False
        return True

    offset = _FOLLOW_OFFSET_RAD.get(follow_position, math.pi)
    base_angle = target_yaw_rad + offset
    ideal_x = float(target_x + math.cos(base_angle) * desired_distance)
    ideal_y = float(target_y + math.sin(base_angle) * desired_distance)
    if _ok(ideal_x, ideal_y):
        return ideal_x, ideal_y, True
    for radius in np.arange(0.2, search_max_radius + 1e-3, 0.2):
        for ang in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
            xx = float(ideal_x + math.cos(ang) * radius)
            yy = float(ideal_y + math.sin(ang) * radius)
            if _ok(xx, yy):
                return xx, yy, True
    return ideal_x, ideal_y, False


def resolve_robot_spawn_tf(args: argparse.Namespace, npc: RandomNpcRuntime, target_track_id: str) -> carla.Transform:
    # Other NPCs' current positions — used to keep robot away from existing walkers.
    other_xys = [(s.x, s.y) for s in npc.get_states() if s.track_id != target_track_id]
    t = npc.get_target_by_track_id(target_track_id)
    if t is None:
        # Race condition: target's transform not materialised yet. Place robot at
        # an offset behind A along the axis (not at A — that's where the target is).
        ax, ay = npc.A
        bx, by = npc.B
        yaw_rad = float(np.arctan2(by - ay, bx - ax))
        x, y, _ok = follow_position_spawn_xy(
            ax, ay, yaw_rad,
            str(args.follow_position), float(args.desired_distance),
            npc.roi_map, forbidden_xys=other_xys,
        )
        return carla.Transform(
            carla.Location(x=x, y=y, z=args.robot_spawn_z),
            carla.Rotation(yaw=float(np.degrees(yaw_rad))),
        )
    target_yaw_rad = float(np.arctan2(npc.B[1] - npc.A[1], npc.B[0] - npc.A[0]))
    z = float(max(args.robot_spawn_z, t.z + 0.25))
    x, y, _ok = follow_position_spawn_xy(
        float(t.x), float(t.y), target_yaw_rad,
        str(args.follow_position), float(args.desired_distance),
        npc.roi_map, forbidden_xys=other_xys,
    )
    robot_yaw = float(np.degrees(np.arctan2(float(t.y) - y, float(t.x) - x)))
    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=robot_yaw),
    )


def yaw_rad_from_velocity(vx: float, vy: float, fallback_deg: float) -> float:
    if abs(vx) + abs(vy) > 1e-4:
        return float(np.arctan2(vy, vx))
    return float(np.deg2rad(fallback_deg))


def step_world(world: carla.World, synchronous_mode: bool) -> None:
    if synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()


def apply_world_mode(world: carla.World, base_settings: carla.WorldSettings, sync_mode: bool, dt: float) -> None:
    s = world.get_settings()
    s.no_rendering_mode = base_settings.no_rendering_mode
    s.substepping = base_settings.substepping
    s.max_substep_delta_time = base_settings.max_substep_delta_time
    s.max_substeps = base_settings.max_substeps
    s.synchronous_mode = bool(sync_mode)
    s.fixed_delta_seconds = float(dt) if sync_mode else None
    world.apply_settings(s)


def _to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3][:, :, ::-1]


def _depth_to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    d = (
        arr[:, :, 2].astype(np.float32)
        + arr[:, :, 1].astype(np.float32) * 256.0
        + arr[:, :, 0].astype(np.float32) * 65536.0
    ) / (256.0**3 - 1.0) * 1000.0
    gray = np.clip(np.log1p(d) / np.log1p(100.0) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _depth_to_meters(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return (
        arr[:, :, 2].astype(np.float32)
        + arr[:, :, 1].astype(np.float32) * 256.0
        + arr[:, :, 0].astype(np.float32) * 65536.0
    ) / (256.0**3 - 1.0) * 1000.0


def _instance_to_rgb(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    b = arr[:, :, 0].astype(np.uint32)
    g = arr[:, :, 1].astype(np.uint32)
    r = arr[:, :, 2].astype(np.uint32)
    inst_id = (r << 16) | (g << 8) | b
    out = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    out[:, :, 0] = ((inst_id * 37) % 255).astype(np.uint8)
    out[:, :, 1] = ((inst_id * 67) % 255).astype(np.uint8)
    out[:, :, 2] = ((inst_id * 97) % 255).astype(np.uint8)
    out[inst_id == 0] = 0
    return out


def _lidar_to_points(lidar_data) -> Optional[np.ndarray]:
    if lidar_data is None:
        return None
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    return pts.copy()


def _draw_bboxes_on(img: Optional[np.ndarray], bboxes, stale: int) -> Optional[np.ndarray]:
    if img is None or not bboxes:
        return img
    img = img.copy()
    for b in bboxes:
        x1, y1 = int(b["x1"]), int(b["y1"])
        x2, y2 = int(b["x2"]), int(b["y2"])
        tid = int(b["track_id"])
        is_tgt = bool(b.get("is_target", False))
        if is_tgt:
            color = (0, 220, 0) if stale == 0 else (0, 140, 0)
        else:
            color = (255, 215, 0) if stale == 0 else (160, 130, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ID {tid}{' (TGT)' if is_tgt else ''}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1), (x1 + tw + 8, y1 + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def _build_policy(args: argparse.Namespace):
    policy_kwargs = {
        "desired_distance": float(args.desired_distance),
        "follow_position": str(args.follow_position),
    }
    if args.use_perception:
        perception_kwargs = dict(
            reid_mode=args.reid_mode,
            reid_kpr_config=args.reid_kpr_config,
            reid_device=args.reid_device,
            lost_policy=args.lost_policy,
            kinematics_mode=args.kinematics_mode,
            kf_pos_sigma=args.kf_pos_sigma,
            kf_vel_sigma_q=args.kf_vel_sigma_q,
        )
    else:
        perception_kwargs = None

    if args.planner == "custom":
        from followbench.interfaces.policy_factory import build_custom_policy  # noqa: PLC0415
        return build_custom_policy(args, scene="plaza")
    if args.planner == "dwa_traj":
        if perception_kwargs is not None:
            from baselines.adapters.dwa_traj_perception_adapter import DwaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            return DwaTrajPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs)
        from baselines.adapters.dwa_traj_adapter import DwaTrajFollowerPolicy  # noqa: PLC0415
        return DwaTrajFollowerPolicy(dt=args.dt, **policy_kwargs)
    if args.planner == "sfm":
        if perception_kwargs is not None:
            from baselines.adapters.sfm_perception_adapter import SfmPerceptionFollowerPolicy  # noqa: PLC0415
            return SfmPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs)
        from baselines.adapters.sfm_adapter import SfmFollowerPolicy  # noqa: PLC0415
        return SfmFollowerPolicy(dt=args.dt, **policy_kwargs)
    if args.planner == "rda":
        if perception_kwargs is not None:
            from baselines.adapters.rda_perception_adapter import RdaPerceptionFollowerPolicy  # noqa: PLC0415
            return RdaPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs)
        from baselines.adapters.rda_adapter import RdaFollowerPolicy  # noqa: PLC0415
        return RdaFollowerPolicy(dt=args.dt, **policy_kwargs)
    if args.planner == "rda_lidar":
        if perception_kwargs is not None:
            from baselines.adapters.rda_lidar_perception_adapter import RdaLidarPerceptionFollowerPolicy  # noqa: PLC0415
            return RdaLidarPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs)
        from baselines.adapters.rda_lidar_adapter import RdaLidarFollowerPolicy  # noqa: PLC0415
        return RdaLidarFollowerPolicy(dt=args.dt, **policy_kwargs)
    if args.planner == "gt_planner":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner gt_planner "
                "(gt_planner already consumes full GT + lidar)."
            )
        from baselines.adapters.gt_planner_adapter import GtPlannerFollowerPolicy  # noqa: PLC0415
        return GtPlannerFollowerPolicy(dt=args.dt, **policy_kwargs, grid_npz=args.grid_npz)
    if args.planner == "rda_traj":
        traj_kwargs = dict(
            use_lidar_obstacles=not bool(args.rda_traj_disable_lidar_obstacles),
        )
        if perception_kwargs is not None:
            from baselines.adapters.rda_traj_perception_adapter import RdaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            return RdaTrajPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs, **traj_kwargs)
        from baselines.adapters.rda_traj_adapter import RdaTrajFollowerPolicy  # noqa: PLC0415
        return RdaTrajFollowerPolicy(dt=args.dt, **policy_kwargs, **traj_kwargs)
    if args.planner == "rda_search":
        search_kwargs = dict(
            gt_target_always_visible=not bool(args.rda_search_use_camera_visibility),
            protect_target_for_side_follow=not bool(args.rda_search_disable_target_protection),
        )
        if perception_kwargs is not None:
            from baselines.adapters.rda_search_perception_adapter import RdaSearchPerceptionFollowerPolicy  # noqa: PLC0415
            return RdaSearchPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs, **search_kwargs)
        from baselines.adapters.rda_search_adapter import RdaSearchFollowerPolicy  # noqa: PLC0415
        return RdaSearchFollowerPolicy(dt=args.dt, **policy_kwargs, **search_kwargs)
    if args.planner == "bso_hfc":
        if perception_kwargs is not None:
            from baselines.adapters.bso_hfc_perception_adapter import BsoHfcPerceptionFollowerPolicy  # noqa: PLC0415
            return BsoHfcPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs)
        from baselines.adapters.bso_hfc_adapter import BsoHfcFollowerPolicy  # noqa: PLC0415
        return BsoHfcFollowerPolicy(dt=args.dt, **policy_kwargs)
    if args.planner == "trackvla":
        if perception_kwargs is not None:
            raise SystemExit("--use-perception is not compatible with --planner trackvla")
        from baselines.adapters.trackvla_adapter import TrackVlaFollowerPolicy  # noqa: PLC0415
        return TrackVlaFollowerPolicy(
            dt=args.dt,
            instruction=args.trackvla_instruction,
            ckpt_path=args.trackvla_ckpt,
            render_overlay=bool(args.debug),
        )
    if args.planner == "oa_vat":
        if perception_kwargs is not None:
            raise SystemExit("--use-perception is not compatible with --planner oa_vat")
        from baselines.adapters.oa_vat_adapter import OaVatFollowerPolicy  # noqa: PLC0415
        return OaVatFollowerPolicy(
            dt=args.dt,
            text_query=args.oa_vat_text_query,
            kf_mode=args.oa_vat_kf_mode,
            online_enhance=args.oa_vat_online_enhance,
            dinov3_model=args.oa_vat_dinov3,
            match_thresh=args.oa_vat_match_thresh,
            reacq_match_thresh=args.oa_vat_reacq_thresh,
            yolo_conf_thresh=args.oa_vat_yolo_conf,
            target_area_ratio=args.oa_vat_target_area_ratio,
        )
    # pid (default)
    if perception_kwargs is not None:
        from baselines.adapters.pid_perception_adapter import PidPerceptionFollowerPolicy  # noqa: PLC0415
        return PidPerceptionFollowerPolicy(dt=args.dt, **policy_kwargs, **perception_kwargs)
    from baselines.adapters.pid_adapter import PIDFollowerPolicy  # noqa: PLC0415
    return PIDFollowerPolicy(dt=args.dt, **policy_kwargs)


def main() -> None:
    args = build_parser()
    num_walkers_overridden = args.num_walkers is not None
    if args.num_walkers is None:
        args.num_walkers = int(DIFFICULTY_PRESETS[args.difficulty]["num_walkers"])
    if args.num_walkers < 1:
        raise SystemExit("--num-walkers must be at least 1.")
    if args.teleop and args.no_ui:
        raise SystemExit("--teleop requires the pygame UI; remove --no-ui.")
    if args.teleop and not HAS_PYGAME:
        raise SystemExit("--teleop requires pygame, but pygame could not be imported.")
    if args.teleop and args.use_perception:
        raise SystemExit("--teleop is incompatible with --use-perception.")
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be at least 1.")
    if args.num_episodes > 1 and not args.teleop:
        raise SystemExit("--num-episodes > 1 is supported only with --teleop.")
    if args.num_episodes > 1 and args.eval_trial_id is not None:
        raise SystemExit("--eval-trial-id cannot be fixed when --num-episodes > 1.")
    if args.sim_speed <= 0.0:
        raise SystemExit("--sim-speed must be greater than zero.")
    eval_args = argparse.Namespace(**vars(args))
    # Record the effective count so explicit --num-walkers overrides are
    # grouped correctly in evaluation output.
    eval_args.num_walkers = int(args.num_walkers)
    if args.teleop:
        eval_args.planner = "teleop"

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_stamp)
    os.makedirs(out_dir, exist_ok=True)
    gt_path = os.path.join(out_dir, "npc_gt_stream.jsonl")
    vis_path = os.path.join(out_dir, "visibility_stream.jsonl")
    calib_path = os.path.join(out_dir, "robot_sensor_calibration.json")

    client = carla.Client(args.host, args.port)
    # 20 s default times out under N-way parallel boot (eval_multi_gpu.sh)
    # because every CARLA simultaneously reads Town10HD_Opt from the same
    # SSD. 60 s leaves ample slack on a single-instance run too.
    client.set_timeout(float(os.environ.get("CARLA_CLIENT_TIMEOUT_S", "300")))
    world = connect_world(client, args.map_name, args.load_map)
    orig_settings = world.get_settings()
    use_sync_mode = True
    apply_world_mode(world, orig_settings, sync_mode=use_sync_mode, dt=args.dt)
    print(f"[EP] world_mode=SYNC dt={args.dt}")

    npc = RandomNpcRuntime(
        world=world,
        grid_npz=args.grid_npz,
        flow_points_json=args.flow_points_json,
        target_track_id=args.target_track_id,
        mode=args.flow_mode,
        spawn_z=args.robot_spawn_z,
        rng_seed=args.npc_rng_seed,
    )
    robot_cls = G1RobotRuntime if args.robot_type == "g1" else ScoutRobotRuntime
    robot = robot_cls(world=world, dt=args.dt)
    robot.rescue_lift_z = float(args.robot_rescue_lift_z)
    policy = _build_policy(args)
    vis_checker = InstanceVisibilityChecker(threshold=args.visibility_threshold)
    eval_logger = EvaluationLogger.from_args(
        eval_args, scenario_type="plaza"
    ) if args.enable_evaluation else None
    collision_monitor = None
    termination_reason = None
    if eval_logger is not None:
        print(f"[EVAL] run_dir={eval_logger.run_dir}")

    dbg_vis = None
    if args.debug:
        from followbench.visualization.debug_visualizer import DebugVisualizer  # noqa: PLC0415
        dbg_vis = DebugVisualizer(half_size=30.0)

    spawned = False
    following = False
    teleop_started = False
    paused = False
    paused_at = 0.0
    active_target_track_id = args.target_track_id
    _DIFFICULTIES = list(DIFFICULTY_PRESETS.keys())
    _DIFF_COLORS = {"easy": (80, 220, 80), "hard": (240, 200, 40),
                    "boss": (240, 120, 40), "legend": (220, 60, 60)}
    current_difficulty: str = args.difficulty
    npc.set_difficulty(current_difficulty)
    npc.params["num_walkers"] = int(args.num_walkers)
    if num_walkers_overridden:
        print(f"[EP] num_walkers override={args.num_walkers}")

    # Weather is drawn from a stream derived from the episode seed (paired across
    # policies) with difficulty-dependent weights (boss/legend biased toward
    # dusk/rain/fog).  Without a seed, a fresh unseeded stream is used.
    _weather_rng = weather_mod.seeded_rng(args.npc_rng_seed)

    def _select_and_apply_weather() -> str:
        if args.weather == "random":
            name = weather_mod.sample_for_difficulty(current_difficulty, _weather_rng)
        else:
            name = args.weather
        weather_mod.apply(world, name)
        print(f"[EP] weather → {name.upper()} ({weather_mod.WEATHER_LABELS_CN[name]})")
        return name

    current_weather: str = _select_and_apply_weather()
    weather_history: List[Dict] = [{"tick": 0, "weather": current_weather}]
    difficulty_history: List[Dict] = [{"tick": 0, "difficulty": current_difficulty}]

    _notifs: List[Tuple[str, float]] = []
    _LIVE_VIEWER_PATH = module_path("visualization", "live_viewer.py")
    _debug_proc: List[Optional[subprocess.Popen]] = [None]

    def _notify(msg: str, duration: float = 3.0) -> None:
        _notifs.append((msg, time.time() + duration))

    ui_enabled = (not args.no_ui) and HAS_PYGAME
    if ui_enabled:
        pygame.init()
        panel_w = 400
        panel_h = 260
        hud_h = 170
        win_w = panel_w * 3
        win_h = hud_h + panel_h * 2
        pygame.display.set_mode((win_w, win_h))
        title = "Random Teleoperation" if args.teleop else "Random Episode Manager"
        pygame.display.set_caption(
            f"{title} | G spawn | F start/stop | WASD drive | R reset | Q quit"
        )
        ui_font = pygame.font.SysFont("monospace", 18)
        latest_panels = {
            "rgb_left": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "rgb": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "rgb_right": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "depth": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "instance": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
            "lidar": np.zeros((panel_h, panel_w, 3), dtype=np.uint8),
        }
    elif not args.no_ui:
        print("[WARN] pygame not available, key UI disabled.")

    running = True
    tick = 0
    completed_episodes = 0
    _cam_yaw = 0.0
    _cam_pitch = -20.0

    def on_signal(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)

    gt_f = open(gt_path, "w", encoding="utf-8")
    vis_f = open(vis_path, "w", encoding="utf-8")

    print(f"[EP] flow_mode={args.flow_mode} difficulty={current_difficulty}")
    print(f"[EP] output_dir={out_dir}")
    print("[EP] ready. Press G to spawn, then F to start following.")

    def do_spawn() -> None:
        nonlocal spawned, collision_monitor
        if spawned:
            return
        # Cleanup stale robot actors from previous runs.
        for a in world.get_actors():
            try:
                is_scout = "vehicle.scout" in a.type_id
                is_g1 = "walker.pedestrian.g1" in a.type_id
                is_known_role = a.attributes.get("role_name", "") in {"followbench_robot", "sfm_follow_robot", "scout_follow"}
                if is_scout or is_g1 or is_known_role:
                    a.destroy()
            except Exception:
                pass
        npc.spawn()
        if not npc.agents:
            raise RuntimeError("No NPC spawned, cannot place robot near target.")
        # Tick a few times until walker transforms have actually materialised — CARLA
        # often returns (0,0) for a freshly-spawned walker even after one tick.
        tgt_actor = npc.get_actor_by_track_id(active_target_track_id)
        for _ in range(8):
            step_world(world, use_sync_mode)
            if tgt_actor is None:
                tgt_actor = npc.get_actor_by_track_id(active_target_track_id)
            if tgt_actor is not None:
                _loc = tgt_actor.get_location()
                if abs(_loc.x) + abs(_loc.y) > 1e-3:
                    break
        tgt_init = npc.get_target_by_track_id(active_target_track_id)
        if tgt_actor is None or tgt_init is None:
            raise RuntimeError(f"target {active_target_track_id} missing after spawn")

        robot_tf = resolve_robot_spawn_tf(args, npc=npc, target_track_id=active_target_track_id)
        print(
            f"[EP] robot_spawn at=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f},{robot_tf.location.z:.2f})"
        )
        # Walker positions to avoid in the spawn fallback ring search.
        _spawn_forbidden_xys = [(s.x, s.y) for s in npc.get_states()]
        robot.spawn(robot_tf, roi_map=npc.roi_map,
                    forbidden_xys=_spawn_forbidden_xys, min_clearance_m=0.7)
        step_world(world, use_sync_mode)

        rloc = robot.actor.get_location()
        tloc = tgt_actor.get_location()
        if abs(tloc.x) < 1e-4 and abs(tloc.y) < 1e-4:
            step_world(world, use_sync_mode)
            tloc = tgt_actor.get_location()
        if abs(rloc.x) < 1e-4 and abs(rloc.y) < 1e-4:
            step_world(world, use_sync_mode)
            rloc = robot.actor.get_location()
        dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
        if dist0 > (float(args.desired_distance) + 1.5):
            target_yaw_rad = float(np.arctan2(npc.B[1] - npc.A[1], npc.B[0] - npc.A[0]))
            _correction_other_xys = [(s.x, s.y) for s in npc.get_states() if s.track_id != active_target_track_id]
            new_x, new_y, _ok = follow_position_spawn_xy(
                float(tloc.x), float(tloc.y), target_yaw_rad,
                str(args.follow_position), float(args.desired_distance),
                npc.roi_map, forbidden_xys=_correction_other_xys,
            )
            fix_yaw = float(np.degrees(np.arctan2(float(tloc.y) - new_y, float(tloc.x) - new_x)))
            fix_tf = carla.Transform(
                carla.Location(x=new_x, y=new_y, z=robot_tf.location.z),
                carla.Rotation(yaw=fix_yaw),
            )
            for _ in range(8):
                robot.force_place(fix_tf)
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
                dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
                if dist0 <= (float(args.desired_distance) + 1.5):
                    break
            print(f"[EP] corrected robot_xy=({rloc.x:.2f},{rloc.y:.2f}) dist={dist0:.2f}m")
        if dist0 > (float(args.desired_distance) + 1.5):
            raise RuntimeError(
                f"G1/robot spawn did not materialize near the target after retries: "
                f"robot=({rloc.x:.2f},{rloc.y:.2f}) "
                f"target=({tloc.x:.2f},{tloc.y:.2f}) dist={dist0:.2f}m"
            )
        print(f"[EP] robot_target_init_dist={dist0:.2f}m")

        robot.spawn_sensors(
            image_w=args.sensor_image_w,
            image_h=args.sensor_image_h,
            cam_fov=args.sensor_fov_deg,
            lidar_range=args.sensor_lidar_range,
            lidar_scan_height_agl=args.sensor_lidar_height_agl,
            enable_side_views=True,
        )
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(robot.get_calibration(), f, indent=2)
        if eval_logger is not None:
            eval_logger.write_meta(
                eval_args,
                robot_actor=robot.actor,
                target_actor=tgt_actor,
                extras={
                    "difficulty": current_difficulty,
                    "weather": current_weather,
                    "flow_mode_resolved": npc.flow_mode,
                    "axis_A": list(npc.A),
                    "axis_B": list(npc.B),
                    "axis_len_m": float(npc.axis_len),
                    "rng_seed": int(npc.rng_seed),
                    "num_walkers": len(npc.agents),
                    "npc_speed_range": [float(npc.params["min_speed"]), float(npc.params["max_speed"])],
                    "target_speed": float(npc.target_speed),
                    "lookalike_track_ids": list(npc.lookalike_track_ids),
                    "difficulty_history": list(difficulty_history),
                    "weather_history": list(weather_history),
                    "robot_type": args.robot_type,
                    "sim_speed": args.sim_speed,
                    **({
                        "control_mode": "human_keyboard",
                        "operator_id": args.operator_id,
                        "session_episode": completed_episodes + 1,
                        "session_num_episodes": args.num_episodes,
                    } if args.teleop else {}),
                },
            )
            collision_monitor = HumanCollisionMonitor(
                world,
                robot.actor,
                enabled=not bool(args.disable_collision_sensor),
            )
        npc.set_motion_active(False)
        spawned = True
        if dbg_vis is not None:
            rs = robot.get_state()
            dbg_vis.start(rs.x, rs.y, args.grid_npz)
        print(f"[EP] spawned actors (idle) | walkers={len(npc.agents)} mode={npc.flow_mode}")

        # Spectator looking down on the axis midpoint.
        try:
            cx, cy = npc.axis_mid
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=float(cx), y=float(cy), z=60.0),
                carla.Rotation(pitch=-85.0, yaw=0.0, roll=0.0),
            ))
        except Exception as _e:
            print(f"[EP][WARN] spectator auto-position failed: {_e}")

    def do_reset(*, announce: bool = True) -> None:
        nonlocal spawned, following, teleop_started, tick, collision_monitor, termination_reason
        if spawned:
            npc.destroy()
            robot.destroy()
        if collision_monitor is not None:
            collision_monitor.destroy()
            collision_monitor = None
        if dbg_vis is not None:
            dbg_vis.stop()
        spawned = False
        following = False
        teleop_started = False
        if args.teleop and ui_enabled:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        tick = 0
        termination_reason = None
        vis_checker.reset()
        if announce:
            print("[EP] reset done. Press G to spawn.")

    def finalize_current_eval(reason: Optional[str]) -> None:
        if eval_logger is None:
            return
        result = eval_logger.finalize(reason)
        if result is not None:
            print(f"[EVAL] result: {os.path.join(eval_logger.run_dir, 'eval_result.json')}")
            print(
                f"[EVAL] obstacle_avoidance_success={result.get('obstacle_avoidance_success')} "
                f"search_success={result.get('search_success')} "
                f"TVR={result.get('target_visibility_ratio')}"
            )

    def finish_episode(reason: str) -> None:
        nonlocal completed_episodes, eval_logger, running
        nonlocal auto_spawn_done, auto_follow_done
        nonlocal current_weather, weather_history
        finalize_current_eval(reason)
        completed_episodes += 1
        print(f"[EP] completed episode {completed_episodes}/{args.num_episodes} reason={reason}")
        if completed_episodes >= args.num_episodes:
            running = False
            return
        do_reset(announce=False)
        current_weather = _select_and_apply_weather()
        weather_history = [{"tick": 0, "weather": current_weather}]
        auto_spawn_done = False
        auto_follow_done = False
        eval_logger = EvaluationLogger.from_args(
            eval_args, scenario_type="plaza"
        ) if args.enable_evaluation else None
        if eval_logger is not None:
            print(f"[EVAL] next run_dir={eval_logger.run_dir}")
        print(
            f"[EP] ready for episode {completed_episodes + 1}/{args.num_episodes}. "
            "Press G to spawn, then F to start."
        )

    def do_toggle_follow() -> None:
        nonlocal following, start_t, teleop_started, teleop_start_tick
        nonlocal _cam_yaw, _cam_pitch
        if not spawned:
            return
        following = not following
        if args.teleop and following and not teleop_started:
            start_t = sim_clock()
            teleop_started = True
            teleop_start_tick = tick
        npc.set_motion_active(following)
        policy.reset()
        vis_checker.reset()
        if args.teleop and ui_enabled:
            if following:
                if robot.actor is not None:
                    _cam_yaw = float(robot.actor.get_transform().rotation.yaw)
                _cam_pitch = -20.0
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
            else:
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
                robot.hold_still()
        print(f"[EP] following={'ON' if following else 'OFF'} target={active_target_track_id}")

    # Sim clock: seconds of simulated time (world steps * dt). Used instead of
    # wall clock so auto-spawn/follow timing and duration are reproducible.
    sim_steps = 0

    def sim_clock() -> float:
        return sim_steps * float(args.dt)

    start_t = sim_clock()
    teleop_start_tick = 0
    auto_spawn_done = False
    auto_follow_done = False
    rgb_intrinsics_cached: Optional[dict] = None
    rgb_extr_cached: Optional[dict] = None
    lidar_extr_cached: Optional[dict] = None
    rgb_intrinsics_left_cached: Optional[dict] = None
    rgb_extr_left_cached: Optional[dict] = None
    rgb_intrinsics_right_cached: Optional[dict] = None
    rgb_extr_right_cached: Optional[dict] = None
    last_rgb_np: Optional[np.ndarray] = None
    last_lidar_np: Optional[np.ndarray] = None
    last_depth_np: Optional[np.ndarray] = None
    last_rgb_left_np: Optional[np.ndarray] = None
    last_rgb_right_np: Optional[np.ndarray] = None
    last_depth_left_np: Optional[np.ndarray] = None
    last_depth_right_np: Optional[np.ndarray] = None

    sim_pacer = SimulationPacer(args.dt, args.sim_speed, enabled=args.teleop)

    def step_main_loop(frame_started_at: float) -> None:
        nonlocal sim_steps
        step_world(world, use_sync_mode)
        sim_steps += 1
        sim_pacer.end_frame(frame_started_at)

    try:
        while running:
            frame_started_at = sim_pacer.begin_frame()
            now_t = sim_clock()
            episode_end_reason = None
            duration_active = (not args.teleop) or teleop_started
            duration_elapsed = (
                max(tick - teleop_start_tick, 0) * args.dt
                if args.teleop else now_t - start_t
            )
            if duration_active and duration_elapsed >= args.duration_sec:
                if args.teleop:
                    finish_episode(termination_reason or "duration_limit")
                    if running:
                        step_main_loop(frame_started_at)
                        continue
                break

            elapsed = now_t - start_t
            if args.auto_spawn_sec >= 0.0 and (not auto_spawn_done) and elapsed >= args.auto_spawn_sec:
                do_spawn()
                auto_spawn_done = True
            if args.auto_follow_sec >= 0.0 and (not auto_follow_done) and elapsed >= args.auto_follow_sec:
                if not spawned:
                    do_spawn()
                    auto_spawn_done = True
                if not following:
                    do_toggle_follow()
                auto_follow_done = True

            if ui_enabled:
                key_state = pygame.key.get_pressed()
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        running = False
                    elif e.type == pygame.MOUSEMOTION and args.teleop and following:
                        _cam_yaw = (_cam_yaw + e.rel[0] * MANUAL_MOUSE_SENS) % 360.0
                        _cam_pitch = float(np.clip(
                            _cam_pitch - e.rel[1] * MANUAL_MOUSE_SENS * 0.4,
                            -60.0,
                            10.0,
                        ))
                    elif e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_q:
                            running = False
                        elif e.key == pygame.K_g and not spawned:
                            do_spawn()
                        elif e.key == pygame.K_r:
                            do_reset()
                        elif e.key == pygame.K_f and spawned:
                            do_toggle_follow()
                        # In teleoperation, D is the robot's steer-right key.
                        # Difficulty is fixed at launch so a steering input must
                        # never mutate the experimental condition.
                        elif e.key == pygame.K_d and not args.teleop:
                            idx = _DIFFICULTIES.index(current_difficulty)
                            current_difficulty = _DIFFICULTIES[(idx + 1) % len(_DIFFICULTIES)]
                            npc.set_difficulty(current_difficulty)
                            if num_walkers_overridden:
                                npc.params["num_walkers"] = int(args.num_walkers)
                                _notify(f"Difficulty → {current_difficulty.upper()}  Walkers fixed at {args.num_walkers}")
                            else:
                                args.num_walkers = int(
                                    DIFFICULTY_PRESETS[current_difficulty]["num_walkers"]
                                )
                            current_weather = _select_and_apply_weather()
                            difficulty_history.append({"tick": tick, "difficulty": current_difficulty})
                            weather_history.append({"tick": tick, "weather": current_weather})
                            if eval_logger is not None:
                                eval_logger.update_extras({
                                    "difficulty": current_difficulty,
                                    "weather": current_weather,
                                    "difficulty_history": list(difficulty_history),
                                    "weather_history": list(weather_history),
                                })
                            _notify(f"Difficulty → {current_difficulty.upper()}  Weather → {current_weather}", 4.0)
                        elif e.key == pygame.K_v:
                            if _debug_proc[0] is None or _debug_proc[0].poll() is not None:
                                _debug_proc[0] = subprocess.Popen(
                                    [sys.executable, _LIVE_VIEWER_PATH, "--npz", args.grid_npz])
                                _notify("Debug viewer ON")
                            else:
                                _debug_proc[0].terminate()
                                _debug_proc[0] = None
                                _notify("Debug viewer OFF")
                        elif e.key == pygame.K_SPACE:
                            paused = not paused
                            if paused:
                                paused_at = sim_clock()
                                _notify("PAUSED  (SPACE to resume)", 9999.0)
                                if spawned:
                                    npc.set_motion_active(False)
                                    robot.hold_still()
                            else:
                                start_t += sim_clock() - paused_at
                                _notifs.clear()
                                _notify("RESUMED")
                                if spawned and following:
                                    npc.set_motion_active(True)

                if not spawned:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        line1 = (
                            f"State: spawned={spawned} follow={following} target={active_target_track_id} "
                            f"flow={npc.flow_mode}"
                        )
                        line2 = (
                            "Keys: G spawn  F follow  D difficulty  V debug  R reset  SPACE pause  Q quit"
                            if not args.teleop else
                            "Keys: G spawn  F follow  WASD drive  Mouse look  R reset  SPACE pause  Q quit"
                        )
                        diff_label = (
                            f"DIFFICULTY: {current_difficulty.upper()}  "
                            f"({'LOCKED BY LAUNCH' if args.teleop else 'D to cycle'})"
                        )
                        for idx, text in enumerate([line1, line2]):
                            img = ui_font.render(text, True, (230, 235, 240))
                            surf.blit(img, (16, 20 + idx * 36))
                        diff_surf = ui_font.render(diff_label, True, _DIFF_COLORS[current_difficulty])
                        surf.blit(diff_surf, (16, 20 + 2 * 36))
                        now_n = time.time()
                        active = [(m, e) for m, e in _notifs if e > now_n]
                        _notifs[:] = active
                        for ni, (msg, _) in enumerate(reversed(active[-4:])):
                            ns = ui_font.render(f"▶ {msg}", True, (255, 230, 80))
                            surf.blit(ns, (16, surf.get_height() - 28 - ni * 26))
                        pygame.display.flip()

            if paused:
                if spawned:
                    robot.hold_still()
                step_main_loop(frame_started_at)
                continue

            if spawned:
                npc.step(
                    dt=args.dt,
                    robot_actor=robot.actor if robot.actor is not None else None,
                    target_track_id=active_target_track_id,
                )
                if npc.is_episode_done() and following:
                    if termination_reason is None:
                        termination_reason = "target_completed_route"
                    following = False
                    npc.set_motion_active(False)
                    robot.hold_still()
                    print(f"[EP] episode DONE  reason={termination_reason}  (target reached B)")
                    episode_end_reason = termination_reason
                npc_states = npc.get_states()
                target_state = next((s for s in npc_states if s.track_id == active_target_track_id), None)
                target_actor = npc.get_actor_by_track_id(active_target_track_id)
                robot_state = robot.get_state()
                v_cmd = None
                w_cmd = None
                planner_cost_ms = None
                policy_debug_info = {}

                visible = False
                pix_count = 0
                inst_img = robot.get_sensor_data("instance")
                inst_img_left = robot.get_sensor_data("instance_left")
                inst_img_right = robot.get_sensor_data("instance_right")
                rgb_img = robot.get_sensor_data("rgb")
                depth_img = robot.get_sensor_data("depth")
                lidar_img = robot.get_sensor_data("lidar")
                rgb_img_left = robot.get_sensor_data("rgb_left")
                rgb_img_right = robot.get_sensor_data("rgb_right")
                depth_img_left = robot.get_sensor_data("depth_left")
                depth_img_right = robot.get_sensor_data("depth_right")
                if target_actor is not None and "instance" in robot.sensors:
                    visible, pix_count = vis_checker.evaluate_multi(
                        views=[
                            (inst_img, robot.sensors.get("instance")),
                            (inst_img_left, robot.sensors.get("instance_left")),
                            (inst_img_right, robot.sensors.get("instance_right")),
                        ],
                        target_actor=target_actor,
                        image_w=args.sensor_image_w,
                        image_h=args.sensor_image_h,
                        fov_deg=args.sensor_fov_deg,
                    )
                if ui_enabled:
                    r = _to_rgb(rgb_img)
                    rl = _to_rgb(rgb_img_left)
                    rr = _to_rgb(rgb_img_right)
                    d = _depth_to_rgb(depth_img)
                    ins = _instance_to_rgb(inst_img)
                    lbev = _lidar_to_bev(lidar_img, w=400, h=260)
                    _dbg_fn = getattr(policy, "get_debug_info", None)
                    _dbg_info = _dbg_fn() if callable(_dbg_fn) else {}
                    bboxes_by_view = _dbg_info.get("track_bboxes_by_view") or {}
                    _stale = int(_dbg_info.get("track_bboxes_age", 0))
                    if r is not None:
                        r = _draw_bboxes_on(r, bboxes_by_view.get("front"), _stale)
                        latest_panels["rgb"] = cv2.resize(r, (400, 260))
                    if rl is not None:
                        rl = _draw_bboxes_on(rl, bboxes_by_view.get("left"), _stale)
                        latest_panels["rgb_left"] = cv2.resize(rl, (400, 260))
                    if rr is not None:
                        rr = _draw_bboxes_on(rr, bboxes_by_view.get("right"), _stale)
                        latest_panels["rgb_right"] = cv2.resize(rr, (400, 260))
                    if d is not None:
                        latest_panels["depth"] = cv2.resize(d, (400, 260))
                    if ins is not None:
                        latest_panels["instance"] = cv2.resize(ins, (400, 260))
                    if lbev is not None:
                        latest_panels["lidar"] = lbev

                if args.teleop and following:
                    v_cmd, w_cmd = _apply_manual_robot_control(robot, key_state)
                    policy_debug_info = {"mode": "teleop"}
                    if robot.actor is not None:
                        _update_manual_spectator(
                            world.get_spectator(),
                            robot.actor,
                            _cam_yaw,
                            _cam_pitch,
                        )
                elif target_state is not None and following:
                    target_velocity_yaw_rad = yaw_rad_from_velocity(
                        target_state.vx, target_state.vy, fallback_deg=target_state.yaw_deg,
                    )
                    target_actor_yaw_rad = float(np.deg2rad(target_state.yaw_deg))
                    if rgb_intrinsics_cached is None:
                        calib = robot.get_calibration()
                        rgb_calib = calib.get("rgb", {})
                        lidar_calib = calib.get("lidar", {})
                        rgb_left_calib = calib.get("rgb_left", {})
                        rgb_right_calib = calib.get("rgb_right", {})
                        rgb_intrinsics_cached = rgb_calib.get("intrinsics")
                        rgb_extr_cached = rgb_calib.get("extrinsics_robot_to_sensor")
                        lidar_extr_cached = lidar_calib.get("extrinsics_robot_to_sensor")
                        rgb_intrinsics_left_cached = rgb_left_calib.get("intrinsics")
                        rgb_extr_left_cached = rgb_left_calib.get("extrinsics_robot_to_sensor")
                        rgb_intrinsics_right_cached = rgb_right_calib.get("intrinsics")
                        rgb_extr_right_cached = rgb_right_calib.get("extrinsics_robot_to_sensor")
                    rgb_np = _to_rgb(rgb_img)
                    if rgb_np is not None:
                        last_rgb_np = rgb_np
                    lidar_np = _lidar_to_points(lidar_img)
                    if lidar_np is not None:
                        last_lidar_np = lidar_np
                    depth_np = _depth_to_meters(depth_img)
                    if depth_np is not None:
                        last_depth_np = depth_np
                    rgb_left_np = _to_rgb(rgb_img_left)
                    if rgb_left_np is not None:
                        last_rgb_left_np = rgb_left_np
                    rgb_right_np = _to_rgb(rgb_img_right)
                    if rgb_right_np is not None:
                        last_rgb_right_np = rgb_right_np
                    depth_left_np = _depth_to_meters(depth_img_left)
                    if depth_left_np is not None:
                        last_depth_left_np = depth_left_np
                    depth_right_np = _depth_to_meters(depth_img_right)
                    if depth_right_np is not None:
                        last_depth_right_np = depth_right_np
                    obs = FollowObservation(
                        tick=tick,
                        dt=args.dt,
                        robot=robot_state,
                        target=target_state,
                        npcs=npc_states,
                        target_visible=visible,
                        target_pixel_count=pix_count,
                        extras={
                            "target_yaw_rad": target_actor_yaw_rad,
                            "target_actor_yaw_rad": target_actor_yaw_rad,
                            "target_velocity_yaw_rad": target_velocity_yaw_rad,
                            "target_yaw_source": "actor",
                        },
                        rgb_image=last_rgb_np,
                        depth_image=last_depth_np,
                        lidar_points=last_lidar_np,
                        rgb_intrinsics=rgb_intrinsics_cached,
                        rgb_extrinsics_robot_to_sensor=rgb_extr_cached,
                        lidar_extrinsics_robot_to_sensor=lidar_extr_cached,
                        rgb_image_left=last_rgb_left_np,
                        rgb_image_right=last_rgb_right_np,
                        depth_image_left=last_depth_left_np,
                        depth_image_right=last_depth_right_np,
                        rgb_intrinsics_left=rgb_intrinsics_left_cached,
                        rgb_intrinsics_right=rgb_intrinsics_right_cached,
                        rgb_extrinsics_left_robot_to_sensor=rgb_extr_left_cached,
                        rgb_extrinsics_right_robot_to_sensor=rgb_extr_right_cached,
                    )
                    _t_policy = time.perf_counter()
                    action = policy.act(obs)
                    planner_cost_ms = (time.perf_counter() - _t_policy) * 1000.0
                    _dbg_fn_eval = getattr(policy, "get_debug_info", None)
                    policy_debug_info = _dbg_fn_eval() if callable(_dbg_fn_eval) else {}
                    v_cmd = float(action.v_mps)
                    w_cmd = float(action.w_radps)
                    robot.apply_velocity_command(v_cmd, w_cmd)
                else:
                    robot.hold_still()

                eval_clearance = None
                eval_collision = False
                eval_collision_source = None
                eval_termination_triggered = False
                if eval_logger is not None:
                    eval_clearance = eval_logger.compute_clearance(
                        robot_state, target_state, npc_states, active_target_track_id,
                    )
                    actor_to_track = actor_track_lookup(npc_states)
                    sensor_events = (
                        enrich_sensor_collision_events(
                            collision_monitor.drain(),
                            tick=tick, time_s=now_t - start_t,
                            episode_active=following,
                            actor_to_track=actor_to_track,
                            target_track_id=active_target_track_id,
                        )
                        if collision_monitor is not None
                        else []
                    )
                    geometry_event = (
                        make_geometry_collision_event(
                            tick, now_t - start_t, eval_clearance,
                            active_target_track_id,
                            margin=float(args.human_collision_margin),
                        )
                        if following
                        else None
                    )
                    eval_collision_events = list(sensor_events)
                    if geometry_event is not None:
                        eval_collision_events.append(geometry_event)
                    if eval_collision_events:
                        eval_logger.write_collision_events(eval_collision_events)
                        eval_collision = True
                        eval_collision_source = eval_collision_events[0].get("source")
                        if following and bool(args.terminate_on_human_collision):
                            termination_reason = "human_collision"
                            eval_termination_triggered = True
                    eval_logger.write_step(
                        tick=tick, time_s=now_t - start_t,
                        episode_active=following, paused=paused,
                        target_track_id=active_target_track_id,
                        robot_state=robot_state, target_state=target_state, npc_states=npc_states,
                        visible=visible, pixel_count=pix_count,
                        visibility_threshold=args.visibility_threshold,
                        command=None if v_cmd is None else {"v_mps": v_cmd, "w_radps": w_cmd},
                        planner_cost_ms=planner_cost_ms,
                        debug_info=policy_debug_info,
                        collision=eval_collision,
                        collision_source=eval_collision_source,
                        termination_triggered=eval_termination_triggered,
                        clearance=eval_clearance,
                    )
                    if eval_termination_triggered:
                        print("[EVAL] terminating episode: human_collision")
                        npc.set_motion_active(False)
                        robot.hold_still()
                        episode_end_reason = termination_reason

                if dbg_vis is not None and robot_state is not None:
                    _dbg = getattr(policy, "get_debug_info", None)
                    _dbg_info = _dbg() if callable(_dbg) else {}
                    _lidar_world = debug_lidar_world_points(
                        robot, robot_state, lidar_img, last_lidar_np, lidar_extr_cached,
                    )
                    dbg_vis.update(
                        tick=tick, robot_state=robot_state, npc_states=npc_states,
                        active_target_id=active_target_track_id,
                        obstacles=_dbg_info.get("obstacles"),
                        traj_points=_dbg_info.get("traj_points"),
                        global_path=_dbg_info.get("global_path"),
                        lidar_points_world=_lidar_world,
                    )

                gt_f.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": now_t - start_t,
                            "target_track_id": active_target_track_id,
                            "robot": {"x": robot_state.x, "y": robot_state.y, "z": robot_state.z,
                                      "yaw_rad": robot_state.yaw_rad, "speed": robot_state.speed},
                            "npcs": [s.__dict__ for s in npc_states],
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                vis_f.write(
                    json.dumps(
                        {
                            "tick": tick, "time_s": now_t - start_t,
                            "target_track_id": active_target_track_id,
                            "target_visible": bool(visible),
                            "target_pixel_count": int(pix_count),
                            "threshold": int(args.visibility_threshold),
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        line1 = (
                            f"State: spawned={spawned} follow={following} target={active_target_track_id} "
                            f"flow={npc.flow_mode} walkers={len(npc.agents)}"
                        )
                        line2 = (
                            "Keys: G spawn  F follow  D difficulty  V debug  R reset  SPACE pause  Q quit"
                            if not args.teleop else
                            "Keys: WASD drive  Mouse look  F pause  R reset  SPACE pause  Q quit"
                        )
                        line3 = (
                            f"Visibility: pix={pix_count} thr={args.visibility_threshold} visible={int(visible)}"
                            f"  | weather={current_weather}"
                        )
                        diff_label = (
                            f"DIFFICULTY: {current_difficulty.upper()}  "
                            f"({'LOCKED BY LAUNCH' if args.teleop else 'D to cycle'})"
                        )
                        for idx, text in enumerate([line1, line2, line3]):
                            img = ui_font.render(text, True, (230, 235, 240))
                            surf.blit(img, (16, 16 + idx * 34))
                        diff_surf = ui_font.render(diff_label, True, _DIFF_COLORS[current_difficulty])
                        surf.blit(diff_surf, (16, 16 + 3 * 34))
                        now_n = time.time()
                        active = [(m, e) for m, e in _notifs if e > now_n]
                        _notifs[:] = active
                        for ni, (msg, _) in enumerate(reversed(active[-4:])):
                            ns = ui_font.render(f"▶ {msg}", True, (255, 230, 80))
                            surf.blit(ns, (16, surf.get_height() - 28 - ni * 26))
                        labels = [
                            ("RGB Left", 0, 0, "rgb_left"),
                            ("RGB Front", 1, 0, "rgb"),
                            ("RGB Right", 2, 0, "rgb_right"),
                            ("Depth", 0, 1, "depth"),
                            ("Instance", 1, 1, "instance"),
                            ("LiDAR BEV", 2, 1, "lidar"),
                        ]
                        base_y = 170
                        for title, cx, cy, key in labels:
                            x = cx * 400
                            y = base_y + cy * 260
                            arr = latest_panels[key]
                            surf.blit(pygame.surfarray.make_surface(arr.transpose(1, 0, 2)), (x, y))
                            pygame.draw.rect(surf, (60, 60, 60), pygame.Rect(x, y, 400, 260), 1)
                            t = ui_font.render(title, True, (255, 220, 0))
                            surf.blit(t, (x + 6, y + 6))
                        pygame.display.flip()

            if episode_end_reason is not None:
                if args.teleop:
                    finish_episode(episode_end_reason)
                else:
                    break
                if running:
                    step_main_loop(frame_started_at)
                    continue

            step_main_loop(frame_started_at)
            tick += 1
    finally:
        if collision_monitor is not None:
            collision_monitor.destroy()
        gt_f.close()
        vis_f.close()
        npc.destroy()
        robot.destroy()
        world.apply_settings(orig_settings)
        if _debug_proc[0] is not None:
            try:
                _debug_proc[0].terminate()
            except Exception:
                pass
        if ui_enabled:
            pygame.quit()
        if dbg_vis is not None:
            dbg_vis.stop()
        if eval_logger is not None:
            finalize_current_eval(termination_reason)
        print("[EP] cleanup done.")
        print(f"[EP] GT stream: {gt_path}")
        print(f"[EP] visibility stream: {vis_path}")
        print(f"[EP] calibration: {calib_path}")


if __name__ == "__main__":
    main()
