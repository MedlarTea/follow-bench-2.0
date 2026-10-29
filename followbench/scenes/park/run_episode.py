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

SCENE = "park"

from followbench.paths import module_path, scene_asset, scene_runs_dir  # noqa: E402


from followbench.sim.core_types import FollowObservation, RobotState


from followbench.visualization.lidar_vis import debug_lidar_world_points, lidar_to_bev as _lidar_to_bev
from followbench.scenes.park.npc_runtime import NpcRuntime, PassageQueueConfig
from followbench.sim.robot_runtime import ScoutRobotRuntime
from followbench.sim.humanoid_runtime import G1RobotRuntime
from followbench.sim.visibility_instance import InstanceVisibilityChecker
from followbench.sim.visibility_instance import _target_bbox_roi as _target_bbox_roi_xyxy
import followbench.sim.weather as weather_mod
from followbench.evaluation.collision import HumanCollisionMonitor, actor_track_lookup
from followbench.evaluation.logging import (
    EvaluationLogger,
    enrich_sensor_collision_events,
    make_geometry_collision_event,
)
from followbench.evaluation.schemas import add_evaluation_args
from followbench.evaluation.sim_clock import SimulationPacer
from followbench.sim.carla_world import connect_world  # noqa: E402

# Optional park-specific metrics plug-in (not shipped with the release).
try:
    from followbench.scenes.park.park_metrics import ParkMetrics
    _HAS_PARK_METRICS = True
except ImportError:
    _HAS_PARK_METRICS = False

try:
    import pygame

    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False


MANUAL_WALKER_SPEED_MPS = 1.8
MANUAL_ROBOT_V_MPS      = 2.0
MANUAL_ROBOT_W_RADPS    = 2.2
MANUAL_SPRINT_MULT      = 1.8
# 3rd-person camera
MANUAL_CAM_DIST         = 4.5   # metres behind actor
MANUAL_CAM_HEIGHT       = 2.0   # metres above actor
MANUAL_MOUSE_SENS       = 0.20  # degrees per pixel


class TargetDistanceMonitor:
    """Fail the episode when the robot stays far from the target for too long."""

    def __init__(
        self,
        *,
        warmup_sec: float,
        fail_dist_m: float,
        fail_sec: float,
        fail_steps: int = 0,
    ) -> None:
        self.warmup_sec = max(float(warmup_sec), 0.0)
        self.fail_dist_m = max(float(fail_dist_m), 0.0)
        self.fail_sec = max(float(fail_sec), 0.0)
        self.fail_steps = max(int(fail_steps), 0)
        self.duration_s = 0.0
        self.max_duration_s = 0.0
        self.steps = 0
        self.max_steps = 0

    def reset(self) -> None:
        self.duration_s = 0.0
        self.max_duration_s = 0.0
        self.steps = 0
        self.max_steps = 0

    def update(self, *, dt: float, eval_time_s: float, following: bool, distance_m: Optional[float]) -> dict:
        payload = {
            "enabled": True,
            "distance_m": distance_m,
            "warmup_sec": float(self.warmup_sec),
            "fail_dist_m": float(self.fail_dist_m),
            "fail_sec": float(self.fail_sec),
            "fail_steps": int(self.fail_steps),
            "duration_s": float(self.duration_s),
            "max_duration_s": float(self.max_duration_s),
            "steps": int(self.steps),
            "max_steps": int(self.max_steps),
            "active": False,
            "termination_triggered": False,
            "termination_reason": None,
        }
        if not following or distance_m is None or eval_time_s < self.warmup_sec or distance_m <= self.fail_dist_m:
            self.duration_s = 0.0
            self.steps = 0
            payload["duration_s"] = 0.0
            payload["steps"] = 0
            return payload
        self.duration_s += max(float(dt), 0.0)
        self.steps += 1
        self.max_duration_s = max(self.max_duration_s, self.duration_s)
        self.max_steps = max(self.max_steps, self.steps)
        payload["duration_s"] = float(self.duration_s)
        payload["max_duration_s"] = float(self.max_duration_s)
        payload["steps"] = int(self.steps)
        payload["max_steps"] = int(self.max_steps)
        payload["active"] = True
        # Step threshold takes priority when configured (>0); otherwise fall back to seconds.
        triggered_by_steps = self.fail_steps > 0 and self.steps > self.fail_steps
        triggered_by_sec = self.fail_steps <= 0 and self.duration_s > self.fail_sec
        if triggered_by_steps or triggered_by_sec:
            payload["termination_triggered"] = True
            payload["termination_reason"] = "target_too_far"
        return payload


def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Episode manager: decoupled NPC + robot + follow policy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--load-map", action="store_true")
    p.add_argument("--map-name", type=str, default="Town10HD_Opt")
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--duration-sec", type=float, default=300.0)
    p.add_argument("--num-walkers", type=int, default=35)
    p.add_argument(
        "--npc-rng-seed",
        type=int,
        default=None,
        help="NPC random seed. Omit for a fresh seed each run; set an integer to reproduce.",
    )
    p.add_argument(
        "--npc-flow-mode",
        choices=["ab_cd", "pair_12_34", "pair_13_24", "free_3"],
        default="ab_cd",
        help="NPC flow mode. pair_12_34=(1<->2,3<->4), pair_13_24=(1<->3,2<->4), free_3=3-point random triangle.",
    )
    p.add_argument("--npc-min-speed", type=float, default=0.7, help="Initial NPC min speed (m/s)")
    p.add_argument("--npc-max-speed", type=float, default=1.3, help="Initial NPC max speed (m/s)")
    p.add_argument("--npc-navigation-mode", choices=["direct", "astar"], default="direct")
    p.add_argument(
        "--npc-strict-flow-endpoints",
        action="store_true",
        help="Spawn at fixed flow endpoints and navigate to paired endpoints without random retargeting",
    )
    p.add_argument("--npc-replan-on-avoid", action="store_true")
    p.add_argument("--npc-avoid-replan-threshold", type=float, default=1.15)
    p.add_argument("--npc-astar-replan-cooldown-sec", type=float, default=0.8)
    p.add_argument("--npc-spawn-retries", type=int, default=3,
                   help="How many full NPC crowd respawn attempts to allow when the target is missing.")
    p.add_argument("--npc-spawn-attempts-per-agent", type=int, default=12,
                   help="How many spawn candidates to try for each NPC before counting it as failed.")
    p.add_argument("--npc-spawn-clearance-m", type=float, default=0.9,
                   help="Minimum initial spacing between spawned NPCs.")
    p.add_argument(
        "--use-orca-npc",
        action="store_true",
        help="Use ORCA velocity solver for NPC crowd instead of SFM repulsion. "
             "Produces more natural, orderly queuing through narrow passages.",
    )
    p.add_argument("--orca-tau", type=float, default=3.0, help="ORCA time horizon (seconds).")
    p.add_argument("--orca-neighbor-dist", type=float, default=5.0, help="ORCA neighbour detection radius (m).")
    p.add_argument(
        "--use-passage-queue",
        action="store_true",
        help="Enable passage queuing state machine: NPCs loop spawn→approach→ORCA through door→dest.",
    )
    p.add_argument("--passage-door-y", type=float, default=149.0, help="Y-coordinate of door centre-line.")
    p.add_argument("--passage-door-x", type=float, default=-39.0, help="X-coordinate of door centre.")
    p.add_argument("--passage-approach-a-y", type=float, default=147.5, help="Approach waypoint Y on A side.")
    p.add_argument("--passage-approach-b-y", type=float, default=150.5, help="Approach waypoint Y on B side.")
    p.add_argument("--passage-trigger-dist", type=float, default=3.0, help="Distance to approach point that triggers PASSING.")
    p.add_argument("--passage-exit-y-atob", type=float, default=150.5, help="Y threshold for A→B crossing completion.")
    p.add_argument("--passage-exit-y-btoa", type=float, default=146.5, help="Y threshold for B→A crossing completion.")
    p.add_argument("--passage-max-passing", type=int, default=1, help="Max simultaneous NPCs in PASSING (throat is 0.5 m wide).")
    p.add_argument("--passage-min-dist-after-cross", type=float, default=6.0, help="NPC must be this far from door before gate re-opens.")
    p.add_argument("--target-track-id", type=str, default="N01")
    p.add_argument("--visibility-threshold", type=int, default=400)
    p.add_argument("--grid-npz", type=str, default=scene_asset(SCENE, "gridmap_roi.npz"))
    p.add_argument("--flow-points-json", type=str, default=scene_asset(SCENE, "gridmap_roi_flow_points.json"))
    p.add_argument("--robot-spawn-x", type=float, default=-60.0)
    p.add_argument("--robot-spawn-y", type=float, default=5.0)
    p.add_argument("--robot-spawn-z", type=float, default=0.5)
    p.add_argument("--robot-spawn-yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--robot-spawn-mode",
        choices=["fixed", "roi_random", "near_target"],
        default="near_target",
        help="Robot spawn strategy. near_target spawns inside ROI near target NPC.",
    )
    p.add_argument("--robot-type", choices=["scout", "g1"], default="scout")
    p.add_argument("--robot-spawn-min-dist", type=float, default=2.5, help="Min robot-target spawn distance (near_target)")
    p.add_argument("--robot-spawn-max-dist", type=float, default=8.0, help="Max robot-target spawn distance (near_target)")
    p.add_argument("--robot-spawn-retries", type=int, default=10,
                   help="How many constrained robot spawn candidates to try before failing the episode init.")
    p.add_argument("--robot-spawn-clearance-m", type=float, default=1.1,
                   help="Minimum initial spacing between the robot and background NPCs.")
    p.add_argument("--sensor-image-w", type=int, default=800)
    p.add_argument("--sensor-image-h", type=int, default=600)
    p.add_argument("--sensor-fov-deg", type=float, default=90.0)
    p.add_argument("--sensor-lidar-range", type=float, default=30.0)
    p.add_argument("--sensor-lidar-height-agl", type=float, default=0.50,
                   help="Single-line LiDAR scan plane height above ground (m).")
    p.add_argument("--robot-rescue-lift-z", type=float, default=0.20)
    p.add_argument(
        "--planner",
        choices=["pid", "dwa_traj", "dwa_traj_depth_tpt", "sfm", "rda", "rda_lidar", "gt_planner", "rda_traj", "rda_search", "rda_depth_tpt", "bso_hfc", "trackvla", "oa_vat", "custom"],
        default="pid",
        help="Follow policy: pid (default), dwa_traj, dwa_traj_depth_tpt, sfm, rda, rda_lidar, rda_traj, rda_search, rda_depth_tpt, bso_hfc, trackvla, oa_vat",
    )
    from followbench.interfaces.policy_factory import add_policy_factory_args  # noqa: PLC0415
    add_policy_factory_args(p)
    p.add_argument("--follow-position", choices=["back", "left_side", "right_side"], default="back")
    p.add_argument("--desired-distance", type=float, default=1.5)
    p.add_argument(
        "--rda-search-use-camera-visibility",
        action="store_true",
        help="Let rda_search enter search modes from camera visibility. Default keeps GT-target runs in follow mode.",
    )
    p.add_argument(
        "--rda-search-disable-target-protection",
        action="store_true",
        help="Disable the side-follow target protected zone injected for rda_search.",
    )
    p.add_argument(
        "--rda-traj-disable-lidar-obstacles",
        action="store_true",
        help="Disable lidar/NPC obstacle injection for rda_traj (legacy behavior).",
    )
    p.add_argument(
        "--use-perception",
        action="store_true",
        help="Wrap the chosen --planner with a YOLO+depth+ReID perception "
             "frontend (multi-view). Target and NPC positions come from the "
             "tracker instead of GT. Compatible with every planner except "
             "rda_depth_tpt (which already has its own perception path). "
             "The --reid-* and --lost-policy flags below are only consulted "
             "when --use-perception is set.",
    )
    p.add_argument(
        "--reid-mode",
        choices=["basic", "kpr"],
        default="basic",
        help="Appearance ReID backbone (only when --use-perception): basic "
             "(ResNet, lightweight) or kpr (SOLIDER-Swin, occlusion-robust).",
    )
    p.add_argument(
        "--reid-kpr-config",
        type=str,
        default="kpr_occ_duke_test",
        help="KPR config name (only used when --reid-mode=kpr).",
    )
    p.add_argument(
        "--reid-device",
        type=str,
        default="auto",
        help="Device for the ReID backbone (auto/cuda/cpu).",
    )
    p.add_argument(
        "--lost-policy",
        choices=["gt_fallback", "brake", "last_known"],
        default="last_known",
        help="Behaviour when the perception frontend reports target lost: "
             "gt_fallback (use GT target), brake (zero command), "
             "last_known (drive toward last seen position; default). "
             "rda_search only accepts last_known.",
    )
    p.add_argument(
        "--kinematics-mode",
        choices=["ema", "kf"],
        default="kf",
        help="Per-track velocity estimator used by the depth tracker "
             "(only when --use-perception is set). "
             "'kf' (default) = constant-velocity Kalman filter (filterpy) — smoother "
             "vx/vy; recommended when downstream planners predict target trajectories "
             "(e.g. rda_traj + --use-perception). "
             "'ema' = finite-difference + EMA (lighter, use when vx/vy is not critical).",
    )
    p.add_argument(
        "--kf-pos-sigma",
        type=float,
        default=0.20,
        help="(--kinematics-mode=kf only) Measurement noise std-dev in metres. "
             "Increase to trust the depth tracker less and rely more on the "
             "CV-model prediction. Default 0.20 m.",
    )
    p.add_argument(
        "--kf-vel-sigma-q",
        type=float,
        default=0.05,
        help="(--kinematics-mode=kf only) Process noise std-dev on velocity "
             "(m/s). Reflects how quickly pedestrian velocity can change. "
             "Default 0.05 m/s.",
    )
    p.add_argument(
        "--trackvla-instruction",
        type=str,
        default="Follow the person",
        help="(planner=trackvla only) Natural-language instruction fed to the "
             "VLM each tick.",
    )
    p.add_argument(
        "--trackvla-ckpt",
        type=str,
        default=None,
        help="(planner=trackvla only) Override the default .pt checkpoint path "
             "under data/trackvla/ckpts/. Default: epoch04_step024000.pt.",
    )
    p.add_argument(
        "--trackvla-auto-caption",
        action="store_true",
        default=True,
        help="(planner=trackvla only) On the first frame with a GT target "
             "bbox, crop the person and run Qwen3-VL to generate "
             "'Follow the person wearing <clothes>.' Disable with "
             "--trackvla-no-auto-caption.",
    )
    p.add_argument(
        "--trackvla-no-auto-caption",
        dest="trackvla_auto_caption",
        action="store_false",
        help="(planner=trackvla only) Force the plain --trackvla-instruction.",
    )
    p.add_argument(
        "--trackvla-caption-ckpt",
        type=str,
        default=None,
        help="(planner=trackvla only) Qwen3-VL checkpoint dir; "
             "default data/trackvla/LLM_hf/qwen3-vl-4b-instruct.",
    )
    p.add_argument(
        "--trackvla-caption-device",
        type=str,
        default=None,
        help="(planner=trackvla only) Captioner device (cuda/cpu). "
             "Default: cuda if available; VRAM released after first frame.",
    )
    p.add_argument("--oa-vat-text-query", type=str, default="person",
                   help="(planner=oa_vat) YOLOe text prompt for first-frame and re-acq detection")
    p.add_argument("--oa-vat-kf-mode", type=str, default="confaware",
                   choices=["none", "standard", "confaware"],
                   help="(planner=oa_vat) Kalman filter mode for occlusion gap-fill")
    p.add_argument("--oa-vat-online-enhance", action="store_true", default=True,
                   help="(planner=oa_vat) EMA-update reference DINOv3 feature every N frames")
    p.add_argument("--oa-vat-no-online-enhance", dest="oa_vat_online_enhance",
                   action="store_false", help="(planner=oa_vat) disable online enhance")
    p.add_argument("--oa-vat-dinov3", type=str, default="dinov3_vitb16",
                   choices=["dinov3_vitb16", "dinov3_vits16"],
                   help="(planner=oa_vat) DINOv3 backbone")
    p.add_argument("--oa-vat-match-thresh", type=float, default=0.4,
                   help="(planner=oa_vat) DINOv3 cosine threshold for first-frame match")
    p.add_argument("--oa-vat-reacq-thresh", type=float, default=0.5,
                   help="(planner=oa_vat) DINOv3 cosine threshold for re-acquisition")
    p.add_argument("--oa-vat-yolo-conf", type=float, default=0.1,
                   help="(planner=oa_vat) YOLOe confidence threshold")
    p.add_argument("--oa-vat-target-area-ratio", type=float, default=0.12,
                   help="(planner=oa_vat) Desired bbox area / frame area for distance control")
    p.add_argument(
        "--teleop",
        action="store_true",
        help="Use the existing keyboard manual-robot control for the full episode and log it as planner=teleop.",
    )
    p.add_argument(
        "--operator-id",
        type=str,
        default="anonymous",
        help="Anonymous operator identifier stored in episode_meta.json when --teleop is enabled.",
    )
    p.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of teleoperation episodes to run in the same CARLA/UI session.",
    )
    p.add_argument("--sim-speed", type=float, default=1.0,
                   help="Teleop simulation speed relative to wall clock (1.0=realtime).")
    p.add_argument("--no-ui", action="store_true", help="Disable pygame key UI")
    p.add_argument("--debug", action="store_true", help="Enable 2D matplotlib top-down debug visualizer")
    p.add_argument("--auto-spawn-sec", type=float, default=-1.0, help="Auto trigger spawn after N seconds")
    p.add_argument("--auto-follow-sec", type=float, default=-1.0, help="Auto trigger follow after N seconds")
    p.add_argument("--output-dir", type=str, default=scene_runs_dir(SCENE))
    p.add_argument("--draw-debug", action="store_true")
    p.add_argument("--draw-laser", action="store_true", help="Draw a laser line from robot to target while following")
    p.add_argument("--enable-metrics", action="store_true", help="Enable park scenario metrics logging")
    p.add_argument("--async-mode", action="store_true", help="Run CARLA world in asynchronous mode")
    p.add_argument(
        "--async-after-spawn",
        action="store_true",
        help="When async-mode is enabled, keep sync mode until initial spawn finishes",
    )
    p.add_argument(
        "--use-scheduled-flow",
        action="store_true",
        help="After spawn, configure NPCs with pre-planned A* paths and corner batch gate.",
    )
    p.add_argument("--scheduled-flow-dwell-s", type=float, default=0.0,
                   help="Dwell time at endpoint before cycling back (scheduled flow).")
    p.add_argument(
        "--difficulty",
        choices=["easy", "hard", "boss", "legend"],
        default="easy",
        help="Crowd difficulty preset (easy/hard/boss/legend). Change at runtime with D key.",
    )
    p.add_argument(
        "--weather",
        choices=["random"] + weather_mod.CHOICES,
        default="random",
        help="Weather preset. 'random' re-rolls each time difficulty is selected.",
    )
    add_evaluation_args(p)
    p.add_argument("--robot-target-far-fail-dist-m", type=float, default=3.0)
    p.add_argument("--robot-target-far-fail-sec", type=float, default=2.0)
    p.add_argument(
        "--robot-target-far-fail-steps",
        type=int,
        default=100,
        help="Long-term target loss threshold N_lost: the episode fails once the robot-target distance exceeds "
             "--robot-target-far-fail-dist-m for more than N_lost consecutive steps (target_too_far). "
             "Set <=0 to fall back to --robot-target-far-fail-sec.",
    )
    p.add_argument("--robot-target-far-warmup-sec", type=float, default=0.0)
    return p.parse_args()


def make_robot_spawn_tf_fixed(args: argparse.Namespace) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=args.robot_spawn_x, y=args.robot_spawn_y, z=args.robot_spawn_z),
        carla.Rotation(yaw=args.robot_spawn_yaw_deg),
    )


_FOLLOW_OFFSET_RAD = {
    "back":       math.pi,
    "front":      0.0,
    "left_side":  -math.pi / 2.0,
    "right_side": math.pi / 2.0,
}


def active_camera_view_for_follow_position(follow_position: str) -> str:
    if str(follow_position) == "left_side":
        return "right"
    if str(follow_position) == "right_side":
        return "left"
    return "front"


def follow_position_spawn_xy(target_x, target_y, target_yaw_rad,
                              follow_position, desired_distance,
                              roi_map, forbidden_xys=None,
                              min_clearance_m: float = 0.7,
                              search_max_radius: float = 2.0):
    """Pick a spawn (x, y) at the requested follow offset, avoiding existing
    walkers. ``forbidden_xys`` is a list of (x, y) world coords the candidate
    must stay at least ``min_clearance_m`` away from."""
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


def target_route_yaw_rad(npc: NpcRuntime, target_track_id: str) -> Optional[float]:
    for ag in npc.agents:
        if ag.track_id != target_track_id:
            continue
        loc = ag.walker.get_location()
        for wx, wy, _ in ag.wps:
            dx = float(wx) - float(loc.x)
            dy = float(wy) - float(loc.y)
            if float(np.hypot(dx, dy)) > 0.30:
                return float(np.arctan2(dy, dx))
        return None
    return None


def align_target_to_route_yaw(npc: NpcRuntime, target_track_id: str) -> Optional[float]:
    yaw_rad = target_route_yaw_rad(npc, target_track_id)
    if yaw_rad is None:
        return None
    actor = npc.get_actor_by_track_id(target_track_id)
    if actor is not None:
        tf = actor.get_transform()
        tf.rotation.yaw = float(np.degrees(yaw_rad))
        actor.set_transform(tf)
    return yaw_rad


def resolve_robot_spawn_tf(
    args: argparse.Namespace,
    npc: NpcRuntime,
    target_track_id: str,
    target_yaw_rad: Optional[float] = None,
) -> carla.Transform:
    if args.robot_spawn_mode == "fixed":
        return make_robot_spawn_tf_fixed(args)

    other_xys = [(s.x, s.y) for s in npc.get_states() if s.track_id != target_track_id]
    if args.robot_spawn_mode == "near_target":
        t = npc.get_target_by_track_id(target_track_id)
        if t is not None:
            vx, vy = float(t.vx), float(t.vy)
            if target_yaw_rad is None:
                target_yaw_rad = (float(np.arctan2(vy, vx))
                                  if abs(vx) + abs(vy) > 1e-4
                                  else float(np.deg2rad(t.yaw_deg)))
            z = float(max(args.robot_spawn_z, t.z + 0.25))
            x, y, _ok = follow_position_spawn_xy(
                float(t.x), float(t.y), target_yaw_rad,
                str(args.follow_position), float(args.desired_distance),
                npc.roi_map, forbidden_xys=other_xys,
            )
            robot_yaw = float(np.degrees(target_yaw_rad))
            return carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(yaw=robot_yaw),
            )

    # roi_random fallback
    loc = npc.roi_map.sample_world_location(npc.rng, z=args.robot_spawn_z, jitter=True)
    return carla.Transform(
        carla.Location(x=float(loc.x), y=float(loc.y), z=float(loc.z)),
        carla.Rotation(yaw=args.robot_spawn_yaw_deg),
    )


def yaw_rad_from_velocity(vx: float, vy: float, fallback_deg: float) -> float:
    if abs(vx) + abs(vy) > 1e-4:
        return float(np.arctan2(vy, vx))
    return float(np.deg2rad(fallback_deg))


def _distance_xy(loc_a: carla.Location, loc_b: carla.Location) -> float:
    return float(np.hypot(float(loc_a.x) - float(loc_b.x), float(loc_a.y) - float(loc_b.y)))


def _is_robot_spawn_clear(
    npc: NpcRuntime,
    candidate_loc: carla.Location,
    target_track_id: str,
    clearance_m: float,
) -> bool:
    for ag in npc.agents:
        if ag.track_id == target_track_id or ag.walker is None:
            continue
        try:
            ag_loc = ag.walker.get_location()
        except Exception:
            continue
        if _distance_xy(candidate_loc, ag_loc) < clearance_m:
            return False
    return True


def build_robot_spawn_candidates(
    args: argparse.Namespace,
    npc: NpcRuntime,
    target_loc: carla.Location,
    target_yaw_rad: float,
    target_track_id: str,
) -> List[carla.Transform]:
    if str(args.robot_spawn_mode) != "near_target":
        return [resolve_robot_spawn_tf(args, npc, target_track_id, target_yaw_rad)]

    follow_position = str(args.follow_position)
    desired_distance = float(args.desired_distance)
    robot_z = float(max(args.robot_spawn_z, float(target_loc.z) + 0.25))
    robot_yaw = float(np.degrees(target_yaw_rad))
    clearance_m = float(max(0.1, args.robot_spawn_clearance_m))

    offset = _FOLLOW_OFFSET_RAD.get(follow_position, math.pi)
    base_angle = target_yaw_rad + offset
    radial_dir = np.array([math.cos(base_angle), math.sin(base_angle)], dtype=np.float32)
    tangent_dir = np.array([-radial_dir[1], radial_dir[0]], dtype=np.float32)

    distance_offsets = [0.0, 0.25, 0.5, 0.8, 1.1, 1.4]
    tangential_offsets = [0.0, -0.25, 0.25, -0.5, 0.5]
    radial_jitter = [0.0, 0.15, -0.15]
    search_radii = [0.0, 0.5, 0.9, 1.3]

    candidates: List[carla.Transform] = []
    seen = set()

    for dist_extra in distance_offsets:
        dist = desired_distance + dist_extra
        for search_radius in search_radii:
            base_x, base_y, ok = follow_position_spawn_xy(
                float(target_loc.x),
                float(target_loc.y),
                target_yaw_rad,
                follow_position,
                dist,
                npc.roi_map,
                search_max_radius=float(search_radius),
            )
            if not ok and search_radius <= 0.0:
                continue
            base_xy = np.array([float(base_x), float(base_y)], dtype=np.float32)
            for tan_offset in tangential_offsets:
                for radial_offset in radial_jitter:
                    xy = base_xy + tangent_dir * float(tan_offset) + radial_dir * float(radial_offset)
                    x = float(xy[0])
                    y = float(xy[1])
                    key = (round(x, 2), round(y, 2))
                    if key in seen:
                        continue
                    if not npc.roi_map.is_world_walkable(x, y):
                        continue
                    loc = carla.Location(x=x, y=y, z=robot_z)
                    if not _is_robot_spawn_clear(npc, loc, target_track_id, clearance_m):
                        continue
                    seen.add(key)
                    candidates.append(
                        carla.Transform(
                            loc,
                            carla.Rotation(yaw=robot_yaw),
                        )
                    )

    if not candidates:
        candidates.append(resolve_robot_spawn_tf(args, npc, target_track_id, target_yaw_rad))
    return candidates[: max(1, int(args.robot_spawn_retries))]


def spawn_robot_with_retries(
    args: argparse.Namespace,
    world: carla.World,
    synchronous_mode: bool,
    npc: NpcRuntime,
    robot: ScoutRobotRuntime,
    target_track_id: str,
    target_loc: carla.Location,
    target_yaw_rad: float,
) -> carla.Transform:
    candidates = build_robot_spawn_candidates(
        args=args,
        npc=npc,
        target_loc=target_loc,
        target_yaw_rad=target_yaw_rad,
        target_track_id=target_track_id,
    )
    min_sep = max(2.0, float(args.desired_distance))
    last_error: Optional[Exception] = None
    forbidden_xys = [(s.x, s.y) for s in npc.get_states() if str(s.track_id) != str(target_track_id)]

    for attempt_idx, robot_tf in enumerate(candidates, start=1):
        npc.hide_conflicting_initial_distractors(
            [target_loc, robot_tf.location],
            target_track_id=target_track_id,
            min_sep_m=min_sep,
        )
        step_world(world, synchronous_mode)
        robot.destroy()
        try:
            print(
                f"[EP] robot_spawn attempt={attempt_idx}/{len(candidates)} "
                f"mode={args.robot_spawn_mode} "
                f"at=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f},{robot_tf.location.z:.2f})"
            )
            robot.spawn(robot_tf, roi_map=npc.roi_map,
                        forbidden_xys=forbidden_xys, min_clearance_m=0.7)
            return robot_tf
        except RuntimeError as exc:
            last_error = exc
            robot.destroy()
            print(
                f"[EP][WARN] robot spawn attempt {attempt_idx}/{len(candidates)} failed: {exc}"
            )

    raise RuntimeError(
        f"robot_spawn_failed after {len(candidates)} attempt(s): {last_error or 'unknown error'}"
    )


def _compute_target_bbox_for_trackvla(
    args,
    target_actor,
    robot,
    rgb_intrinsics,
):
    """Project the target actor's 3D bounding box into front-camera pixel coords.

    Returns (u0, v0, u1, v1) or None. Only computed when the active planner is
    ``trackvla`` AND auto-captioning is enabled — keeps other runs CARLA-free.
    """
    if getattr(args, "planner", None) != "trackvla":
        return None
    if not getattr(args, "trackvla_auto_caption", False):
        return None
    if target_actor is None or rgb_intrinsics is None:
        return None
    rgb_sensor = robot.sensors.get("rgb") if hasattr(robot, "sensors") else None
    if rgb_sensor is None:
        return None
    try:
        return _target_bbox_roi_xyxy(
            target_actor,
            rgb_sensor,
            int(rgb_intrinsics.get("width", 0)),
            int(rgb_intrinsics.get("height", 0)),
            float(args.sensor_fov_deg),
        )
    except Exception:
        return None


def draw_status(world: carla.World, text: str) -> None:
    world.debug.draw_string(
        carla.Location(x=0.0, y=0.0, z=3.0),
        text,
        draw_shadow=False,
        color=carla.Color(255, 255, 255),
        life_time=0.1,
    )


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


def _apply_manual_target_control(
    target_actor: Optional[carla.Actor],
    key_state,
    cam_yaw: float = 0.0,
) -> None:
    """Apply WASD walker control using camera yaw as the reference direction."""
    if target_actor is None or key_state is None:
        return
    move_forward = float(key_state[pygame.K_w]) - float(key_state[pygame.K_s])
    move_right   = float(key_state[pygame.K_d]) - float(key_state[pygame.K_a])
    sprint = bool(key_state[pygame.K_LSHIFT] or key_state[pygame.K_RSHIFT])
    speed  = MANUAL_WALKER_SPEED_MPS * (MANUAL_SPRINT_MULT if sprint else 1.0)

    if abs(move_forward) + abs(move_right) < 1e-6:
        target_actor.apply_control(
            carla.WalkerControl(direction=carla.Vector3D(1.0, 0.0, 0.0), speed=0.0, jump=False)
        )
        return

    yaw_rad = math.radians(cam_yaw)
    # CARLA/UE left-handed: forward=(cos,sin), right=(-sin,cos) at yaw angle
    dx = move_forward * math.cos(yaw_rad) - move_right * math.sin(yaw_rad)
    dy = move_forward * math.sin(yaw_rad) + move_right * math.cos(yaw_rad)
    n  = math.hypot(dx, dy)
    if n < 1e-6:
        return
    dx /= n; dy /= n
    target_actor.apply_control(
        carla.WalkerControl(
            direction=carla.Vector3D(dx, dy, 0.0),
            speed=float(speed),
            jump=False,
        )
    )


def _update_manual_spectator(
    spectator: carla.Actor,
    actor: carla.Actor,
    cam_yaw: float,
    cam_pitch: float,
) -> None:
    """Position the spectator in 3rd-person behind `actor`."""
    loc = actor.get_location()
    yaw_rad   = math.radians(cam_yaw)
    pitch_rad = math.radians(cam_pitch)
    # Offset: behind and slightly above
    ox = -MANUAL_CAM_DIST * math.cos(yaw_rad) * math.cos(pitch_rad)
    oy = -MANUAL_CAM_DIST * math.sin(yaw_rad) * math.cos(pitch_rad)
    oz =  MANUAL_CAM_HEIGHT - MANUAL_CAM_DIST * math.sin(pitch_rad)
    spectator.set_transform(carla.Transform(
        carla.Location(x=loc.x + ox, y=loc.y + oy, z=loc.z + max(oz, 0.5)),
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
    scale = MANUAL_SPRINT_MULT if sprint else 1.0
    v_cmd = MANUAL_ROBOT_V_MPS * scale * forward
    w_cmd = MANUAL_ROBOT_W_RADPS * scale * turn
    robot.apply_velocity_command(v_cmd, w_cmd)
    return v_cmd, w_cmd


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
    """Decode CARLA depth RGBA buffer into HxW float32 metres."""
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
    # deterministic pseudo-color mapping by id
    out[:, :, 0] = ((inst_id * 37) % 255).astype(np.uint8)
    out[:, :, 1] = ((inst_id * 67) % 255).astype(np.uint8)
    out[:, :, 2] = ((inst_id * 97) % 255).astype(np.uint8)
    out[inst_id == 0] = 0
    return out


def _lidar_to_points(lidar_data) -> Optional[np.ndarray]:
    """Decode CARLA LiDAR raw buffer into an (N, 4) float32 array [x, y, z, intensity]
    in the LiDAR sensor frame. Returns None when no scan is available."""
    if lidar_data is None:
        return None
    pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    return pts.copy()


def _draw_bboxes_on(img: Optional[np.ndarray], bboxes, stale: int) -> Optional[np.ndarray]:
    """Overlay perception-tracker bboxes onto a view's RGB image (returns a copy).

    ``bboxes`` is a list of dicts with keys {track_id, x1, y1, x2, y2, conf,
    is_target}. Green for target, yellow for others; dim when ``stale > 0``.
    Safe to call with None / empty list — returns the input unchanged.
    """
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


def main() -> None:
    args = build_parser()
    if args.teleop and args.no_ui:
        raise SystemExit("--teleop requires the pygame UI; remove --no-ui.")
    if args.teleop and not HAS_PYGAME:
        raise SystemExit("--teleop requires pygame, but pygame could not be imported.")
    if args.teleop and args.use_perception:
        raise SystemExit("--teleop controls the robot directly and is incompatible with --use-perception.")
    if args.num_episodes < 1:
        raise SystemExit("--num-episodes must be at least 1.")
    if args.num_episodes > 1 and not args.teleop:
        raise SystemExit("--num-episodes > 1 is currently supported only with --teleop.")
    if args.num_episodes > 1 and args.eval_trial_id is not None:
        raise SystemExit("--eval-trial-id cannot be fixed when --num-episodes > 1.")
    if args.sim_speed <= 0.0:
        raise SystemExit("--sim-speed must be greater than zero.")
    if args.teleop and args.async_mode:
        raise SystemExit("--teleop time scaling requires synchronous mode.")
    eval_args = argparse.Namespace(**vars(args))
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
    requested_async = bool(args.async_mode)
    use_sync_mode = (not requested_async) or bool(args.async_after_spawn)
    apply_world_mode(world, orig_settings, sync_mode=use_sync_mode, dt=args.dt)
    print(f"[EP] world_mode={'SYNC' if use_sync_mode else 'ASYNC'} dt={args.dt}")

    _passage_cfg: PassageQueueConfig | None = None
    if getattr(args, "use_passage_queue", False):
        _passage_cfg = PassageQueueConfig(
            door_y=float(getattr(args, "passage_door_y", 148.0)),
            door_x=float(getattr(args, "passage_door_x", -36.5)),
            approach_a_y=float(getattr(args, "passage_approach_a_y", 146.0)),
            approach_b_y=float(getattr(args, "passage_approach_b_y", 152.0)),
            trigger_dist=float(getattr(args, "passage_trigger_dist", 1.5)),
            exit_y_atob=float(getattr(args, "passage_exit_y_atob", 150.5)),
            exit_y_btoa=float(getattr(args, "passage_exit_y_btoa", 146.5)),
            max_passing=int(getattr(args, "passage_max_passing", 1)),
            min_dist_after_cross=float(getattr(args, "passage_min_dist_after_cross", 6.0)),
        )
        print(f"[EP] Passage queuing enabled: door_x={_passage_cfg.door_x} door_y={_passage_cfg.door_y} "
              f"max_passing={_passage_cfg.max_passing}")

    npc = NpcRuntime(
        world=world,
        grid_npz=args.grid_npz,
        flow_points_json=args.flow_points_json,
        flow_mode=args.npc_flow_mode,
        rng_seed=args.npc_rng_seed,
        spawn_z=args.robot_spawn_z,
        use_orca=bool(getattr(args, "use_orca_npc", False)),
        passage_config=_passage_cfg,
    )
    # Propagate ORCA tuning params.
    npc.params["orca_tau"] = float(getattr(args, "orca_tau", 3.0))
    npc.params["orca_neighbor_dist"] = float(getattr(args, "orca_neighbor_dist", 5.0))
    npc.params["orca_dt"] = float(args.dt)
    robot_cls = G1RobotRuntime if args.robot_type == "g1" else ScoutRobotRuntime
    robot = robot_cls(world=world, dt=args.dt)
    robot.rescue_lift_z = float(args.robot_rescue_lift_z)
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
        policy = build_custom_policy(args, scene="park")
    elif args.planner == "dwa_traj":
        if perception_kwargs is not None:
            from baselines.adapters.dwa_traj_perception_adapter import DwaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            policy = DwaTrajPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.dwa_traj_adapter import DwaTrajFollowerPolicy  # noqa: PLC0415
            policy = DwaTrajFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "dwa_traj_depth_tpt":
        # No perception wrapper — this adapter already runs its own perception.
        from baselines.adapters.dwa_traj_depth_tpt_adapter import DwaTrajDepthTptFollowerPolicy  # noqa: PLC0415
        policy = DwaTrajDepthTptFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "sfm":
        if perception_kwargs is not None:
            from baselines.adapters.sfm_perception_adapter import SfmPerceptionFollowerPolicy  # noqa: PLC0415
            policy = SfmPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.sfm_adapter import SfmFollowerPolicy  # noqa: PLC0415
            policy = SfmFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda":
        if perception_kwargs is not None:
            from baselines.adapters.rda_perception_adapter import RdaPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.rda_adapter import RdaFollowerPolicy  # noqa: PLC0415
            policy = RdaFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "rda_lidar":
        if perception_kwargs is not None:
            from baselines.adapters.rda_lidar_perception_adapter import RdaLidarPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaLidarPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.rda_lidar_adapter import RdaLidarFollowerPolicy  # noqa: PLC0415
            policy = RdaLidarFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "gt_planner":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner gt_planner "
                "(gt_planner already consumes full GT + lidar)."
            )
        from baselines.adapters.gt_planner_adapter import GtPlannerFollowerPolicy  # noqa: PLC0415
        policy = GtPlannerFollowerPolicy(dt=args.dt, **policy_kwargs, grid_npz=args.grid_npz)
    elif args.planner == "rda_traj":
        traj_kwargs = dict(
            use_lidar_obstacles=not bool(args.rda_traj_disable_lidar_obstacles),
        )
        if perception_kwargs is not None:
            from baselines.adapters.rda_traj_perception_adapter import RdaTrajPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaTrajPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs, **traj_kwargs)
        else:
            from baselines.adapters.rda_traj_adapter import RdaTrajFollowerPolicy  # noqa: PLC0415
            policy = RdaTrajFollowerPolicy(dt=args.dt, **policy_kwargs, **traj_kwargs)
    elif args.planner == "rda_search":
        search_kwargs = dict(
            gt_target_always_visible=not bool(args.rda_search_use_camera_visibility),
            protect_target_for_side_follow=not bool(args.rda_search_disable_target_protection),
        )
        if perception_kwargs is not None:
            from baselines.adapters.rda_search_perception_adapter import RdaSearchPerceptionFollowerPolicy  # noqa: PLC0415
            policy = RdaSearchPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs, **search_kwargs)
        else:
            from baselines.adapters.rda_search_adapter import RdaSearchFollowerPolicy  # noqa: PLC0415
            policy = RdaSearchFollowerPolicy(dt=args.dt, **policy_kwargs, **search_kwargs)
    elif args.planner == "rda_depth_tpt":
        # No perception wrapper — this adapter already runs its own perception.
        from baselines.adapters.rda_depth_tpt_adapter import RdaDepthTptFollowerPolicy  # noqa: PLC0415
        policy = RdaDepthTptFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "bso_hfc":
        if perception_kwargs is not None:
            from baselines.adapters.bso_hfc_perception_adapter import BsoHfcPerceptionFollowerPolicy  # noqa: PLC0415
            policy = BsoHfcPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.bso_hfc_adapter import BsoHfcFollowerPolicy  # noqa: PLC0415
            policy = BsoHfcFollowerPolicy(dt=args.dt, **policy_kwargs)
    elif args.planner == "trackvla":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner trackvla "
                "(trackvla is itself an end-to-end vision-language-action policy)."
            )
        from baselines.adapters.trackvla_adapter import TrackVlaFollowerPolicy  # noqa: PLC0415
        policy = TrackVlaFollowerPolicy(
            dt=args.dt,
            instruction=args.trackvla_instruction,
            ckpt_path=args.trackvla_ckpt,
            render_overlay=bool(args.debug),
            auto_caption=bool(args.trackvla_auto_caption),
            caption_ckpt_path=args.trackvla_caption_ckpt,
            caption_device=args.trackvla_caption_device,
        )
    elif args.planner == "oa_vat":
        if perception_kwargs is not None:
            raise SystemExit(
                "--use-perception is not compatible with --planner oa_vat "
                "(oa_vat carries its own YOLOe + ORTrack + DINOv3 perception)."
            )
        from baselines.adapters.oa_vat_adapter import OaVatFollowerPolicy  # noqa: PLC0415
        policy = OaVatFollowerPolicy(
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
    else:  # pid (default)
        if perception_kwargs is not None:
            from baselines.adapters.pid_perception_adapter import PidPerceptionFollowerPolicy  # noqa: PLC0415
            policy = PidPerceptionFollowerPolicy(
                dt=args.dt, **policy_kwargs, **perception_kwargs)
        else:
            from baselines.adapters.pid_adapter import PIDFollowerPolicy  # noqa: PLC0415
            policy = PIDFollowerPolicy(dt=args.dt, **policy_kwargs)
    vis_checker = InstanceVisibilityChecker(threshold=args.visibility_threshold)
    eval_logger = EvaluationLogger.from_args(
        eval_args, scenario_type="park"
    ) if args.enable_evaluation else None
    collision_monitor = None
    termination_reason = None
    target_distance_monitor = TargetDistanceMonitor(
        warmup_sec=float(args.robot_target_far_warmup_sec),
        fail_dist_m=float(args.robot_target_far_fail_dist_m),
        fail_sec=float(args.robot_target_far_fail_sec),
        fail_steps=int(getattr(args, "robot_target_far_fail_steps", 0)),
    )
    if eval_logger is not None:
        print(f"[EVAL] run_dir={eval_logger.run_dir}")

    # 2-D debug visualizer (separate process, only when --debug is set)
    dbg_vis = None
    if args.debug:
        from followbench.visualization.debug_visualizer import DebugVisualizer
        dbg_vis = DebugVisualizer(half_size=40.0)

    spawned = False
    following = False
    teleop_started = False
    paused = False
    paused_at = 0.0
    manual_mode   = "none"   # none | target | robot
    _cam_yaw      = 0.0     # 3rd-person camera yaw   (degrees, mouse-driven)
    _cam_pitch    = -20.0   # 3rd-person camera pitch  (degrees)
    active_target_track_id = args.target_track_id
    metrics_path = os.path.join(out_dir, "park_metrics.json")
    _use_metrics = bool(getattr(args, "enable_metrics", False)) and _HAS_PARK_METRICS
    scene_metrics = ParkMetrics(target_track_id=args.target_track_id) if _use_metrics else None
    if _use_metrics:
        print("[EP] ParkMetrics enabled — will save to", metrics_path)
    running = True
    tick = 0
    completed_episodes = 0
    desired_num_walkers = int(args.num_walkers)
    desired_min_speed = float(args.npc_min_speed)
    desired_max_speed = float(args.npc_max_speed)
    _DIFFICULTIES = ["easy", "hard", "boss", "legend"]
    _DIFF_COLORS   = {"easy": (80, 220, 80), "hard": (240, 200, 40),
                      "boss": (240, 120, 40), "legend": (220, 60, 60)}
    current_difficulty: str = args.difficulty
    npc.set_difficulty(current_difficulty)

    # Weather is drawn from a stream derived from the episode seed (paired across
    # policies) with difficulty-dependent weights (boss/legend biased toward
    # dusk/rain/fog).  npc.rng_seed is the resolved seed (CLI or time fallback).
    _weather_rng = weather_mod.seeded_rng(npc.rng_seed)

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

    # ── HUD notification system ───────────────────────────────────────────────
    _notifs: List[Tuple[str, float]] = []   # (message, expire_wall_time)
    _LIVE_VIEWER_PATH = module_path("visualization", "live_viewer.py")
    _debug_proc: List[Optional[subprocess.Popen]] = [None]  # mutable container for closure

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
        title = "Park Teleoperation" if args.teleop else "Episode Manager"
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

    def on_signal(_sig, _frm):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)

    gt_f = open(gt_path, "w", encoding="utf-8")
    vis_f = open(vis_path, "w", encoding="utf-8")

    print("[EP] ready. Press G to spawn actors, then F to start following.")
    print(f"[EP] output_dir={out_dir}")
    print(f"[EP] npc_flow_mode={args.npc_flow_mode} nav_mode={args.npc_navigation_mode}")

    def do_spawn() -> None:
        nonlocal spawned, use_sync_mode, collision_monitor
        if spawned:
            return
        # Cleanup stale robots from previous runs to avoid visual confusion.
        for a in world.get_actors():
            try:
                is_scout = "vehicle.scout" in a.type_id
                is_g1 = "walker.pedestrian.g1" in a.type_id
                is_known_role = a.attributes.get("role_name", "") in {"followbench_robot", "sfm_follow_robot", "scout_follow"}
                if is_scout or is_g1 or is_known_role:
                    a.destroy()
            except Exception:
                pass
        npc.params["min_speed"] = float(desired_min_speed)
        npc.params["max_speed"] = float(desired_max_speed)
        npc.params["navigation_mode"] = str(args.npc_navigation_mode)
        npc.params["strict_flow_endpoints"] = bool(args.npc_strict_flow_endpoints)
        npc.params["replan_on_avoid"] = bool(args.npc_replan_on_avoid)
        npc.params["avoid_replan_threshold"] = float(args.npc_avoid_replan_threshold)
        npc.params["astar_replan_cooldown_sec"] = float(args.npc_astar_replan_cooldown_sec)
        npc.params["spawn_attempts_per_agent"] = int(max(1, args.npc_spawn_attempts_per_agent))
        npc.params["spawn_min_clearance_m"] = float(max(0.1, args.npc_spawn_clearance_m))
        npc_spawn_retries = int(max(1, args.npc_spawn_retries))
        target_present = False
        for npc_attempt in range(1, npc_spawn_retries + 1):
            npc.destroy()
            npc.spawn(desired_num_walkers)
            if len(npc.agents) == 0:
                print(f"[EP][WARN] npc spawn attempt {npc_attempt}/{npc_spawn_retries} produced no walkers.")
                continue
            step_world(world, use_sync_mode)
            target_present = (
                npc.get_target_by_track_id(active_target_track_id) is not None
                and npc.get_actor_by_track_id(active_target_track_id) is not None
            )
            if target_present:
                break
            print(
                f"[EP][WARN] npc spawn attempt {npc_attempt}/{npc_spawn_retries} "
                f"missing target {active_target_track_id}; retrying crowd init."
            )
        if len(npc.agents) == 0:
            raise RuntimeError("No NPC spawned, cannot place robot near target.")
        if not target_present:
            raise RuntimeError(f"target_spawn_failed: missing {active_target_track_id} after {npc_spawn_retries} crowd spawn attempt(s)")
        # CARLA needs a tick to materialize walker transforms before reading positions.
        step_world(world, use_sync_mode)

        # Scheduled flow decides the target's actual initial route and may
        # move walkers, so configure it before placing the robot.
        if getattr(args, "use_scheduled_flow", False):
            npc.setup_scheduled_flow(
                dwell_s=float(getattr(args, "scheduled_flow_dwell_s", 0.0)),
                stagger_scale=1.0,
                target_track_id=str(getattr(args, "target_track_id", "") or ""),
            )
            step_world(world, use_sync_mode)
            print("[EP] scheduled flow configured.")

        tgt_init = npc.get_target_by_track_id(active_target_track_id)
        tgt_actor = npc.get_actor_by_track_id(active_target_track_id)
        if tgt_actor is None or tgt_init is None:
            tgt_init = npc.get_states()[0]
            active_id = tgt_init.track_id
            print(f"[EP][WARN] target {active_target_track_id} missing, fallback to {active_id}")
            tgt_actor = npc.get_actor_by_track_id(active_id)
        else:
            active_id = active_target_track_id

        target_yaw_rad = align_target_to_route_yaw(npc, active_id)
        if target_yaw_rad is None and tgt_init is not None:
            target_yaw_rad = float(np.deg2rad(float(tgt_init.yaw_deg)))
        if target_yaw_rad is not None:
            step_world(world, use_sync_mode)
            tgt_init = npc.get_target_by_track_id(active_id)
            tgt_actor = npc.get_actor_by_track_id(active_id)

        robot_tf = resolve_robot_spawn_tf(
            args,
            npc=npc,
            target_track_id=active_id,
            target_yaw_rad=target_yaw_rad,
        )
        _spawn_forbidden_xys = [(s.x, s.y) for s in npc.get_states()]
        if tgt_actor is not None:
            reserved_target_loc = tgt_actor.get_location()
        elif tgt_init is not None:
            reserved_target_loc = carla.Location(x=float(tgt_init.x), y=float(tgt_init.y), z=float(tgt_init.z))
        else:
            reserved_target_loc = None
        if reserved_target_loc is not None:
            robot_tf = spawn_robot_with_retries(
                args=args,
                world=world,
                synchronous_mode=use_sync_mode,
                npc=npc,
                robot=robot,
                target_track_id=active_id,
                target_loc=reserved_target_loc,
                target_yaw_rad=target_yaw_rad if target_yaw_rad is not None else float(np.deg2rad(float(tgt_init.yaw_deg))),
            )
        else:
            print(
                f"[EP] robot_spawn mode={args.robot_spawn_mode} "
                f"at=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f},{robot_tf.location.z:.2f})"
            )
            robot.spawn(robot_tf, roi_map=npc.roi_map,
                        forbidden_xys=_spawn_forbidden_xys, min_clearance_m=0.7)
        # Materialize actor transforms before reading coordinates.
        step_world(world, use_sync_mode)
        # Hard correction: ensure spawn is within configured max distance to target.
        if tgt_init is not None and robot.actor is not None and tgt_actor is not None:
            rloc = robot.actor.get_location()
            tloc = tgt_actor.get_location()
            # Some CARLA builds may still return a transient zero pose right after spawn.
            if abs(tloc.x) < 1e-4 and abs(tloc.y) < 1e-4:
                step_world(world, use_sync_mode)
                tloc = tgt_actor.get_location()
            if abs(rloc.x) < 1e-4 and abs(rloc.y) < 1e-4:
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
            dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
            print(
                f"[EP][DEBUG] target={active_id} actor_id={tgt_actor.id} "
                f"planned_robot_xy=({robot_tf.location.x:.2f},{robot_tf.location.y:.2f}) "
                f"target_xy=({tloc.x:.2f},{tloc.y:.2f}) "
                f"robot_xy=({rloc.x:.2f},{rloc.y:.2f}) dist={dist0:.2f}m"
            )
            if dist0 > (float(args.desired_distance) + 1.5):
                if target_yaw_rad is None:
                    target_yaw_rad = float(np.deg2rad(float(tgt_init.yaw_deg)))
                _correction_other_xys = [(s.x, s.y) for s in npc.get_states() if s.track_id != active_target_track_id]
                new_x, new_y, _ok = follow_position_spawn_xy(
                    float(tloc.x), float(tloc.y), target_yaw_rad,
                    str(args.follow_position), float(args.desired_distance),
                    npc.roi_map, forbidden_xys=_correction_other_xys,
                )
                fix_yaw = float(np.degrees(target_yaw_rad))
                fix_tf = carla.Transform(
                    carla.Location(x=new_x, y=new_y, z=robot_tf.location.z),
                    carla.Rotation(yaw=fix_yaw),
                )
                robot.force_place(fix_tf)
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
                dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
                print(
                    f"[EP][DEBUG] corrected robot_xy=({rloc.x:.2f},{rloc.y:.2f}) "
                    f"target_xy=({tloc.x:.2f},{tloc.y:.2f}) dist={dist0:.2f}m"
                )
            # If still at origin, force place again with planned transform.
            if abs(rloc.x) < 1e-4 and abs(rloc.y) < 1e-4:
                robot.force_place(robot_tf)
                step_world(world, use_sync_mode)
                rloc = robot.actor.get_location()
                dist0 = float(np.hypot(rloc.x - tloc.x, rloc.y - tloc.y))
                print(
                    f"[EP][DEBUG] force_place retry robot_xy=({rloc.x:.2f},{rloc.y:.2f}) "
                    f"target_xy=({tloc.x:.2f},{tloc.y:.2f}) dist={dist0:.2f}m"
                )
            print(f"[EP] robot_target_init_dist={dist0:.2f}m (target={active_id})")
            if args.draw_debug:
                world.debug.draw_string(
                    carla.Location(x=tloc.x, y=tloc.y, z=tloc.z + 2.8),
                    "TARGET_INIT",
                    draw_shadow=False,
                    color=carla.Color(255, 255, 0),
                    life_time=5.0,
                )
                world.debug.draw_string(
                    carla.Location(x=rloc.x, y=rloc.y, z=rloc.z + 2.8),
                    "ROBOT_SPAWN",
                    draw_shadow=False,
                    color=carla.Color(0, 255, 0),
                    life_time=5.0,
                )
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
                    "npc_rng_seed": int(npc.rng_seed),
                    "num_walkers": len(npc.agents),
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
        if scene_metrics is not None:
            interceptor_ids = {ag.track_id for ag in npc.agents if ag.is_interceptor}
            scene_metrics.set_interceptor_ids(interceptor_ids)
            print(f"[EP][Metrics] interceptors={sorted(interceptor_ids)}")
        spawned = True
        if dbg_vis is not None:
            rs = robot.get_state()
            if rs is not None:
                dbg_vis.start(rs.x, rs.y, args.grid_npz)
        print(
            f"[EP] spawned all actors (idle) | walkers={desired_num_walkers} "
            f"speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"
        )
        # ── Auto-position spectator ───────────────────────────────────────────
        try:
            if _passage_cfg is not None:
                # Passage scenario: fly to just above and behind the passage
                _cx = float(_passage_cfg.door_x) - 8.0   # slight offset to see both sides
                _cy = float(_passage_cfg.door_y)
                _z  = 35.0
                _pitch = -70.0
            else:
                # Centre on flow points so the spectator looks down on the actual scene.
                _fps = getattr(npc, "flow_points", [])
                if _fps:
                    _cx = float(sum(fp.x for fp in _fps) / len(_fps))
                    _cy = float(sum(fp.y for fp in _fps) / len(_fps))
                    _span = max(
                        max(fp.x for fp in _fps) - min(fp.x for fp in _fps),
                        max(fp.y for fp in _fps) - min(fp.y for fp in _fps),
                    )
                    _z = max(50.0, _span * 1.2)
                else:
                    _rm = npc.roi_map
                    _cx = float((_rm.world_min[0] + _rm.world_max[0]) / 2.0)
                    _cy = float((_rm.world_min[1] + _rm.world_max[1]) / 2.0)
                    _z  = 80.0
                _pitch = -85.0
            world.get_spectator().set_transform(carla.Transform(
                carla.Location(x=_cx, y=_cy, z=_z),
                carla.Rotation(pitch=_pitch, yaw=0.0, roll=0.0),
            ))
            print(f"[EP] spectator -> ({_cx:.1f}, {_cy:.1f}, z={_z:.1f})")
        except Exception as _e:
            print(f"[EP][WARN] spectator auto-position failed: {_e}")
        if args.debug:
            print("[EP][Debug] writing state to /tmp/npc_debug_state.json each tick")
            print(f"[EP][Debug] launch viewer:  conda run -n carla38 python "
                  f"{module_path('visualization', 'live_viewer.py')} "
                  f"--npz {args.grid_npz} --half 30")
        if requested_async and args.async_after_spawn and use_sync_mode:
            apply_world_mode(world, orig_settings, sync_mode=False, dt=args.dt)
            use_sync_mode = False
            print("[EP] world_mode switched to ASYNC after initial spawn.")

    def do_reset(*, announce: bool = True) -> None:
        nonlocal spawned, following, teleop_started, tick, manual_mode
        nonlocal collision_monitor, termination_reason
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
        if manual_mode != "none" and ui_enabled:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
        manual_mode = "none"
        tick = 0
        vis_checker.reset()
        target_distance_monitor.reset()
        termination_reason = None
        if announce:
            print("[EP] reset done. Press G to spawn with current params.")

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
        print(
            f"[EP] completed episode {completed_episodes}/{args.num_episodes} "
            f"reason={reason}"
        )
        if completed_episodes >= args.num_episodes:
            running = False
            return

        do_reset(announce=False)
        current_weather = _select_and_apply_weather()
        weather_history = [{"tick": 0, "weather": current_weather}]
        auto_spawn_done = False
        auto_follow_done = False
        eval_logger = EvaluationLogger.from_args(
            eval_args, scenario_type="park"
        ) if args.enable_evaluation else None
        if eval_logger is not None:
            print(f"[EVAL] next run_dir={eval_logger.run_dir}")
        print(
            f"[EP] ready for episode {completed_episodes + 1}/{args.num_episodes}. "
            "Press G to spawn, then F to start."
        )

    def do_toggle_follow() -> None:
        nonlocal following, start_t, teleop_started, teleop_start_tick
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
        if args.teleop:
            if following and manual_mode != "robot":
                do_toggle_manual_robot()
            elif not following and manual_mode == "robot":
                do_toggle_manual_robot()
                robot.hold_still()
        print(f"[EP] following={'ON' if following else 'OFF'} target={active_target_track_id}")

    def do_toggle_manual_target() -> None:
        nonlocal manual_mode, _cam_yaw, _cam_pitch
        if not spawned:
            return
        if manual_mode == "target":
            manual_mode = "none"
            if ui_enabled:
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
            print("[EP] manual target OFF — algorithm resumes")
        else:
            manual_mode = "target"
            # Sync camera yaw to actor's current facing
            _ta = npc.get_actor_by_track_id(active_target_track_id)
            if _ta is not None:
                _cam_yaw = float(_ta.get_transform().rotation.yaw)
            _cam_pitch = -20.0
            if ui_enabled:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
            print(f"[EP] manual target ON ({active_target_track_id}) | WASD move, Mouse look, Shift sprint, ESC release")

    def do_toggle_manual_robot() -> None:
        nonlocal manual_mode, _cam_yaw, _cam_pitch
        if not spawned:
            return
        if manual_mode == "robot":
            manual_mode = "none"
            if ui_enabled:
                pygame.mouse.set_visible(True)
                pygame.event.set_grab(False)
            suffix = "episode paused" if args.teleop else "algorithm resumes"
            print(f"[EP] manual robot OFF — {suffix}")
        else:
            manual_mode = "robot"
            if robot.actor is not None:
                _cam_yaw = float(robot.actor.get_transform().rotation.yaw)
            _cam_pitch = -20.0
            if ui_enabled:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
            print("[EP] manual robot ON | WASD drive, Mouse look, Shift sprint")

    # Sim clock: seconds of simulated time (world steps * dt). Used instead of
    # wall clock so auto-spawn/follow timing and duration are reproducible.
    sim_steps = 0

    def sim_clock() -> float:
        return sim_steps * float(args.dt)

    start_t = sim_clock()
    teleop_start_tick = 0
    auto_spawn_done = False
    auto_follow_done = False
    active_camera_view = active_camera_view_for_follow_position(args.follow_position)
    # Cached after spawn_sensors() so we don't keep dict-walking the calibration every tick.
    rgb_intrinsics_cached: Optional[dict] = None
    rgb_extr_cached: Optional[dict] = None
    lidar_extr_cached: Optional[dict] = None
    rgb_intrinsics_left_cached: Optional[dict] = None
    rgb_extr_left_cached: Optional[dict] = None
    rgb_intrinsics_right_cached: Optional[dict] = None
    rgb_extr_right_cached: Optional[dict] = None
    # Last-good decoded sensor frames. Used as a fallback for the very first
    # ticks after spawn_sensors() before the asynchronous sensor listeners have
    # delivered their first payload, so we never feed None to a planner that
    # expects rgb_image / lidar_points to be present.
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
                else:
                    termination_reason = termination_reason or "duration_limit"
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
                for e in pygame.event.get():
                    if e.type == pygame.QUIT:
                        running = False
                    elif e.type == pygame.MOUSEMOTION and manual_mode != "none":
                        _cam_yaw   = (_cam_yaw + e.rel[0] * MANUAL_MOUSE_SENS) % 360.0
                        _cam_pitch = float(np.clip(
                            _cam_pitch - e.rel[1] * MANUAL_MOUSE_SENS * 0.4, -60.0, 10.0))
                    elif e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_q:
                            running = False
                        elif e.key == pygame.K_ESCAPE:
                            if manual_mode != "none":
                                if args.teleop:
                                    _notify("Teleop is locked for this episode; press F to pause")
                                else:
                                    manual_mode = "none"
                                    if ui_enabled:
                                        pygame.mouse.set_visible(True)
                                        pygame.event.set_grab(False)
                                    _notify("Manual control OFF — algorithm resumed")
                        elif e.key == pygame.K_g and not spawned:
                            do_spawn()
                        elif e.key == pygame.K_r:
                            do_reset()
                        elif e.key == pygame.K_m and spawned and not args.teleop:
                            do_toggle_manual_target()
                        elif e.key == pygame.K_b and spawned and not args.teleop:
                            do_toggle_manual_robot()
                        elif e.key == pygame.K_f and spawned:
                            do_toggle_follow()
                        # Teleop fixes the difficulty at process launch.  D is
                        # always steer-right there, including while paused.
                        elif e.key == pygame.K_d and manual_mode == "none" and not args.teleop:
                            idx = _DIFFICULTIES.index(current_difficulty)
                            current_difficulty = _DIFFICULTIES[(idx + 1) % len(_DIFFICULTIES)]
                            npc.set_difficulty(current_difficulty)
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
                                    [sys.executable, _LIVE_VIEWER_PATH,
                                     "--npz", args.grid_npz])
                                _notify("NPC debug viewer: ON  (V to close)")
                            else:
                                _debug_proc[0].terminate()
                                _debug_proc[0] = None
                                _notify("NPC debug viewer: OFF")
                        elif e.key == pygame.K_t and spawned:
                            states = npc.get_states()
                            if states:
                                ids = [s.track_id for s in states]
                                if active_target_track_id not in ids:
                                    active_target_track_id = ids[0]
                                else:
                                    idx = ids.index(active_target_track_id)
                                    active_target_track_id = ids[(idx + 1) % len(ids)]
                                _notify(f"Target → {active_target_track_id}")
                        # Keep every crowd knob fixed during teleoperation.
                        elif e.key == pygame.K_UP and not args.teleop:
                            desired_num_walkers = min(120, desired_num_walkers + 2)
                            _notify(f"Walkers → {desired_num_walkers}  (R+G to apply)")
                        elif e.key == pygame.K_DOWN and not args.teleop:
                            desired_num_walkers = max(2, desired_num_walkers - 2)
                            _notify(f"Walkers → {desired_num_walkers}  (R+G to apply)")
                        elif e.key == pygame.K_LEFTBRACKET and not args.teleop:
                            desired_min_speed = max(0.2, desired_min_speed - 0.1)
                            desired_max_speed = max(desired_max_speed, desired_min_speed + 0.1)
                            _notify(f"Min speed → {desired_min_speed:.2f}")
                        elif e.key == pygame.K_RIGHTBRACKET and not args.teleop:
                            desired_min_speed = min(desired_max_speed - 0.1, desired_min_speed + 0.1)
                            _notify(f"Min speed → {desired_min_speed:.2f}")
                        elif e.key == pygame.K_MINUS and not args.teleop:
                            desired_max_speed = max(desired_min_speed + 0.1, desired_max_speed - 0.1)
                            _notify(f"Max speed → {desired_max_speed:.2f}")
                        elif e.key == pygame.K_EQUALS and not args.teleop:
                            desired_max_speed = min(3.0, desired_max_speed + 0.1)
                            _notify(f"Max speed → {desired_max_speed:.2f}")
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

                # Draw idle HUD before actors are spawned.
                if not spawned:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        line1 = (
                            f"State: spawned={spawned} follow={following} manual={manual_mode} target={active_target_track_id} "
                            f"| desired walkers={desired_num_walkers} speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"
                        )
                        line2 = (
                            "Keys: G spawn  F follow  D difficulty  V debug-viewer  M manual  B manual-robot  T target  R reset  SPACE pause  Q quit"
                            if not args.teleop else
                            "Keys: G spawn  F follow  D steer-right  R reset  SPACE pause  Q quit"
                        )
                        line3 = "Tune: Up/Down walkers | [ ] min_speed | - = max_speed  (R+G to apply)"
                        diff_label = (
                            f"DIFFICULTY: {current_difficulty.upper()}  "
                            f"({'LOCKED BY LAUNCH' if args.teleop else 'D to cycle'})"
                        )
                        for idx, text in enumerate([line1, line2, line3]):
                            img = ui_font.render(text, True, (230, 235, 240))
                            surf.blit(img, (16, 20 + idx * 36))
                        diff_surf = ui_font.render(diff_label, True, _DIFF_COLORS[current_difficulty])
                        surf.blit(diff_surf, (16, 20 + 3 * 36))
                        # Notifications
                        now_n = time.time()
                        active = [(m, e) for m, e in _notifs if e > now_n]
                        _notifs[:] = active
                        for ni, (msg, _) in enumerate(reversed(active[-4:])):
                            ns = ui_font.render(f"▶ {msg}", True, (255, 230, 80))
                            surf.blit(ns, (16, surf.get_height() - 28 - ni * 26))
                        pygame.display.flip()

            if paused:
                # Keep applying zero-speed controls every tick so walkers truly stop.
                # Without this, CARLA physics continues applying the last non-zero
                # WalkerControl, causing walkers to drift and invalidate their routes.
                if spawned:
                    npc.step(
                        debug_draw_ids=args.draw_debug,
                        label_life=max(args.dt * 2.0, 0.12),
                        robot_actor=robot.actor if robot.actor is not None else None,
                        target_track_id=active_target_track_id,
                        manual_track_id=active_target_track_id if manual_mode == "target" else None,
                    )
                    robot.hold_still()
                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        overlay = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
                        overlay.fill((0, 0, 0, 140))
                        surf.blit(overlay, (0, 0))
                        pause_text = ui_font.render("PAUSED — press SPACE to resume", True, (255, 220, 0))
                        px = (surf.get_width() - pause_text.get_width()) // 2
                        py = (surf.get_height() - pause_text.get_height()) // 2
                        surf.blit(pause_text, (px, py))
                        pygame.display.flip()
                step_main_loop(frame_started_at)
                continue

            if spawned:
                key_state = pygame.key.get_pressed() if ui_enabled else None
                npc.step(
                    debug_draw_ids=args.draw_debug,
                    label_life=max(args.dt * 2.0, 0.12),
                    robot_actor=robot.actor if robot.actor is not None else None,
                    target_track_id=active_target_track_id,
                    manual_track_id=active_target_track_id if manual_mode == "target" else None,
                )
                if npc.is_episode_done() and following:
                    if termination_reason is None:
                        termination_reason = "target_completed_route"
                    following = False
                    npc.set_motion_active(False)
                    robot.hold_still()
                    print(f"[EP] episode DONE  reason={termination_reason}  "
                          "(target completed P0→P1 evaluation leg)")
                    episode_end_reason = termination_reason
                npc_states = npc.get_states()
                target_state = next((s for s in npc_states if s.track_id == active_target_track_id), None)
                target_actor = npc.get_actor_by_track_id(active_target_track_id)
                if manual_mode == "target":
                    _apply_manual_target_control(target_actor, key_state, cam_yaw=_cam_yaw)
                    if target_actor is not None:
                        _update_manual_spectator(
                            world.get_spectator(), target_actor, _cam_yaw, _cam_pitch)
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
                if active_camera_view != "front":
                    inst_img = None
                    rgb_img = None
                    depth_img = None
                if active_camera_view != "left":
                    inst_img_left = None
                    rgb_img_left = None
                    depth_img_left = None
                if active_camera_view != "right":
                    inst_img_right = None
                    rgb_img_right = None
                    depth_img_right = None
                active_instance_img = {
                    "front": inst_img,
                    "left": inst_img_left,
                    "right": inst_img_right,
                }.get(active_camera_view)
                active_depth_img = {
                    "front": depth_img,
                    "left": depth_img_left,
                    "right": depth_img_right,
                }.get(active_camera_view)
                if target_actor is not None and "instance" in robot.sensors:
                    active_instance_sensor = {
                        "front": robot.sensors.get("instance"),
                        "left": robot.sensors.get("instance_left"),
                        "right": robot.sensors.get("instance_right"),
                    }.get(active_camera_view)
                    visible, pix_count = vis_checker.evaluate_multi(
                        views=[(active_instance_img, active_instance_sensor)],
                        target_actor=target_actor,
                        image_w=args.sensor_image_w,
                        image_h=args.sensor_image_h,
                        fov_deg=args.sensor_fov_deg,
                    )
                if ui_enabled:
                    # Show all three RGB views live, bypassing the planner's single
                    # active_camera_view gating that nulls the off-views above.
                    r = _to_rgb(robot.get_sensor_data("rgb"))
                    rl = _to_rgb(robot.get_sensor_data("rgb_left"))
                    rr = _to_rgb(robot.get_sensor_data("rgb_right"))
                    d = _depth_to_rgb(active_depth_img)
                    ins = _instance_to_rgb(active_instance_img)
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

                if manual_mode == "robot" and following:
                    v_cmd, w_cmd = _apply_manual_robot_control(robot, key_state)
                    policy_debug_info = {"mode": "teleop"} if args.teleop else {}
                    if robot.actor is not None:
                        _update_manual_spectator(
                            world.get_spectator(), robot.actor, _cam_yaw, _cam_pitch)
                elif target_state is not None and following:
                    target_velocity_yaw_rad = yaw_rad_from_velocity(
                        target_state.vx,
                        target_state.vy,
                        fallback_deg=target_state.yaw_deg,
                    )
                    target_actor_yaw_rad = float(np.deg2rad(target_state.yaw_deg))
                    if rgb_intrinsics_cached is None:
                        calib = robot.get_calibration()
                        rgb_calib = calib.get("rgb", {}) if active_camera_view == "front" else {}
                        lidar_calib = calib.get("lidar", {})
                        rgb_left_calib = calib.get("rgb_left", {}) if active_camera_view == "left" else {}
                        rgb_right_calib = calib.get("rgb_right", {}) if active_camera_view == "right" else {}
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
                            "target_bbox_xyxy": _compute_target_bbox_for_trackvla(
                                args, target_actor, robot, rgb_intrinsics_cached,
                            ),
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
                    # Robot-side safety layer: slow down / brake near pedestrians.
                    nearest_npc_dist = float("inf")
                    for s in npc_states:
                        d = float(np.hypot(robot_state.x - s.x, robot_state.y - s.y))
                        if d < nearest_npc_dist:
                            nearest_npc_dist = d

                    v_cmd = float(action.v_mps)
                    w_cmd = float(action.w_radps)
                    # if nearest_npc_dist < 2.8:
                    #     # Linear speed attenuation as pedestrians get close.
                    #     scale = float(np.clip((nearest_npc_dist - 1.0) / (2.8 - 1.0), 0.0, 1.0))
                    #     v_cmd *= scale
                    # if nearest_npc_dist < 1.0:
                    #     # Emergency stop zone.
                    #     v_cmd = 0.0
                    robot.apply_velocity_command(v_cmd, w_cmd)
                    if args.draw_laser:
                        world.debug.draw_line(
                            carla.Location(
                                x=robot_state.x,
                                y=robot_state.y,
                                z=robot_state.z + 1.2,
                            ),
                            carla.Location(
                                x=target_state.x,
                                y=target_state.y,
                                z=target_state.z + 1.4,
                            ),
                            thickness=0.02,
                            color=carla.Color(50, 10, 10),
                            life_time=max(args.dt * 2.0, 0.12),
                        )
                else:
                    robot.hold_still()

                eval_clearance = None
                eval_collision = False
                eval_collision_source = None
                eval_termination_triggered = False
                if eval_logger is not None:
                    eval_clearance = eval_logger.compute_clearance(
                        robot_state,
                        target_state,
                        npc_states,
                        active_target_track_id,
                    )
                    actor_to_track = actor_track_lookup(npc_states)
                    sensor_events = (
                        enrich_sensor_collision_events(
                            collision_monitor.drain(),
                            tick=tick,
                            time_s=now_t - start_t,
                            episode_active=following,
                            actor_to_track=actor_to_track,
                            target_track_id=active_target_track_id,
                        )
                        if collision_monitor is not None
                        else []
                    )
                    geometry_event = (
                        make_geometry_collision_event(
                            tick,
                            now_t - start_t,
                            eval_clearance,
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
                    target_center_dist_m = None
                    if target_state is not None and robot_state is not None:
                        target_center_dist_m = float(
                            np.hypot(robot_state.x - target_state.x, robot_state.y - target_state.y)
                        )
                    target_far_status = target_distance_monitor.update(
                        dt=float(args.dt),
                        eval_time_s=now_t - start_t,
                        following=following,
                        distance_m=target_center_dist_m,
                    )
                    if (
                        not eval_termination_triggered
                        and bool(target_far_status.get("termination_triggered", False))
                    ):
                        termination_reason = str(target_far_status.get("termination_reason"))
                        eval_termination_triggered = True
                    if isinstance(policy_debug_info, dict):
                        policy_debug_info["target_far"] = target_far_status
                    eval_logger.write_step(
                        tick=tick,
                        time_s=now_t - start_t,
                        episode_active=following,
                        paused=paused,
                        target_track_id=active_target_track_id,
                        robot_state=robot_state,
                        target_state=target_state,
                        npc_states=npc_states,
                        visible=visible,
                        pixel_count=pix_count,
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
                        print(f"[EVAL] terminating episode: {termination_reason}")
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
                        tick=tick,
                        robot_state=robot_state,
                        npc_states=npc_states,
                        active_target_id=active_target_track_id,
                        obstacles=_dbg_info.get("obstacles"),
                        traj_points=_dbg_info.get("traj_points"),
                        global_path=_dbg_info.get("global_path"),
                        goal_point=_dbg_info.get("goal_point"),
                        search_samples=_dbg_info.get("search_samples"),
                        search_goal=_dbg_info.get("search_goal"),
                        predicted_target_traj=(
                            _dbg_info.get("predicted_target_traj")
                            or _dbg_info.get("target_predicted_traj")
                        ),
                        map_occupied_cells=_dbg_info.get("map_occupied_cells"),
                        map_observed_free_cells=_dbg_info.get("map_observed_free_cells"),
                        map_outline=_dbg_info.get("map_outline"),
                        map_occupancy_rgba=_dbg_info.get("map_occupancy_rgba"),
                        map_esdf_rgba=_dbg_info.get("map_esdf_rgba"),
                        map_hybrid_rgba=_dbg_info.get("map_hybrid_rgba"),
                        map_debug_extent=_dbg_info.get("map_debug_extent"),
                        map_debug_mode=_dbg_info.get("map_debug_mode"),
                        lidar_range_max=_dbg_info.get("lidar_range_max"),
                        lidar_points_world=_lidar_world,
                    )

                if scene_metrics is not None:
                    scene_metrics.update(
                        npc_states=npc_states,
                        robot_state=robot_state,
                        target_state=target_state,
                        dt=args.dt,
                        following=following,
                        cam_visible=bool(visible),
                    )

                gt_f.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": now_t - start_t,
                            "target_track_id": active_target_track_id,
                            "robot": {"x": robot_state.x, "y": robot_state.y, "z": robot_state.z, "yaw_rad": robot_state.yaw_rad, "speed": robot_state.speed},
                            "npcs": [s.__dict__ for s in npc_states],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                vis_f.write(
                    json.dumps(
                        {
                            "tick": tick,
                            "time_s": now_t - start_t,
                            "target_track_id": active_target_track_id,
                            "target_visible": bool(visible),
                            "target_pixel_count": int(pix_count),
                            "threshold": int(args.visibility_threshold),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # if args.draw_debug and target_state is not None:
                #     world.debug.draw_string(
                #         carla.Location(
                #             x=target_state.x,
                #             y=target_state.y,
                #             z=target_state.z + 2.6,
                #         ),
                #         f"TGT {active_target_track_id} vis={int(visible)} pix={pix_count}",
                #         draw_shadow=False,
                #         color=carla.Color(255, 255, 0),
                #         life_time=max(args.dt * 2.0, 0.12),
                #     )

                # Debug: write state file for live_viewer.py (every 3 ticks)
                if (args.debug or (_debug_proc[0] is not None and _debug_proc[0].poll() is None)) and tick % 3 == 0:
                    npc.write_debug_state(
                        tick=tick,
                        robot_state=robot_state,
                        target_track_id=active_target_track_id or "",
                    )

                draw_status(
                    world,
                    f"spawned={spawned} follow={following} manual={manual_mode} target={active_target_track_id} pix={pix_count}",
                )
                if ui_enabled:
                    surf = pygame.display.get_surface()
                    if surf is not None:
                        surf.fill((18, 22, 28))
                        line1 = (
                            f"State: spawned={spawned} follow={following} manual={manual_mode} target={active_target_track_id} "
                            f"| walkers={desired_num_walkers} speed=[{desired_min_speed:.2f},{desired_max_speed:.2f}]"
                        )
                        if args.teleop:
                            line2 = "Keys: WASD drive  Shift sprint  Mouse look  F pause  R reset  SPACE pause  Q quit"
                        else:
                            line2 = "Keys: G spawn  F follow  D difficulty  V debug-viewer  M manual  B manual-robot  T target  R reset  SPACE pause  Q quit"
                        line3 = (
                            f"Visibility: pix={pix_count} thr={args.visibility_threshold} visible={int(visible)}"
                            f"  |  walkers={desired_num_walkers}  speed=[{desired_min_speed:.1f},{desired_max_speed:.1f}]"
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
                        # Notifications
                        now_n = time.time()
                        active = [(m, e) for m, e in _notifs if e > now_n]
                        _notifs[:] = active
                        for ni, (msg, _) in enumerate(reversed(active[-4:])):
                            ns = ui_font.render(f"▶ {msg}", True, (255, 230, 80))
                            surf.blit(ns, (16, surf.get_height() - 28 - ni * 26))

                        labels = [
                            ("RGB Left",  0, 0, "rgb_left"),
                            ("RGB Front", 1, 0, "rgb"),
                            ("RGB Right", 2, 0, "rgb_right"),
                            ("Depth",     0, 1, "depth"),
                            ("Instance",  1, 1, "instance"),
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
                    if running:
                        step_main_loop(frame_started_at)
                        continue
                else:
                    break

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
        if scene_metrics is not None:
            scene_metrics.save_json(metrics_path)
            print(scene_metrics.one_line_summary())
        if eval_logger is not None:
            finalize_current_eval(termination_reason)
        print("[EP] cleanup done.")
        print(f"[EP] GT stream: {gt_path}")
        print(f"[EP] visibility stream: {vis_path}")
        print(f"[EP] calibration: {calib_path}")


if __name__ == "__main__":
    main()
