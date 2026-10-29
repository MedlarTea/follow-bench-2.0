from __future__ import annotations

import os
import re
from typing import Any, Mapping, Optional

from followbench.paths import EVAL_LOGS_DIR

DEFAULT_LOG_ROOT = str(EVAL_LOGS_DIR)

DEFAULT_SIM_DT = 0.1
DEFAULT_TARGET_LOSS_DIST_M = 3.0
DEFAULT_TARGET_LOSS_N_LOST_STEPS = 100
DEFAULT_TARGET_LOSS_WARMUP_SEC = 0.0

DEFAULT_ROBOT_RADIUS = 0.50
DEFAULT_ROBOT_LENGTH = 1.00
DEFAULT_ROBOT_WIDTH = 0.80
DEFAULT_HUMAN_RADIUS = 0.35
DEFAULT_TARGET_RADIUS = 0.35
DEFAULT_ZONE_THRESHOLDS = [0.45, 1.20]
DEFAULT_MAX_SEARCH_TIME_S = 5.0
DEFAULT_HUMAN_COLLISION_MARGIN = 0.0
DEFAULT_TARGET_IDENTITY_ASSOCIATE_GATE_M = 1.5
DEFAULT_TARGET_IDENTITY_WRONG_DIST_M = 7.0
DEFAULT_TARGET_IDENTITY_STARTUP_GRACE_SEC = 3.0
DEFAULT_TARGET_IDENTITY_CONFIRM_SEC = 3.0
DEFAULT_TARGET_IDENTITY_FAIL_SEC = 5.0


def add_evaluation_args(parser) -> None:
    parser.add_argument(
        "--enable-evaluation",
        action="store_true",
        help="Enable FollowBench 2.0 evaluation logging and per-episode eval_result.json.",
    )
    parser.add_argument(
        "--eval-log-root",
        type=str,
        default=DEFAULT_LOG_ROOT,
        help="Root directory for evaluation logs.",
    )
    parser.add_argument(
        "--eval-scenario-type",
        type=str,
        default=None,
        help="Scenario type used in evaluation log paths. Defaults to the scenario directory name.",
    )
    parser.add_argument(
        "--eval-trial-id",
        type=int,
        default=None,
        help="Trial id. If omitted, the next id under the same config directory is selected.",
    )
    parser.add_argument(
        "--max-search-time-s",
        type=float,
        default=DEFAULT_MAX_SEARCH_TIME_S,
        help="Maximum continuous target-invisible time before search_success becomes false.",
    )
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument(
        "--terminate-on-human-collision",
        dest="terminate_on_human_collision",
        action="store_true",
        default=True,
        help="End the episode when the robot collides with any pedestrian.",
    )
    collision_group.add_argument(
        "--no-terminate-on-human-collision",
        dest="terminate_on_human_collision",
        action="store_false",
        help="Record human collisions without ending the episode.",
    )
    parser.add_argument(
        "--human-collision-margin",
        type=float,
        default=DEFAULT_HUMAN_COLLISION_MARGIN,
        help="Geometry fallback margin in metres. min human surface distance <= margin is a collision.",
    )
    parser.add_argument(
        "--disable-collision-sensor",
        action="store_true",
        help="Disable CARLA sensor.other.collision and rely on geometry fallback only.",
    )
    identity_group = parser.add_mutually_exclusive_group()
    identity_group.add_argument(
        "--terminate-on-target-identity-loss",
        dest="terminate_on_target_identity_loss",
        action="store_true",
        default=True,
        help="End perception episodes when the tracker consistently locks onto a non-target pedestrian.",
    )
    identity_group.add_argument(
        "--no-terminate-on-target-identity-loss",
        dest="terminate_on_target_identity_loss",
        action="store_false",
        help="Log perception target identity mismatches without ending the episode.",
    )
    parser.add_argument(
        "--target-identity-associate-gate-m",
        type=float,
        default=DEFAULT_TARGET_IDENTITY_ASSOCIATE_GATE_M,
        help="Tracker-to-GT association gate in metres for target identity checks.",
    )
    parser.add_argument(
        "--target-identity-wrong-dist-m",
        type=float,
        default=DEFAULT_TARGET_IDENTITY_WRONG_DIST_M,
        help="Tracker must be at least this far from the true target before a non-target association counts as wrong.",
    )
    parser.add_argument(
        "--target-identity-startup-grace-sec",
        type=float,
        default=DEFAULT_TARGET_IDENTITY_STARTUP_GRACE_SEC,
        help="Initial evaluation time ignored by the target identity watchdog.",
    )
    parser.add_argument(
        "--target-identity-confirm-sec",
        type=float,
        default=DEFAULT_TARGET_IDENTITY_CONFIRM_SEC,
        help="Continuous wrong-target duration that marks the identity watchdog status as suspect.",
    )
    parser.add_argument(
        "--target-identity-fail-sec",
        type=float,
        default=DEFAULT_TARGET_IDENTITY_FAIL_SEC,
        help="Continuous wrong-target duration that terminates the episode as target_identity_lost.",
    )


def format_distance(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text if text else "0"


def trial_dir_name(trial_id: int, timestamp: str) -> str:
    return f"trial_{int(trial_id):03d}_{timestamp}"


def next_trial_id(config_dir: str) -> int:
    if not os.path.isdir(config_dir):
        return 1
    max_id = 0
    pattern = re.compile(r"^trial_(\d+)(?:_|$)")
    for name in os.listdir(config_dir):
        match = pattern.match(name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def num_pedestrians_from_total(num_walkers_total: int) -> int:
    return max(int(num_walkers_total) - 1, 0)


def scenario_name_from_args(args: Any, scenario_type: str) -> str:
    scenario_type = str(scenario_type or "unknown")
    if scenario_type == "corner":
        flow = getattr(args, "npc_flow_mode", None)
        nav = getattr(args, "npc_navigation_mode", None)
        return "_".join(str(p) for p in (scenario_type, flow, nav) if p)
    if scenario_type == "park":
        difficulty = getattr(args, "difficulty", None)
        return "_".join(str(p) for p in (scenario_type, difficulty) if p)
    if scenario_type == "passage":
        return "passage"
    if scenario_type == "plaza":
        difficulty = getattr(args, "difficulty", None)
        flow = getattr(args, "flow_mode", None)
        return "_".join(str(p) for p in (scenario_type, flow, difficulty) if p)
    return scenario_type


def scenario_variant_from_args(args: Any, scenario_type: str) -> dict:
    common_keys = [
        "robot_spawn_mode",
        "robot_spawn_clearance_m",
        "robot_spawn_retry_attempts",
        "npc_min_speed",
        "npc_max_speed",
        "npc_rng_seed",
    ]
    scenario_keys = {
        "corner": [
            "npc_flow_mode",
            "npc_navigation_mode",
            "npc_pair_spawn_mode",
            "npc_pair_goal_policy",
            "npc_pair_target_band_half_width_m",
            "right_hand_lane_sep_m",
            "side_follow_robot_yaw_policy",
            "target_lane_bias_mode",
            "target_lane_bias_scale",
            "target_lane_min_offset_m",
            "target_lane_max_offset_m",
            "target_lateral_jitter_m",
            "target_turnaround_forward_m",
            "target_turnaround_merge_m",
            "target_turnaround_side_m",
            "target_turnaround_samples",
            "target_turnaround_speed_mps",
            "crowd_turn_sublane_mode",
            "crowd_turn_radius_jitter_m",
            "crowd_turn_min_offset_keep_ratio",
            "crowd_turn_samples",
            "target_required_legs",
            "max_active_sec",
            "npc_replan_on_block",
            "npc_block_replan_window_sec",
            "npc_block_replan_progress_thresh",
            "npc_block_replan_risk_thresh",
            "npc_astar_replan_cooldown_sec",
        ],
        "park": [
            "npc_flow_mode",
            "npc_navigation_mode",
            "difficulty",
            "use_scheduled_flow",
            "scheduled_flow_dwell_s",
            "npc_strict_flow_endpoints",
            "npc_replan_on_avoid",
            "npc_avoid_replan_threshold",
            "npc_astar_replan_cooldown_sec",
            "use_orca_npc",
        ],
        "passage": [
            "routes_json",
            "flow_points_json",
        ],
        "plaza": [
            "flow_mode",
            "difficulty",
            "flow_points_json",
        ],
    }
    variant = {}
    for key in common_keys + scenario_keys.get(str(scenario_type or ""), []):
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                variant[key] = value
    return variant


def state_to_dict(state: Any) -> Optional[dict]:
    if state is None:
        return None
    out = {}
    for key in ("track_id", "actor_id", "x", "y", "z", "vx", "vy", "vz", "yaw_deg", "speed", "npc_state", "wp_x", "wp_y"):
        if hasattr(state, key):
            value = getattr(state, key)
            if value is not None:
                out[key] = value
    return out


def robot_to_dict(robot_state: Any) -> Optional[dict]:
    if robot_state is None:
        return None
    return {
        "x": float(robot_state.x),
        "y": float(robot_state.y),
        "z": float(robot_state.z),
        "yaw_rad": float(robot_state.yaw_rad),
        "speed": float(getattr(robot_state, "speed", 0.0)),
    }


def planner_mode_from_debug(debug_info: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not debug_info:
        return None
    for key in ("search_mode", "mode", "task_mode", "state_name", "planner_mode"):
        value = debug_info.get(key)
        if value is not None:
            return str(value)
    return None


def actor_footprint(actor: Any, fallback_radius: float, fallback_length: float | None = None, fallback_width: float | None = None) -> dict:
    radius = float(fallback_radius)
    length = None if fallback_length is None else float(fallback_length)
    width = None if fallback_width is None else float(fallback_width)
    try:
        extent = actor.bounding_box.extent
        length = float(extent.x * 2.0)
        width = float(extent.y * 2.0)
        radius = float(max(extent.x, extent.y))
    except Exception:
        pass
    out = {"radius": radius}
    if length is not None:
        out["length"] = length
    if width is not None:
        out["width"] = width
    return out
