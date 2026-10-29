from __future__ import annotations

import json
import os
from typing import Any


def write_schema_docs(run_dir: str) -> None:
    _write_json(os.path.join(run_dir, "eval_result_schema.json"), eval_result_schema())
    _write_json(os.path.join(run_dir, "episode_data_schema.json"), episode_data_schema())


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def field(display_name: str, unit: str | None, dtype: str, description: str, formula: str | None = None, source: str | None = None) -> dict[str, Any]:
    return {
        "display_name": display_name,
        "unit": unit,
        "type": dtype,
        "description": description,
        "formula": formula,
        "source": source,
    }


def eval_result_schema() -> dict[str, Any]:
    return {
        "file": "eval_result.json",
        "note": "Machine-readable metric values. This companion schema explains field meaning, units, formulas, and data sources.",
        "fields": {
            "max_steps": field("Effective Steps", "ticks", "int", "Number of active, non-paused evaluation ticks.", "|S|", "episode_step.jsonl"),
            "max_search_steps": field("Max Search Steps", "ticks", "int", "Allowed maximum consecutive target-invisible ticks.", "round(max_search_time_s / dt)", "episode_meta.json"),
            "obstacle_avoidance_success": field("Obstacle Avoidance Success", None, "bool", "True when no active robot-human collision is detected.", "human_collision_count == 0 and no active collision step", "collision_events.jsonl, episode_step.jsonl"),
            "target_identity_success": field("Target Identity Success", None, "bool", "True when perception did not terminate from sustained non-target identity lock.", "termination_reason != target_identity_lost", "episode_meta.json, episode_step.jsonl/target_identity"),
            "target_identity_switch_count": field("Target Identity Switch Count", "events", "int", "Number of sustained non-target identity mismatch episodes seen by the watchdog.", "max(target_identity.switch_count)", "episode_step.jsonl/target_identity"),
            "max_identity_mismatch_duration_s": field("Max Identity Mismatch Duration", "s", "float", "Longest continuous duration where perception was associated to a non-target pedestrian far from the true target.", "max(target_identity.max_mismatch_duration_s)", "episode_step.jsonl/target_identity"),
            "identity_failure_track_id": field("Identity Failure Track", None, "string|null", "GT track id associated with the perception target when identity failure triggered.", None, "episode_step.jsonl/target_identity"),
            "target_visibility_ratio": field("Target Visibility Ratio", "ratio", "float", "Fraction of active ticks where target is visible.", "N_visible / max_steps", "episode_step.jsonl/visibility.target_visible"),
            "search_success": field("Search Success", None, "bool", "True when the longest continuous target-lost interval does not exceed max_search_steps.", "max_consecutive_invisible_steps <= max_search_steps", "episode_step.jsonl/visibility.target_visible"),
            "search_path_length": field("Search Path Length", "m", "float", "Robot path length accumulated while target is invisible.", "sum(||p_t - p_{t-1}||) for invisible ticks", "episode_step.jsonl/robot"),
            "avg_robot_target_dist_no_radius": field("Average Robot-Target Center Distance", "m", "float", "Average center distance between robot and target, without subtracting radii.", "mean(||p_robot - p_target||)", "episode_step.jsonl/robot,target"),
            "path_length": field("Path Length", "m", "float", "Robot path length over active evaluation ticks.", "sum(||p_t - p_{t-1}||)", "episode_step.jsonl/robot"),
            "avg_velocity": field("Average Velocity", "m/s", "float", "Average robot speed computed from position differences.", "mean(||(p_t - p_{t-1}) / dt||)", "episode_step.jsonl/robot"),
            "avg_acceleration": field("Average Acceleration", "m/s^2", "float", "Average acceleration norm computed from velocity differences.", "mean(||(v_t - v_{t-1}) / dt||)", "episode_step.jsonl/robot"),
            "avg_jerk": field("Average Jerk", "m/s^3", "float", "Average jerk norm computed from acceleration differences.", "mean(||(a_t - a_{t-1}) / dt||)", "episode_step.jsonl/robot"),
            "time_in_target_personal_zone": field("Time In Target Personal Zone", "s", "float", "Time where robot-target surface distance is inside the configured personal zone band.", "sum(dt for zone_min < d_target_surface < zone_max)", "episode_step.jsonl, episode_meta.json"),
            "time_in_human_private_zone": field("Time In Human Private Zone", "s", "float", "Time where robot is closer than zone_min to any non-target pedestrian.", "sum(dt for min_human_surface_dist < zone_min)", "episode_step.jsonl"),
            "time_in_target_search": field("Time In Target Search", "s", "float", "Total active time where target is not visible.", "sum(dt for target_visible == false)", "episode_step.jsonl/visibility.target_visible"),
            "total_time": field("Total Evaluation Time", "s", "float", "Active evaluation time.", "max_steps * dt", "episode_meta.json, episode_step.jsonl"),
            "active_total_time": field("Active Total Time", "s", "float", "Active non-paused evaluation time; currently equivalent to total_time.", "max_steps * dt", "episode_step.jsonl"),
            "termination_reason": field("Termination Reason", None, "string|null", "Reason episode ended, if explicitly recorded.", None, "episode_meta.json"),
            "task_success": field("Task Success", None, "bool", "Follow-Bench 1.0 / EVT-Bench criterion: no collision, no long-term target loss, and robot-target distance <= target_loss_dist_m at termination. Evaluated from the final state regardless of termination reason.", "collision_free && !long_term_target_loss && final_robot_target_dist <= target_loss_dist_m", "episode_step.jsonl, collision_events.jsonl, episode_meta.json"),
            "collision_free": field("Collision Free", None, "bool", "True when no active collision of any type (pedestrian, target, obstacle) is recorded by sensor, map monitor, step flag, or geometry fallback.", "obstacle_avoidance_success && collision_count == 0", "collision_events.jsonl, episode_step.jsonl"),
            "long_term_target_loss": field("Long-Term Target Loss", None, "bool", "True when robot-target center distance exceeds target_loss_dist_m for more than N_lost consecutive steps.", "max_consecutive_target_loss_steps > target_loss_n_lost_steps or termination_reason == target_too_far", "episode_step.jsonl/robot,target"),
            "final_robot_target_dist": field("Final Robot-Target Distance", "m", "float|null", "Robot-target center distance at the last active step with valid poses.", "||p_robot(T) - p_target(T)||", "episode_step.jsonl/robot,target"),
            "final_within_loss_dist": field("Final Within Loss Distance", None, "bool", "True when the final robot-target distance is at most target_loss_dist_m.", "final_robot_target_dist <= target_loss_dist_m", "episode_step.jsonl"),
            "max_consecutive_target_loss_steps": field("Max Consecutive Target Loss Steps", "ticks", "int", "Longest run of consecutive steps with robot-target distance > target_loss_dist_m.", None, "episode_step.jsonl"),
            "target_loss_steps": field("Target Loss Steps", "ticks", "int", "Total steps with robot-target distance > target_loss_dist_m (short losses included).", None, "episode_step.jsonl"),
            "target_loss_ratio": field("Target Loss Ratio", "ratio", "float", "Fraction of active steps in target loss.", "target_loss_steps / max_steps", "episode_step.jsonl"),
            "target_loss_dist_m": field("Target Loss Distance", "m", "float", "Robot-target center distance threshold for target loss.", None, "episode_meta.json/target_loss.dist_m"),
            "target_loss_n_lost_steps": field("N_lost", "ticks", "int", "Consecutive-step threshold beyond which a target loss becomes long-term.", None, "episode_meta.json/target_loss.n_lost_steps"),
            "target_route_completed": field("Target Route Completed", None, "bool", "True when the episode ended because the target finished its trajectory (full horizon).", "termination_reason in {target_round_trip_done, target_completed_route, target_completed_a_to_b}", "episode_meta.json"),
            "collision_count": field("Collision Count", "events", "int", "Number of active collision events.", "count(active collision_events)", "collision_events.jsonl"),
            "human_collision_count": field("Human Collision Count", "events", "int", "Number of active robot-human collision events.", "count(active human collision_events)", "collision_events.jsonl"),
            "geometry_collision_detected": field("Geometry Collision Detected", None, "bool", "True when geometry fallback detects overlap in active steps.", "any(surface_dist <= human_collision_margin)", "episode_step.jsonl/clearance"),
            "min_human_surface_dist": field("Minimum Human Surface Distance", "m", "float|null", "Minimum robot-to-non-target pedestrian surface distance in active steps.", "min(min_human_surface_dist)", "episode_step.jsonl/clearance"),
            "planner_cost_ms_mean": field("Policy Total Mean Time (legacy)", "ms", "float|null", "Legacy timing field. Equivalent to policy_total_ms_mean.", "mean(timing.policy_total_ms)", "episode_step.jsonl/timing"),
            "planner_cost_ms_p95": field("Policy Total P95 Time (legacy)", "ms", "float|null", "Legacy timing field. Equivalent to policy_total_ms_p95.", "percentile95(timing.policy_total_ms)", "episode_step.jsonl/timing"),
            "policy_total_ms_mean": field("Policy Total Mean Time", "ms", "float|null", "Mean wall-clock time of policy.act(obs), including perception, planner, and wrapper overhead.", "mean(timing.policy_total_ms)", "episode_step.jsonl/timing.policy_total_ms"),
            "policy_total_ms_p95": field("Policy Total P95 Time", "ms", "float|null", "95th percentile wall-clock time of policy.act(obs).", "percentile95(timing.policy_total_ms)", "episode_step.jsonl/timing.policy_total_ms"),
            "planner_core_ms_mean": field("Planner Core Mean Time", "ms", "float|null", "Mean pure planner/controller execution time, excluding perception when available.", "mean(timing.planner_core_ms)", "episode_step.jsonl/timing.planner_core_ms"),
            "planner_core_ms_p95": field("Planner Core P95 Time", "ms", "float|null", "95th percentile pure planner/controller execution time.", "percentile95(timing.planner_core_ms)", "episode_step.jsonl/timing.planner_core_ms"),
            "perception_total_ms_mean": field("Perception Mean Time", "ms", "float|null", "Mean perception-layer total time, when perception timing is available.", "mean(timing.perception_total_ms)", "episode_step.jsonl/timing.perception_total_ms"),
            "perception_total_ms_p95": field("Perception P95 Time", "ms", "float|null", "95th percentile perception-layer total time.", "percentile95(timing.perception_total_ms)", "episode_step.jsonl/timing.perception_total_ms"),
            "wrapper_overhead_ms_mean": field("Wrapper Overhead Mean Time", "ms", "float|null", "Mean overhead not attributed to perception_total_ms or planner_core_ms.", "mean(timing.wrapper_overhead_ms)", "episode_step.jsonl/timing.wrapper_overhead_ms"),
            "wrapper_overhead_ms_p95": field("Wrapper Overhead P95 Time", "ms", "float|null", "95th percentile wrapper overhead.", "percentile95(timing.wrapper_overhead_ms)", "episode_step.jsonl/timing.wrapper_overhead_ms"),
            "perception_detection_ms_mean": field("Detection Mean Time", "ms", "float|null", "Mean detector time, e.g. YOLO forward pass.", "mean(timing.perception.detection_ms)", "episode_step.jsonl/timing.perception"),
            "perception_detection_ms_p95": field("Detection P95 Time", "ms", "float|null", "95th percentile detector time.", "percentile95(timing.perception.detection_ms)", "episode_step.jsonl/timing.perception"),
            "perception_tracking_ms_mean": field("Tracking Mean Time", "ms", "float|null", "Mean tracker/association time.", "mean(timing.perception.tracking_ms)", "episode_step.jsonl/timing.perception"),
            "perception_tracking_ms_p95": field("Tracking P95 Time", "ms", "float|null", "95th percentile tracker/association time.", "percentile95(timing.perception.tracking_ms)", "episode_step.jsonl/timing.perception"),
            "perception_reid_ms_mean": field("ReID Mean Time", "ms", "float|null", "Mean ReID feature extraction/matching time.", "mean(timing.perception.reid_ms)", "episode_step.jsonl/timing.perception"),
            "perception_reid_ms_p95": field("ReID P95 Time", "ms", "float|null", "95th percentile ReID time.", "percentile95(timing.perception.reid_ms)", "episode_step.jsonl/timing.perception"),
            "perception_mapping_ms_mean": field("Mapping Mean Time", "ms", "float|null", "Mean depth projection, BEV/world mapping, or multi-view fusion time.", "mean(timing.perception.mapping_ms)", "episode_step.jsonl/timing.perception"),
            "perception_mapping_ms_p95": field("Mapping P95 Time", "ms", "float|null", "95th percentile mapping time.", "percentile95(timing.perception.mapping_ms)", "episode_step.jsonl/timing.perception"),
            "perception_fsm_ms_mean": field("FSM Mean Time", "ms", "float|null", "Mean target selection/lost/reacquire FSM time.", "mean(timing.perception.fsm_ms)", "episode_step.jsonl/timing.perception"),
            "perception_fsm_ms_p95": field("FSM P95 Time", "ms", "float|null", "95th percentile FSM time.", "percentile95(timing.perception.fsm_ms)", "episode_step.jsonl/timing.perception"),
        },
    }


def episode_data_schema() -> dict[str, Any]:
    return {
        "files": {
            "episode_meta.json": {
                "description": "Episode configuration, geometry, thresholds, and termination metadata.",
                "key_fields": {
                    "scenario_type": "Scenario family, e.g. corner/park/passage.",
                    "scenario_name": "Auto-generated scenario configuration name.",
                    "scenario_variant": "Key scenario parameters used for filtering and reproduction.",
                    "num_walkers_total": "Total pedestrians spawned, including target.",
                    "num_pedestrians": "Non-target pedestrian count used in H naming.",
                    "robot/target/human": "Geometry radii and footprints used for clearance and collision metrics.",
                    "collision": "Collision sensor and geometry fallback settings.",
                },
            },
            "episode_step.jsonl": {
                "description": "One JSON object per tick. Metrics use active records where episode_active=true and paused=false.",
                "key_fields": {
                    "planner_cost_ms": "Legacy per-step policy total time. Same as timing.policy_total_ms.",
                    "timing.policy_total_ms": "Whole policy.act(obs) wall-clock time.",
                    "timing.perception_total_ms": "Perception layer total time when available.",
                    "timing.planner_core_ms": "Pure planner/controller time when available.",
                    "timing.perception": "Perception sub-timings: detection, tracking, reid, mapping, fsm.",
                    "perception": "Compact perception diagnostics: FSM state, target lock/lost flags, active track id, target bbox/view, tracker target xy, bbox counts by view, and planner-facing target state.",
                    "planner": "Compact planner diagnostics: search/reacquire mode, target protection flags, search anchor/goal, hidden-step count, and counts for obstacles/trajectory/search samples.",
                    "clearance": "Robot-target and robot-human surface distances.",
                    "target_identity": "Perception-vs-GT identity watchdog diagnostics, present during evaluation.",
                },
            },
            "collision_events.jsonl": {
                "description": "One JSON object per collision event. Sources are collision_sensor and geometry.",
                "key_fields": {
                    "episode_active": "Whether event happened inside the active evaluation interval.",
                    "source": "collision_sensor or geometry.",
                    "is_target": "True if the collided pedestrian is the target.",
                },
            },
        }
    }
