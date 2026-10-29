"""
RDA-MPC follower with RGB+depth perception (depth-TPT variant).

Mirrors ``adapters.rda_lidar_adapter.py`` but the perception backbone is
``PedTrackerDepth``: YOLO bbox -> depth -> back-project to world BEV.
"""
from __future__ import annotations

import os
import sys
import time


from baselines.adapters.rda_lidar_adapter import RdaLidarFollowerPolicy
from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter
from baselines.perception.gt.gt_visibility import gt_target_xy
from baselines.perception.sensor.people_pipeline import PeopleTrackerPipeline

try:
    from followbench.perception.ped_tracker_depth import PedTrackerDepth, ViewInputs

    _HAS_TRACKER = True
except ImportError:
    _HAS_TRACKER = False


class RdaDepthTptFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        desired_distance: float = 1.5,
        follow_position: str = "back",
        yolo_model: str = "yolo11s.pt",
        tracker_cfg: str = "botsort.yaml",
        tracker_conf: float = 0.25,
        tracker_iou: float = 0.7,
        tracker_imgsz: int = 640,
        tracker_device: str = "cuda",
        tracker_yolo_stride: int = 1,
        tracker_max_range_m: float = 15.0,
        receding: int = 10,
        iter_num: int = 4,
        process_num: int = 1,
        max_obs_num: int = 5,
    ) -> None:
        if not _HAS_TRACKER:
            raise ImportError(
                "PedTrackerDepth not found. Make sure followbench.perception "
                "is present and ultralytics is installed."
            )

        self._inner = RdaLidarFollowerPolicy(
            dt=dt,
            desired_distance=desired_distance,
            follow_position=follow_position,
            receding=receding,
            iter_num=iter_num,
            process_num=process_num,
            max_obs_num=max_obs_num,
            use_npc_gt_obstacles=True,
        )
        self._tracker = PedTrackerDepth(
            model_name=yolo_model,
            tracker=tracker_cfg,
            conf=tracker_conf,
            iou=tracker_iou,
            imgsz=tracker_imgsz,
            device=tracker_device,
            yolo_stride=tracker_yolo_stride,
            max_range_m=tracker_max_range_m,
            dt=dt,
        )
        self._people = PeopleTrackerPipeline(
            tracker=self._tracker,
            tracker_update_fn=_run_depth_tracker,
            synthetic_target_track_id="T_DPT",
            synthetic_npc_track_id_fn=_depth_tpt_npc_track_id,
        )
        self._last_debug: dict = {
            "obstacles": [],
            "traj_points": [],
            "tracked_peds": [],
            "track_bboxes_by_view": {},
            "track_bboxes_age": 0,
        }

    def reset(self) -> None:
        self._inner.reset()
        self._people.reset()
        self._last_debug = {
            "obstacles": [],
            "traj_points": [],
            "tracked_peds": [],
            "track_bboxes_by_view": {},
            "track_bboxes_age": 0,
        }

    def get_debug_info(self) -> dict:
        return self._last_debug

    def act(self, obs: FollowObservation) -> FollowAction:
        t_perception = time.perf_counter()
        result = self._people.update(obs)
        measured_perception_ms = (time.perf_counter() - t_perception) * 1000.0
        tracked_peds = result.tracked_peds
        target_ped = result.target_ped
        modified_obs = result.modified_obs
        timing = result.timing
        perception_total_ms = float(timing.get("percep_ms", measured_perception_ms))
        gt_xy = gt_target_xy(obs)

        if result.using_tracker_target:
            gt_str = (
                f"  gt=({gt_xy[0]:.2f},{gt_xy[1]:.2f})  err={result.target_error_m:.2f}m"
                if gt_xy is not None and result.target_error_m is not None
                else ("  gt=N/A" if gt_xy is None else f"  gt=({gt_xy[0]:.2f},{gt_xy[1]:.2f})")
            )
            print(
                f"[DPT] tick={obs.tick}  target_id={result.target_tracker_id}"
                f"  pos=({target_ped.x:.2f},{target_ped.y:.2f})"
                f"{gt_str}"
                f"  npcs={len(modified_obs.npcs)}  total_tracks={len(tracked_peds)}"
                f"  t_track={timing.get('track_ms', 0.0):.1f}ms"
                f" (yolo={timing.get('yolo_ms', 0.0):.1f}"
                f" depth={timing.get('depth_ms', 0.0):.1f}"
                f" proj={timing.get('project_ms', 0.0):.1f}"
                f" yolo_run={int(timing.get('ran_yolo', 0.0))})"
                f"  t_id={timing.get('id_ms', 0.0):.1f}ms"
                f"  t_percep={timing.get('percep_ms', 0.0):.1f}ms"
            )
        else:
            gt_str = f"  gt=({gt_xy[0]:.2f},{gt_xy[1]:.2f})" if gt_xy is not None else "  gt=N/A"
            print(
                f"[DPT] tick={obs.tick}  target lost — falling back to GT"
                f"{gt_str}"
                f"  total_tracks={len(tracked_peds)}"
                f"  t_track={timing.get('track_ms', 0.0):.1f}ms"
                f" (yolo={timing.get('yolo_ms', 0.0):.1f}"
                f" depth={timing.get('depth_ms', 0.0):.1f}"
                f" proj={timing.get('project_ms', 0.0):.1f}"
                f" yolo_run={int(timing.get('ran_yolo', 0.0))})"
                f"  t_id={timing.get('id_ms', 0.0):.1f}ms"
                f"  t_percep={timing.get('percep_ms', 0.0):.1f}ms"
            )

        t_planner = time.perf_counter()
        action = self._inner.act(modified_obs)
        planner_core_ms = (time.perf_counter() - t_planner) * 1000.0

        inner_debug = self._inner.get_debug_info()
        tracker_xy = (float(target_ped.x), float(target_ped.y)) if target_ped is not None else None
        self._last_debug = {
            "obstacles": inner_debug.get("obstacles", []),
            "traj_points": inner_debug.get("traj_points", []),
            "goal_point": inner_debug.get("goal_point"),
            "tracked_peds": [
                {
                    "track_id": tp.track_id,
                    "x": tp.x,
                    "y": tp.y,
                    "is_target": (tp.track_id == result.target_tracker_id),
                }
                for tp in tracked_peds
            ],
            "track_bboxes_by_view": result.track_bboxes_by_view,
            "track_bboxes_age": int(
                (getattr(self._tracker, "last_yolo_age_by_view", {}) or {}).get("front", 0)
            ),
            "target_track_id": result.target_tracker_id,
            "target_pos": {
                "tracker": tracker_xy,
                "gt": gt_xy,
                "err_m": result.target_error_m,
            },
            "timing": {
                "perception_total_ms": perception_total_ms,
                "planner_core_ms": planner_core_ms,
                "perception": {
                    "detection_ms": timing.get("yolo_ms", 0.0),
                    "tracking_ms": timing.get("track_ms", 0.0),
                    "reid_ms": timing.get("id_ms", 0.0),
                    "mapping_ms": float(timing.get("depth_ms", 0.0)) + float(timing.get("project_ms", 0.0)) + float(timing.get("merge_ms", 0.0)),
                    "fsm_ms": None,
                    "other_ms": None,
                    "total_ms": perception_total_ms,
                    "raw": dict(timing),
                },
                "track_ms": timing.get("track_ms", 0.0),
                "id_ms": timing.get("id_ms", 0.0),
                "percep_ms": perception_total_ms,
                "yolo_ms": timing.get("yolo_ms", 0.0),
                "depth_ms": timing.get("depth_ms", 0.0),
                "project_ms": timing.get("project_ms", 0.0),
                "ran_yolo": bool(timing.get("ran_yolo", 0.0)),
            },
        }
        return action


def _run_depth_tracker(tracker, obs: FollowObservation) -> list:
    """Build a multi-view views dict from the observation (front + optional
    left/right) and forward it to the tracker. Missing views are skipped."""
    views: dict = {}
    if obs.rgb_image is not None:
        views["front"] = ViewInputs(
            rgb=obs.rgb_image,
            depth=obs.depth_image,
            intrinsics=obs.rgb_intrinsics,
            extrinsics=obs.rgb_extrinsics_robot_to_sensor,
        )
    if obs.rgb_image_left is not None:
        views["left"] = ViewInputs(
            rgb=obs.rgb_image_left,
            depth=obs.depth_image_left,
            intrinsics=obs.rgb_intrinsics_left,
            extrinsics=obs.rgb_extrinsics_left_robot_to_sensor,
        )
    if obs.rgb_image_right is not None:
        views["right"] = ViewInputs(
            rgb=obs.rgb_image_right,
            depth=obs.depth_image_right,
            intrinsics=obs.rgb_intrinsics_right,
            extrinsics=obs.rgb_extrinsics_right_robot_to_sensor,
        )
    return tracker.update(
        views=views,
        robot_x=float(obs.robot.x),
        robot_y=float(obs.robot.y),
        robot_z=float(obs.robot.z),
        robot_yaw=float(obs.robot.yaw_rad),
    )


def _depth_tpt_npc_track_id(track_id: int) -> str:
    return f"D{track_id:03d}" if int(track_id) >= 0 else "DL00"
