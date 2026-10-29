"""
DebugVisualizer — real-time 2D top-down debug view for FollowBench episodes.

Runs a separate process so matplotlib never blocks the CARLA simulation loop.
The main process calls update() each tick; frames are dropped if the visualizer
is busy (queue full) so the simulation is never stalled.

Uses multiprocessing "spawn" context so the child starts completely clean —
this is critical because by the time start() is called, pygame is already
initialized in the parent and a fork() would inherit its X11 connection,
preventing Tk from opening a second display connection.

The view combines a scene-level static ROI map with planner-side per-tick
debug overlays:

- Static ROI image:
  background crop from the scenario grid asset. White = walkable ROI,
  dark grey = static wall/obstacle cells in the offline map, light grey =
  outside the ROI but still inside the crop.
- Local occupancy grid:
  planner-side LiDAR local map around the robot. In the default hybrid view,
  occupancy classes remain visible while observed-free cells also carry a
  semi-transparent ESDF heatmap. Very light grey = unknown, yellow =
  inflated obstacle buffer, dark grey = occupied hit cells, grey outline =
  current local-map bounds, and the colored free-space tint indicates ESDF
  distance within observed-free cells. Press V in the debug window to cycle
  occupancy / esdf / hybrid map display modes.
- Robot and NPCs:
  green rectangle with heading arrow for the robot, blue circles for NPCs,
  orange circle for the active target.
- Planner overlays:
  green curve = A* global path on the ROI map, red curve = planned robot
  trajectory, orange dashed line = predicted target trajectory, dark sample
  points = search candidates, red x = selected follow/search goal, pink
  polygons = obstacle boxes passed into the planner, cyan dots = single-line
  LiDAR hits (world XY, range < 10 m).

This window is meant to explain what the planner believes locally, not to be
photorealistic or sensor-faithful. In particular, the local occupancy grid is a
planner debug product derived from LiDAR hits and inflation, so it may not
match the static ROI map one-to-one.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import sys
import time
from typing import List, Optional

import numpy as np


# ── Robot geometry ────────────────────────────────────────────────────────────
_ROBOT_LENGTH = 0.6
_ROBOT_WIDTH  = 0.4
_NPC_RADIUS   = 0.25


def _rect_vertices(x: float, y: float, yaw: float,
                   length: float = _ROBOT_LENGTH,
                   width: float  = _ROBOT_WIDTH) -> np.ndarray:
    ca, sa = math.cos(yaw), math.sin(yaw)
    hw, hl = width / 2.0, length / 2.0
    local = [( hl,  hw), (-hl,  hw), (-hl, -hw), ( hl, -hw)]
    return np.array([
        [x + lx * ca - ly * sa,
         y + lx * sa + ly * ca]
        for lx, ly in local
    ])


# ── Occupancy-map helpers ─────────────────────────────────────────────────────

def _build_occ_image(npz_path: str, cx: float, cy: float, half: float):
    try:
        data = np.load(npz_path, allow_pickle=True)
        world_min = np.array(data["world_min"], dtype=np.float64)
        world_max = np.array(data["world_max"], dtype=np.float64)
        roi_mask = np.array(data["roi_mask"],      dtype=np.uint8)
        obs_raw  = np.array(data["obstacle_raw"],  dtype=np.uint8)

        nrows, ncols = roi_mask.shape
        res_x = (world_max[0] - world_min[0]) / ncols
        res_y = (world_max[1] - world_min[1]) / nrows

        gx_lo = max(0,      int((cx - half - world_min[0]) / res_x))
        gx_hi = min(ncols,  int((cx + half - world_min[0]) / res_x) + 1)
        gy_lo = max(0,      int((cy - half - world_min[1]) / res_y))
        gy_hi = min(nrows,  int((cy + half - world_min[1]) / res_y) + 1)

        roi_crop = roi_mask[gy_lo:gy_hi, gx_lo:gx_hi]
        obs_crop = obs_raw [gy_lo:gy_hi, gx_lo:gx_hi]

        # roi_mask==1  → white   (walkable corner)
        # obstacle_raw→ dark grey (actual walls, un-inflated)
        # elsewhere   → light grey (out-of-ROI non-wall space)
        img = np.full((*roi_crop.shape, 3), 200, dtype=np.uint8)  # default light grey
        img[obs_crop == 1] = [110, 110, 110]   # walls → dark grey
        img[roi_crop == 1] = [255, 255, 255]   # walkable → white

        ext = [
            world_min[0] + gx_lo * res_x,
            world_min[0] + gx_hi * res_x,
            world_min[1] + gy_lo * res_y,
            world_min[1] + gy_hi * res_y,
        ]
        return img, ext
    except Exception as e:
        print(f"[DebugVis] occ-map load failed: {e}", flush=True)
        return None, None


def _map_center_from_npz(npz_path: str):
    try:
        data = np.load(npz_path, allow_pickle=True)
        world_min = np.array(data["world_min"], dtype=np.float64)
        world_max = np.array(data["world_max"], dtype=np.float64)
        roi_mask = np.array(data["roi_mask"], dtype=np.uint8)

        nrows, ncols = roi_mask.shape
        ys, xs = np.where(roi_mask == 1)
        if xs.size == 0 or ys.size == 0:
            return float((world_min[0] + world_max[0]) * 0.5), float((world_min[1] + world_max[1]) * 0.5)

        res_x = (world_max[0] - world_min[0]) / ncols
        res_y = (world_max[1] - world_min[1]) / nrows
        gx_lo = int(xs.min())
        gx_hi = int(xs.max()) + 1
        gy_lo = int(ys.min())
        gy_hi = int(ys.max()) + 1

        cx = world_min[0] + 0.5 * (gx_lo + gx_hi) * res_x
        cy = world_min[1] + 0.5 * (gy_lo + gy_hi) * res_y
        return float(cx), float(cy)
    except Exception:
        return None


# ── Queue drain helper ────────────────────────────────────────────────────────

def _drain(queue: multiprocessing.Queue):
    init_msg  = None
    frame_msg = None
    stop_flag = False
    while True:
        try:
            m = queue.get_nowait()
        except Exception:
            break
        t = m.get("type")
        if t == "init":
            init_msg = m
        elif t == "frame":
            frame_msg = m
        elif t == "stop":
            stop_flag = True
    return init_msg, frame_msg, stop_flag


# ── Server process entry point ────────────────────────────────────────────────
# NOTE: must be a module-level function for multiprocessing "spawn" to pickle it.

def _server_main(queue: multiprocessing.Queue, extra_sys_paths: list) -> None:  # noqa: C901
    for p in reversed(extra_sys_paths):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    import matplotlib
    for _be in ("TkAgg", "Qt5Agg", "GTK3Agg", "Agg"):
        try:
            matplotlib.use(_be)
            import matplotlib.pyplot as _chk  # noqa: F401
            break
        except Exception:
            continue

    from matplotlib.animation import FuncAnimation
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    print(f"[DebugVis] server started, backend={matplotlib.get_backend()}", flush=True)

    # ── Block until we receive the init message ───────────────────────────────
    init_msg = None
    while init_msg is None:
        try:
            m = queue.get(timeout=30)
        except Exception:
            print("[DebugVis] timed out waiting for init", flush=True)
            return
        if m.get("type") == "stop":
            return
        if m.get("type") == "init":
            init_msg = m

    half = float(init_msg.get("half_size", 30.0))
    npz  = str(init_msg["grid_npz"])
    center = _map_center_from_npz(npz)
    if center is None:
        cx = float(init_msg["robot_init_x"])
        cy = float(init_msg["robot_init_y"])
    else:
        cx, cy = center

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)

    occ_img, ext = _build_occ_image(npz, cx, cy, half)
    if occ_img is not None:
        ax.imshow(occ_img, origin="lower", extent=ext,
                  interpolation="bilinear", zorder=0)
    local_map_img = ax.imshow(
        np.zeros((2, 2, 4), dtype=np.uint8),
        origin="lower",
        extent=[cx - 1.0, cx + 1.0, cy - 1.0, cy + 1.0],
        interpolation="nearest",
        zorder=0.8,
        visible=False,
    )

    robot_patch = mpatches.Polygon(
        _rect_vertices(cx, cy, 0.0),
        closed=True, facecolor="green", edgecolor="darkgreen",
        linewidth=1.2, zorder=3,
    )
    ax.add_patch(robot_patch)

    tick_text = ax.text(
        cx - half + 0.4, cy + half - 1.5,
        "tick=0", fontsize=10.5, fontweight="bold", color="#1f1f1f", zorder=10,
    )
    map_mode_text = ax.text(
        cx - half + 0.4, cy + half - 2.9,
        "map=occupancy", fontsize=9.5, fontweight="bold", color="#202020", zorder=10,
    )
    hint_text = ax.text(
        cx - half + 0.4, cy + half - 4.2,
        "click here / V: cycle map mode", fontsize=8.5, color="#303030", zorder=10,
    )
    mode_button_patch = mpatches.FancyBboxPatch(
        (cx - half + 0.15, cy + half - 4.7),
        4.8,
        3.7,
        boxstyle="round,pad=0.12,rounding_size=0.12",
        facecolor=(1.0, 1.0, 1.0, 0.88),
        edgecolor="#8e8e8e",
        linewidth=1.1,
        zorder=9.5,
    )
    ax.add_patch(mode_button_patch)
    tick_text.set_zorder(10)
    map_mode_text.set_zorder(10)
    hint_text.set_zorder(10)

    global_path_line, = ax.plot([], [], color="limegreen", linewidth=2.0,
                                alpha=0.9, zorder=5.8, solid_capstyle="round")
    traj_line, = ax.plot([], [], color="red", linewidth=1.5,
                         alpha=0.85, zorder=6, solid_capstyle="round")
    pred_line, = ax.plot([], [], color="#c77d2b", linewidth=0.9,
                         alpha=0.5, zorder=5.5, linestyle="--")
    map_free_scatter = ax.scatter([], [], s=5, c="#8ecae6", alpha=0.22, zorder=1, visible=False)
    map_occ_scatter = ax.scatter([], [], s=10, c="#444444", alpha=0.5, zorder=1.2, visible=False)
    search_scatter = ax.scatter([], [], s=12, c="#2f2f2f", alpha=0.42, zorder=6.5)
    lidar_scatter = ax.scatter([], [], s=14, c="#00e5ff", alpha=0.85, zorder=5.4,
                               edgecolors="#00838f", linewidths=0.3)
    goal_scatter = ax.scatter([], [], s=36, c="crimson", marker="x",
                              linewidths=1.4, alpha=0.8, zorder=2.5)
    map_outline_line, = ax.plot([], [], color="#666666", linewidth=0.8,
                                alpha=0.8, zorder=1)
    lidar_range_patch = mpatches.Circle(
        (cx, cy),
        radius=0.0,
        fill=False,
        edgecolor="#4f6d7a",
        linestyle="--",
        linewidth=0.9,
        alpha=0.5,
        zorder=1.1,
        visible=False,
    )
    ax.add_patch(lidar_range_patch)

    # Mutable state shared with the animation callback (use a dict to avoid
    # nonlocal assignment issues in Python 3.8).
    S = {
        "heading_arrow": None,
        "npc_patches":   [],
        "obs_patches":   [],
        "map_mode": str(init_msg.get("map_debug_mode", "hybrid")),
    }

    def _cycle_map_mode(_event):
        key = getattr(_event, "key", None)
        if key is None or str(key).lower() != "v":
            return
        _advance_map_mode()

    def _advance_map_mode():
        order = ("occupancy", "esdf", "hybrid")
        try:
            idx = order.index(S["map_mode"])
        except ValueError:
            idx = 0
        S["map_mode"] = order[(idx + 1) % len(order)]

    def _click_cycle_map_mode(_event):
        if getattr(_event, "inaxes", None) is not ax:
            return
        if _event.xdata is None or _event.ydata is None:
            return
        x0, y0 = mode_button_patch.get_x(), mode_button_patch.get_y()
        x1 = x0 + mode_button_patch.get_width()
        y1 = y0 + mode_button_patch.get_height()
        if x0 <= float(_event.xdata) <= x1 and y0 <= float(_event.ydata) <= y1:
            _advance_map_mode()

    fig.canvas.mpl_connect("key_press_event", _cycle_map_mode)
    fig.canvas.mpl_connect("button_press_event", _click_cycle_map_mode)

    print(f"[DebugVis] window opened at ({cx:.1f}, {cy:.1f}) ±{half}m", flush=True)

    # ── Animation callback — driven by Tk event loop at `interval` ms ─────────
    def _update(_frame):
        _, frame_msg, stop_flag = _drain(queue)

        if stop_flag:
            ani.event_source.stop()
            plt.close(fig)
            return []

        if frame_msg is None:
            return []

        rd   = frame_msg.get("robot", {})
        rx   = float(rd.get("x",       0.0))
        ry   = float(rd.get("y",       0.0))
        ryaw = float(rd.get("yaw_rad", 0.0))

        # Robot rectangle
        robot_patch.set_xy(_rect_vertices(rx, ry, ryaw))

        # Heading arrow (remove & re-add is cheapest for annotate)
        if S["heading_arrow"] is not None:
            try:
                S["heading_arrow"].remove()
            except Exception:
                pass
        arrow_len = _ROBOT_LENGTH * 0.7
        S["heading_arrow"] = ax.annotate(
            "",
            xy=(rx + arrow_len * math.cos(ryaw),
                ry + arrow_len * math.sin(ryaw)),
            xytext=(rx, ry),
            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2.0),
            zorder=4,
        )

        # NPC circles (grow pool as needed, hide extras)
        npc_list = frame_msg.get("npcs", [])
        while len(S["npc_patches"]) < len(npc_list):
            p = mpatches.Circle((0.0, 0.0), _NPC_RADIUS,
                                facecolor="cornflowerblue", edgecolor="navy",
                                linewidth=0.8, zorder=2)
            ax.add_patch(p)
            S["npc_patches"].append(p)
        for i, p in enumerate(S["npc_patches"]):
            if i < len(npc_list):
                nd = npc_list[i]
                p.set_center((float(nd.get("x", 0)), float(nd.get("y", 0))))
                is_tgt = bool(nd.get("is_target", False))
                p.set_facecolor("orange" if is_tgt else "cornflowerblue")
                p.set_edgecolor("darkorange" if is_tgt else "navy")
                p.set_radius(_NPC_RADIUS * 1.4 if is_tgt else _NPC_RADIUS)
                p.set_visible(True)
            else:
                p.set_visible(False)

        # Obstacle polygons
        obs_list = frame_msg.get("obstacles", [])
        while len(S["obs_patches"]) < len(obs_list):
            p = mpatches.Polygon(
                [[0, 0], [1, 0], [0, 1]], closed=True,
                facecolor="pink", edgecolor="deeppink",
                alpha=0.55, linewidth=0.8, zorder=5,
            )
            ax.add_patch(p)
            S["obs_patches"].append(p)
        for i, p in enumerate(S["obs_patches"]):
            if i < len(obs_list):
                p.set_xy(np.array(obs_list[i]["vertices"], dtype=float))
                p.set_visible(True)
            else:
                p.set_visible(False)

        global_pts = frame_msg.get("global_path", [])
        if global_pts:
            gxs = [p[0] for p in global_pts]
            gys = [p[1] for p in global_pts]
            global_path_line.set_data(gxs, gys)
            global_path_line.set_visible(True)
        else:
            global_path_line.set_visible(False)

        # Planned trajectory (red curve)
        traj_pts = frame_msg.get("traj_points", [])
        if traj_pts:
            # Prepend robot position so the curve starts from the robot
            rd2 = frame_msg.get("robot", {})
            xs = [float(rd2.get("x", 0))] + [p[0] for p in traj_pts]
            ys = [float(rd2.get("y", 0))] + [p[1] for p in traj_pts]
            traj_line.set_data(xs, ys)
            traj_line.set_visible(True)
        else:
            traj_line.set_visible(False)
        pred_pts = frame_msg.get("predicted_target_traj", [])
        if pred_pts:
            pred_line.set_data([p[0] for p in pred_pts], [p[1] for p in pred_pts])
            pred_line.set_visible(True)
        else:
            pred_line.set_visible(False)

        free_pts = frame_msg.get("map_observed_free_cells", [])
        requested_mode = frame_msg.get("map_debug_mode", "hybrid")
        if S["map_mode"] not in {"occupancy", "esdf", "hybrid"}:
            S["map_mode"] = requested_mode
        map_rgba_by_mode = {
            "occupancy": frame_msg.get("map_occupancy_rgba"),
            "esdf": frame_msg.get("map_esdf_rgba"),
            "hybrid": frame_msg.get("map_hybrid_rgba"),
        }
        map_rgba = map_rgba_by_mode.get(S["map_mode"])
        map_extent = frame_msg.get("map_debug_extent")
        if map_rgba is not None and map_extent is not None:
            local_map_img.set_data(np.array(map_rgba, dtype=np.uint8))
            local_map_img.set_extent(map_extent)
            local_map_img.set_visible(True)
            map_free_scatter.set_visible(False)
            map_occ_scatter.set_visible(False)
        else:
            local_map_img.set_visible(False)
            map_free_scatter.set_visible(True)
            map_occ_scatter.set_visible(True)
            if free_pts:
                map_free_scatter.set_offsets(np.array(free_pts, dtype=float))
            else:
                map_free_scatter.set_offsets(np.empty((0, 2), dtype=float))
        occ_pts = frame_msg.get("map_occupied_cells", [])
        if occ_pts:
            map_occ_scatter.set_offsets(np.array(occ_pts, dtype=float))
        else:
            map_occ_scatter.set_offsets(np.empty((0, 2), dtype=float))

        sample_pts = frame_msg.get("search_samples", [])
        if sample_pts:
            search_scatter.set_offsets(np.array(sample_pts, dtype=float))
        else:
            search_scatter.set_offsets(np.empty((0, 2), dtype=float))

        goal_pt = frame_msg.get("goal_point")
        if goal_pt:
            goal_scatter.set_offsets(np.array([goal_pt], dtype=float))
        else:
            goal_scatter.set_offsets(np.empty((0, 2), dtype=float))

        map_outline = frame_msg.get("map_outline", [])
        if map_outline:
            map_outline_line.set_data([p[0] for p in map_outline], [p[1] for p in map_outline])
            map_outline_line.set_visible(True)
        else:
            map_outline_line.set_visible(False)

        lidar_pts = frame_msg.get("lidar_points_world", [])
        if lidar_pts:
            lidar_scatter.set_offsets(np.array(lidar_pts, dtype=float))
            lidar_scatter.set_visible(True)
        else:
            lidar_scatter.set_offsets(np.empty((0, 2), dtype=float))
            lidar_scatter.set_visible(False)

        lidar_range_max = frame_msg.get("lidar_range_max")
        if lidar_range_max is not None and float(lidar_range_max) > 0.0:
            lidar_range_patch.center = (rx, ry)
            lidar_range_patch.set_radius(float(lidar_range_max))
            lidar_range_patch.set_visible(True)
        else:
            lidar_range_patch.set_visible(False)

        tick_text.set_text(f"tick={frame_msg.get('tick', 0)}")
        map_mode_text.set_text(f"map={S['map_mode']}")
        return []

    # interval=50 ms → up to 20 fps UI refresh (faster than the 10 Hz simulation dt=0.1 s)
    ani = FuncAnimation(fig, _update, interval=50, blit=False, cache_frame_data=False)

    # plt.show() runs the Tk mainloop — this blocks until the window is closed.
    plt.show()
    print("[DebugVis] server exited", flush=True)


# ── Public client API ─────────────────────────────────────────────────────────

class DebugVisualizer:
    """
    Client handle for the debug visualization process.

    Usage::

        vis = DebugVisualizer()
        vis.start(robot_init_x, robot_init_y, grid_npz_path)   # after actors spawn
        # each tick:
        vis.update(tick, robot_state, npc_states, active_target_id)
        # on reset / exit:
        vis.stop()
    """

    def __init__(self, half_size: float = 30.0) -> None:
        self._half_size = half_size
        self._queue: Optional[multiprocessing.Queue] = None
        self._proc:  Optional[multiprocessing.Process] = None

    def start(self, robot_init_x: float, robot_init_y: float,
              grid_npz: str) -> None:
        self.stop()

        # "spawn" creates a clean child process — essential because by the time
        # this is called, pygame (and its X11 connection) is already active in
        # the parent, which would prevent Tk from opening a display if we forked.
        ctx = multiprocessing.get_context("spawn")
        self._queue = ctx.Queue(maxsize=4)

        # Pass sys.path so the child (which starts with a minimal path) can
        # import numpy and other project modules.
        extra_paths = [p for p in sys.path if p]

        self._proc = ctx.Process(
            target=_server_main,
            args=(self._queue, extra_paths),
            daemon=True,
            name="DebugVis",
        )
        self._proc.start()
        self._queue.put({
            "type":         "init",
            "robot_init_x": float(robot_init_x),
            "robot_init_y": float(robot_init_y),
            "grid_npz":     str(grid_npz),
            "half_size":    float(self._half_size),
            "map_debug_mode": "hybrid",
        })

    def update(
        self,
        tick:             int,
        robot_state,
        npc_states:       list,
        active_target_id: str,
        obstacles:        Optional[List[dict]] = None,
        traj_points:      Optional[List[List[float]]] = None,
        global_path:      Optional[List[List[float]]] = None,
        goal_point:       Optional[List[float]] = None,
        search_samples:   Optional[List[List[float]]] = None,
        search_goal:      Optional[List[float]] = None,
        predicted_target_traj: Optional[List[List[float]]] = None,
        map_occupied_cells: Optional[List[List[float]]] = None,
        map_observed_free_cells: Optional[List[List[float]]] = None,
        map_outline: Optional[List[List[float]]] = None,
        map_occupancy_rgba = None,
        map_esdf_rgba = None,
        map_hybrid_rgba = None,
        map_debug_extent: Optional[List[float]] = None,
        map_debug_mode: Optional[str] = None,
        lidar_range_max: Optional[float] = None,
        lidar_points_world: Optional[List[List[float]]] = None,
    ) -> None:
        """Send one frame to the visualizer (non-blocking; dropped if queue full).

        obstacles   : list of {"vertices": [[x,y], ...]} dicts (pink polygons)
        traj_points : list of [x, y] waypoints for the planned trajectory (red curve)
        global_path : list of [x, y] waypoints for the A* global path (green curve)
        goal_point  : generic current follow/planning goal, drawn as a red x
        lidar_points_world : filtered single-line LiDAR hits in world XY (cyan dots)
        """
        if self._queue is None:
            return
        msg = {
            "type": "frame",
            "tick": int(tick),
            "robot": {
                "x":       float(robot_state.x),
                "y":       float(robot_state.y),
                "yaw_rad": float(robot_state.yaw_rad),
            },
            "npcs": [
                {
                    "x":         float(n.x),
                    "y":         float(n.y),
                    "is_target": (n.track_id == active_target_id),
                }
                for n in npc_states
            ],
            "obstacles":   obstacles   or [],
            "traj_points": traj_points or [],
            "global_path": global_path or [],
            "goal_point": goal_point,
            "search_samples": search_samples or [],
            "search_goal": search_goal,
            "predicted_target_traj": predicted_target_traj or [],
            "map_occupied_cells": map_occupied_cells or [],
            "map_observed_free_cells": map_observed_free_cells or [],
            "map_outline": map_outline or [],
            "map_occupancy_rgba": map_occupancy_rgba,
            "map_esdf_rgba": map_esdf_rgba,
            "map_hybrid_rgba": map_hybrid_rgba,
            "map_debug_extent": map_debug_extent,
            "map_debug_mode": map_debug_mode or "hybrid",
            "lidar_range_max": None if lidar_range_max is None else float(lidar_range_max),
            "lidar_points_world": lidar_points_world or [],
        }
        try:
            self._queue.put_nowait(msg)
        except Exception:
            pass

    def stop(self) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait({"type": "stop"})
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
        self._queue = None
        self._proc  = None

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()
