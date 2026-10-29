#!/usr/bin/env python3
"""
Park-scenario route painter.

Loads the park gridmap as background and overlays:
  - Target trajectory (npc_trajectory_cache.json) — 3 segments, cyan/white
  - Existing distractor routes (distractor_routes_cache.json) — orange, read-only

Draw new interference routes on top.  New routes are saved to
  assets/scenes/park/extra_routes.json  (separate from the existing caches).

Controls
--------
  N              Start a new route (auto-numbered new_0, new_1, …)
  Left-click     Add waypoint (snapped to nearest walkable cell)
  Right-click    Finish current route
  C              Same as right-click
  Z              Undo last point (drawing) / last route (idle)
  S              Save new routes to extra_routes.json
  Q / Esc        Quit

Usage
-----
  conda run -n multiRobots python park_route_painter.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches

# ── Paths ────────────────────────────────────────────────────────────────────
from followbench.paths import scene_assets_dir  # noqa: E402

ASSETS_DIR = scene_assets_dir("park")
NPZ_PATH   = ASSETS_DIR / "gridmap_roi.npz"
TRAJ_PATH  = ASSETS_DIR / "npc_trajectory_cache.json"
DIST_PATH  = ASSETS_DIR / "distractor_routes_cache.json"
OUT_PATH   = ASSETS_DIR / "extra_routes.json"

# ── Colour palette ───────────────────────────────────────────────────────────
_SEG_COLORS  = ["#00E5FF", "#40C4FF", "#80D8FF"]   # cyan family → target segs
_DIST_COLORS = ["#FF9800", "#FFB74D", "#FFCC02"]   # orange family → distractor routes
_NEW_COLORS  = [                                    # vivid → new routes
    "#FF4444", "#B2FF59", "#EA80FC",
    "#FFD740", "#FF6E40", "#69FF47",
]

def _new_color(idx: int) -> str:
    return _NEW_COLORS[idx % len(_NEW_COLORS)]

# ── Grid loader ──────────────────────────────────────────────────────────────

def _load_grid():
    data      = np.load(NPZ_PATH, allow_pickle=True)
    free_grid = np.array(data["free_grid"],    dtype=np.uint8)
    roi_mask  = np.array(data["roi_mask"],     dtype=np.uint8)
    obs_raw   = np.array(data["obstacle_raw"], dtype=np.uint8)
    world_min = np.array(data["world_min"],    dtype=np.float64)[:2]
    world_max = np.array(data["world_max"],    dtype=np.float64)[:2]
    H, W      = free_grid.shape
    res       = float((world_max[0] - world_min[0]) / W)
    walkable  = (free_grid > 0) & (roi_mask > 0)
    return walkable, free_grid, roi_mask, obs_raw, world_min, world_max, res

# ── Snap helper ──────────────────────────────────────────────────────────────

def _build_snap(walkable, world_min, res):
    gy_arr, gx_arr = np.where(walkable)
    wx_arr = world_min[0] + (gx_arr + 0.5) * res
    wy_arr = world_min[1] + (gy_arr + 0.5) * res
    pts    = np.stack([wx_arr, wy_arr], axis=1)
    try:
        from scipy.spatial import cKDTree
        kd = cKDTree(pts)
        def snap(wx, wy):
            _, i = kd.query([wx, wy])
            return float(pts[i, 0]), float(pts[i, 1])
    except ImportError:
        def snap(wx, wy):
            d = np.hypot(pts[:, 0] - wx, pts[:, 1] - wy)
            return float(pts[int(np.argmin(d)), 0]), float(pts[int(np.argmin(d)), 1])
    return snap

# ── Route save ───────────────────────────────────────────────────────────────

def _save_routes(routes, world_min, world_max, res, walkable):
    H, W = walkable.shape
    out  = {
        "grid_meta": {
            "world_min":  [float(world_min[0]), float(world_min[1])],
            "world_max":  [float(world_max[0]), float(world_max[1])],
            "resolution": float(res),
            "shape":      [int(H), int(W)],
        },
        "routes": {
            str(i): [[round(x, 3), round(y, 3)] for x, y in r]
            for i, r in enumerate(routes)
        },
    }
    tmp = OUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, OUT_PATH)
    print(f"[painter] saved {len(routes)} new route(s) → {OUT_PATH}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not NPZ_PATH.exists():
        raise SystemExit(f"NPZ not found: {NPZ_PATH}")

    walkable, free_grid, roi_mask, obs_raw, world_min, world_max, res = _load_grid()
    snap = _build_snap(walkable, world_min, res)
    H, W = walkable.shape

    # ── Background image ─────────────────────────────────────────────────────
    img = np.full((H, W, 3), 20, dtype=np.uint8)
    img[obs_raw  == 1] = [70,  70,  70]
    img[roi_mask == 1] = [180, 185, 190]
    img[walkable]      = [210, 215, 220]
    extent = [float(world_min[0]), float(world_max[0]),
              float(world_min[1]), float(world_max[1])]

    # ROI crop for display
    ys_w, xs_w = np.where(roi_mask > 0)
    PAD  = 10.0
    xlim = (world_min[0] + xs_w.min() * res - PAD,
            world_min[0] + xs_w.max() * res + PAD)
    ylim_data = (world_min[1] + ys_w.min() * res - PAD,
                 world_min[1] + ys_w.max() * res + PAD)
    # inverted: large Y at top of screen → matches CARLA top-down view
    ylim_screen = (ylim_data[1], ylim_data[0])

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.manager.set_window_title(
        "Park Route Painter  |  N=new  Click=add  RClick/C=finish  Z=undo  S=save  Q=quit"
    )

    ax.imshow(img, origin="lower", extent=extent,
              interpolation="nearest", zorder=0, alpha=0.80)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim_screen)

    # ── Overlay: target segments ──────────────────────────────────────────────
    if TRAJ_PATH.exists():
        with open(TRAJ_PATH) as f:
            traj = json.load(f)
        for si, seg in enumerate(traj["segments"]):
            xs = [p[0] for p in seg]
            ys = [p[1] for p in seg]
            col = _SEG_COLORS[si % len(_SEG_COLORS)]
            ax.plot(xs, ys, color=col, lw=2.5, alpha=0.7, zorder=2,
                    solid_capstyle="round")
            ax.scatter(xs[0], ys[0], s=120, c=col, marker="o",
                       edgecolors="white", linewidths=1.0, zorder=5)
            ax.text(xs[0], ys[0], f" seg{si}",
                    fontsize=9, color=col, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
        print(f"[painter] loaded {len(traj['segments'])} target segments")
    else:
        print(f"[painter] WARN: {TRAJ_PATH} not found")

    # ── Overlay: existing distractor routes ──────────────────────────────────
    if DIST_PATH.exists():
        with open(DIST_PATH) as f:
            dist = json.load(f)
        for di, route in enumerate(dist["routes"]):
            xs = [p[0] for p in route]
            ys = [p[1] for p in route]
            col = _DIST_COLORS[di % len(_DIST_COLORS)]
            ax.plot(xs, ys, color=col, lw=2.0, alpha=0.75, zorder=2,
                    linestyle="--", solid_capstyle="round")
            ax.scatter(xs[0], ys[0], s=80, c=col, marker="s",
                       edgecolors="white", linewidths=0.8, zorder=5)
            ax.text(xs[0], ys[0], f" D{di}",
                    fontsize=9, color=col, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
        print(f"[painter] loaded {len(dist['routes'])} distractor routes")
    else:
        print(f"[painter] WARN: {DIST_PATH} not found")

    # ── Load existing extra_routes if present ────────────────────────────────
    existing_new: List[List[Tuple[float, float]]] = []
    new_route_artists = []
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            ex = json.load(f)
        for i, pts_raw in ex["routes"].items():
            pts  = [(p[0], p[1]) for p in pts_raw]
            existing_new.append(pts)
            col  = _new_color(int(i))
            xs   = [p[0] for p in pts]
            ys   = [p[1] for p in pts]
            ln,  = ax.plot(xs, ys, color=col, lw=2.5, zorder=3,
                           solid_capstyle="round", alpha=0.95)
            sc   = ax.scatter(xs, ys, c=col, s=30, zorder=4,
                              edgecolors="white", linewidths=0.5)
            lb   = ax.text(xs[0], ys[0], f" R{i}",
                           fontsize=10, color=col, fontweight="bold", zorder=9,
                           path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
            new_route_artists.append((ln, sc, lb))
        print(f"[painter] loaded {len(existing_new)} existing extra routes")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=_SEG_COLORS[0],  label="Target route (seg0/1/2)"),
        mpatches.Patch(color=_DIST_COLORS[0], label="Distractor routes (D0/1/2)"),
        mpatches.Patch(color=_NEW_COLORS[0],  label="New routes (R0, R1, …)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.7, facecolor="#1A1A2E", labelcolor="white",
              edgecolor="#555577")

    # ── Drawing state ─────────────────────────────────────────────────────────
    new_routes: List[List[Tuple[float, float]]] = list(existing_new)
    cur_pts:    List[Tuple[float, float]]        = []
    drawing     = False

    cur_line,   = ax.plot([], [], lw=2.0, zorder=7, color="#FFFF00",
                          linestyle="--", solid_capstyle="round")
    cur_dots    = ax.scatter([], [], s=40, zorder=8, color="#FFFF00",
                             edgecolors="white", linewidths=0.6)
    preview_ln, = ax.plot([], [], lw=1.0, zorder=6, color="#FFFFFF",
                          linestyle=":", alpha=0.5)
    snap_dot    = ax.scatter([], [], s=25, zorder=9, color="#FFFFFF",
                             alpha=0.7, marker="+")

    status_txt  = ax.text(
        0.01, 0.99, "",
        transform=ax.transAxes, fontsize=9, color="#EEEEEE", va="top", zorder=10,
        bbox=dict(facecolor="#0D1117", alpha=0.8, edgecolor="none", pad=3),
    )
    coord_txt   = ax.text(
        0.99, 0.99, "",
        transform=ax.transAxes, fontsize=8, color="#AAAAAA",
        va="top", ha="right", zorder=10,
        bbox=dict(facecolor="#0D1117", alpha=0.7, edgecolor="none", pad=2),
    )

    fig.tight_layout(pad=0.3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_status():
        n   = len(new_routes)
        idx = len(new_routes) - len(existing_new)   # count only session-new
        col = _new_color(len(new_routes))
        if drawing:
            msg = (f"Drawing R{len(new_routes)}  [{len(cur_pts)} pts]  —  "
                   f"click=add  RClick/C=finish  Z=undo last pt")
            status_txt.set_color(col)
        else:
            msg = (f"{len(new_routes)} new route(s)  —  "
                   f"N=new  Z=undo last route  S=save  Q=quit")
            status_txt.set_color("#EEEEEE")
        status_txt.set_text(msg)

    def _redraw_cur():
        col = _new_color(len(new_routes))
        if cur_pts:
            xs = [p[0] for p in cur_pts]
            ys = [p[1] for p in cur_pts]
            cur_line.set_data(xs, ys)
            cur_dots.set_offsets(np.c_[xs, ys])
            cur_line.set_color(col)
            cur_dots.set_facecolor(col)
        else:
            cur_line.set_data([], [])
            cur_dots.set_offsets(np.empty((0, 2)))

    def _commit_route():
        nonlocal drawing
        if len(cur_pts) < 2:
            cur_pts.clear(); drawing = False
            _redraw_cur(); _update_status(); fig.canvas.draw_idle()
            return
        idx = len(new_routes)
        col = _new_color(idx)
        pts = list(cur_pts)
        new_routes.append(pts)
        cur_pts.clear(); drawing = False
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ln,  = ax.plot(xs, ys, color=col, lw=2.5, zorder=3,
                       solid_capstyle="round", alpha=0.95)
        sc   = ax.scatter(xs, ys, c=col, s=30, zorder=4,
                          edgecolors="white", linewidths=0.5)
        lb   = ax.text(xs[0], ys[0], f" R{idx}",
                       fontsize=10, color=col, fontweight="bold", zorder=9,
                       path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
        new_route_artists.append((ln, sc, lb))
        _redraw_cur(); preview_ln.set_data([], [])
        _update_status(); fig.canvas.draw_idle()
        print(f"[painter] route R{idx} committed: {len(pts)} wps  "
              f"start=({pts[0][0]:.1f},{pts[0][1]:.1f}) "
              f"end=({pts[-1][0]:.1f},{pts[-1][1]:.1f})")

    # ── Events ────────────────────────────────────────────────────────────────

    def on_press(ev):
        nonlocal drawing
        if ev.inaxes != ax or ev.xdata is None:
            return
        if ev.button == 1 and drawing:
            sx, sy = snap(ev.xdata, ev.ydata)
            cur_pts.append((sx, sy))
            _redraw_cur(); _update_status(); fig.canvas.draw_idle()
        elif ev.button == 3 and drawing:
            _commit_route()

    def on_motion(ev):
        if ev.xdata is None:
            snap_dot.set_offsets(np.empty((0, 2)))
            preview_ln.set_data([], [])
            fig.canvas.draw_idle()
            return
        sx, sy = snap(ev.xdata, ev.ydata)
        snap_dot.set_offsets([[sx, sy]])
        col = _new_color(len(new_routes)) if drawing else "#888888"
        snap_dot.set_facecolor(col)
        coord_txt.set_text(f"({sx:.2f}, {sy:.2f})")
        if drawing and cur_pts:
            lx, ly = cur_pts[-1]
            preview_ln.set_data([lx, sx], [ly, sy])
            preview_ln.set_color(col)
        else:
            preview_ln.set_data([], [])
        fig.canvas.draw_idle()

    def on_key(ev):
        nonlocal drawing
        k = ev.key
        if k in ("q", "escape"):
            plt.close("all"); sys.exit(0)
        elif k == "n":
            if drawing:
                _commit_route()
            drawing = True; cur_pts.clear()
            _redraw_cur(); _update_status(); fig.canvas.draw_idle()
            print(f"[painter] starting route R{len(new_routes)}")
        elif k == "c" and drawing:
            _commit_route()
        elif k == "z":
            if drawing and cur_pts:
                cur_pts.pop()
                _redraw_cur(); preview_ln.set_data([], [])
                _update_status(); fig.canvas.draw_idle()
            elif not drawing and new_routes:
                new_routes.pop()
                ln, sc, lb = new_route_artists.pop()
                ln.remove(); sc.remove(); lb.remove()
                _update_status(); fig.canvas.draw_idle()
                print(f"[painter] undid route R{len(new_routes)}")
        elif k == "s":
            if drawing:
                _commit_route()
            if not new_routes:
                print("[painter] nothing to save"); return
            _save_routes(new_routes, world_min, world_max, res, walkable)
            status_txt.set_text(f"Saved {len(new_routes)} routes → {OUT_PATH.name}")
            status_txt.set_color("#44FF88")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",  on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event",     on_key)

    _update_status()
    print("[painter] ready — N to start a new route, Q to quit")
    plt.show()


if __name__ == "__main__":
    main()
