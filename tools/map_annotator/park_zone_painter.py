#!/usr/bin/env python3
"""
Park bottleneck zone painter.

Mark narrow corners (polygon) and junction choke-points (circle) on the
park gridmap.  Existing routes are shown as read-only reference overlays.

Zone types
----------
  narrow   — direction semaphore: only one travel-direction allowed at a time
  junction — capacity cap: at most K NPCs inside simultaneously

Controls
--------
  Tab            Switch shape mode: polygon ↔ circle
  T              Toggle zone type: narrow ↔ junction
  Left-click     Polygon: add vertex  |  Circle: set centre
  Scroll / +/-   Circle: grow / shrink radius (before confirming)
  Right-click    Polygon: close & commit  |  Circle: commit
  Enter          Same as right-click
  Z              Undo last vertex (polygon) / last zone (idle)
  S              Save  →  assets/scenes/park/bottleneck_zones.json
  Q / Esc        Quit

Output JSON
-----------
{
  "zones": [
    {"id": "Z0", "type": "narrow",   "shape": "polygon",
     "vertices": [[x,y], ...]},
    {"id": "Z1", "type": "junction", "shape": "circle",
     "x": 0.0, "y": 0.0, "radius": 3.0}
  ]
}

Usage
-----
  conda run -n carla38 python zone_painter.py
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
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle

# ── Paths ────────────────────────────────────────────────────────────────────
from followbench.paths import scene_assets_dir  # noqa: E402

ASSETS_DIR = scene_assets_dir("park")
NPZ_PATH   = ASSETS_DIR / "gridmap_roi.npz"
TRAJ_PATH  = ASSETS_DIR / "npc_trajectory_cache.json"
DIST_PATH  = ASSETS_DIR / "distractor_routes_cache.json"
EXTRA_PATH = ASSETS_DIR / "extra_routes.json"
OUT_PATH   = ASSETS_DIR / "bottleneck_zones.json"

# ── Colours ──────────────────────────────────────────────────────────────────
_TYPE_STYLE = {
    "narrow":   {"face": "#FF4444", "edge": "#FF0000", "alpha": 0.30},
    "junction": {"face": "#FFD740", "edge": "#FFA000", "alpha": 0.30},
}
_SEG_COLORS  = ["#00E5FF", "#40C4FF", "#80D8FF"]
_DIST_COLORS = ["#FF9800", "#FFB74D", "#FFCC02"]
_EXTRA_COLORS = ["#FF4444", "#B2FF59", "#EA80FC", "#FFD740", "#FF6E40"]

# ── Grid ─────────────────────────────────────────────────────────────────────

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

# ── Save ─────────────────────────────────────────────────────────────────────

def _save(zones: list):
    tmp = OUT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"zones": zones}, f, indent=2)
    os.replace(tmp, OUT_PATH)
    print(f"[zone_painter] saved {len(zones)} zone(s) → {OUT_PATH}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not NPZ_PATH.exists():
        raise SystemExit(f"NPZ not found: {NPZ_PATH}")

    walkable, free_grid, roi_mask, obs_raw, world_min, world_max, res = _load_grid()
    H, W = free_grid.shape

    img = np.full((H, W, 3), 20, dtype=np.uint8)
    img[obs_raw  == 1] = [70,  70,  70]
    img[roi_mask == 1] = [180, 185, 190]
    img[walkable]      = [210, 215, 220]
    extent = [float(world_min[0]), float(world_max[0]),
              float(world_min[1]), float(world_max[1])]

    ys_w, xs_w = np.where(roi_mask > 0)
    PAD  = 10.0
    xlim = (world_min[0] + xs_w.min() * res - PAD,
            world_min[0] + xs_w.max() * res + PAD)
    ylim_data   = (world_min[1] + ys_w.min() * res - PAD,
                   world_min[1] + ys_w.max() * res + PAD)
    ylim_screen = (ylim_data[1], ylim_data[0])   # inverted → matches CARLA

    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.manager.set_window_title(
        "Zone Painter  |  Tab=shape  T=type  Click=add  RClick/Enter=commit  Z=undo  S=save  Q=quit"
    )
    ax.imshow(img, origin="lower", extent=extent,
              interpolation="nearest", zorder=0, alpha=0.80)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim_screen)

    # ── Reference overlays ───────────────────────────────────────────────────
    if TRAJ_PATH.exists():
        with open(TRAJ_PATH) as f:
            traj = json.load(f)
        for si, seg in enumerate(traj["segments"]):
            xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
            col = _SEG_COLORS[si % len(_SEG_COLORS)]
            ax.plot(xs, ys, color=col, lw=2.0, alpha=0.55, zorder=2)
            ax.text(xs[0], ys[0], f" seg{si}", fontsize=8, color=col,
                    fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])

    if DIST_PATH.exists():
        with open(DIST_PATH) as f:
            dist = json.load(f)
        for di, route in enumerate(dist["routes"]):
            xs = [p[0] for p in route]; ys = [p[1] for p in route]
            col = _DIST_COLORS[di % len(_DIST_COLORS)]
            ax.plot(xs, ys, color=col, lw=1.8, alpha=0.55, zorder=2, linestyle="--")
            ax.text(xs[0], ys[0], f" D{di}", fontsize=8, color=col,
                    fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])

    if EXTRA_PATH.exists():
        with open(EXTRA_PATH) as f:
            ex = json.load(f)
        for i, pts in ex["routes"].items():
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            col = _EXTRA_COLORS[int(i) % len(_EXTRA_COLORS)]
            ax.plot(xs, ys, color=col, lw=1.8, alpha=0.55, zorder=2)
            ax.text(xs[0], ys[0], f" R{i}", fontsize=8, color=col,
                    fontweight="bold", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])

    # ── Load existing zones ──────────────────────────────────────────────────
    committed_zones: list = []        # JSON-serialisable dicts
    committed_artists: list = []      # matplotlib patches per zone

    def _draw_committed(z: dict):
        st = _TYPE_STYLE[z["type"]]
        if z["shape"] == "polygon":
        	verts = np.array(z["vertices"])
        	patch = MplPolygon(verts, closed=True,
                               facecolor=st["face"], edgecolor=st["edge"],
                               linewidth=1.8, alpha=st["alpha"], zorder=10)
        	ax.add_patch(patch)
        	cx, cy = verts.mean(axis=0)
        	lbl = ax.text(cx, cy, f" {z['id']}\n{z['type']}",
                          fontsize=8, color=st["edge"], ha="center", va="center",
                          fontweight="bold", zorder=11,
                          path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
        	return patch, lbl
        else:
        	patch = MplCircle((z["x"], z["y"]), z["radius"],
                              facecolor=st["face"], edgecolor=st["edge"],
                              linewidth=1.8, alpha=st["alpha"], zorder=10)
        	ax.add_patch(patch)
        	lbl = ax.text(z["x"], z["y"], f" {z['id']}\n{z['type']}",
                          fontsize=8, color=st["edge"], ha="center", va="center",
                          fontweight="bold", zorder=11,
                          path_effects=[pe.withStroke(linewidth=2, foreground="#0D1117")])
        	return patch, lbl

    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            existing = json.load(f)
        for z in existing.get("zones", []):
            committed_zones.append(z)
            committed_artists.append(_draw_committed(z))
        print(f"[zone_painter] loaded {len(committed_zones)} existing zone(s)")

    # ── Drawing state ────────────────────────────────────────────────────────
    shape_mode = ["polygon"]   # "polygon" | "circle"
    zone_type  = ["narrow"]    # "narrow"  | "junction"
    cur_verts: List[Tuple[float, float]] = []   # polygon in-progress
    circle_cx  = [None]        # circle centre x
    circle_cy  = [None]        # circle centre y
    circle_r   = [3.0]         # circle radius

    # In-progress artists
    cur_line,  = ax.plot([], [], lw=1.8, zorder=12, color="#FFFF00",
                         linestyle="--", solid_capstyle="round")
    cur_dots   = ax.scatter([], [], s=40, zorder=13, color="#FFFF00",
                            edgecolors="white", linewidths=0.6)
    preview_ln, = ax.plot([], [], lw=1.0, zorder=11, color="#FFFFFF",
                          linestyle=":", alpha=0.5)
    circ_preview = MplCircle((0, 0), circle_r[0],
                             facecolor="none", edgecolor="#FFFF00",
                             linewidth=1.5, linestyle="--", zorder=12, visible=False)
    ax.add_patch(circ_preview)

    status_txt = ax.text(
        0.01, 0.99, "", transform=ax.transAxes, fontsize=9,
        color="#EEEEEE", va="top", zorder=20,
        bbox=dict(facecolor="#0D1117", alpha=0.8, edgecolor="none", pad=3),
    )
    coord_txt = ax.text(
        0.99, 0.99, "", transform=ax.transAxes, fontsize=8,
        color="#AAAAAA", va="top", ha="right", zorder=20,
        bbox=dict(facecolor="#0D1117", alpha=0.7, edgecolor="none", pad=2),
    )
    legend_handles = [
        mpatches.Patch(color=_TYPE_STYLE["narrow"]["face"],
                       edgecolor=_TYPE_STYLE["narrow"]["edge"],   label="narrow (direction lock)"),
        mpatches.Patch(color=_TYPE_STYLE["junction"]["face"],
                       edgecolor=_TYPE_STYLE["junction"]["edge"], label="junction (capacity cap)"),
        mpatches.Patch(color=_SEG_COLORS[0],   label="target segs"),
        mpatches.Patch(color=_DIST_COLORS[0],  label="distractor routes"),
        mpatches.Patch(color=_EXTRA_COLORS[0], label="extra trunk routes"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
              framealpha=0.7, facecolor="#1A1A2E", labelcolor="white",
              edgecolor="#555577")
    fig.tight_layout(pad=0.3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _style():
        return _TYPE_STYLE[zone_type[0]]

    def _update_status():
        n   = len(committed_zones)
        sm  = shape_mode[0]
        zt  = zone_type[0]
        col = _style()["edge"]
        if sm == "polygon":
            if cur_verts:
                msg = (f"[polygon/{zt}]  {len(cur_verts)} pts  —  "
                       f"click=add vertex  RClick/Enter=close  Z=undo pt")
            else:
                msg = (f"[polygon/{zt}]  click to start  |  "
                       f"Tab=circle  T=type  Z=undo zone  S=save")
        else:
            if circle_cx[0] is None:
                msg = (f"[circle/{zt}]  click=set centre  |  "
                       f"Tab=polygon  T=type  Z=undo zone  S=save")
            else:
                msg = (f"[circle/{zt}]  r={circle_r[0]:.1f}m  —  "
                       f"scroll/+/-=resize  RClick/Enter=commit")
        status_txt.set_text(f"zones={n}  {msg}")
        status_txt.set_color(col)

    def _redraw_poly():
        col = _style()["edge"]
        if cur_verts:
            xs = [p[0] for p in cur_verts]
            ys = [p[1] for p in cur_verts]
            cur_line.set_data(xs, ys)
            cur_dots.set_offsets(np.c_[xs, ys])
            cur_line.set_color(col)
            cur_dots.set_facecolor(col)
        else:
            cur_line.set_data([], [])
            cur_dots.set_offsets(np.empty((0, 2)))

    def _commit_polygon():
        if len(cur_verts) < 3:
            cur_verts.clear()
            _redraw_poly()
            preview_ln.set_data([], [])
            _update_status(); fig.canvas.draw_idle()
            return
        zid = f"Z{len(committed_zones)}"
        z = {"id": zid, "type": zone_type[0], "shape": "polygon",
             "vertices": [[round(x, 3), round(y, 3)] for x, y in cur_verts]}
        committed_zones.append(z)
        committed_artists.append(_draw_committed(z))
        cur_verts.clear()
        _redraw_poly()
        preview_ln.set_data([], [])
        _update_status(); fig.canvas.draw_idle()
        print(f"[zone_painter] {zid} polygon/{z['type']} committed "
              f"({len(z['vertices'])} vertices)")

    def _commit_circle():
        if circle_cx[0] is None:
            return
        zid = f"Z{len(committed_zones)}"
        z = {"id": zid, "type": zone_type[0], "shape": "circle",
             "x": round(circle_cx[0], 3), "y": round(circle_cy[0], 3),
             "radius": round(circle_r[0], 2)}
        committed_zones.append(z)
        committed_artists.append(_draw_committed(z))
        circle_cx[0] = None
        circ_preview.set_visible(False)
        _update_status(); fig.canvas.draw_idle()
        print(f"[zone_painter] {zid} circle/{z['type']} committed "
              f"centre=({z['x']:.1f},{z['y']:.1f}) r={z['radius']:.1f}")

    def _commit_current():
        if shape_mode[0] == "polygon":
            _commit_polygon()
        else:
            _commit_circle()

    # ── Events ────────────────────────────────────────────────────────────────

    def on_press(ev):
        if ev.inaxes != ax or ev.xdata is None:
            return
        if ev.button == 1:
            if shape_mode[0] == "polygon":
                cur_verts.append((float(ev.xdata), float(ev.ydata)))
                _redraw_poly()
                _update_status(); fig.canvas.draw_idle()
            else:
                # Set / move circle centre
                circle_cx[0] = float(ev.xdata)
                circle_cy[0] = float(ev.ydata)
                circ_preview.set_center((circle_cx[0], circle_cy[0]))
                circ_preview.set_radius(circle_r[0])
                circ_preview.set_edgecolor(_style()["edge"])
                circ_preview.set_visible(True)
                _update_status(); fig.canvas.draw_idle()
        elif ev.button == 3:
            _commit_current()

    def on_scroll(ev):
        if shape_mode[0] != "circle" or circle_cx[0] is None:
            return
        if ev.button == "up":
            circle_r[0] = min(circle_r[0] * 1.15, 50.0)
        else:
            circle_r[0] = max(circle_r[0] / 1.15, 0.5)
        circ_preview.set_radius(circle_r[0])
        _update_status(); fig.canvas.draw_idle()

    def on_motion(ev):
        if ev.xdata is None:
            preview_ln.set_data([], [])
            coord_txt.set_text("")
            fig.canvas.draw_idle()
            return
        coord_txt.set_text(f"({ev.xdata:.2f}, {ev.ydata:.2f})")
        if shape_mode[0] == "polygon" and cur_verts:
            lx, ly = cur_verts[-1]
            preview_ln.set_data([lx, ev.xdata], [ly, ev.ydata])
            preview_ln.set_color(_style()["edge"])
        else:
            preview_ln.set_data([], [])
        fig.canvas.draw_idle()

    def on_key(ev):
        k = ev.key
        if k in ("q", "escape"):
            plt.close("all"); sys.exit(0)

        elif k == "tab":
            if shape_mode[0] == "polygon":
                shape_mode[0] = "circle"
                cur_verts.clear(); _redraw_poly()
                preview_ln.set_data([], [])
            else:
                shape_mode[0] = "polygon"
                circle_cx[0]  = None
                circ_preview.set_visible(False)
            _update_status(); fig.canvas.draw_idle()

        elif k == "t":
            zone_type[0] = "junction" if zone_type[0] == "narrow" else "narrow"
            circ_preview.set_edgecolor(_style()["edge"])
            _update_status(); fig.canvas.draw_idle()

        elif k in ("enter", "return"):
            _commit_current()

        elif k in ("+", "="):
            if shape_mode[0] == "circle" and circle_cx[0] is not None:
                circle_r[0] = min(circle_r[0] * 1.15, 50.0)
                circ_preview.set_radius(circle_r[0])
                _update_status(); fig.canvas.draw_idle()

        elif k == "-":
            if shape_mode[0] == "circle" and circle_cx[0] is not None:
                circle_r[0] = max(circle_r[0] / 1.15, 0.5)
                circ_preview.set_radius(circle_r[0])
                _update_status(); fig.canvas.draw_idle()

        elif k == "z":
            if shape_mode[0] == "polygon" and cur_verts:
                cur_verts.pop()
                _redraw_poly(); preview_ln.set_data([], [])
                _update_status(); fig.canvas.draw_idle()
            elif shape_mode[0] == "circle" and circle_cx[0] is not None:
                circle_cx[0] = None
                circ_preview.set_visible(False)
                _update_status(); fig.canvas.draw_idle()
            elif committed_zones:
                committed_zones.pop()
                patch, lbl = committed_artists.pop()
                patch.remove(); lbl.remove()
                _update_status(); fig.canvas.draw_idle()
                print(f"[zone_painter] undid zone Z{len(committed_zones)}")

        elif k == "s":
            _commit_current()
            if not committed_zones:
                print("[zone_painter] nothing to save"); return
            _save(committed_zones)
            status_txt.set_text(f"Saved {len(committed_zones)} zones → {OUT_PATH.name}")
            status_txt.set_color("#44FF88")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",  on_press)
    fig.canvas.mpl_connect("scroll_event",        on_scroll)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event",     on_key)

    _update_status()
    print("[zone_painter] ready — Tab=shape  T=type  Q=quit")
    plt.show()


if __name__ == "__main__":
    main()
