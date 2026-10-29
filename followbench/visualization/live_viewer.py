#!/usr/bin/env python3
"""
Live NPC debug viewer — reads /tmp/npc_debug_state.json written by the simulation.

Standalone script (not importable); the episode runners launch it as a subprocess:
    python followbench/visualization/live_viewer.py --npz assets/scenes/park/gridmap_roi.npz --half 30

Controls:
    +/-    → zoom in / out
    q/Esc  → quit
"""
import argparse, json, math, os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--state", default="/tmp/npc_debug_state.json")
parser.add_argument("--npz",   default="")
parser.add_argument("--half",  type=float, default=40.0)
parser.add_argument("--fps",   type=float, default=10.0)
args = parser.parse_args()

STATE_FILE = args.state
NPZ_PATH   = args.npz
HALF       = args.half
INTERVAL   = int(1000.0 / max(args.fps, 1.0))

_COLORS = {
    "target":  ("#FF8C00", "#8B4500"),
    "transit": ("#4A90D9", "#1B4F8A"),
    "staged":  ("#A0D468", "#4A7C20"),
    "waiting": ("#BDBDBD", "#757575"),
    "stuck":   ("#E53935", "#B71C1C"),
}
_NPC_R     = 0.3
_FP_COLORS = ["#00EDC0", "#1EA0DC", "#FF823C", "#E040FB", "#FFEB3B"]

# ── Load npz for background ───────────────────────────────────────────────────
_world_min = _world_max = _free_grid = _roi_mask = _obs_raw = None
_resolution = 0.5

def _load_npz():
    global _world_min, _world_max, _free_grid, _roi_mask, _obs_raw, _resolution
    if not NPZ_PATH or not os.path.exists(NPZ_PATH):
        return
    d = np.load(NPZ_PATH, allow_pickle=True)
    _world_min  = np.array(d["world_min"],    dtype=np.float64)
    _world_max  = np.array(d["world_max"],    dtype=np.float64)
    _free_grid  = np.array(d["free_grid"],    dtype=np.uint8)
    _roi_mask   = np.array(d["roi_mask"],     dtype=np.uint8)
    _obs_raw    = np.array(d["obstacle_raw"], dtype=np.uint8)
    nrows, ncols = _free_grid.shape
    _resolution = float((_world_max[0] - _world_min[0]) / ncols)
    print(f"[Viewer] map loaded: {nrows}×{ncols}, res={_resolution}m")

_load_npz()

def _build_occ_image(cx, cy, half):
    if _free_grid is None:
        return None, None
    nrows, ncols = _free_grid.shape
    gx_lo = max(0,      int((cx - half - _world_min[0]) / _resolution))
    gx_hi = min(ncols,  int((cx + half - _world_min[0]) / _resolution) + 1)
    gy_lo = max(0,      int((cy - half - _world_min[1]) / _resolution))
    gy_hi = min(nrows,  int((cy + half - _world_min[1]) / _resolution) + 1)
    rc  = _roi_mask[gy_lo:gy_hi, gx_lo:gx_hi]
    oc  = _obs_raw [gy_lo:gy_hi, gx_lo:gx_hi]
    img = np.full((*rc.shape, 3), 180, dtype=np.uint8)
    img[oc == 1] = [90,  90,  90]
    img[rc == 1] = [240, 240, 240]
    ext = [
        _world_min[0] + gx_lo * _resolution, _world_min[0] + gx_hi * _resolution,
        _world_min[1] + gy_lo * _resolution, _world_min[1] + gy_hi * _resolution,
    ]
    return img, ext

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor("#1A1A2E")
ax.set_facecolor("#1A1A2E")
ax.set_aspect("equal")
ax.invert_yaxis()   # CARLA world Y increases "downward" in top-down view
ax.axis("off")
fig.tight_layout(pad=0.4)
fig.canvas.manager.set_window_title("NPC Live Debug Viewer")

_S = {"cx": 0.0, "cy": 0.0, "init": False}
ax.set_xlim(-HALF, HALF)
ax.set_ylim(HALF, -HALF)   # inverted to match CARLA top-down orientation

def _set_view(cx, cy):
    ax.set_xlim(cx - HALF, cx + HALF)
    ax.set_ylim(cy + HALF, cy - HALF)   # inverted: top > bottom in data coords
    _S["cx"] = cx; _S["cy"] = cy

_occ_im      = [None]
_con_sc      = [None]
_fp_arts     = []
_robot_patch = [None]
_robot_arrow = [None]
_npc_circles = []
_npc_labels  = []
_npc_wp_dots = []
_npc_path_lns= []

_status = ax.text(0.01, 0.99, "waiting for simulation…",
                  transform=ax.transAxes, fontsize=9,
                  color="#CCCCCC", va="top", zorder=20)

ax.legend(handles=[
    mpatches.Patch(color=_COLORS["target"][0],  label="target (N01)"),
    mpatches.Patch(color=_COLORS["transit"][0], label="in transit"),
    mpatches.Patch(color=_COLORS["staged"][0],  label="staged"),
    mpatches.Patch(color=_COLORS["waiting"][0], label="waiting"),
    mpatches.Patch(color=_COLORS["stuck"][0],   label="STUCK"),
    mpatches.Patch(color="#FF4444", alpha=0.4,   label="connector"),
], loc="lower right", fontsize=7, framealpha=0.6,
   facecolor="#222233", labelcolor="white", edgecolor="#555577")

_last_mtime = 0.0

def _ensure(pool, n, factory):
    while len(pool) < n:
        pool.append(factory())

def refresh(_frame=None):
    global _last_mtime, HALF

    try:
        mtime = os.path.getmtime(STATE_FILE)
    except OSError:
        return
    if mtime == _last_mtime:
        return
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
    except Exception:
        return
    _last_mtime = mtime

    tick      = int(state.get("tick", 0))
    sim_t     = float(state.get("sim_t", 0.0))
    robot     = state.get("robot", {})
    npcs      = state.get("npcs", [])
    paths     = state.get("npc_paths", [])
    con_cells = state.get("connector_cells_world", [])
    flow_pts  = state.get("flow_points_world", [])

    rx   = float(robot.get("x", _S["cx"]))
    ry   = float(robot.get("y", _S["cy"]))
    ryaw = float(robot.get("yaw_rad", 0.0))

    tgt = next((n for n in npcs if n.get("is_target")), None)
    ncx = float(tgt["x"]) if tgt else rx
    ncy = float(tgt["y"]) if tgt else ry
    if not _S["init"] or abs(ncx - _S["cx"]) > 2.0 or abs(ncy - _S["cy"]) > 2.0:
        _set_view(ncx, ncy)

    if not _S["init"]:
        img, ext = _build_occ_image(ncx, ncy, HALF * 2)
        if img is not None:
            _occ_im[0] = ax.imshow(img, origin="lower", extent=ext,
                                   interpolation="nearest", zorder=0, alpha=0.75)
        if con_cells:
            ca = np.array(con_cells, dtype=float)
            _con_sc[0] = ax.scatter(ca[:,0], ca[:,1], s=18, c="#FF4444",
                                    alpha=0.35, marker="s", linewidths=0, zorder=1)
        for idx, fp in enumerate(flow_pts):
            col = _FP_COLORS[idx % len(_FP_COLORS)]
            sc = ax.scatter([float(fp[0])], [float(fp[1])], s=130, c=col,
                            marker="D", edgecolors="white", linewidths=1.5, zorder=9)
            tx = ax.text(float(fp[0])+0.5, float(fp[1])+0.5,
                         str(fp[2]) if len(fp)>2 else f"P{idx}",
                         fontsize=8, color=col, fontweight="bold", zorder=10)
            _fp_arts.append((sc, tx))
        _S["init"] = True

    # Robot
    rl, rw = 0.6, 0.4
    ca2, sa2 = math.cos(ryaw), math.sin(ryaw)
    verts = np.array([[rx+(lx*ca2-ly*sa2), ry+(lx*sa2+ly*ca2)]
                      for lx,ly in [(rl/2,rw/2),(-rl/2,rw/2),(-rl/2,-rw/2),(rl/2,-rw/2)]])
    if _robot_patch[0] is None:
        _robot_patch[0] = mpatches.Polygon(verts, closed=True,
            facecolor="#00C853", edgecolor="#007B33", linewidth=1.5, zorder=6)
        ax.add_patch(_robot_patch[0])
    else:
        _robot_patch[0].set_xy(verts)
    if _robot_arrow[0] is not None:
        try: _robot_arrow[0].remove()
        except Exception: pass
    _robot_arrow[0] = ax.annotate("",
        xy=(rx+rl*0.8*ca2, ry+rl*0.8*sa2), xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color="#00C853", lw=2.0), zorder=7)

    # NPCs
    path_by_id = {p["track_id"]: p for p in paths}
    _ensure(_npc_circles, len(npcs), lambda: ax.add_patch(
        mpatches.Circle((0,0), _NPC_R, facecolor="#4A90D9", edgecolor="#1B4F8A",
                        linewidth=1.0, zorder=4)))
    _ensure(_npc_labels, len(npcs), lambda: ax.text(
        0, 0, "", fontsize=6, color="white", ha="center", va="bottom", zorder=6,
        bbox=dict(boxstyle="round,pad=0.1", fc="#00000088", ec="none")))
    _ensure(_npc_wp_dots, len(npcs), lambda: ax.scatter(
        [], [], s=35, c="#FFFF00", marker="x", linewidths=1.5, zorder=8))
    _ensure(_npc_path_lns, len(npcs), lambda: ax.plot(
        [], [], linewidth=1.0, alpha=0.75, zorder=3, solid_capstyle="round")[0])

    for i, nd in enumerate(npcs):
        nx = float(nd.get("x", 0)); ny = float(nd.get("y", 0))
        tid = str(nd.get("track_id", f"N{i:02d}"))
        st  = "target" if nd.get("is_target") else str(nd.get("state", "transit"))
        fc, ec = _COLORS.get(st, _COLORS["transit"])
        r = _NPC_R*1.6 if st=="target" else (_NPC_R*1.3 if st=="stuck" else _NPC_R)
        c = _npc_circles[i]
        c.set_center((nx,ny)); c.set_radius(r)
        c.set_facecolor(fc); c.set_edgecolor(ec); c.set_visible(True)
        lbl = _npc_labels[i]
        lbl.set_position((nx, ny+r+0.15)); lbl.set_text(tid)
        lbl.set_color(fc); lbl.set_visible(True)
        sc = _npc_wp_dots[i]
        wp_x = nd.get("wp_x"); wp_y = nd.get("wp_y")
        if wp_x is not None and wp_y is not None:
            sc.set_offsets([[float(wp_x), float(wp_y)]]); sc.set_visible(True)
        else:
            sc.set_visible(False)
        ln = _npc_path_lns[i]
        pi = path_by_id.get(tid)
        if pi and pi.get("wps"):
            wps = pi["wps"]
            ln.set_data([nx]+[w[0] for w in wps], [ny]+[w[1] for w in wps])
            ln.set_color("#FF8C00" if st=="target" else
                         "#FF4444" if st=="stuck" else "#7AB8E8")
            ln.set_linewidth(2.0 if st=="target" else 1.0); ln.set_visible(True)
        else:
            ln.set_visible(False)

    for i in range(len(npcs), len(_npc_circles)):
        _npc_circles[i].set_visible(False)
        _npc_labels[i].set_visible(False)
        _npc_wp_dots[i].set_visible(False)
        _npc_path_lns[i].set_visible(False)

    n_tr = sum(1 for n in npcs if n.get("state")=="transit")
    n_st = sum(1 for n in npcs if n.get("state")=="stuck")
    _status.set_text(f"tick={tick}  t={sim_t:.1f}s  transit={n_tr}  stuck={n_st}")
    fig.canvas.draw_idle()

def _on_key(event):
    global HALF
    if event.key in ("q", "escape"):
        plt.close("all"); import sys; sys.exit(0)
    elif event.key == "+":
        HALF = max(5.0, HALF*0.8); _set_view(_S["cx"], _S["cy"])
    elif event.key == "-":
        HALF = HALF*1.25; _set_view(_S["cx"], _S["cy"])

fig.canvas.mpl_connect("key_press_event", _on_key)
print(f"[Viewer] watching {STATE_FILE}  |  npz={'set' if NPZ_PATH else 'not set'}")

ani = FuncAnimation(fig, refresh, interval=INTERVAL, blit=False, cache_frame_data=False)
plt.show()
