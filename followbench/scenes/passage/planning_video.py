"""RGB + planning BEV video recorder for passage episodes.

Frame layout (top -> bottom):
  HUD bar      : time, perception state, distance, far-timer, commands
  Front RGB    : robot front camera with perception bboxes
  Planning BEV : grid map, GT humans/target, perception target, MPC trajectories
  Timeline     : robot-target distance, cmd_v / target speed, perception state band
"""
from __future__ import annotations

import collections
import json
import math
import os
import shutil
import subprocess
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

W = 800
HUD_H = 64
RGB_H = 600
BEV_H = 600
TL_H = 160
BEV_PPM = 50.0  # pixels per metre -> 16 m x 12 m window
TRAIL_SEC = 10.0
TL_WINDOW_SEC = 30.0

# RGB colours (aligned with debug_vis.debug_visualizer conventions)
C_TRACKING = (60, 200, 60)
C_REID = (230, 60, 60)
C_OTHER = (150, 150, 150)
C_ROBOT = (0, 170, 0)
C_TARGET_GT = (255, 140, 0)
C_HUMAN = (70, 120, 230)
C_PERC = (205, 30, 180)
C_TRACK = (200, 140, 0)
C_GOAL = (220, 40, 40)
C_MPC_OPT = (230, 60, 60)
C_MPC_REF = (120, 120, 125)
C_OBS = (255, 120, 180)
C_PRED = (255, 150, 30)

STATE_COLORS = {"tracking": C_TRACKING, "reid": C_REID}


def _state_color(state: Optional[str]) -> Tuple[int, int, int]:
    return STATE_COLORS.get(str(state), C_OTHER)


def _xy(p) -> Optional[Tuple[float, float]]:
    try:
        if p is None:
            return None
        return float(p[0]), float(p[1])
    except (TypeError, ValueError, IndexError):
        return None


class _GridBackground:
    def __init__(self, grid_npz: Optional[str]) -> None:
        self.ok = False
        if not grid_npz or not os.path.exists(grid_npz):
            return
        data = np.load(grid_npz, allow_pickle=True)
        meta = {}
        if "__meta__" in data.files:
            try:
                meta = json.loads(str(data["__meta__"]))
            except (TypeError, ValueError):
                meta = {}
        self.res = float(meta.get("resolution", 0.5))
        self.world_min = np.asarray(data["world_min"], dtype=np.float64)[:2]
        raw = np.asarray(data["obstacle_raw"], dtype=np.uint8) > 0
        infl = np.asarray(data["obstacle_inflated"], dtype=np.uint8) > 0 if "obstacle_inflated" in data.files else raw
        roi = np.asarray(data["roi_mask"], dtype=np.uint8) > 0 if "roi_mask" in data.files else np.ones_like(raw)
        lut = np.zeros(raw.shape + (3,), dtype=np.uint8)
        lut[:] = (188, 190, 196)      # outside ROI: light grey
        lut[roi] = (246, 246, 248)     # walkable ROI: near-white
        lut[infl] = (150, 150, 155)    # inflated obstacle buffer: mid grey
        lut[raw] = (96, 96, 100)       # raw obstacle: dark grey
        self.lut = lut
        self.h, self.w = raw.shape
        self.ok = True

    def render(self, x0: float, y0: float, width: int, height: int, ppm: float) -> np.ndarray:
        if not self.ok:
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:] = (40, 40, 46)
            return img
        xs = x0 + (np.arange(width) + 0.5) / ppm
        ys = y0 + (np.arange(height) + 0.5) / ppm
        gx = np.floor((xs - self.world_min[0]) / self.res).astype(np.int64)
        gy = np.floor((ys - self.world_min[1]) / self.res).astype(np.int64)
        vx = (gx >= 0) & (gx < self.w)
        vy = (gy >= 0) & (gy < self.h)
        img = self.lut[np.clip(gy, 0, self.h - 1)][:, np.clip(gx, 0, self.w - 1)].copy()
        img[~vy, :] = (20, 20, 20)
        img[:, ~vx] = (20, 20, 20)
        return img


class PlanningVideoRecorder:
    def __init__(
        self,
        run_dir: str,
        fps: float,
        grid_npz: Optional[str],
        dt: float,
        target_track_id: str,
        desired_distance: float,
        fail_dist_m: float,
        robot_length: float = 0.62,
        robot_width: float = 0.58,
        human_radius: float = 0.35,
        title: str = "",
        filename: str = "rgb_planning.mp4",
    ) -> None:
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, filename)
        self.fps = max(float(fps), 1.0)
        self.dt = float(dt)
        self.target_track_id = str(target_track_id)
        self.desired_distance = float(desired_distance)
        self.fail_dist_m = float(fail_dist_m)
        self.robot_length = float(robot_length)
        self.robot_width = float(robot_width)
        self.human_radius = float(human_radius)
        self.title = title
        self.frame_h = HUD_H + RGB_H + BEV_H + TL_H
        self.grid = _GridBackground(grid_npz)
        n_trail = int(TRAIL_SEC / max(self.dt, 1e-3))
        self.robot_trail: Deque[Tuple[float, float]] = collections.deque(maxlen=n_trail)
        self.target_trail: Deque[Tuple[float, float]] = collections.deque(maxlen=n_trail)
        self.history: List[Dict[str, Any]] = []
        self._prev_robot_xy: Optional[Tuple[float, float]] = None
        self._robot_speed_ema = 0.0
        self.last_frame: Optional[np.ndarray] = None
        self.frames_written = 0
        self._closed = False
        self._proc: Optional[subprocess.Popen] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._open()

    # ── writer backends ─────────────────────────────────────────────────────
    def _open(self) -> None:
        if shutil.which("ffmpeg") is not None:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{W}x{self.frame_h}", "-r", f"{self.fps:.6f}", "-i", "-",
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart",
                self.path,
            ]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[VIS] planning video -> {self.path} ({W}x{self.frame_h} @ {self.fps:.0f}Hz, ffmpeg)")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[VIS][WARN] ffmpeg failed ({exc}); falling back to cv2")
                self._proc = None
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (W, self.frame_h))
        if writer.isOpened():
            self._writer = writer
            print(f"[VIS] planning video -> {self.path} (cv2/mp4v)")
        else:
            print(f"[VIS][ERROR] cannot open video writer for {self.path}")

    def _emit(self, frame_rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(bgr.tobytes())
            except (BrokenPipeError, ValueError):
                self._proc = None
                return
        elif self._writer is not None:
            self._writer.write(bgr)
        else:
            return
        self.frames_written += 1

    # ── public API ──────────────────────────────────────────────────────────
    def write(
        self,
        *,
        tick: int,
        time_s: float,
        rgb_front: Optional[np.ndarray],
        robot_state,
        target_state,
        npc_states: Sequence,
        debug: Optional[dict],
        v_cmd: Optional[float],
        w_cmd: Optional[float],
        visible: bool,
        extra_hud: str = "",
    ) -> None:
        if self._closed:
            return
        debug = debug if isinstance(debug, dict) else {}
        rx, ry, ryaw = float(robot_state.x), float(robot_state.y), float(robot_state.yaw_rad)
        if self._prev_robot_xy is not None:
            inst = math.hypot(rx - self._prev_robot_xy[0], ry - self._prev_robot_xy[1]) / max(self.dt, 1e-3)
            self._robot_speed_ema = 0.7 * self._robot_speed_ema + 0.3 * inst
        self._prev_robot_xy = (rx, ry)
        self.robot_trail.append((rx, ry))
        tgt_xy = None
        tgt_speed = 0.0
        dist = None
        if target_state is not None:
            tgt_xy = (float(target_state.x), float(target_state.y))
            tgt_speed = float(math.hypot(target_state.vx, target_state.vy))
            self.target_trail.append(tgt_xy)
            dist = math.hypot(rx - tgt_xy[0], ry - tgt_xy[1])

        state = debug.get("perception_state")
        target_lost = bool(debug.get("target_lost", False))
        far = debug.get("target_far") if isinstance(debug.get("target_far"), dict) else {}
        self.history.append({
            "t": float(time_s), "dist": dist, "v_cmd": v_cmd, "v_robot": self._robot_speed_ema,
            "v_tgt": tgt_speed, "state": state, "lost": target_lost,
        })

        hud = self._render_hud(tick, time_s, state, target_lost, dist, far, v_cmd, w_cmd,
                               tgt_speed, visible, debug, extra_hud)
        rgb = self._render_rgb(rgb_front, state, target_lost)
        bev = self._render_bev(rx, ry, ryaw, tgt_xy, target_state, npc_states, debug, v_cmd, target_lost)
        tl = self._render_timeline()
        frame = np.concatenate([hud, rgb, bev, tl], axis=0)
        self.last_frame = frame
        self._emit(frame)

    def close(self, final_reason: Optional[str] = None, success: Optional[bool] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if self.last_frame is not None:
            banner = self.last_frame.copy()
            txt = f"END: {final_reason or 'n/a'}  success={success}"
            color = (40, 160, 40) if success else (200, 40, 40)
            y0 = HUD_H + RGB_H // 2 - 40
            cv2.rectangle(banner, (0, y0), (W, y0 + 80), color, -1)
            cv2.putText(banner, txt, (20, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            for _ in range(int(self.fps * 2)):
                self._emit(banner)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                pass
        if self._writer is not None:
            self._writer.release()
        print(f"[VIS] planning video closed: {self.path} frames={self.frames_written}")

    # ── panels ──────────────────────────────────────────────────────────────
    def _render_hud(self, tick, time_s, state, target_lost, dist, far, v_cmd, w_cmd,
                    tgt_speed, visible, debug, extra_hud) -> np.ndarray:
        bar = np.zeros((HUD_H, W, 3), dtype=np.uint8)
        bar[:] = (24, 24, 28)
        f = cv2.FONT_HERSHEY_SIMPLEX
        st_txt = f"{state or 'n/a'}{' / LOST' if target_lost else ''}"
        sc = _state_color(state)
        (tw, th), _ = cv2.getTextSize(st_txt, f, 0.6, 2)
        cv2.rectangle(bar, (8, 6), (8 + tw + 12, 6 + th + 10), sc, -1)
        cv2.putText(bar, st_txt, (14, 6 + th + 4), f, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        dist_txt = "n/a" if dist is None else f"{dist:.2f}m"
        far_dur = float(far.get("duration_s", 0.0) or 0.0)
        far_sec = float(far.get("fail_sec", 0.0) or 0.0)
        line1 = (f"t={time_s:6.1f}s tick={tick}  dist={dist_txt}  far={far_dur:.1f}/{far_sec:.1f}s  "
                 f"vis={int(bool(visible))}")
        cv2.putText(bar, line1, (tw + 32, 6 + th + 4), f, 0.5, (230, 230, 235), 1, cv2.LINE_AA)
        vc = "n/a" if v_cmd is None else f"{v_cmd:+.2f}"
        wc = "n/a" if w_cmd is None else f"{w_cmd:+.2f}"
        pt = debug.get("planner_target") if isinstance(debug.get("planner_target"), dict) else {}
        line2 = (f"cmd v={vc} w={wc}  v_robot={self._robot_speed_ema:.2f}  v_tgt={tgt_speed:.2f}  "
                 f"plan_tgt_v={float(pt.get('perception_target_speed', 0.0) or 0.0):.2f}  "
                 f"tracks={len(debug.get('tracked_peds') or [])}  {extra_hud}")
        cv2.putText(bar, line2, (10, HUD_H - 10), f, 0.45, (200, 210, 220), 1, cv2.LINE_AA)
        return bar

    def _render_rgb(self, rgb, state, target_lost) -> np.ndarray:
        canvas = np.zeros((RGB_H, W, 3), dtype=np.uint8)
        if rgb is not None and rgb.size > 0:
            ih, iw = rgb.shape[:2]
            s = min(W / iw, RGB_H / ih)
            nw, nh = max(int(iw * s), 1), max(int(ih * s), 1)
            x0, y0 = (W - nw) // 2, (RGB_H - nh) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.putText(canvas, "Front RGB", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)
        if self.title:
            cv2.putText(canvas, self.title, (10, RGB_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, 0), (W - 1, RGB_H - 1), _state_color(state), 4)
        return canvas

    def _render_bev(self, rx, ry, ryaw, tgt_xy, target_state, npc_states, debug, v_cmd, target_lost) -> np.ndarray:
        cx, cy = (rx, ry) if tgt_xy is None else ((rx + tgt_xy[0]) / 2.0, (ry + tgt_xy[1]) / 2.0)
        x0 = cx - W / (2 * BEV_PPM)
        y0 = cy - BEV_H / (2 * BEV_PPM)
        img = self.grid.render(x0, y0, W, BEV_H, BEV_PPM)

        def P(x, y) -> Tuple[int, int]:
            return int(round((x - x0) * BEV_PPM)), int(round((y - y0) * BEV_PPM))

        def R(m) -> int:
            return max(int(round(m * BEV_PPM)), 1)

        def polyline(pts, color, thick=2, dots=False):
            pts = [p for p in (_xy(q) for q in pts or []) if p is not None]
            if len(pts) >= 2:
                arr = np.array([P(*p) for p in pts], dtype=np.int32)
                cv2.polylines(img, [arr], False, color, thick, cv2.LINE_AA)
            if dots:
                for p in pts:
                    cv2.circle(img, P(*p), 3, color, -1, cv2.LINE_AA)

        # 1 m grid lines
        for gx in range(int(math.floor(x0)), int(math.ceil(x0 + W / BEV_PPM)) + 1):
            px = P(gx, y0)[0]
            cv2.line(img, (px, 0), (px, BEV_H), (218, 218, 222), 1)
        for gy in range(int(math.floor(y0)), int(math.ceil(y0 + BEV_H / BEV_PPM)) + 1):
            py = P(x0, gy)[1]
            cv2.line(img, (0, py), (W, py), (218, 218, 222), 1)

        # trails
        polyline(list(self.target_trail), (180, 120, 0), 2)
        polyline(list(self.robot_trail), (0, 120, 0), 2)

        # planner obstacles
        for ob in debug.get("obstacles") or []:
            verts = ob.get("vertices") if isinstance(ob, dict) else None
            if verts:
                arr = np.array([P(float(v[0]), float(v[1])) for v in verts], dtype=np.int32)
                cv2.polylines(img, [arr], True, C_OBS, 2, cv2.LINE_AA)

        # GT humans
        for s in npc_states or []:
            if str(s.track_id) == self.target_track_id:
                continue
            p = P(float(s.x), float(s.y))
            cv2.circle(img, p, R(self.human_radius), C_HUMAN, 2, cv2.LINE_AA)
            sp = math.hypot(s.vx, s.vy)
            if 0.05 < sp < 5.0:
                q = P(float(s.x) + s.vx * 0.8, float(s.y) + s.vy * 0.8)
                cv2.arrowedLine(img, p, q, C_HUMAN, 1, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(img, str(s.track_id), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        C_HUMAN, 1, cv2.LINE_AA)

        # GT target + distance rings
        if tgt_xy is not None:
            p = P(*tgt_xy)
            cv2.circle(img, p, R(self.desired_distance), (60, 160, 60), 1, cv2.LINE_AA)
            cv2.circle(img, p, R(self.fail_dist_m), (200, 60, 60), 1, cv2.LINE_AA)
            cv2.circle(img, p, R(self.human_radius), C_TARGET_GT, -1, cv2.LINE_AA)
            if target_state is not None:
                q = P(tgt_xy[0] + target_state.vx * 1.0, tgt_xy[1] + target_state.vy * 1.0)
                cv2.arrowedLine(img, p, q, C_TARGET_GT, 2, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(img, f"{self.target_track_id}(GT)", (p[0] + 10, p[1] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TARGET_GT, 1, cv2.LINE_AA)

        # perception tracks
        for tp in debug.get("tracked_peds") or []:
            if not isinstance(tp, dict):
                continue
            p = P(float(tp.get("x", 0.0)), float(tp.get("y", 0.0)))
            col = C_PERC if tp.get("is_target") else C_TRACK
            cv2.drawMarker(img, p, col, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
            cv2.putText(img, f"D{tp.get('track_id')}", (p[0] + 6, p[1] + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, col, 1, cv2.LINE_AA)

        # planner target (what the MPC actually chases)
        pt = debug.get("planner_target") if isinstance(debug.get("planner_target"), dict) else {}
        ptxy = _xy(pt.get("perception_target_filtered_xy"))
        if ptxy is not None:
            p = P(*ptxy)
            cv2.drawMarker(img, p, C_PERC, cv2.MARKER_DIAMOND, 18, 2, cv2.LINE_AA)
            label = "plan_tgt (LAST-KNOWN, v=0)" if target_lost else "plan_tgt"
            cv2.putText(img, label, (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        C_PERC, 1, cv2.LINE_AA)

        polyline(debug.get("predicted_target_traj"), C_PERC, 1, dots=True)
        polyline(debug.get("goal_traj"), C_GOAL, 1, dots=True)
        polyline(debug.get("mpc_ref_traj"), C_MPC_REF, 1)
        polyline(debug.get("mpc_opt_traj"), C_MPC_OPT, 3, dots=True)
        gp = _xy(debug.get("goal_point"))
        if gp is not None:
            cv2.drawMarker(img, P(*gp), C_GOAL, cv2.MARKER_STAR, 20, 2, cv2.LINE_AA)

        # robot footprint
        c, s = math.cos(ryaw), math.sin(ryaw)
        hl, hw = self.robot_length / 2, self.robot_width / 2
        corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        arr = np.array([P(rx + c * a - s * b, ry + s * a + c * b) for a, b in corners], dtype=np.int32)
        cv2.fillPoly(img, [arr], (0, 140, 0))
        cv2.polylines(img, [arr], True, C_ROBOT, 2, cv2.LINE_AA)
        hv = max(abs(v_cmd or 0.0), 0.0)
        cv2.arrowedLine(img, P(rx, ry), P(rx + c * (0.5 + hv), ry + s * (0.5 + hv)), C_ROBOT, 2,
                        cv2.LINE_AA, tipLength=0.25)

        # legend + scale
        legend = [
            (C_ROBOT, "robot (arrow ~ cmd_v)"), (C_TARGET_GT, "target GT (rings 1.5/3.0m)"),
            (C_HUMAN, "other humans GT"), (C_TRACK, "perception tracks"),
            (C_PERC, "planner target"), (C_PRED, "pred target traj"),
            (C_GOAL, "goal traj / goal pt"), (C_MPC_OPT, "MPC optimal traj"),
            (C_MPC_REF, "MPC ref traj"), (C_OBS, "planner obstacles"),
        ]
        ly = 22
        cv2.rectangle(img, (W - 250, 4), (W - 4, 8 + 18 * len(legend)), (20, 20, 24), -1)
        for col, txt in legend:
            cv2.line(img, (W - 244, ly - 4), (W - 224, ly - 4), col, 3)
            cv2.putText(img, txt, (W - 218, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
            ly += 18
        cv2.line(img, (12, BEV_H - 14), (12 + int(BEV_PPM), BEV_H - 14), (255, 255, 255), 2)
        cv2.putText(img, "1 m", (14, BEV_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "Planning BEV", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)
        return img

    def _render_timeline(self) -> np.ndarray:
        img = np.zeros((TL_H, W, 3), dtype=np.uint8)
        img[:] = (18, 18, 22)
        if not self.history:
            return img
        t_now = self.history[-1]["t"]
        t0 = t_now - TL_WINDOW_SEC
        rows = [h for h in self.history if h["t"] >= t0]
        L, Rm, top, band_h = 40, 10, 8, 14
        plot_h = TL_H - top - band_h - 22
        pw = W - L - Rm

        def X(t) -> int:
            return L + int((t - t0) / TL_WINDOW_SEC * pw)

        def Yd(d) -> int:  # distance axis 0..5 m
            return top + int((1.0 - min(max(d, 0.0), 5.0) / 5.0) * plot_h)

        for d, col in ((self.desired_distance, (60, 160, 60)), (self.fail_dist_m, (200, 60, 60))):
            cv2.line(img, (L, Yd(d)), (W - Rm, Yd(d)), col, 1)
            cv2.putText(img, f"{d:.1f}m", (2, Yd(d) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

        def series(key, col, scale=1.0, thick=1):
            pts = [(X(h["t"]), Yd(h[key] * scale)) for h in rows if h.get(key) is not None]
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, col, thick, cv2.LINE_AA)

        # speeds plotted on the same axis (1 m/s == 1 m)
        series("v_tgt", C_TARGET_GT)
        series("v_cmd", C_MPC_OPT)
        series("v_robot", C_ROBOT)
        series("dist", (255, 255, 255), thick=2)

        by = TL_H - band_h - 18
        for i in range(len(rows) - 1):
            col = _state_color(rows[i]["state"])
            cv2.rectangle(img, (X(rows[i]["t"]), by), (max(X(rows[i + 1]["t"]), X(rows[i]["t"]) + 1), by + band_h),
                          col, -1)
            if rows[i]["lost"]:
                cv2.line(img, (X(rows[i]["t"]), by + band_h + 2), (X(rows[i]["t"]), by + band_h + 5),
                         (255, 255, 255), 1)
        f = cv2.FONT_HERSHEY_SIMPLEX
        txt = [((255, 255, 255), "dist"), (C_TARGET_GT, "v_tgt"), (C_MPC_OPT, "cmd_v"), (C_ROBOT, "v_robot"),
               (C_TRACKING, "tracking"), (C_REID, "reid"), ((255, 255, 255), "| = target_lost")]
        x = L
        for col, t in txt:
            cv2.putText(img, t, (x, TL_H - 4), f, 0.38, col, 1, cv2.LINE_AA)
            x += cv2.getTextSize(t, f, 0.38, 1)[0][0] + 14
        cv2.putText(img, f"last {TL_WINDOW_SEC:.0f}s", (W - 70, TL_H - 4), f, 0.38, (180, 180, 180), 1)
        return img


__all__ = ["PlanningVideoRecorder"]
