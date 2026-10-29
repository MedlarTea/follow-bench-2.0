"""End-to-end TrackVLA follower adapter.

Wraps `OfflineUnifiedAgent` (vendored under
``baselines/third_party/trackvla/socialRPF/``). On every tick, takes the
front-camera RGB, runs ViT (DINOv3 + SigLIP) → Qwen3-4B VLM → planner head, and
turns the second predicted waypoint into a (v, w) command (dt=0.1 s).

Single-view "forward" only. Side cameras are ignored.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

_THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
_LEARNING_DIR = os.path.join(_THIRD_PARTY_DIR, "trackvla")
if _LEARNING_DIR not in sys.path:
    sys.path.insert(0, _LEARNING_DIR)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from followbench.sim.core_types import FollowAction, FollowObservation
from followbench.sim.follow_policy_adapter import FollowerPolicyAdapter

try:
    from socialRPF.baseline.offline_agent_unified import OfflineUnifiedAgent
    from socialRPF.constants import DATA_ROOT, IMAGE_SIZE

    _HAS_TRACKVLA = True
    _TRACKVLA_IMPORT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_TRACKVLA = False
    _TRACKVLA_IMPORT_ERR = str(_e)

# Auto-caption helper (Qwen3-VL). Imported lazily below so the adapter still
# loads when the optional captioner dependency/checkpoint is missing.
try:
    from baselines.adapters._qwen3vl_captioner import (
        Qwen3VLCaptioner,
        _bbox_crop_with_padding,
    )
    _HAS_QWEN3VL_CAPTIONER = True
    _QWEN3VL_IMPORT_ERR: Optional[str] = None
except ImportError as _e:
    _HAS_QWEN3VL_CAPTIONER = False
    _QWEN3VL_IMPORT_ERR = str(_e)


_DEFAULT_CAPTION_CKPT_REL = "LLM_hf/qwen3-vl-4b-instruct"


_DEFAULT_CKPT_REL = (
    "ckpts/260417-qwen3-4b-unified-nav-qa-view-fwd-ws16-bs16-acc1-glb256-"
    "qabs16-ep5-lrb2e-5-lrh2e-4-bn10.0-bq1.0/model_epoch04_step024000.pt"
)


def _resize_with_padding(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(side / w, side / h)
    rw, rh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = side - rh, side - rw
    top, left = pad_h // 2, pad_w // 2
    return cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=0,
    )


_TRAIN_TIME_DT = 0.1  # Waypoint spacing the checkpoint was trained with.


class TrackVlaFollowerPolicy(FollowerPolicyAdapter):
    def __init__(
        self,
        dt: float,
        instruction: str = "Follow the person",
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        render_overlay: bool = False,
        # ── Auto-caption (optional, requires Qwen3-VL checkpoint) ────────────
        auto_caption: bool = True,
        caption_ckpt_path: Optional[str] = None,
        caption_device: Optional[str] = None,
    ) -> None:
        if not _HAS_TRACKVLA:
            raise ImportError(
                f"TrackVLA imports failed: {_TRACKVLA_IMPORT_ERR}. Make sure "
                "baselines/third_party/trackvla/socialRPF/ is present "
                "and transformers/peft are installed in the active env."
            )

        if ckpt_path is None:
            ckpt_path = os.path.join(DATA_ROOT, _DEFAULT_CKPT_REL)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"TrackVLA ckpt not found: {ckpt_path}. Did the rsync to "
                f"{DATA_ROOT} finish?"
            )

        self._dt_carla = float(dt)
        # ``_instruction`` is the live prompt fed to the VLM each tick. It starts
        # as the user-supplied base (or the default "Follow the person") and may
        # be rewritten once on the first good frame if auto_caption is enabled.
        self._instruction = str(instruction)
        self._render_overlay = bool(render_overlay)
        self._image_size = int(IMAGE_SIZE)

        # ── Auto-caption state ──────────────────────────────────────────────
        # Treat any instruction that differs from the default as an explicit
        # user override and do NOT rewrite it — even if auto_caption=True.
        self._user_supplied_instruction = (str(instruction) != "Follow the person")
        self._auto_caption = bool(auto_caption) and not self._user_supplied_instruction
        self._caption_ckpt_path = caption_ckpt_path or os.path.join(
            DATA_ROOT, _DEFAULT_CAPTION_CKPT_REL,
        )
        self._caption_device = caption_device
        self._caption_done = False          # becomes True after first successful call
        self._captioner: Optional[Qwen3VLCaptioner] = None

        # Two dt-related decisions when sim != training rate:
        #   1) waypoint→velocity uses _TRAIN_TIME_DT so command magnitudes match
        #      what the planner head was trained to produce.
        #   2) inference is throttled to ~10 Hz; intermediate sim ticks reuse
        #      the previous command. Otherwise the 31-step coarse history would
        #      cover 31*dt_carla seconds (1.55 s at 20 Hz) instead of the 3.1 s
        #      training window — the model would perceive the world as moving
        #      at 2× speed.
        self._inference_stride = max(1, int(round(_TRAIN_TIME_DT / self._dt_carla)))
        self._tick_counter = 0
        self._last_action_cached = (0.0, 0.0, 0.0)
        if self._inference_stride != 1 or abs(self._dt_carla - _TRAIN_TIME_DT) > 1e-3:
            print(
                f"[TRACKVLA] sim dt={self._dt_carla:.3f}s, model dt={_TRAIN_TIME_DT}s "
                f"→ inference_stride={self._inference_stride} (skip {self._inference_stride-1} ticks between calls).",
                flush=True,
            )

        print(f"[TRACKVLA] loading ckpt {ckpt_path}", flush=True)
        t0 = time.perf_counter()
        self._agent = OfflineUnifiedAgent(
            ckpt_path=ckpt_path, device=device, dt=_TRAIN_TIME_DT,
        )
        print(f"[TRACKVLA] ready ({(time.perf_counter()-t0):.1f}s)", flush=True)

        self._last_traj_world: List[List[float]] = []
        self._last_rendered: Optional[np.ndarray] = None
        self._last_action = (0.0, 0.0, 0.0)

    def reset(self) -> None:
        self._agent.reset()
        self._last_traj_world = []
        self._last_rendered = None
        self._last_action = (0.0, 0.0, 0.0)
        self._tick_counter = 0
        self._last_action_cached = (0.0, 0.0, 0.0)
        # Forget any previously-generated caption so a new episode re-captions
        # on its own first usable frame.
        self._caption_done = False

    def get_debug_info(self) -> dict:
        info = {
            "obstacles": [],
            "traj_points": list(self._last_traj_world),
            "instruction": self._instruction,
            "trackvla_action": list(self._last_action),
        }
        if self._last_rendered is not None:
            info["rendered_front"] = self._last_rendered
        return info

    # ── Auto-caption ──────────────────────────────────────────────────────────

    def _maybe_generate_caption(self, obs: FollowObservation) -> None:
        """First-frame hook: crop the target patch via the GT bbox, run
        Qwen3-VL once to describe the person's clothing, then splice the phrase
        into ``self._instruction`` as ``"Follow the person <phrase>."``.

        Any failure (missing bbox, missing checkpoint, VLM error) degrades
        gracefully — we keep the current instruction and skip further attempts
        for this episode.
        """
        if self._caption_done or not self._auto_caption:
            return

        bbox = obs.extras.get("target_bbox_xyxy") if obs.extras else None
        if bbox is None or obs.rgb_image is None:
            # Defer to a later tick when the target is visible & RGB is ready.
            return

        patch = _bbox_crop_with_padding(obs.rgb_image, bbox)
        if patch is None:
            return

        # Lazy-load the captioner on first use so other planners pay nothing.
        if self._captioner is None:
            if not _HAS_QWEN3VL_CAPTIONER:
                print(f"[TRACKVLA] auto-caption disabled: {_QWEN3VL_IMPORT_ERR}",
                      flush=True)
                self._caption_done = True
                return
            try:
                print(f"[TRACKVLA] loading Qwen3-VL captioner "
                      f"from {self._caption_ckpt_path}", flush=True)
                t0 = time.perf_counter()
                self._captioner = Qwen3VLCaptioner(
                    ckpt_dir=self._caption_ckpt_path,
                    device=self._caption_device or "cuda",
                )
                print(f"[TRACKVLA] captioner ready "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
            except Exception as e:
                print(f"[TRACKVLA] captioner init failed: {e} — "
                      f"falling back to plain 'Follow the person'.", flush=True)
                self._caption_done = True
                return

        try:
            t0 = time.perf_counter()
            phrase = self._captioner.describe(patch)
            ms = (time.perf_counter() - t0) * 1000.0
            self._instruction = f"Follow the person {phrase}."
            print(f"[TRACKVLA] captioned in {ms:.0f} ms: {self._instruction!r}",
                  flush=True)
        except Exception as e:
            print(f"[TRACKVLA] caption generation failed: {e} — "
                  f"keeping existing instruction.", flush=True)
        finally:
            # Regardless of success/failure, free the captioner's VRAM so
            # TrackVLA's navigation VLM has the full GPU for the rest of the
            # episode.
            try:
                self._captioner.release()
            except Exception:
                pass
            self._captioner = None
            self._caption_done = True

    def act(self, obs: FollowObservation) -> FollowAction:
        if obs.rgb_image is None:
            print("[TRACKVLA] no rgb_image — emitting zero action", flush=True)
            return FollowAction(v_mps=0.0, w_radps=0.0)

        # One-shot: the first tick where a GT bbox is available gets a VLM-
        # generated clothing caption spliced into the prompt. After that, the
        # captioner is released and this is a no-op. (Safe to run every tick.)
        self._maybe_generate_caption(obs)

        # Hold previous command on intermediate ticks to keep model query rate
        # close to the 10 Hz training rate.
        if self._tick_counter % self._inference_stride != 0:
            self._tick_counter += 1
            v_mps, _, w_radps = self._last_action_cached
            return FollowAction(v_mps=v_mps, w_radps=w_radps)

        rgb = _resize_with_padding(obs.rgb_image, self._image_size)
        t0 = time.perf_counter()
        action, traj, rendered = self._agent.predict_frame(
            rgb, self._instruction, render=self._render_overlay,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._tick_counter += 1

        vx = float(action[0])
        vy = float(action[1])
        wz = float(action[2])
        # Handedness flip: model is trained in a right-handed frame (+y=left,
        # +theta=CCW). CARLA's world is left-handed (+yaw_deg=CW from above),
        # and apply_velocity_command does yaw_deg += w*dt directly. So +theta
        # from the model means physical CCW, which CARLA reaches with negative
        # w_radps. Without this flip the robot turns toward the wrong side.
        v_mps = vx
        w_radps = -wz
        self._last_action = (vx, vy, wz)
        self._last_action_cached = (v_mps, vy, w_radps)

        # Transform robot-frame traj → CARLA world. Model is right-handed
        # (+x forward, +y left); CARLA world is left-handed (+yaw_deg=CW). The
        # forward axis aligns; the lateral axis is mirrored, so ty contributes
        # with the opposite sign of the standard rotation matrix.
        traj_world: List[List[float]] = []
        if traj is not None and traj.shape[0] > 0:
            cy = math.cos(float(obs.robot.yaw_rad))
            sy = math.sin(float(obs.robot.yaw_rad))
            rx, ry = float(obs.robot.x), float(obs.robot.y)
            for i in range(traj.shape[0]):
                tx = float(traj[i, 0])
                ty = float(traj[i, 1]) if traj.shape[1] >= 2 else 0.0
                wx = rx + tx * cy + ty * sy
                wy = ry + tx * sy - ty * cy
                traj_world.append([wx, wy])
        self._last_traj_world = traj_world
        self._last_rendered = rendered

        print(
            f"[TRACKVLA] tick={obs.tick}  v={v_mps:+.3f} w={w_radps:+.3f}"
            f"  vy_drop={vy:+.3f}  infer={infer_ms:.1f}ms"
            f"  instr={self._instruction!r}",
            flush=True,
        )
        return FollowAction(v_mps=v_mps, w_radps=w_radps)


__all__ = ["TrackVlaFollowerPolicy"]
