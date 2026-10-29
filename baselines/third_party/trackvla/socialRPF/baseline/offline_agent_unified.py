"""
OfflineUnifiedAgent — Habitat-free offline inference agent for OpenTrackVLA-Unified.

This is a stripped-down version of `trained_agent_multiview_unified.GTBBoxAgentUnified`,
containing only what's needed to run the unified model on user-supplied RGB frames
(e.g. a local video). Everything Habitat-related is removed:

    - No `import habitat` / `habitat_sim`.
    - No subclassing `AgentConfig`.
    - No `act(observations, episode_id, instruction)` (which used Habitat sensor names).

The class exposes a clean `predict_frame(rgb)` interface that:
    1) encodes the frame's coarse / fine vision tokens (DINO + SigLIP)
    2) maintains a rolling coarse-token history (length=`history`)
    3) calls `unified_model.forward_navigation` to predict `n_waypoints` waypoints
    4) converts the second waypoint into a [vx, vy, wz] velocity command (dt=0.1s)
    5) renders a copy of the frame with the predicted trajectory + instruction overlay

Only single-view (`view_list=["forward"]`) checkpoints are supported by this agent.
For multi-view checkpoints, extend `predict_frame` to accept a dict of views.
"""

from __future__ import annotations

import os
import os.path as osp
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from socialRPF.constants import VIEW_YAWS, DATA_ROOT
from socialRPF.model.cache_gridpool import (
    VisionCacheConfig,
    VisionFeatureCacher,
    grid_pool_tokens,
)
from socialRPF.model.openTrackVLA_multiview_mixed import load_model_config
from socialRPF.model.openTrackVLA_multiview_unified_alpha import (
    OpenTrackVLAUnifiedAlpha as UnifiedModel,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class OfflineUnifiedAgent:
    """Habitat-free inference wrapper around `OpenTrackVLAUnifiedAlpha`.

    Parameters
    ----------
    ckpt_path : str
        Path to the trained `.pt` checkpoint. `model_config.json` must sit next
        to it.
    device : str | torch.device | None
        Compute device. Defaults to CUDA if available.
    history : int
        Coarse-token history length (default 31, matches the agent used in
        Habitat eval).
    image_size : int
        Vision encoder input resolution (default 384, matches VisionCacheConfig).
    vision_feat_dim : int
        Concatenated DINO+SigLIP feature dim (default 1536).
    dt : float
        Time step (s) used to convert the predicted waypoint into a velocity.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str | torch.device] = None,
        history: int = 31,
        image_size: int = 384,
        vision_feat_dim: int = 1536,
        dt: float = 0.1,
    ):
        if not osp.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.ckpt_path = ckpt_path
        self.history = history
        self.image_size = image_size
        self.vision_feat_dim = vision_feat_dim
        self.dt = dt

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ---- Model ----
        self.model_config = None
        self.unified_model: Optional[UnifiedModel] = None
        self._init_unified_model()

        self.view_list: List[str] = list(self.model_config.view_list)
        if self.view_list != ["forward"]:
            print(
                f"[OfflineAgent][WARN] view_list={self.view_list} != ['forward']. "
                "OfflineUnifiedAgent only feeds the forward camera; non-forward "
                "views will be missing."
            )

        # ---- Vision encoder (lazy) ----
        self._vision_cache: Optional[VisionFeatureCacher] = None

        # ---- Rolling coarse-token history per view ----
        self._coarse_hist_tokens: Dict[str, deque] = {
            v: deque(maxlen=self.history) for v in self.view_list
        }
        self._last_predicted_traj: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear coarse-token history (start a new episode)."""
        for v in self.view_list:
            self._coarse_hist_tokens[v].clear()
        self._last_predicted_traj = None

    @property
    def last_predicted_traj(self) -> Optional[np.ndarray]:
        """(n_waypoints, 3) numpy array of the most recent prediction (or None)."""
        return self._last_predicted_traj

    def predict_frame(
        self,
        rgb: np.ndarray,
        instruction: str,
        render: bool = True,
    ) -> Tuple[List[float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Run a single inference step on one RGB frame.

        Parameters
        ----------
        rgb : np.ndarray, shape (H, W, 3), dtype uint8, RGB
            Forward camera frame. Should already be resized/padded to
            `image_size` x `image_size`. (Caller is responsible for resizing.)
        instruction : str
            Navigation instruction text.
        render : bool
            If True, overlay predicted trajectory + instruction onto a copy of
            the frame and return it as the third tuple element.

        Returns
        -------
        action : list of 3 floats
            [vx, vy, wz] velocity command derived from the second waypoint.
        traj : np.ndarray | None, shape (n_waypoints, 3)
            Predicted waypoints (x, y, theta) in the agent frame.
        rendered : np.ndarray | None, shape (H, W, 3), uint8 RGB
            Frame with overlay (only if `render=True`).
        """
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"`rgb` must be (H, W, 3) uint8, got shape={getattr(rgb, 'shape', None)}"
            )

        # Single-view forward only
        rgb_dict = {"forward": rgb}
        action = self._navigation_action(rgb_dict, instruction)

        traj = self._last_predicted_traj
        rendered = (
            self._render_frame_with_traj(rgb, traj, instruction)
            if render else None
        )
        return action, traj, rendered

    # ------------------------------------------------------------------
    # Model initialization
    # ------------------------------------------------------------------

    def _init_unified_model(self) -> None:
        print(f"[OfflineAgent] Loading checkpoint: {self.ckpt_path}")
        ckpt_dir = osp.dirname(self.ckpt_path)

        cfg_path = osp.join(ckpt_dir, "model_config.json")
        if not osp.isfile(cfg_path):
            raise FileNotFoundError(
                f"model_config.json not found next to ckpt: {cfg_path}"
            )

        self.model_config = load_model_config(cfg_path)
        # Resolve LLM path under DATA_ROOT/LLM_hf/<basename>.
        self.model_config.llm_name = osp.join(
            DATA_ROOT, "LLM_hf", self.model_config.llm_name
        )

        # mmap=True keeps the 7.7 GB state_dict on disk and pages individual
        # tensors in only when load_state_dict copies them. CPU peak drops from
        # ~16 GB (model + full state_dict) to ~9 GB (model alone).
        try:
            obj = torch.load(
                self.ckpt_path, map_location="cpu", mmap=True, weights_only=True,
            )
        except Exception:
            # Older torch builds or non-mmappable serialization formats fall
            # back to the regular path; if memory is the issue this will OOM
            # again, but at least the failure mode is the same as before.
            obj = torch.load(self.ckpt_path, map_location="cpu")

        model = UnifiedModel(self.model_config, vision_feat_dim=self.vision_feat_dim).eval()
        msd = obj.get("model_state") or obj.get("model_state_dict")
        if msd is None:
            raise RuntimeError(
                f"Checkpoint {self.ckpt_path} does not contain "
                "'model_state' or 'model_state_dict'."
            )
        model.load_state_dict(msd, strict=True)
        del msd, obj
        # Cast small fp32 heads (planner/projector/embedder) to bf16 to match the
        # LLM dtype and keep the GPU footprint at ~9 GB instead of ~12 GB.
        if self.device.type == "cuda":
            model = model.to(dtype=torch.bfloat16)
        model = model.to(self.device)
        self.unified_model = model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print("[OfflineAgent] Unified model loaded.")

    def _ensure_vision_cache(self) -> Optional[VisionFeatureCacher]:
        if self._vision_cache is None:
            cfg = VisionCacheConfig(
                image_size=self.image_size,
                batch_size=1,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
            self._vision_cache = VisionFeatureCacher(cfg).eval()
        return self._vision_cache

    # ------------------------------------------------------------------
    # Vision token encoding
    # ------------------------------------------------------------------

    def _encode_frame_tokens(
        self, rgb_np: np.ndarray
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode an RGB frame into (V_coarse(4,C), V_fine(64,C)) tokens."""
        from PIL import Image  # local import keeps top-level deps light

        enc = self._ensure_vision_cache()
        pil = Image.fromarray(rgb_np.astype(np.uint8))
        tok_dino, Hp, Wp = enc._encode_dino([pil])
        tok_sigl = enc._encode_siglip([pil], out_hw=(Hp, Wp))
        Vt_cat = torch.cat([tok_dino, tok_sigl], dim=-1)  # (1, P, C)
        target_dtype = next(self.unified_model.parameters()).dtype
        Vfine = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=64)[0].to(target_dtype)
        Vcoarse = grid_pool_tokens(Vt_cat, Hp, Wp, out_tokens=4)[0].to(target_dtype)
        return Vcoarse, Vfine

    # ------------------------------------------------------------------
    # Navigation forward
    # ------------------------------------------------------------------

    def _navigation_action(
        self,
        rgb_frame_dict: Dict[str, np.ndarray],
        instruction: Optional[str],
    ) -> List[float]:
        """Predict waypoints and convert to a [vx, vy, wz] velocity command."""
        if self.unified_model is None:
            return [0.0, 0.0, 0.0]

        Vc_dict, Vf_dict = {}, {}
        for v in self.view_list:
            rgb = rgb_frame_dict[v]
            vc, vf = self._encode_frame_tokens(rgb)
            Vc_dict[v] = vc
            Vf_dict[v] = vf

        # Update coarse history
        for v in self.view_list:
            self._coarse_hist_tokens[v].append(Vc_dict[v].cpu())

        # Build coarse token sequence (with left-padding using first frame)
        H = self.history
        coarse_list: List[torch.Tensor] = []
        coarse_tidx: List[torch.Tensor] = []

        lengths = [len(self._coarse_hist_tokens[v]) for v in self.view_list]
        T = min(lengths)
        trim_len = min(H, T)
        missing = H - trim_len

        first_tok: Optional[torch.Tensor] = None
        pending_pad = missing

        for t in range(H):
            if t < missing:
                continue
            tok_per_view = []
            for v in self.view_list:
                tok = self._coarse_hist_tokens[v][t - missing].to(self.device)
                tok_per_view.append(tok)
            tok_views = torch.cat(tok_per_view, dim=0)  # (V*4, C)

            if first_tok is None:
                first_tok = tok_views
                for pt in range(pending_pad):
                    coarse_list.append(first_tok)
                    coarse_tidx.append(
                        torch.full((tok_views.size(0),), pt, device=self.device)
                    )
                pending_pad = 0

            coarse_list.append(tok_views)
            coarse_tidx.append(
                torch.full((tok_views.size(0),), t, device=self.device)
            )

        coarse_tokens = torch.cat(coarse_list, dim=0).unsqueeze(0)
        coarse_tidx_t = torch.cat(coarse_tidx, dim=0).unsqueeze(0)

        # Fine tokens (current frame only)
        fine_tokens = torch.cat(
            [Vf_dict[v] for v in self.view_list], dim=0
        ).to(self.device).unsqueeze(0)
        fine_tidx = torch.full(
            (1, fine_tokens.size(1)),
            fill_value=H,
            dtype=torch.long,
            device=self.device,
        )

        instr = [instruction or "follow the person"]

        target_dtype = next(self.unified_model.parameters()).dtype
        yaw_hist = torch.tensor(
            [VIEW_YAWS[v] for v in self.view_list] * H,
            dtype=target_dtype,
        ).unsqueeze(0)
        yaw_curr = torch.tensor(
            [VIEW_YAWS[v] for v in self.view_list],
            dtype=target_dtype,
        ).unsqueeze(0)

        # NOTE: yaw_hist / yaw_curr 必须用关键字传，否则浮点张量会被当成
        # token ids 进入 embedding 路径，报 "indices must be Long/Int"。
        with torch.inference_mode():
            tau = self.unified_model.forward_navigation(
                coarse_tokens, coarse_tidx_t,
                fine_tokens, fine_tidx,
                instructions=instr,
                yaw_hist=yaw_hist,
                yaw_curr=yaw_curr,
            )  # (1, n_waypoints, 3)

        tau_cpu = tau.detach().float().cpu().numpy()
        self._last_predicted_traj = tau_cpu[0]

        # Use the second waypoint to derive a velocity (matches Habitat agent).
        wp = tau[0, 1] if tau.shape[1] >= 2 else tau[0, 0]
        x = float(wp[0].item())
        y = float(wp[1].item())
        theta = float(wp[2].item()) if wp.numel() >= 3 else 0.0
        vx = x / self.dt
        vy = y / self.dt
        wz = theta / self.dt
        return [vx, vy, wz]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _render_frame_with_traj(
        self,
        rgb_frame_np: np.ndarray,
        traj_xyz: Optional[np.ndarray],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Overlay predicted trajectory and instruction onto a copy of the frame."""
        from PIL import Image, ImageDraw, ImageFont
        import textwrap

        try:
            img = Image.fromarray(rgb_frame_np.astype(np.uint8), mode="RGB")
            draw = ImageDraw.Draw(img)

            # Instruction text (top-left)
            if instruction:
                try:
                    font = ImageFont.truetype(
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
                    )
                except Exception:
                    font = ImageFont.load_default()

                wrapped = textwrap.wrap(instruction, width=50)
                line_height = 20
                total_h = len(wrapped) * line_height
                pad = 5
                max_w = max(
                    [draw.textlength(line, font=font) for line in wrapped] or [0]
                )
                draw.rectangle(
                    [10 - pad, 10 - pad, 10 + max_w + pad, 10 + total_h + pad],
                    fill=(0, 0, 0, 180),
                )
                y_off = 10
                for line in wrapped:
                    draw.text((10, y_off), line, fill=(0, 255, 0), font=font)
                    y_off += line_height

            if (
                traj_xyz is None
                or not isinstance(traj_xyz, np.ndarray)
                or traj_xyz.size == 0
            ):
                return np.asarray(img)

            w, h = img.size
            base_x = w // 2
            base_y = int(h * 0.86)
            scale = 120.0  # px per meter
            pts = []
            npts = min(int(traj_xyz.shape[0]), 64)
            for i in range(npts):
                x = float(traj_xyz[i, 0])
                y = float(traj_xyz[i, 1]) if traj_xyz.shape[1] >= 2 else 0.0
                px = base_x - int(y * scale)
                py = base_y - int(x * scale)
                pts.append((px, py))

            for i in range(1, len(pts)):
                draw.line([pts[i - 1], pts[i]], fill=(0, 0, 0), width=8)
            for i in range(1, len(pts)):
                draw.line([pts[i - 1], pts[i]], fill=(0, 255, 180), width=4)
            if pts:
                r = 4
                sx, sy = pts[0]
                draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(0, 255, 0))
            return np.asarray(img)
        except Exception as e:
            print(f"[OfflineAgent][WARN] rendering failed: {e}")
            return rgb_frame_np
