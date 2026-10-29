"""Person ReID feature extractor — wraps ``ReIDNet`` and provides batched inference.

Input  : list of HxWx3 BGR (or RGB) image patches (any size)
Output : (N, 512) L2-normalised numpy embeddings (float32)
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from followbench.paths import WEIGHTS_DIR

from .net import ReIDNet

_DEFAULT_PATCH_SIZE = (64, 128)  # (W, H), matches the trained checkpoint
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_default_ckpt() -> Optional[str]:
    """Locate the bundled checkpoint, falling back to the legacy in-repo location."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("FOLLOWBENCH_REID_CKPT", ""),
        # Canonical location (data/weights/reid_basic/ckpt.t7).
        os.path.join(str(WEIGHTS_DIR), "reid_basic", "ckpt.t7"),
        # Legacy in-repo location (kept for users who haven't migrated yet).
        os.path.join(here, "checkpoint", "ckpt.t7"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
        return True
    except Exception:
        return False


class PersonReIDExtractor:
    """Stateless 64×128 patch → 512-d embedding extractor."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        patch_size: Sequence[int] = _DEFAULT_PATCH_SIZE,
    ) -> None:
        ckpt_path = model_path or _resolve_default_ckpt()
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                "Could not locate the ReID checkpoint (ckpt.t7). "
                "Set $FOLLOWBENCH_REID_CKPT or place it at "
                "data/weights/reid_basic/ckpt.t7."
            )

        if device == "auto":
            self.device = "cuda" if _cuda_usable() else "cpu"
        else:
            self.device = device if (device != "cuda" or _cuda_usable()) else "cpu"

        self._patch_w, self._patch_h = int(patch_size[0]), int(patch_size[1])

        self.net = ReIDNet(reid=True)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # torch < 2.4 has no ``weights_only`` kwarg.
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("net_dict", ckpt)
        self.net.load_state_dict(state_dict, strict=False)
        # The pretrained checkpoint has a 751-class head; keep it but never use it
        # (we always read embeddings via ``reid=True``).
        self.net.classifier[-1] = nn.Linear(256, 2)
        self.net.to(self.device).eval()

        self._norm = T.Compose([
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    @property
    def patch_size(self) -> tuple:
        return (self._patch_w, self._patch_h)

    def _preprocess(self, patches: List[np.ndarray]) -> torch.Tensor:
        # ``ToTensor`` expects HxWxC uint8/float in [0,1]; we feed float32.
        tensors = []
        for p in patches:
            if p.shape[0] != self._patch_h or p.shape[1] != self._patch_w:
                # Caller is expected to resize; bail loudly to surface bugs early.
                raise ValueError(
                    f"PersonReIDExtractor expects {self._patch_w}x{self._patch_h} "
                    f"patches; got {p.shape[1]}x{p.shape[0]}."
                )
            t = self._norm(p.astype(np.float32) / 255.0).unsqueeze(0)
            tensors.append(t)
        return torch.cat(tensors, dim=0).float()

    @torch.no_grad()
    def __call__(self, patches: List[np.ndarray]) -> np.ndarray:
        if not patches:
            return np.zeros((0, 512), dtype=np.float32)
        batch = self._preprocess(patches).to(self.device)
        feats = self.net(batch)
        return feats.detach().cpu().numpy().astype(np.float32)
