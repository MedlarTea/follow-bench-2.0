"""KPR (Knowledge Proxy ReID) feature extractor — wraps the local deep-person-reid.

Same interface as ``reid_model.PersonReIDExtractor`` so callers can swap
backbones without code changes:

    extr = KPRReIDExtractor(config_name='kpr_occ_duke_test')
    embeddings = extr(list_of_uint8_HxWxC_patches)   # (N, 512) float32

Patches are resized internally to whatever ``cfg.data.height x cfg.data.width``
the chosen config asks for (default 384x128 for occluded-Duke / 256x128 for
Market). The first part embedding (``bn_foreg`` — the global foreground
representation) is L2-normalised and returned. KPR's other 8 part embeddings
are intentionally discarded — empirically the foreground head alone gives
strong cosine similarity for our target-person classifier and keeps the
downstream ``RidgeReIDClassifier`` 1-to-1 with the basic ResNet path.

Local layout (everything under ``followbench/perception/reid_kpr/``):
    deep_person_reid/
        torchreid/...                  full inference codebase
        configs/kpr/solider/*.yaml     model configs
        pretrained_models/*.pth.tar    10 pretrained KPR variants
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DPR_ROOT = os.path.join(_HERE, "deep_person_reid")
_CONFIG_DIR = os.path.join(_DPR_ROOT, "configs", "kpr", "solider")

# Default config + matching weight file (bundled).
DEFAULT_KPR_CONFIG = "kpr_occ_duke_test"


def list_available_configs() -> List[str]:
    if not os.path.isdir(_CONFIG_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_CONFIG_DIR)
        if f.endswith("_test.yaml")
    )


def _resolve_config_path(name_or_path: str) -> str:
    if os.path.isabs(name_or_path) and os.path.isfile(name_or_path):
        return name_or_path
    name = name_or_path
    if not name.endswith(".yaml"):
        name = name + ".yaml"
    candidate = os.path.join(_CONFIG_DIR, name)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"KPR config '{name_or_path}' not found. Looked in {_CONFIG_DIR}. "
        f"Available: {list_available_configs()}"
    )


def _cuda_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _ = (torch.zeros(1, device="cuda") + 1).item()
        return True
    except Exception:
        return False


class KPRReIDExtractor:
    """KPR-based person ReID feature extractor with the basic-extractor interface."""

    def __init__(
        self,
        config_name: str = DEFAULT_KPR_CONFIG,
        device: str = "auto",
        embedding_index: int = 0,
        verbose: bool = False,
    ) -> None:
        if not os.path.isdir(_DPR_ROOT):
            raise FileNotFoundError(
                f"deep_person_reid not found at {_DPR_ROOT}. Run the install "
                f"steps for the KPR backbone first."
            )

        cfg_path = _resolve_config_path(config_name)

        # torchreid expects to be importable AND its config-relative weight paths
        # require the working directory to be the deep-person-reid root.
        if _DPR_ROOT not in sys.path:
            sys.path.insert(0, _DPR_ROOT)
        original_cwd = os.getcwd()
        try:
            os.chdir(_DPR_ROOT)

            from torchreid.scripts.builder import build_config
            from torchreid.tools.feature_extractor import KPRFeatureExtractor
            from torchreid.utils.tools import extract_test_embeddings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfg = build_config(config_path=cfg_path)
                extractor = KPRFeatureExtractor(cfg, verbose=verbose)
        finally:
            os.chdir(original_cwd)

        # Move to chosen device.
        if device == "auto":
            self.device = "cuda" if _cuda_usable() else "cpu"
        else:
            self.device = device if (device != "cuda" or _cuda_usable()) else "cpu"
        extractor.model.to(self.device).eval()

        self._cfg = cfg
        self._extractor = extractor
        self._extract_test_embeddings = extract_test_embeddings
        self._embedding_index = int(embedding_index)
        self._patch_h = int(cfg.data.height)
        self._patch_w = int(cfg.data.width)
        self._mean = torch.tensor(cfg.data.norm_mean, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(cfg.data.norm_std, device=self.device).view(1, 3, 1, 1)
        self._test_embeddings = list(cfg.model.kpr.test_embeddings)
        self._normalize_feature = bool(cfg.test.normalize_feature)

    @property
    def patch_size(self) -> tuple:
        # (W, H) — matches the basic extractor convention.
        return (self._patch_w, self._patch_h)

    def _resize_to_patch(self, patches: List[np.ndarray]) -> np.ndarray:
        out = np.empty((len(patches), self._patch_h, self._patch_w, 3), dtype=np.uint8)
        for i, p in enumerate(patches):
            if p.shape[0] != self._patch_h or p.shape[1] != self._patch_w:
                out[i] = cv2.resize(
                    p, (self._patch_w, self._patch_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                out[i] = p
        return out

    @torch.no_grad()
    def __call__(self, patches: List[np.ndarray]) -> np.ndarray:
        if not patches:
            return np.zeros((0, 512), dtype=np.float32)
        batch = self._resize_to_patch(patches)
        t = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float()
        t.div_(255.0).sub_(self._mean).div_(self._std)
        outputs = self._extractor.model(images=t)
        embeds, _vis, _parts, _pix = self._extract_test_embeddings(
            outputs, self._test_embeddings
        )
        if self._normalize_feature:
            embeds = torch.nn.functional.normalize(embeds, dim=-1, p=2)
        # (B, num_parts, D) → (B, D) using the foreground head.
        idx = max(0, min(self._embedding_index, embeds.shape[1] - 1))
        feats = embeds[:, idx, :]
        return feats.detach().cpu().numpy().astype(np.float32)
