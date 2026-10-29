"""Repo-relative weight-path resolution for the perception stack.

All checkpoints live under ``<repo_root>/data/weights/``; callers should go
through the helpers below rather than hard-coding filenames or paths. That
way:

  - a developer who clones the repo anywhere on disk gets the right paths
    without touching any Python file;
  - we avoid the "same file copied 3 times in 3 cwds" problem that arose
    when ultralytics auto-downloaded YOLO weights into whatever directory
    ``scripts/run_episode.sh`` was launched from.

Usage:

    from followbench.weights import yolo_weight, reid_basic_weight, traj_weight

    yolo_weight()                   # → <repo>/data/weights/yolo/yolo11s.pt
    yolo_weight("yolo11n.pt")       # → <repo>/data/weights/yolo/yolo11n.pt
    reid_basic_weight()             # → <repo>/data/weights/reid_basic/ckpt.t7
    traj_weight("sgan.pt")          # → <repo>/data/weights/traj_predictor/sgan.pt

KPR ReID weights live under ``data/weights/reid_kpr/``. Access to them is
transparent through the symlink
``followbench/perception/reid_kpr/deep_person_reid/pretrained_models``,
so the 12 KPR YAML configs (which reference ``pretrained_models/kpr_*.pth.tar``)
continue to work without modification.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from followbench.paths import REPO_ROOT, WEIGHTS_DIR  # noqa: F401  (re-exported)

# Per-category sub-directories.
YOLO_DIR: Path = WEIGHTS_DIR / "yolo"
REID_BASIC_DIR: Path = WEIGHTS_DIR / "reid_basic"
REID_KPR_DIR: Path = WEIGHTS_DIR / "reid_kpr"
TRAJ_DIR: Path = WEIGHTS_DIR / "traj_predictor"


def yolo_weight(name: str = "yolo11s.pt") -> str:
    """Return the absolute path to a YOLO checkpoint.

    ``name`` may be:
      - a bare filename (e.g. ``"yolo11s.pt"``) — resolved against
        ``data/weights/yolo/``;
      - an absolute path that already exists on disk — returned unchanged.

    If the target file is *not* present yet, the intended absolute path is
    still returned so that ultralytics' auto-download writes the file into
    the central location on first use (instead of scattering copies across
    whichever directory ``scripts/run_episode.sh`` was invoked from).
    """
    if os.path.isabs(name) and os.path.isfile(name):
        return name
    return str(YOLO_DIR / name)


def reid_basic_weight(name: str = "ckpt.t7") -> Optional[str]:
    """Return the absolute path to the basic (ResNet) ReID checkpoint.

    Returns ``None`` if the file is missing so the caller can gracefully
    fall back (the ``PersonReIDExtractor`` already has a candidate chain).
    """
    p = REID_BASIC_DIR / name
    return str(p) if p.is_file() else None


def traj_weight(name: str) -> str:
    """Return the absolute path to a trajectory-predictor checkpoint
    (``sgan.pt``, ``csgan.pt``, ``csgan2.pt``)."""
    return str(TRAJ_DIR / name)


__all__ = [
    "REPO_ROOT", "WEIGHTS_DIR",
    "YOLO_DIR", "REID_BASIC_DIR", "REID_KPR_DIR", "TRAJ_DIR",
    "yolo_weight", "reid_basic_weight", "traj_weight",
]
