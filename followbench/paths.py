"""Canonical filesystem locations for FollowBench 2.0.

Every path the benchmark needs is derived from the repository root here, so
modules never have to count ``..`` segments relative to their own file.

Layout::

    <repo>/
      assets/scenes/<scene>/   static scene assets (gridmaps, routes, flow points)
      data/                    external artifacts (weights, checkpoints) - not in git
      outputs/                 everything a run produces - not in git
                               (override the location with $FB_OUTPUT_ROOT)
        runs/<scene>/          per-episode run directories
        eval_logs/             default evaluation log root
        logs_batch/<scene>/    batch evaluation logs
        batch_manifests/<scene>/
        results/               aggregated tables / plots
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_DIR.parent

ASSETS_DIR: Path = REPO_ROOT / "assets"
SCENE_ASSETS_DIR: Path = ASSETS_DIR / "scenes"

DATA_DIR: Path = REPO_ROOT / "data"
WEIGHTS_DIR: Path = DATA_DIR / "weights"

OUTPUTS_DIR: Path = Path(os.environ.get("FB_OUTPUT_ROOT") or (REPO_ROOT / "outputs")).resolve()
RUNS_DIR: Path = OUTPUTS_DIR / "runs"
EVAL_LOGS_DIR: Path = OUTPUTS_DIR / "eval_logs"
BATCH_LOGS_DIR: Path = OUTPUTS_DIR / "logs_batch"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"

SCENES = ("passage", "corner", "park", "plaza")


def scene_assets_dir(scene: str) -> Path:
    """Directory holding the static assets of ``scene``."""
    return SCENE_ASSETS_DIR / scene


def scene_asset(scene: str, name: str) -> str:
    """Absolute path (as ``str``) of asset ``name`` for ``scene``."""
    return str(scene_assets_dir(scene) / name)


def scene_runs_dir(scene: str) -> str:
    """Default ``--output-dir`` for single episodes of ``scene``."""
    return str(RUNS_DIR / scene)


def module_path(*parts: str) -> str:
    """Absolute path of a file inside the ``followbench`` package."""
    return str(PACKAGE_DIR.joinpath(*parts))
