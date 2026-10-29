"""Sampling helpers for planner-side local search candidate generation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def korobov_lattice(n: int, dim: int = 2, a: Sequence[int] = (1, 7)) -> np.ndarray:
    k = np.arange(1, n + 1)
    return (np.outer(k, np.asarray(a[:dim], dtype=int)) % n) / n


def generate_korobov_semicircle_samples(
    last_point: Sequence[float],
    current_point: Sequence[float],
    inner_radius: float,
    outer_radius: float,
    num_points: int,
) -> np.ndarray:
    if num_points <= 0:
        return np.empty((0, 2), dtype=float)

    lattice = korobov_lattice(num_points, 2)
    radii = np.sqrt(inner_radius * inner_radius + (outer_radius * outer_radius - inner_radius * inner_radius) * lattice[:, 0])
    angles = np.pi * lattice[:, 1]

    last = np.asarray(last_point, dtype=float)
    current = np.asarray(current_point, dtype=float)
    direction = current - last
    direction_angle = np.arctan2(direction[1], direction[0])
    angles += direction_angle - np.pi / 2.0

    xs = radii * np.cos(angles) + current[0]
    ys = radii * np.sin(angles) + current[1]
    return np.column_stack([xs, ys])
