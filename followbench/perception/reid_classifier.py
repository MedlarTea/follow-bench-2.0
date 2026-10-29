"""Online Ridge classifier for ReID confidence scoring.

Distilled from ``follow/follow_server/src/reid/Classisifer_strong.py``: keeps
two FIFO sample banks (target=positive, others=negative) and refits a Ridge
regression each time the banks change. ``predict`` returns a scalar confidence
(>0 ≈ target-like, <0 ≈ distractor).

We deliberately drop the dual short/long-term cache from the original — the
shorter pipeline is plenty for our 10 Hz CARLA loop and removes an entire
class of book-keeping bugs.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

try:
    from sklearn.linear_model import Ridge
    _HAS_SKLEARN = True
except ImportError:
    Ridge = None  # type: ignore[assignment]
    _HAS_SKLEARN = False


class RidgeReIDClassifier:
    def __init__(
        self,
        alpha: float = 1.0,
        max_pos_samples: int = 64,
        max_neg_samples: int = 128,
        min_train_pairs: int = 4,
    ) -> None:
        if not _HAS_SKLEARN:
            raise RuntimeError(
                "RidgeReIDClassifier requires scikit-learn (already in environment.multiRobots.yml)."
            )
        self._alpha = float(alpha)
        self._max_pos = int(max_pos_samples)
        self._max_neg = int(max_neg_samples)
        self._min_train_pairs = int(min_train_pairs)

        self._pos: List[np.ndarray] = []
        self._neg: List[np.ndarray] = []
        self._clf: Optional[Ridge] = None
        self._dirty: bool = False

    def reset(self) -> None:
        self._pos.clear()
        self._neg.clear()
        self._clf = None
        self._dirty = False

    @property
    def trained(self) -> bool:
        return self._clf is not None

    @property
    def num_positive(self) -> int:
        return len(self._pos)

    @property
    def num_negative(self) -> int:
        return len(self._neg)

    def add_positive(self, features: Iterable[np.ndarray]) -> None:
        for f in features:
            self._pos.append(np.asarray(f, dtype=np.float32).flatten())
        if len(self._pos) > self._max_pos:
            self._pos = self._pos[-self._max_pos:]
        self._dirty = True

    def add_negative(self, features: Iterable[np.ndarray]) -> None:
        for f in features:
            self._neg.append(np.asarray(f, dtype=np.float32).flatten())
        if len(self._neg) > self._max_neg:
            self._neg = self._neg[-self._max_neg:]
        self._dirty = True

    def maybe_refit(self) -> bool:
        if not self._dirty:
            return self._clf is not None
        if len(self._pos) < self._min_train_pairs or len(self._neg) < self._min_train_pairs:
            self._dirty = False
            return self._clf is not None
        X = np.vstack(self._pos + self._neg)
        y = np.concatenate([
            np.ones(len(self._pos), dtype=np.float32),
            np.zeros(len(self._neg), dtype=np.float32),
        ])
        self._clf = Ridge(alpha=self._alpha, random_state=1)
        self._clf.fit(X, y)
        self._dirty = False
        return True

    def predict(self, feature: np.ndarray) -> float:
        if self._clf is None:
            return 0.0
        return float(self._clf.predict(np.asarray(feature, dtype=np.float32).reshape(1, -1))[0])

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if self._clf is None or features.size == 0:
            return np.zeros((features.shape[0],), dtype=np.float32)
        return self._clf.predict(features.astype(np.float32)).astype(np.float32)

    def cosine_to_positive_mean(self, feature: np.ndarray) -> float:
        """Fallback similarity used before the classifier has any training data."""
        if not self._pos:
            return 0.0
        f = np.asarray(feature, dtype=np.float32).flatten()
        mean = np.mean(np.stack(self._pos, axis=0), axis=0)
        denom = float(np.linalg.norm(f) * np.linalg.norm(mean))
        if denom <= 1e-9:
            return 0.0
        return float(np.dot(f, mean) / denom)
