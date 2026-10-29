"""Per-track kinematics estimators for the perception stack.

Two implementations with the same interface
``update(track_id, x, y) → (vx, vy, yaw_rad, speed)``
and ``reset() / prune(active_ids)``:

``TrackKinematicsEstimator``
    Frame-to-frame finite differences + EMA smoothing.  Simple, no extra deps,
    but inherits position noise from the depth camera directly into velocity.
    Suitable when depth noise is small or downstream consumers tolerate jitter.

``TrackKinematicsKF``
    Constant-velocity Kalman filter (filterpy).  Measures *only position* from
    the depth tracker and lets the KF estimate velocity from the state dynamics.
    The explicit separation of measurement noise R and process noise Q lets you
    tune how much the filter trusts the noisy depth positions versus its own
    CV-model prediction.  This suppresses the high-frequency velocity jitter
    that causes downstream planners (e.g. rda_traj) to oscillate.

    Key insight compared to cvkf.py (which measures [x, y, vx, vy]):
    here H is 2×4 (position-only), so vx/vy are never directly observed and
    are estimated purely from the KF's dynamics.  Velocity is therefore the
    *derivative of the smoothed position trajectory*, not a noisy frame-diff.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    from filterpy.kalman import KalmanFilter as _FilterPyKF
    _HAS_FILTERPY = True
except ImportError:
    _FilterPyKF = None   # type: ignore[assignment]
    _HAS_FILTERPY = False


# ── Shared protocol ───────────────────────────────────────────────────────────

class TrackKinematicsEstimator:
    """Finite-difference + EMA per-track velocity estimator (original)."""

    def __init__(
        self,
        dt: float,
        ema_alpha: float = 0.1,
        min_speed_for_yaw: float = 0.15,
    ) -> None:
        self.dt = float(dt)
        self.ema_alpha = float(ema_alpha)
        self.min_speed_for_yaw = float(min_speed_for_yaw)
        self._prev_xy: Dict[int, Tuple[float, float]] = {}
        # tid -> {"vx", "vy", "yaw_rad", "speed"}
        self._smooth: Dict[int, Dict[str, float]] = {}

    def reset(self) -> None:
        self._prev_xy.clear()
        self._smooth.clear()

    def update(self, track_id: int, x: float, y: float) -> Tuple[float, float, float, float]:
        """Returns (vx, vy, yaw_rad, speed) for ``track_id``.

        First observation of an id returns zeros; yaw is held when speed dips
        below ``min_speed_for_yaw`` so a momentarily stationary ped keeps a
        sane heading.
        """
        prev = self._prev_xy.get(track_id)
        if prev is None:
            self._prev_xy[track_id] = (float(x), float(y))
            self._smooth[track_id] = {"vx": 0.0, "vy": 0.0, "yaw_rad": 0.0, "speed": 0.0}
            return 0.0, 0.0, 0.0, 0.0

        px, py = prev
        raw_vx = (float(x) - px) / self.dt
        raw_vy = (float(y) - py) / self.dt

        s = self._smooth.setdefault(
            track_id, {"vx": 0.0, "vy": 0.0, "yaw_rad": 0.0, "speed": 0.0}
        )
        a = self.ema_alpha
        s["vx"] = a * raw_vx + (1.0 - a) * s["vx"]
        s["vy"] = a * raw_vy + (1.0 - a) * s["vy"]
        s["speed"] = math.hypot(s["vx"], s["vy"])
        if s["speed"] >= self.min_speed_for_yaw:
            s["yaw_rad"] = math.atan2(s["vy"], s["vx"])
        # else: keep last yaw_rad

        self._prev_xy[track_id] = (float(x), float(y))
        return s["vx"], s["vy"], s["yaw_rad"], s["speed"]

    def prune(self, active_ids: Iterable[int]) -> None:
        keep = {int(i) for i in active_ids}
        for k in list(self._prev_xy.keys()):
            if int(k) not in keep:
                self._prev_xy.pop(k, None)
                self._smooth.pop(k, None)


# ── Kalman-filter implementation ──────────────────────────────────────────────

class TrackKinematicsKF:
    """Per-track constant-velocity Kalman filter (position-only measurement).

    State:    x_kf = [x, y, vx, vy]
    Measurement: z = [x, y]   (position from the depth tracker)

    Motion model (F, constant-velocity):
        x[t] = x[t-1] + vx[t-1]*dt
        y[t] = y[t-1] + vy[t-1]*dt
        vx[t] = vx[t-1]
        vy[t] = vy[t-1]

    Noise tuning:
        R  — measurement noise covariance.  Set pos_sigma to the typical
             one-sigma depth-reprojection error in metres.  Larger values
             trust the tracker less and rely more on the CV prediction,
             producing smoother (but more latent) velocity estimates.
        Q  — process noise covariance.  pos_sigma_q captures unmodelled
             position jitter; vel_sigma_q captures pedestrian acceleration
             (how quickly the CV assumption is violated).

    Because vx/vy are never directly measured (H is position-only), the
    filter derives velocity from the *Kalman-smoothed* position sequence
    rather than raw frame-differences.  This removes the dt-division
    amplification of position noise that plagues the EMA estimator.
    """

    def __init__(
        self,
        dt: float,
        pos_sigma: float = 0.20,        # measurement noise (m); tune to depth reprojection error
        pos_sigma_q: float = 0.01,      # process noise on position (m)
        vel_sigma_q: float = 0.05,      # process noise on velocity (m/s); ~pedestrian accel per tick
        min_speed_for_yaw: float = 0.15,
    ) -> None:
        if not _HAS_FILTERPY:
            raise ImportError(
                "TrackKinematicsKF requires 'filterpy'. "
                "Install with: pip install filterpy"
            )
        self.dt = float(dt)
        self.pos_sigma = float(pos_sigma)
        self.pos_sigma_q = float(pos_sigma_q)
        self.vel_sigma_q = float(vel_sigma_q)
        self.min_speed_for_yaw = float(min_speed_for_yaw)

        # Per-track state: {track_id: KalmanFilter}
        self._filters: Dict[int, object] = {}
        # Last known yaw (held when speed < threshold).
        self._last_yaw: Dict[int, float] = {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_filter(self, x0: float, y0: float) -> object:
        """Create and initialise a fresh KF for a new track."""
        kf = _FilterPyKF(dim_x=4, dim_z=2)

        # State transition: constant-velocity model.
        kf.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0,    self.dt],
            [0.0, 0.0, 1.0,    0.0],
            [0.0, 0.0, 0.0,    1.0],
        ])

        # Observation: measure position only.
        kf.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Measurement noise (position uncertainty from depth reprojection).
        r = self.pos_sigma ** 2
        kf.R = np.diag([r, r])

        # Process noise: how much we expect state to deviate from CV motion.
        qp = self.pos_sigma_q ** 2
        qv = self.vel_sigma_q ** 2
        kf.Q = np.diag([qp, qp, qv, qv])

        # Initial state covariance: be uncertain about velocity at birth.
        kf.P = np.diag([r, r, (self.vel_sigma_q * 5) ** 2, (self.vel_sigma_q * 5) ** 2])

        # Initialise state: position known, velocity zero.
        kf.x = np.array([x0, y0, 0.0, 0.0])
        return kf

    # ── Public interface ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._filters.clear()
        self._last_yaw.clear()

    def update(self, track_id: int, x: float, y: float) -> Tuple[float, float, float, float]:
        """Predict-update cycle for one track; returns (vx, vy, yaw_rad, speed)."""
        tid = int(track_id)
        if tid not in self._filters:
            # First observation: birth the filter.
            self._filters[tid] = self._make_filter(float(x), float(y))
            self._last_yaw[tid] = 0.0
            return 0.0, 0.0, 0.0, 0.0

        kf = self._filters[tid]
        kf.predict()                           # type: ignore[union-attr]
        kf.update(np.array([float(x), float(y)]))  # type: ignore[union-attr]

        vx = float(kf.x[2])                   # type: ignore[index]
        vy = float(kf.x[3])                   # type: ignore[index]
        speed = math.hypot(vx, vy)

        if speed >= self.min_speed_for_yaw:
            yaw_rad = math.atan2(vy, vx)
            self._last_yaw[tid] = yaw_rad
        else:
            yaw_rad = self._last_yaw.get(tid, 0.0)

        return vx, vy, yaw_rad, speed

    def prune(self, active_ids: Iterable[int]) -> None:
        keep = {int(i) for i in active_ids}
        for k in list(self._filters.keys()):
            if k not in keep:
                self._filters.pop(k, None)
                self._last_yaw.pop(k, None)
