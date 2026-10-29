"""Based on https://github.com/fluentrobotics/ComplexityNav"""

from __future__ import annotations

import copy

import numpy as np
from filterpy.kalman import KalmanFilter

from baselines.prediction.backends.cv import CV


class CVKF(CV):
    def __init__(self):
        super().__init__()
        self.dt = None
        self.prediction_horizon = None
        self.filters = None
        self.wrap = np.vectorize(self._wrap)

    def set_params(self, params):
        self.dt = params["dt"]
        self.prediction_horizon = params["prediction_horizon"]
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = params["history_length"]
        self.num_samples = params["predictor"]["num_samples"]

    def get_kf(self):
        kf = KalmanFilter(dim_x=4, dim_z=4)
        kf.x = np.zeros(4)
        kf.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        kf.H = np.eye(4)

        pos_sigma0, vel_sigma0 = 0.1, 0.1
        kf.P = np.diag([pos_sigma0**2, pos_sigma0**2, vel_sigma0**2, vel_sigma0**2])

        pos_sigma_meas, vel_sigma_meas = 0.5, 0.5
        kf.R = np.diag([pos_sigma_meas**2, pos_sigma_meas**2, vel_sigma_meas**2, vel_sigma_meas**2])

        process_sigma = 0.05
        kf.Q = np.diag([process_sigma**2, process_sigma**2, process_sigma**2, process_sigma**2])
        return kf

    def reset(self):
        self.filters = None

    def unroll_kf(self, kf, s):
        kf_ = copy.deepcopy(kf)
        trajectory = []
        for _ in range(self.rollout_steps):
            kf_.predict()
            trajectory.append(kf_.x)
        trajectory = np.stack(trajectory)
        return trajectory

    def get_predictions(self, trajectory):
        if self.filters is None:
            self.filters = [self.get_kf() for _ in range(trajectory.shape[1] - 1)]
            for kf, s in zip(self.filters, trajectory[-1, 1:, :4]):
                kf.x = s
        predictions = []
        for kf, s in zip(self.filters, trajectory[-1, 1:, :4]):
            kf.predict()
            kf.update(s)

            nis = kf.y.T @ np.linalg.inv(kf.S) @ kf.y
            nis = nis.repeat(self.rollout_steps)[:, np.newaxis]

            predicted_state = self.unroll_kf(kf, s)
            predicted_state = np.hstack([predicted_state, nis])
            predictions.append(predicted_state)

        predictions = np.stack(predictions, axis=1)[np.newaxis, np.newaxis, :]
        return predictions

    def predict(self, trajectory):
        predictions = self.get_predictions(trajectory)
        return predictions


__all__ = ["CVKF"]
