from __future__ import annotations

import numpy as np


class CV:
    def __init__(self):
        super().__init__()
        self.dt = None
        self.prediction_horizon = None
        self.wrap = np.vectorize(self._wrap)

    def set_params(self, params):
        self.dt = params["dt"]
        self.prediction_horizon = params["prediction_horizon"]
        self.history_length = params["history_length"]
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1

    def reset(self):
        pass

    def get_predictions(self, trajectory):
        velocity = trajectory[None, -1, 1:, 2:4]
        init_pos = trajectory[None, -1, 1:, 0:2]

        steps = 1 + np.arange(self.prediction_length, dtype=np.float64)[:, None, None]
        steps = np.multiply(velocity, steps) * self.dt
        steps = (init_pos + steps)[None, None]
        steps = np.concatenate((steps[:, :, :-1], self.predict_velocity(steps)), axis=-1)
        return steps

    def predict_velocity(self, steps):
        return (steps[:, :, 1:] - steps[:, :, :-1]) / self.dt

    def predict(self, trajectory):
        predictions = self.get_predictions(trajectory)
        return predictions

    def predictor_cost(self, state, actions, predictions):
        return np.array([0.0])

    def discrete_cost(self, state, actions, predictions):
        N = actions.shape[0]
        S = predictions.shape[1]
        state_ = np.tile(state[None, None, None, None, 0, :2] - state[None, None, None, 1:, :2], (N, S, 1, 1, 1))
        dxdy = np.concatenate((state_, actions[:, None, :, None, :2] - predictions[:, :, :, :, :2]), axis=2)
        winding_nums = np.arctan2(dxdy[:, :, :, :, 1], dxdy[:, :, :, :, 0])
        winding_nums = winding_nums[:, :, 1:] - winding_nums[:, :, :-1]

        if self.discrete_cost_type == "entropy":
            winding_nums = np.mean(winding_nums, axis=2) < 0
            p = np.mean(winding_nums, axis=1)
            entropy = -(p * np.log(p + 1e-8) + (1 - p) * np.log(1 - p + 1e-8))
            entropy = np.mean(entropy, axis=1)[:, None]
            return self.Q_discrete * (entropy ** 2)

        winding_nums = np.abs(np.mean(winding_nums, axis=2))
        dxdy = state[None, 0, :2] - state[1:, :2]
        obs_theta = np.arctan2(state[1:, 3], state[1:, 2])
        alpha = self.wrap(np.arctan2(dxdy[:, 1], dxdy[:, 0]) - obs_theta + np.pi / 2.0) >= 0
        winding_nums = np.multiply(winding_nums, alpha)
        winding_nums = np.multiply(winding_nums, alpha)
        winding_nums = np.mean(winding_nums, axis=-1)
        return -self.Q_discrete * (winding_nums ** 2)

    @staticmethod
    def _wrap(angle):
        while angle >= np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


__all__ = ["CV"]
