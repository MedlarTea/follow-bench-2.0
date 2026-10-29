"""Based on https://github.com/fluentrobotics/ComplexityNav"""

from __future__ import annotations

import os

import numpy as np
import torch

from baselines.prediction.backends.cv import CV
from followbench.paths import WEIGHTS_DIR
from baselines.prediction.utils.optimized_sgan import OptimizedSGAN
from baselines.prediction.utils.sgan_utils import get_generator, optimized_relative_to_abs


class SGAN(CV):
    def __init__(self):
        super().__init__()
        self.model = OptimizedSGAN

    def set_params(self, params):
        self.dt = params["dt"]
        self.prediction_horizon = params["prediction_horizon"]
        self.rollout_steps = int(np.ceil(self.prediction_horizon / self.dt))
        self.prediction_length = int(np.ceil(self.prediction_horizon / self.dt)) + 1
        self.history_length = params["history_length"]

        self.q_goal_norm = np.square(2 / float(self.prediction_horizon))
        self.iskip = 4
        self.oskip = int(np.ceil(0.4 / self.dt))
        self.model_obs_len = self.history_length
        self.model_pred_len = int(np.ceil(self.prediction_horizon / 0.4)) + 1

        self.device = torch.device("cuda" if params["predictor"]["use_gpu"] else "cpu")
        self.num_samples = params["predictor"]["num_samples"]

        # Resolve the predictor checkpoint from the central data/weights/ store.
        checkpoint_dir = os.path.join(str(WEIGHTS_DIR), "traj_predictor")
        path = os.path.join(checkpoint_dir, params["predictor"]["path"])
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint["args"]["pred_len"] = self.model_pred_len
        checkpoint["args"]["obs_len"] = self.model_obs_len
        checkpoint["args"]["batch_size"] = 1
        self.generator = get_generator(self.model, checkpoint, self.device)

        self.deviation_penalty = params["predictor"]["deviation_penalty"]
        if params["predictor"]["use_sgan_action"]:
            self.predict = self._predict

        self.use_mode = params["predictor"]["use_sgan_mode"]

    def create_input(self, trajectory):
        indexes = torch.flip(torch.arange(len(trajectory) - 1, -1, -self.iskip, device=self.device)[: self.history_length + 1], dims=(0,))
        trajectory = trajectory[indexes, :, :2]
        if len(trajectory) <= 1:
            trajectory = torch.nn.functional.pad(trajectory, (0, 0, 0, 0, 1, 0), mode="constant")
        obs_traj = trajectory[1:]
        obs_traj_rel = obs_traj - trajectory[0:-1]
        return obs_traj, obs_traj_rel

    def create_output(self, traj, init_pos):
        N = traj.shape[0]
        S = traj.shape[1]
        init_pos = init_pos[None, None, None, :, :2].expand(N, S, -1, -1, -1)
        traj = torch.cat((init_pos, traj), axis=2)

        new_traj = []
        for t in range(traj.shape[2] - 1):
            for i in range(self.oskip):
                new_traj.append((traj[:, :, t] * (self.oskip - i) + traj[:, :, t + 1] * i) / self.oskip)
        new_traj.append(traj[:, :, -1])
        new_traj = torch.stack(new_traj, axis=2)
        return new_traj[:, :, 1 : self.prediction_length + 1]

    def get_predictions(self, trajectory):
        with torch.no_grad():
            trajectory = torch.tensor(trajectory, device=self.device, dtype=torch.float)
            obs_traj, obs_traj_rel = self.create_input(trajectory)
            seq_start_end = torch.tensor([[0, obs_traj.shape[1]]], dtype=torch.int, device=self.device)
            noise = None
            if self.use_mode:
                noise = torch.ones((obs_traj.shape[0],) + self.generator.noise_dim, device=self.device)
            pred_traj_fake_rel = self.generator(obs_traj, obs_traj_rel, seq_start_end, self.num_samples, noise)
            pred_traj_fake = optimized_relative_to_abs(pred_traj_fake_rel, obs_traj[-1])[None]
            pred_traj_fake = self.create_output(pred_traj_fake[:, :, :], trajectory[-1])

            pred_traj_fake = torch.cat((pred_traj_fake[:, :, :-1], self.predict_velocity(pred_traj_fake)), dim=-1)

            self.ego_traj_fake = pred_traj_fake[:, :, :, 0].cpu().numpy()
            pred_traj_fake = pred_traj_fake[:, :, :, 1:]
        return pred_traj_fake.cpu().numpy()

    def predictor_cost(self, state, actions, predictions):
        cost = np.array([0.0])
        if self.deviation_penalty:
            cost = np.mean(np.linalg.norm(actions[:, None, :, :2] - self.ego_traj_fake[:, :, :, :2], axis=-1), axis=-1)
        return self.Q_dev * (cost**2)

    def predict(self, trajectory):
        predictions = self.get_predictions(trajectory)[0, :, None]
        return predictions


__all__ = ["SGAN"]
