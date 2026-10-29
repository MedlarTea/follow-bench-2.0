"""Named predictor presets used by planner-side trajectory prediction wrappers."""

from __future__ import annotations


PREDICTOR_CONFIGS = {
    "cv": {
        "name": "cv",
        "dt": 0.1,
        "history_length": 8,
        "prediction_horizon": 2.0,
    },
    "cvkf": {
        "name": "cvkf",
        "dt": 0.1,
        "history_length": 1,
        "prediction_horizon": 2.0,
        "predictor": {
            "num_samples": 20,
        },
    },
    "sgan": {
        "name": "sgan",
        "dt": 0.1,
        "history_length": 8,
        "prediction_horizon": 2.0,
        "predictor": {
            "path": "sgan.pt",
            "use_gpu": False,
            "num_samples": 1,
            "deviation_penalty": True,
            "use_sgan_action": False,
            "use_sgan_mode": True,
        },
    },
}
