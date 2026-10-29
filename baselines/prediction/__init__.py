"""Trajectory-history and predictor service wrappers used by planners."""

from importlib import import_module

__all__ = ["TrajectoryHistory", "TrajectoryPredictionService", "get_predictor"]

_EXPORTS = {
    "TrajectoryHistory": ("baselines.prediction.trajectory_buffer", "TrajectoryHistory"),
    "TrajectoryPredictionService": ("baselines.prediction.predictor", "TrajectoryPredictionService"),
}


def __getattr__(name: str):
    if name == "get_predictor":
        from baselines.prediction.backends import get_predictor as backend_get_predictor

        globals()[name] = backend_get_predictor
        return backend_get_predictor
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
