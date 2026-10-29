"""Predictor backend registry for planner-side trajectory prediction."""

from importlib import import_module

__all__ = ["CV", "CVKF", "SGAN", "get_predictor"]

_EXPORTS = {
    "CV": ("baselines.prediction.backends.cv", "CV"),
    "CVKF": ("baselines.prediction.backends.cvkf", "CVKF"),
    "SGAN": ("baselines.prediction.backends.sgan", "SGAN"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def get_predictor(predictor_type, params):
    if predictor_type == "cv":
        predictor_cls = __getattr__("CV")
    elif predictor_type == "cvkf":
        predictor_cls = __getattr__("CVKF")
    elif predictor_type == "sgan":
        predictor_cls = __getattr__("SGAN")
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    predictor = predictor_cls()
    predictor.set_params(params)
    return predictor
