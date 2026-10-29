from __future__ import annotations

from followbench.paths import EVAL_LOGS_DIR, RESULTS_DIR

DEFAULT_LOG_ROOT = str(EVAL_LOGS_DIR)
DEFAULT_RESULTS_DIR = str(RESULTS_DIR)

METHOD_LABELS = {
    "pid": "PID",
    "dwa_traj": "DWA-Traj",
    "dwa_traj_depth_tpt": "DWA-Traj-DTPT",
    "sfm": "SFM",
    "rda": "RDA",
    "rda_lidar": "RDA-LiDAR",
    "rda_traj": "RDA-Traj",
    "rda_search": "RDA-Search",
    "rda_depth_tpt": "RDA-DTPT",
    "bso_hfc": "BSO-HFC",
    "teleop": "Human Teleop",
}

COLORS = {
    "pid": "#4c78a8",
    "dwa_traj": "#f58518",
    "dwa_traj_depth_tpt": "#ffbf79",
    "sfm": "#54a24b",
    "rda": "#e45756",
    "rda_lidar": "#b279a2",
    "rda_traj": "#9d755d",
    "rda_search": "#72b7b2",
    "rda_depth_tpt": "#d37295",
    "bso_hfc": "#13a8a8",
    "teleop": "#7f7f7f",
}

LINE_STYLES = {
    "back": "-",
    "left_side": "--",
    "right_side": ":",
}

PLOTTED_METRICS = {
    "SR": {"unit": "%", "kind": "percent"},
    "ASR": {"unit": "%", "kind": "percent"},
    "TVR": {"unit": "%", "kind": "percent"},
    "TinTPerson": {"unit": "%", "kind": "percent"},
    "TinPrivate": {"unit": "%", "kind": "percent"},
    "PL": {"unit": "m", "kind": "scalar"},
    "AvgVel": {"unit": "m/s", "kind": "scalar"},
    "AvgAcc": {"unit": "m/s^2", "kind": "scalar"},
    "Jerk": {"unit": "m/s^3", "kind": "scalar"},
    "CostMean": {"unit": "ms", "kind": "scalar"},
    "CostP95": {"unit": "ms", "kind": "scalar"},
    "PolicyMean": {"unit": "ms", "kind": "scalar"},
    "PolicyP95": {"unit": "ms", "kind": "scalar"},
    "PlanMean": {"unit": "ms", "kind": "scalar"},
    "PlanP95": {"unit": "ms", "kind": "scalar"},
    "PercMean": {"unit": "ms", "kind": "scalar"},
    "PercP95": {"unit": "ms", "kind": "scalar"},
    "DetMean": {"unit": "ms", "kind": "scalar"},
    "TrackMean": {"unit": "ms", "kind": "scalar"},
    "ReIDMean": {"unit": "ms", "kind": "scalar"},
    "MapMean": {"unit": "ms", "kind": "scalar"},
    "FSMMean": {"unit": "ms", "kind": "scalar"},
}

DEFAULT_PLOT_METRICS = ["SR", "ASR", "TVR", "TinTPerson", "TinPrivate", "Jerk"]
DEFAULT_TABLE_METRICS = [
    "SR",
    "ASR",
    "TVR",
    "TinTPerson",
    "TinPrivate",
    "PL",
    "AvgVel",
    "AvgAcc",
    "Jerk",
    "PolicyMean",
    "PolicyP95",
    "PlanMean",
    "PlanP95",
    "PercMean",
    "PercP95",
]
