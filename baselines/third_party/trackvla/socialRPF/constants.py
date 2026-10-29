import math
import os

# Directory holding the socialRPF/ package (baselines/third_party/trackvla/).
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Self-contained data root for ckpts: followbench2.0/data/trackvla/.
# Override with TRACKVLA_DATA_ROOT to point elsewhere (e.g. shared /data mount).
DATA_ROOT = os.environ.get("TRACKVLA_DATA_ROOT") or os.path.normpath(
    os.path.join(ROOT_PATH, "..", "..", "..", "data", "trackvla")
)

# Multi-view yaws (rad). The unified_alpha checkpoint is single-view "forward"
# but the model still indexes this dict during forward_navigation.
VIEW_YAWS = {
    "forward": 0,
    "left":    math.pi / 2,
    "right":   3 * math.pi / 2,
    "back":    2 * math.pi,
}

IMAGE_SIZE = 384

# Unused at inference, kept to satisfy any stray imports.
EVAL_MULTIVIEW_DATA_SOURCE = []
