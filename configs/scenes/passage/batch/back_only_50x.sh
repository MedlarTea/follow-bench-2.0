#!/usr/bin/env bash

# Passage batch matrix restricted to the end-to-end / self-contained planners
# (back-following only). Used by scripts/multi_gpu/run_pipeline.sh (--preset full) to evaluate trackvla and
# oa_vat across all four difficulties on the same axis as corner's
# PLANNERS_BACK_ONLY group.
PLANNERS_WITH_PERCEPTION=""
PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="easy hard boss legend"

FOLLOW_POSITIONS_MAIN=""
FOLLOW_POSITIONS_BACK_ONLY="back"

DESIRED_DISTANCE="1.5"
NUM_WALKERS_TOTAL="6"
REPEATS="50"

RESTART_CARLA_EVERY_N_SLOTS="25"
SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

ENABLE_EVALUATION="1"
RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"
