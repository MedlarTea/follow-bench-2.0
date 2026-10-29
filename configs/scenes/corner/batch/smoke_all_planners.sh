#!/usr/bin/env bash

# Multi-GPU smoke for the FULL planner / difficulty matrix at REPEATS=1.
# Use with eval_multi_gpu.sh + --num-workers 8 to verify the 8-way parallel
# fan-out actually exercises every (planner, difficulty) combo in one shot.
#
# Total tasks = (3 perception planners + 2 back-only planners) x 4 difficulties
#             = 20 tasks
# Per-task slots:
#   * perception: FOLLOW_POSITIONS_MAIN (2) x REPEATS (1) = 2 slots
#   * back_only : FOLLOW_POSITIONS_BACK_ONLY (1) x REPEATS (1) = 1 slot
# Total slots = 12*2 + 8*1 = 32 slots across 8 workers.

PLANNERS_WITH_PERCEPTION="rda rda_traj rda_search"
PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="easy hard boss legend"

FOLLOW_POSITIONS_MAIN="back left_side"
FOLLOW_POSITIONS_BACK_ONLY="back"

DESIRED_DISTANCE="1.5"

REPEATS="1"

# inner restart is a no-op under --no-launch-carla; scripts/multi_gpu/run_worker.sh owns CARLA
RESTART_CARLA_EVERY_N_SLOTS="0"
SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

ENABLE_EVALUATION="1"
RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"

# write episode.mp4 / episode_3view.mp4 next to each eval_result.json
VISUALIZE="1"
