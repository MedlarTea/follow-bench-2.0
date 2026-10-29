#!/usr/bin/env bash

# 单 planner left-side 补跑 config:
#   - rda_traj only, with --use-perception (YOLO + ReID + depth tracker)
#   - follow_position = left_side(robot 在 target 视觉左侧)
#   - flow_mode = random(runtime 每 episode 抽 parallel/perpendicular/chaotic)
#   - 4 difficulties × 50 repeats = 200 ep total

PLANNERS_WITH_PERCEPTION="rda_search"
PLANNERS_BACK_ONLY=""

DIFFICULTIES="easy hard boss legend"
FLOW_MODES="random"

FOLLOW_POSITIONS_MAIN="left_side"
FOLLOW_POSITIONS_BACK_ONLY=""

DESIRED_DISTANCE="1.5"
REPEATS="50"

SIM_MODE="sync"
AUTO_SPAWN_SEC="2"
AUTO_FOLLOW_SEC="4"
DURATION_SEC="600"

ENABLE_EVALUATION="1"
RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"
