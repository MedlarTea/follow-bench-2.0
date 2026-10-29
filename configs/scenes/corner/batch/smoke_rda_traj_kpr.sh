#!/usr/bin/env bash

# Smoke for new task splitter: should produce 1 x 4 x 2 = 8 tasks
# (rda_traj x {easy,hard,boss,legend} x {back,left_side}).

PLANNERS_WITH_PERCEPTION="rda_traj"
PLANNERS_BACK_ONLY=""

DIFFICULTIES="easy hard boss legend"

FOLLOW_POSITIONS_MAIN="back left_side"
FOLLOW_POSITIONS_BACK_ONLY=""

DESIRED_DISTANCE="1.5"
REPEATS="1"

RESTART_CARLA_EVERY_N_SLOTS="0"
SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

ENABLE_EVALUATION="1"
RESUME="0"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"

REID_MODE="kpr"
REID_KPR_CONFIG="kpr_occ_duke_test"
REID_DEVICE="auto"
