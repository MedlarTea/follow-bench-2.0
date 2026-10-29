#!/usr/bin/env bash

# Planners using the shared perception wrapper.
PLANNERS_WITH_PERCEPTION="rda rda_traj rda_search dwa_traj sfm bso_hfc"

# End-to-end / self-contained planners that should only run back-following.
PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="easy hard boss legend"

FOLLOW_POSITIONS_MAIN="back left_side"
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

