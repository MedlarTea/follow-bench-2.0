#!/usr/bin/env bash

# 24 runs: 6 planners * 2 difficulties * 2 follow positions * 1 repeat.
PLANNERS_WITH_PERCEPTION="dwa_traj sfm bso_hfc rda_search rda rda_traj"

DIFFICULTIES="easy hard"
FOLLOW_POSITIONS_MAIN="back left_side"

DESIRED_DISTANCE="1.5"
NUM_WALKERS_TOTAL="35"
REPEATS="1"

SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

RESUME="0"
STOP_ON_FAILURE="1"
WITH_EPISODE_PLOTS="0"

ENABLE_UI="0"
ENABLE_DEBUG_VIS="0"
ENABLE_WORLD_DEBUG="0"
ENABLE_LASER="0"

WEATHER="random"

RESTART_CARLA_EVERY_N_SLOTS="25"
