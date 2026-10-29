#!/usr/bin/env bash

# Traditional planners using the shared perception frontend.
PLANNERS_WITH_PERCEPTION="dwa_traj sfm bso_hfc rda_search rda rda_traj"

DIFFICULTIES="easy hard boss legend"
FOLLOW_POSITIONS_MAIN="back left_side"

DESIRED_DISTANCE="1.5"
NUM_WALKERS_TOTAL="35"
REPEATS="50"

SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"

# Batch runs are headless by default.
ENABLE_UI="0"
ENABLE_DEBUG_VIS="0"
ENABLE_WORLD_DEBUG="0"
ENABLE_LASER="0"

WEATHER="random"

RESTART_CARLA_EVERY_N_SLOTS="25"

