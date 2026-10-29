#!/usr/bin/env bash
# Smoke config for the multi-GPU park-traditional batch eval.
# Mirrors default_50x.sh but with REPEATS=1.

PLANNERS_WITH_PERCEPTION="dwa_traj sfm bso_hfc rda_search rda rda_traj"

DIFFICULTIES="easy hard boss legend"
FOLLOW_POSITIONS_MAIN="back left_side"

DESIRED_DISTANCE="1.5"
NUM_WALKERS_TOTAL="35"
REPEATS="1"

SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"

ENABLE_UI="0"
ENABLE_DEBUG_VIS="0"
ENABLE_WORLD_DEBUG="0"
ENABLE_LASER="0"

WEATHER="random"

VISUALIZE="0"
RESTART_CARLA_EVERY_N_SLOTS="0"
