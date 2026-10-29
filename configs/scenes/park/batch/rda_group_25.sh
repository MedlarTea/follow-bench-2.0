#!/usr/bin/env bash

# Batch config: run RDA-family planners and restart CARLA every 25 evaluated slots
PLANNERS_WITH_PERCEPTION="rda rda_traj rda_search"

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

# Headless by default
ENABLE_UI="0"
ENABLE_DEBUG_VIS="0"
ENABLE_WORLD_DEBUG="0"
ENABLE_LASER="0"

WEATHER="random"

# Proactively restart CARLA to avoid memory/RPC issues
RESTART_CARLA_EVERY_N_SLOTS="25"
