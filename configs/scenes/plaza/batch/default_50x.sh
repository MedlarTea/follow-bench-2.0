#!/usr/bin/env bash

# Planners using the shared perception wrapper.
PLANNERS_WITH_PERCEPTION="rda_search rda rda_traj"

# End-to-end / self-contained planners that should only run back-following.
# PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="easy hard boss legend"
FLOW_MODES="parallel perpendicular chaotic"

FOLLOW_POSITIONS_MAIN="left_side"
# FOLLOW_POSITIONS_BACK_ONLY="back"

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
