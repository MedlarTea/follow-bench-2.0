#!/usr/bin/env bash

# Random-flow batch matrix restricted to the end-to-end / self-contained
# planners (back-following only). Used by scripts/multi_gpu/run_pipeline.sh (--preset full)
# to evaluate
# trackvla and oa_vat across all difficulties × flow modes.
PLANNERS_WITH_PERCEPTION=""
PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="easy hard boss legend"
FLOW_MODES="parallel perpendicular chaotic"

FOLLOW_POSITIONS_MAIN=""
FOLLOW_POSITIONS_BACK_ONLY="back"

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
