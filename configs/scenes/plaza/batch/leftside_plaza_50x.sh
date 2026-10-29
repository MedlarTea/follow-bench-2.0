#!/usr/bin/env bash

# left-side follow补跑专用 config:
#   - 仅 perception planner (rda / rda_traj / rda_search) 跑 left_side
#   - flow_mode 用 "random" 让 runtime 每 episode 抽一种(parallel / perpendicular / chaotic)
#     这样每 (planner × difficulty) 只跑 50 个,而不是 3 flow × 50 = 150 个
#   - trackvla / oa_vat 不跑 left_side(spec 只让它们跑 back),所以 BACK_ONLY 留空避免重复

PLANNERS_WITH_PERCEPTION="rda rda_traj rda_search"
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
