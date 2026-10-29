#!/usr/bin/env bash
# Smoke config for the multi-GPU random batch eval.
# Mirrors default_50x.sh but with REPEATS=1 so we just sanity-check the
# end-to-end pipeline once per (planner, difficulty, flow_mode).

PLANNERS_WITH_PERCEPTION="dwa_traj sfm bso_hfc rda_search rda rda_traj"
PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="easy hard boss legend"
FLOW_MODES="parallel perpendicular chaotic"

FOLLOW_POSITIONS_MAIN="back left_side"
FOLLOW_POSITIONS_BACK_ONLY="back"

DESIRED_DISTANCE="1.5"
REPEATS="1"

SIM_MODE="sync"
AUTO_SPAWN_SEC="2"
AUTO_FOLLOW_SEC="4"
DURATION_SEC="600"

ENABLE_EVALUATION="1"
RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"

VISUALIZE="1"
RESTART_CARLA_EVERY_N_SLOTS="0"
