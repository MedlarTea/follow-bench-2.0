#!/usr/bin/env bash

# Small smoke-test matrix for the Corner batch runner.
PLANNERS_WITH_PERCEPTION="rda_traj rda rda_search"
PLANNERS_BACK_ONLY="trackvla oa_vat"

DIFFICULTIES="legend"

FOLLOW_POSITIONS_MAIN="left_side back"
FOLLOW_POSITIONS_BACK_ONLY="back"

DESIRED_DISTANCE="1.5"

REPEATS="2"

SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"

ENABLE_EVALUATION="1"
RESUME="0"
STOP_ON_FAILURE="1"
WITH_EPISODE_PLOTS="0"

# Record an mp4 of the robot's first-person view(s) into each trial dir.
# Back-follow -> episode.mp4 (single front cam). Side-follow -> episode_3view.mp4
# (Left | Front | Right stitched). Bbox overlay shows up automatically when a
# perception planner is active.
VISUALIZE="1"
