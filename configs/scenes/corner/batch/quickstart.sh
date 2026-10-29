#!/usr/bin/env bash
# Quick-start batch: needs only the `core` install (CARLA + data/weights + YOLO11).
# MPC w/ Traj. + R18, corner, easy, back, 2 trials of at most 45 s active time.
# Takes about 3 minutes on one GPU. Used by README "Quick start" step 4.
PLANNERS_WITH_PERCEPTION="rda_traj"
PLANNERS_BACK_ONLY=""
DIFFICULTIES="easy"
FOLLOW_POSITIONS_MAIN="back"
FOLLOW_POSITIONS_BACK_ONLY=""

DESIRED_DISTANCE="1.5"
REPEATS="2"
SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"
ENABLE_EVALUATION="1"
RESUME="1"
STOP_ON_FAILURE="1"
WITH_EPISODE_PLOTS="0"
REID_MODE="basic"
EXTRA_EPISODE_ARGS="--no-ui --rda-traj-disable-lidar-obstacles --max-active-sec 45 --duration-sec 120"
