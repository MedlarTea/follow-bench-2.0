#!/usr/bin/env bash
# MPC w/ Traj. + KPR
# fsm_id_switch_thresh=0.05 rerun (perception_pipeline.py default)
# Protocol: back, D=1.5, difficulties easy|hard|boss|legend, 50 trials.
PLANNERS_WITH_PERCEPTION="rda_traj"
PLANNERS_BACK_ONLY=""
DIFFICULTIES="boss"
FOLLOW_POSITIONS_MAIN="back"
FOLLOW_POSITIONS_BACK_ONLY=""

DESIRED_DISTANCE="1.5"
REPEATS="50"
RESTART_CARLA_EVERY_N_SLOTS="0"
SIM_MODE="sync"
AUTO_SPAWN_SEC="1"
AUTO_FOLLOW_SEC="4"
NUM_WALKERS_TOTAL="6"
ENABLE_EVALUATION="1"
RESUME="1"
STOP_ON_FAILURE="0"
WITH_EPISODE_PLOTS="0"
REID_KPR_CONFIG="kpr_occ_duke_test"
REID_DEVICE="auto"
REID_MODE="kpr"
EXTRA_EPISODE_ARGS="--no-ui --rda-traj-disable-lidar-obstacles"
