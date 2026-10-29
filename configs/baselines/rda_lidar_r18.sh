#!/usr/bin/env bash
# Baseline: MPC + R18 (single smoke episode; corner / back / easy)
PLANNER=rda_lidar FOLLOW_POSITION=back DESIRED_DISTANCE=1.5 DIFFICULTY=easy \
  bash scripts/run_episode.sh --scene corner \
    --enable-evaluation --use-perception --reid-mode basic --kinematics-mode kf \
    --auto-spawn-sec 1 --auto-follow-sec 4 "$@"
