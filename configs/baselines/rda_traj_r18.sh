#!/usr/bin/env bash
# Baseline: MPC w/ Traj. + R18 (single smoke episode; corner / back / easy)
# Disable lidar obstacles to match the published Traj. rows.
PLANNER=rda_traj FOLLOW_POSITION=back DESIRED_DISTANCE=1.5 DIFFICULTY=easy \
  bash scripts/run_episode.sh --scene corner \
    --enable-evaluation --use-perception --reid-mode basic --kinematics-mode kf \
    --rda-traj-disable-lidar-obstacles \
    --auto-spawn-sec 1 --auto-follow-sec 4 "$@"
