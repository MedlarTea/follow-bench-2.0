#!/usr/bin/env bash
# Baseline row: TrackVLA (single smoke episode; corner / back / easy)
PLANNER=trackvla FOLLOW_POSITION=back DESIRED_DISTANCE=1.5 DIFFICULTY=easy \
  bash scripts/run_episode.sh --scene corner \
    --enable-evaluation --auto-spawn-sec 1 --auto-follow-sec 4 \
    --trackvla-instruction "Follow the person" "$@"
