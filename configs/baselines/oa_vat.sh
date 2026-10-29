#!/usr/bin/env bash
# Baseline row: OA-VAT (single smoke episode; corner / back / easy)
PLANNER=oa_vat FOLLOW_POSITION=back DESIRED_DISTANCE=1.5 DIFFICULTY=easy \
  bash scripts/run_episode.sh --scene corner \
    --enable-evaluation --auto-spawn-sec 1 --auto-follow-sec 4 "$@"
