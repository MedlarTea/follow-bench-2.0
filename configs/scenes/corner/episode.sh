# shellcheck shell=bash
# Corner scene preset. Sourced by scripts/run_episode.sh (see passage/episode.sh
# for the preset contract). Difficulty drives the crowd preset below.

SCENE_HUMANOID_SUPPORTED=1

# DIFFICULTY -> walkers, speed range, lateral band half-width, min spawn separation.
corner_difficulty_preset() {
  case "$1" in
    easy)   CORNER_WALKERS=8;  MIN_SPEED=0.50; MAX_SPEED=1.00; BAND_HALF=4.0; SPAWN_MIN_SEP=2.0 ;;
    hard)   CORNER_WALKERS=12; MIN_SPEED=0.70; MAX_SPEED=1.30; BAND_HALF=3.0; SPAWN_MIN_SEP=1.4 ;;
    boss)   CORNER_WALKERS=16; MIN_SPEED=0.90; MAX_SPEED=1.55; BAND_HALF=2.5; SPAWN_MIN_SEP=1.0 ;;
    legend) CORNER_WALKERS=20; MIN_SPEED=1.10; MAX_SPEED=1.80; BAND_HALF=2.0; SPAWN_MIN_SEP=0.8 ;;
    *) echo "[corner] unknown DIFFICULTY=$1 (use easy/hard/boss/legend)" >&2; return 1 ;;
  esac
}

scene_episode_args() {
  if [[ "${HUMANOID:-0}" == "1" ]]; then
    PLANNER="${PLANNER:-pid}"
    DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"
    DIFFICULTY="${DIFFICULTY:-easy}"
  else
    PLANNER="${PLANNER:-gt_planner}"
    DESIRED_DISTANCE="${DESIRED_DISTANCE:-2.2}"
    DIFFICULTY="${DIFFICULTY:-legend}"
  fi
  FOLLOW_POSITION="${FOLLOW_POSITION:-back}"
  SIM_MODE="${SIM_MODE:-sync}"

  corner_difficulty_preset "$DIFFICULTY" || exit 1
  local walkers="$CORNER_WALKERS"
  if [[ "${HUMANOID:-0}" != "1" && -n "${NUM_WALKERS:-}" ]]; then
    walkers="$NUM_WALKERS"
  fi
  echo "[corner] DIFFICULTY=$DIFFICULTY  walkers=$walkers  speed=[$MIN_SPEED,$MAX_SPEED]  band=$BAND_HALF  sep=$SPAWN_MIN_SEP"

  require_asset gridmap_roi.npz
  require_asset gridmap_roi_flow_points.json

  EPISODE_ARGS=(
    --load-map
    --map-name Town10HD_Opt
    --grid-npz "$FB_ASSETS_DIR/gridmap_roi.npz"
    --flow-points-json "$FB_ASSETS_DIR/gridmap_roi_flow_points.json"
    --npc-flow-mode pair_12_34
    --npc-pair-spawn-mode paired_endpoint
    --target-spawn-policy endpoint_center
    --crowd-spawn-policy roi_random
    --npc-pair-goal-policy fixed_until_arrival
    --npc-pair-target-band-half-width-m "$BAND_HALF"
    --npc-spawn-min-sep-m "$SPAWN_MIN_SEP"
    --target-centerline-half-width-m 0.8
    --target-wall-clearance-min-m 1.2
    --target-endpoint-sample-radius-m 5.0
    --right-hand-lane-sep-m 0.80
    --target-lane-bias-scale 0.85
    --target-lane-min-offset-m 0.45
    --target-lane-max-offset-m 0.90
    --target-lateral-jitter-m 0.08
    --target-turnaround-forward-m 2.2
    --target-turnaround-merge-m 4.5
    --target-turnaround-side-m 1.6
    --target-turnaround-samples 20
    --target-turnaround-speed-mps 0.40
    --crowd-turn-sublane-mode preserve_sublane
    --crowd-turn-radius-jitter-m 0.8
    --crowd-turn-min-offset-keep-ratio 0.78
    --crowd-turn-samples 14
    --npc-navigation-mode astar
    --npc-replan-on-block
    --npc-block-replan-window-sec 2.0
    --npc-block-replan-progress-thresh 0.25
    --npc-block-replan-risk-thresh 0.5
    --npc-astar-replan-cooldown-sec 0.6
    --robot-spawn-mode behind_target
    --robot-spawn-min-dist 1.5
    --robot-spawn-max-dist 3.0
    --robot-spawn-clearance-m 2.2
    --robot-spawn-retry-attempts 24
    --side-follow-robot-yaw-policy match_target_heading
    --target-lane-bias-mode leave_follow_side_clear
    --num-walkers "$walkers"
    --npc-min-speed "$MIN_SPEED"
    --npc-max-speed "$MAX_SPEED"
    --target-track-id N01
    --target-required-legs 1
    --visibility-threshold 400
    --robot-rescue-lift-z 0.25
    --sensor-image-w 800
    --sensor-image-h 600
    --sensor-fov-deg 90
    --duration-sec 900
    --max-active-sec 180
    --draw-debug
    --draw-laser
    --planner "$PLANNER"
    --follow-position "$FOLLOW_POSITION"
    --desired-distance "$DESIRED_DISTANCE"
    --enable-evaluation
    --output-dir "$FB_RUNS_DIR"
  )
  if [[ "$SIM_MODE" == "async" ]]; then
    EPISODE_ARGS+=(--async-mode --async-after-spawn)
  fi
  # Difficulty label drives weather weights and legend look-alikes (the
  # humanoid runner does not implement these yet).
  if [[ "${HUMANOID:-0}" != "1" ]]; then
    EPISODE_ARGS+=(--difficulty "$DIFFICULTY")
  fi
  # Paired replay: the batch runner exports one fixed seed per (difficulty, slot).
  [[ -n "${FB_NPC_SEED:-}" ]] && EPISODE_ARGS+=(--npc-rng-seed "$FB_NPC_SEED")
  return 0
}

scene_walkers_for() {
  corner_difficulty_preset "$1" >/dev/null || return 1
  echo "$CORNER_WALKERS"
}

scene_batch_defaults() {
  VISUALIZE="${VISUALIZE:-0}"
}

scene_batch_slot_args() {
  local difficulty="$1" needs_perception="$3"
  export DIFFICULTY="$difficulty"
  unset NUM_WALKERS
  SLOT_ARGS=()
  if [[ "$needs_perception" == "1" ]]; then
    SLOT_ARGS+=(--kinematics-mode kf)
  fi
  if [[ "$VISUALIZE" == "1" ]]; then
    SLOT_ARGS+=(--visualize)
  fi
}
