# shellcheck shell=bash
# Plaza scene preset. Sourced by scripts/run_episode.sh (see passage/episode.sh
# for the preset contract). Plaza adds a crowd flow-mode axis.

SCENE_HUMANOID_SUPPORTED=0
SCENE_FLOW_MODES="random parallel perpendicular chaotic"

scene_episode_args() {
  PLANNER="${PLANNER:-gt_planner}"
  FOLLOW_POSITION="${FOLLOW_POSITION:-back}"
  DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"
  DIFFICULTY="${DIFFICULTY:-legend}"
  FLOW_MODE="${FLOW_MODE:-random}"
  DURATION_SEC="${DURATION_SEC:-600}"
  USE_PERCEPTION="${USE_PERCEPTION:-0}"
  ENABLE_EVALUATION="${ENABLE_EVALUATION:-1}"
  DEBUG="${DEBUG:-0}"
  AUTO_SPAWN_SEC="${AUTO_SPAWN_SEC:--1}"
  AUTO_FOLLOW_SEC="${AUTO_FOLLOW_SEC:--1}"

  case "$DIFFICULTY" in
    easy|hard|boss|legend) ;;
    *) echo "[plaza] unknown DIFFICULTY=$DIFFICULTY (expected easy|hard|boss|legend)" >&2; exit 1 ;;
  esac
  case "$FLOW_MODE" in
    random|parallel|perpendicular|chaotic) ;;
    *) echo "[plaza] unknown FLOW_MODE=$FLOW_MODE (expected $SCENE_FLOW_MODES)" >&2; exit 1 ;;
  esac
  require_asset gridmap_roi.npz
  require_asset gridmap_roi_flow_points.json

  EPISODE_ARGS=(
    --load-map
    --map-name Town10HD_Opt
    --grid-npz "$FB_ASSETS_DIR/gridmap_roi.npz"
    --flow-points-json "$FB_ASSETS_DIR/gridmap_roi_flow_points.json"
    --robot-spawn-mode near_target
    --target-track-id N01
    --visibility-threshold 400
    --robot-rescue-lift-z 0.25
    --sensor-image-w 800
    --sensor-image-h 600
    --sensor-fov-deg 90
    --duration-sec "$DURATION_SEC"
    --draw-debug
    --draw-laser
    --planner "$PLANNER"
    --follow-position "$FOLLOW_POSITION"
    --desired-distance "$DESIRED_DISTANCE"
    --difficulty "$DIFFICULTY"
    --flow-mode "$FLOW_MODE"
    --output-dir "$FB_RUNS_DIR"
    --auto-spawn-sec "$AUTO_SPAWN_SEC"
    --auto-follow-sec "$AUTO_FOLLOW_SEC"
  )
  [[ "$USE_PERCEPTION" == "1" ]] && EPISODE_ARGS+=(--use-perception)
  [[ "$ENABLE_EVALUATION" == "1" ]] && EPISODE_ARGS+=(--enable-evaluation)
  [[ "$DEBUG" == "1" ]] && EPISODE_ARGS+=(--debug)
  [[ -n "${NUM_WALKERS:-}" ]] && EPISODE_ARGS+=(--num-walkers "$NUM_WALKERS")
  # Reproducible "random" flow mode: set by run_batch_eval.sh per slot so the
  # same (difficulty, slot) always draws the same crowd flow.
  [[ -n "${FB_NPC_SEED:-}" ]] && EPISODE_ARGS+=(--npc-rng-seed "$FB_NPC_SEED")
  return 0
}

# Walkers the plaza runner spawns per difficulty preset (target included).
scene_walkers_for() {
  case "$1" in
    easy) echo 10 ;; hard) echo 15 ;; boss) echo 20 ;; legend) echo 25 ;;
    *) return 1 ;;
  esac
}

scene_batch_defaults() {
  AUTO_SPAWN_SEC="${AUTO_SPAWN_SEC:-2}"
  DURATION_SEC="${DURATION_SEC:-600}"
  FLOW_MODES="${FLOW_MODES:-random}"
}

scene_batch_slot_args() {
  local difficulty="$1" flow_mode="$2"
  export DIFFICULTY="$difficulty" FLOW_MODE="$flow_mode" DURATION_SEC ENABLE_EVALUATION
  unset NUM_WALKERS
  SLOT_ARGS=(--no-ui)
}
