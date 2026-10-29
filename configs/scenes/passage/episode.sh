# shellcheck shell=bash
# Passage scene preset. Sourced by scripts/run_episode.sh.
#
# Contract (all scene presets):
#   SCENE_HUMANOID_SUPPORTED   1 if followbench.scenes.<scene>.run_episode_humanoid exists
#   scene_episode_args         fill EPISODE_ARGS from env (EPISODE_PRESET=interactive|batch)
#   scene_walkers_for D        total walkers (incl. target) a batch slot spawns at difficulty D
#   scene_batch_defaults       defaults for scene-specific batch-config variables
#   scene_batch_slot_args D F P  fill SLOT_ARGS for one batch slot (difficulty, flow, perception)
#
# Inputs provided by the caller: FB_ASSETS_DIR, FB_RUNS_DIR, HUMANOID (0|1).

SCENE_HUMANOID_SUPPORTED=0

scene_episode_args() {
  PLANNER="${PLANNER:-gt_planner}"
  SIM_MODE="${SIM_MODE:-sync}"
  FOLLOW_POSITION="${FOLLOW_POSITION:-back}"
  DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"

  require_asset gridmap_roi.npz
  require_asset routes.json

  # Walker count and NPC speed range come from the difficulty preset
  # (followbench/scenes/passage/npc_runtime.py DIFFICULTY_PRESETS) unless
  # NUM_WALKERS is set explicitly.
  EPISODE_ARGS=(
    --load-map
    --map-name Town10HD_Opt
    --grid-npz "$FB_ASSETS_DIR/gridmap_roi.npz"
    --routes-json "$FB_ASSETS_DIR/routes.json"
    --robot-spawn-mode near_target
    --robot-spawn-min-dist 0.8
    --robot-spawn-max-dist 2.0
    --target-track-id N01
    --visibility-threshold 400
    --robot-rescue-lift-z 0.25
    --sensor-image-w 800
    --sensor-image-h 600
    --sensor-fov-deg 90
    --duration-sec 300
    --draw-debug
    --draw-laser
  )
  if [[ "$SIM_MODE" == "async" ]]; then
    EPISODE_ARGS+=(--async-mode --async-after-spawn)
  fi
  [[ -n "${NUM_WALKERS:-}" ]] && EPISODE_ARGS+=(--num-walkers "$NUM_WALKERS")
  # Paired replay: the batch runner exports one fixed seed per (difficulty, slot).
  [[ -n "${FB_NPC_SEED:-}" ]] && EPISODE_ARGS+=(--npc-rng-seed "$FB_NPC_SEED")
  EPISODE_ARGS+=(
    --planner "$PLANNER"
    --follow-position "$FOLLOW_POSITION"
    --desired-distance "$DESIRED_DISTANCE"
    --enable-evaluation
    --output-dir "$FB_RUNS_DIR"
  )
}

# Must match DIFFICULTY_PRESETS[*]["num_walkers"] in
# followbench/scenes/passage/npc_runtime.py.  Kept well above the doorway
# concurrency limit (2/4/6/8) so queueing actually occurs.
scene_walkers_for() {
  case "$1" in
    easy) echo 6 ;; hard) echo 10 ;; boss) echo 14 ;; legend) echo 18 ;;
    *) return 1 ;;
  esac
}

scene_batch_defaults() {
  : # no scene-specific defaults; walkers derive from difficulty via scene_walkers_for
}

scene_batch_slot_args() {
  local difficulty="$1" walkers
  walkers="$(scene_walkers_for "$difficulty")" || return 1
  export NUM_WALKERS="$walkers"
  SLOT_ARGS=(--difficulty "$difficulty" --num-walkers "$walkers")
}
