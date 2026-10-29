# shellcheck shell=bash
# Park scene preset. Sourced by scripts/run_episode.sh (see passage/episode.sh
# for the preset contract).
#
# Park keeps two argument sets on purpose:
#   interactive  the hand-run defaults (30 walkers, NPC speed 0.7-1.2 m/s, debug UI on)
#   batch        the exact argument set used for the baseline batch runs
#                (NUM_WALKERS_TOTAL walkers, runner-default NPC speeds, UI/debug toggles)

SCENE_HUMANOID_SUPPORTED=1

park_common_args() {
  EPISODE_ARGS=(
    --load-map
    --map-name Town10HD_Opt
    --grid-npz "$FB_ASSETS_DIR/gridmap_roi.npz"
    --flow-points-json "$FB_ASSETS_DIR/gridmap_roi_flow_points.json"
    --npc-flow-mode free_3
    --npc-navigation-mode astar
    --npc-replan-on-avoid
    --npc-avoid-replan-threshold 1.0
    --npc-astar-replan-cooldown-sec 0.6
    --use-scheduled-flow
    --scheduled-flow-dwell-s 0.0
    --robot-spawn-mode near_target
  )
  # Paired replay: the batch runner exports one fixed seed per (difficulty, slot).
  [[ -n "${FB_NPC_SEED:-}" ]] && EPISODE_ARGS+=(--npc-rng-seed "$FB_NPC_SEED")
  return 0
}

scene_episode_args() {
  SIM_MODE="${SIM_MODE:-sync}"
  FOLLOW_POSITION="${FOLLOW_POSITION:-back}"
  DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"
  require_asset gridmap_roi.npz
  park_common_args

  if [[ "${EPISODE_PRESET:-interactive}" == "batch" ]]; then
    PLANNER="${PLANNER:-gt_planner}"
    EPISODE_ARGS+=(
      --robot-spawn-min-dist 0.5
      --robot-spawn-max-dist 1.0
      --num-walkers "$NUM_WALKERS"
      --target-track-id N01
      --visibility-threshold 400
      --robot-rescue-lift-z 0.25
      --sensor-image-w 800
      --sensor-image-h 600
      --sensor-fov-deg 90
      --duration-sec "$DURATION_SEC"
      --planner "$PLANNER"
      --follow-position "$FOLLOW_POSITION"
      --desired-distance "$DESIRED_DISTANCE"
      --enable-evaluation
      --output-dir "$FB_RUNS_DIR"
    )
    [[ "$SIM_MODE" == "async" ]] && EPISODE_ARGS+=(--async-mode --async-after-spawn)
    [[ "${ENABLE_UI:-0}" != "1" ]] && EPISODE_ARGS+=(--no-ui)
    [[ "${ENABLE_DEBUG_VIS:-0}" == "1" ]] && EPISODE_ARGS+=(--debug)
    [[ "${ENABLE_WORLD_DEBUG:-0}" == "1" ]] && EPISODE_ARGS+=(--draw-debug)
    [[ "${ENABLE_LASER:-0}" == "1" ]] && EPISODE_ARGS+=(--draw-laser)
    return 0
  fi

  if [[ "${HUMANOID:-0}" == "1" ]]; then
    PLANNER="${PLANNER:-rda}"
    NUM_WALKERS="${NUM_WALKERS:-6}"
    EPISODE_ARGS+=(--robot-spawn-clearance-m 2.0 --robot-spawn-retries 24)
  else
    PLANNER="${PLANNER:-gt_planner}"
    NUM_WALKERS="${NUM_WALKERS:-30}"
    EPISODE_ARGS+=(--robot-spawn-min-dist 0.5 --robot-spawn-max-dist 1.0)
  fi
  echo "[park] grid: $FB_ASSETS_DIR/gridmap_roi.npz"
  echo "[park] planner=$PLANNER  sim_mode=$SIM_MODE  walkers=$NUM_WALKERS"
  EPISODE_ARGS+=(
    --num-walkers "$NUM_WALKERS"
    --npc-min-speed 0.7
    --npc-max-speed 1.2
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
  [[ "$SIM_MODE" == "async" ]] && EPISODE_ARGS+=(--async-mode --async-after-spawn)
  EPISODE_ARGS+=(
    --planner "$PLANNER"
    --follow-position "$FOLLOW_POSITION"
    --desired-distance "$DESIRED_DISTANCE"
    --enable-evaluation
    --output-dir "$FB_RUNS_DIR"
  )
  # The humanoid runner always ran with the debug viewer; the Scout runner
  # makes it optional (DEBUG=1 by default).
  if [[ "${HUMANOID:-0}" == "1" || "${DEBUG:-1}" == "1" ]]; then
    EPISODE_ARGS+=(--debug)
  fi
}

scene_walkers_for() { echo "${NUM_WALKERS_TOTAL}"; }

scene_batch_defaults() {
  NUM_WALKERS_TOTAL="${NUM_WALKERS_TOTAL:-35}"
  DURATION_SEC="${DURATION_SEC:-300}"
  WEATHER="${WEATHER:-random}"
  ENABLE_UI="${ENABLE_UI:-0}"
  ENABLE_DEBUG_VIS="${ENABLE_DEBUG_VIS:-0}"
  ENABLE_WORLD_DEBUG="${ENABLE_WORLD_DEBUG:-0}"
  ENABLE_LASER="${ENABLE_LASER:-0}"
}

scene_batch_slot_args() {
  local difficulty="$1"
  export NUM_WALKERS="$NUM_WALKERS_TOTAL" DURATION_SEC
  export ENABLE_UI ENABLE_DEBUG_VIS ENABLE_WORLD_DEBUG ENABLE_LASER
  SLOT_ARGS=(--difficulty "$difficulty" --weather "$WEATHER")
}
