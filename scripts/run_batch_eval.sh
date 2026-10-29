#!/usr/bin/env bash
# Batch evaluation driver for every FollowBench scene.
#
# Usage:
#   scripts/run_batch_eval.sh --scene <passage|corner|park|plaza> [--config NAME|PATH] [options]
#
# --config accepts a path or a bare name resolved as
# configs/scenes/<scene>/batch/<NAME>.sh (default: default_50x).
#
# Options:
#   --batch-id ID              Batch identifier (default: <scene>_batch_<timestamp>).
#                              Re-using an id resumes it (RESUME=1).
#   --no-launch-carla          Do not auto-launch CARLA if it is unreachable.
#   --shutdown-carla-on-exit   Stop the CARLA server this script launched.
#   --max-slots N              Exit 99 after N executed slots (outer wrapper recycles CARLA).
#   -h, --help
#
# Config variables (see configs/scenes/*/batch/*.sh):
#   PLANNERS_WITH_PERCEPTION   planners run with --use-perception over FOLLOW_POSITIONS_MAIN
#   PLANNERS_BACK_ONLY         planners run without perception over FOLLOW_POSITIONS_BACK_ONLY
#   DIFFICULTIES, FLOW_MODES (plaza), REPEATS, DESIRED_DISTANCE, SIM_MODE,
#   AUTO_SPAWN_SEC, AUTO_FOLLOW_SEC, RESUME, STOP_ON_FAILURE, RESTART_CARLA_EVERY_N_SLOTS,
#   REID_MODE, REID_KPR_CONFIG, REID_DEVICE, EXTRA_EPISODE_ARGS, plus scene-specific knobs.
#
# Environment: CARLA_HOST / CARLA_PORT (default 127.0.0.1:2000), CARLA_CMD or
# CARLA_ROOT (headless launch), PYTHON (interpreter, default python),
# FB_OUTPUT_ROOT (default <repo>/outputs).
#
# Outputs:
#   $FB_OUTPUT_ROOT/batch_manifests/<scene>/<batch_id>/  manifest.csv, failures.csv, progress.log, summary.txt
#   $FB_OUTPUT_ROOT/logs_batch/<scene>/<batch_id>/       per-trial evaluation logs
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_EPISODE="$REPO_ROOT/scripts/run_episode.sh"

usage() { sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

SCENE=""
CONFIG_ARG="default_50x"
LAUNCH_CARLA="1"
SHUTDOWN_CARLA_ON_EXIT="0"
BATCH_ID=""
MAX_SLOTS_THIS_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)                  SCENE="${2:?missing value for --scene}"; shift 2 ;;
    --config)                 CONFIG_ARG="${2:?missing value for --config}"; shift 2 ;;
    --batch-id)               BATCH_ID="${2:?missing value for --batch-id}"; shift 2 ;;
    --no-launch-carla)        LAUNCH_CARLA="0"; shift ;;
    --shutdown-carla-on-exit) SHUTDOWN_CARLA_ON_EXIT="1"; shift ;;
    --max-slots)              MAX_SLOTS_THIS_RUN="${2:?missing value for --max-slots}"; shift 2 ;;
    -h|--help)                usage; exit 0 ;;
    *) echo "[BATCH] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SCENE_PRESET="$REPO_ROOT/configs/scenes/$SCENE/episode.sh"
if [[ -z "$SCENE" || ! -f "$SCENE_PRESET" ]]; then
  echo "[BATCH] --scene must be one of: passage corner park plaza (got '${SCENE}')" >&2
  exit 2
fi
if [[ -f "$CONFIG_ARG" ]]; then
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
else
  CONFIG_PATH="$REPO_ROOT/configs/scenes/$SCENE/batch/${CONFIG_ARG%.sh}.sh"
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[BATCH] missing config: $CONFIG_PATH" >&2
  exit 2
fi

# Scene hooks (scene_walkers_for, scene_batch_defaults, scene_batch_slot_args, SCENE_FLOW_MODES).
FB_ASSETS_DIR="$REPO_ROOT/assets/scenes/$SCENE"
require_asset() { [[ -f "$FB_ASSETS_DIR/$1" ]] || { echo "[BATCH] missing asset $FB_ASSETS_DIR/$1" >&2; exit 2; }; }
# shellcheck source=/dev/null
source "$SCENE_PRESET"
# shellcheck source=/dev/null
source "$CONFIG_PATH"

CARLA_HOST="${CARLA_HOST:-127.0.0.1}"
CARLA_PORT="${CARLA_PORT:-2000}"
[[ -z "$BATCH_ID" ]] && BATCH_ID="${SCENE}_batch_$(date +%Y%m%d_%H%M%S)"

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
BATCH_DIR="$FB_OUTPUT_ROOT/batch_manifests/$SCENE/$BATCH_ID"
EVAL_BATCH_ROOT="$FB_OUTPUT_ROOT/logs_batch/$SCENE/$BATCH_ID"
mkdir -p "$BATCH_DIR"
CONFIG_SNAPSHOT="$BATCH_DIR/config_snapshot.sh"
MANIFEST_CSV="$BATCH_DIR/manifest.csv"
FAILURES_CSV="$BATCH_DIR/failures.csv"
PROGRESS_LOG="$BATCH_DIR/progress.log"
SUMMARY_TXT="$BATCH_DIR/summary.txt"
CARLA_PID_FILE="$BATCH_DIR/carla.pid"
CARLA_STDOUT_LOG="$BATCH_DIR/carla_stdout.log"

log() { printf '%s %s\n' "[BATCH]" "$*" | tee -a "$PROGRESS_LOG"; }

# ── defaults + validation ─────────────────────────────────────────────────────
scene_batch_defaults
REPEATS="${REPEATS:-1}"
DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"
SIM_MODE="${SIM_MODE:-sync}"
AUTO_SPAWN_SEC="${AUTO_SPAWN_SEC:-1}"
AUTO_FOLLOW_SEC="${AUTO_FOLLOW_SEC:-4}"
ENABLE_EVALUATION="${ENABLE_EVALUATION:-1}"
RESUME="${RESUME:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
RESTART_CARLA_EVERY_N_SLOTS="${RESTART_CARLA_EVERY_N_SLOTS:-50}"
REID_MODE="${REID_MODE:-basic}"
REID_KPR_CONFIG="${REID_KPR_CONFIG:-kpr_occ_duke_test}"
REID_DEVICE="${REID_DEVICE:-auto}"
EXTRA_EPISODE_ARGS="${EXTRA_EPISODE_ARGS:-}"
# Per setting, give up after REPEATS * MAX_ATTEMPT_FACTOR attempts in one run.
MAX_ATTEMPT_FACTOR="${MAX_ATTEMPT_FACTOR:-2}"
# Under --no-launch-carla, exit 99 after this many consecutive failed slots so
# the outer wrapper can recycle CARLA.
FAIL_FAST_AFTER="${FAIL_FAST_AFTER:-2}"

die() { echo "[BATCH] $*" >&2; exit 2; }
read -r -a planners_perc  <<<"${PLANNERS_WITH_PERCEPTION:-}"
read -r -a planners_back  <<<"${PLANNERS_BACK_ONLY:-}"
read -r -a positions_main <<<"${FOLLOW_POSITIONS_MAIN:-}"
read -r -a positions_back <<<"${FOLLOW_POSITIONS_BACK_ONLY:-}"
read -r -a difficulties   <<<"${DIFFICULTIES:-}"
if [[ -n "${SCENE_FLOW_MODES:-}" ]]; then
  read -r -a flow_modes <<<"${FLOW_MODES:-}"
  (( ${#flow_modes[@]} > 0 )) || die "FLOW_MODES is empty"
  for f in "${flow_modes[@]}"; do
    [[ " $SCENE_FLOW_MODES " == *" $f "* ]] || die "invalid flow mode: $f (expected: $SCENE_FLOW_MODES)"
  done
else
  flow_modes=("")
fi

(( ${#planners_perc[@]} + ${#planners_back[@]} > 0 )) || die "no planners configured"
(( ${#difficulties[@]} > 0 )) || die "no difficulties configured"
for d in "${difficulties[@]}"; do
  case "$d" in easy|hard|boss|legend) ;; *) die "invalid difficulty: $d" ;; esac
done
for p in "${positions_main[@]}" "${positions_back[@]}"; do
  case "$p" in back|left_side|right_side) ;; *) die "invalid follow position: $p" ;; esac
done
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive integer, got: $REPEATS"
case "$REID_MODE" in basic|kpr) ;; *) die "invalid REID_MODE: $REID_MODE (expected basic|kpr)" ;; esac
case "$SIM_MODE" in sync|async) ;; *) die "invalid SIM_MODE: $SIM_MODE (expected sync|async)" ;; esac

DIST_LABEL="$(python3 -c 'import sys; t=f"{float(sys.argv[1]):.2f}".rstrip("0").rstrip("."); print(t or "0")' "$DESIRED_DISTANCE")"

# ── bookkeeping files ─────────────────────────────────────────────────────────
MANIFEST_HEADER="batch_id,scene,planner,difficulty,flow_mode,follow_position,desired_distance,num_walkers_total,repeat_slot,needs_perception,existing_complete_before,status,config_dir,run_dir,start_time,end_time,duration_sec,exit_code,termination_reason,task_success"
FAILURES_HEADER="batch_id,scene,planner,difficulty,flow_mode,follow_position,repeat_slot,status,run_dir,exit_code,termination_reason"

init_csv() {  # file header -- keep rows on resume; rotate files written by an older runner
  local file="$1" header="$2"
  if [[ -s "$file" && "$(head -n 1 "$file")" != "$header" ]]; then
    mv "$file" "${file%.csv}.legacy.csv"
  fi
  [[ -s "$file" ]] || echo "$header" >"$file"
}
init_csv "$MANIFEST_CSV" "$MANIFEST_HEADER"
init_csv "$FAILURES_CSV" "$FAILURES_HEADER"

{
  echo "# FollowBench batch config snapshot ($(date --iso-8601=seconds))"
  echo "SCENE=$SCENE"
  echo "CONFIG_PATH=$CONFIG_PATH"
  echo "EVAL_BATCH_ROOT=$EVAL_BATCH_ROOT"
  for v in PLANNERS_WITH_PERCEPTION PLANNERS_BACK_ONLY FOLLOW_POSITIONS_MAIN FOLLOW_POSITIONS_BACK_ONLY \
           DIFFICULTIES FLOW_MODES REPEATS DESIRED_DISTANCE NUM_WALKERS_TOTAL DURATION_SEC SIM_MODE \
           AUTO_SPAWN_SEC AUTO_FOLLOW_SEC ENABLE_EVALUATION RESUME STOP_ON_FAILURE RESTART_CARLA_EVERY_N_SLOTS \
           REID_MODE REID_KPR_CONFIG REID_DEVICE EXTRA_EPISODE_ARGS VISUALIZE WEATHER ENABLE_UI \
           ENABLE_DEBUG_VIS ENABLE_WORLD_DEBUG ENABLE_LASER CARLA_HOST CARLA_PORT CARLA_CMD CARLA_ROOT; do
    [[ -n "${!v+x}" ]] && printf '%s=%q\n' "$v" "${!v}"
  done
} >"$CONFIG_SNAPSHOT"

# ── CARLA lifecycle ───────────────────────────────────────────────────────────
carla_alive() {
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(1.0); sys.exit(0 if s.connect_ex((sys.argv[1], int(sys.argv[2])))==0 else 1)' \
    "$CARLA_HOST" "$CARLA_PORT"
}

resolve_carla_cmd() {
  if [[ -n "${CARLA_CMD:-}" ]]; then
    printf '%s\n' "$CARLA_CMD"; return 0
  fi
  local candidates=() c
  [[ -n "${CARLA_ROOT:-}" ]] && candidates+=("${CARLA_ROOT%/}/CarlaUE4.sh")
  candidates+=("$REPO_ROOT/CARLA_multi_robots/CarlaUE4.sh")
  candidates+=("$HOME/CARLA_0.9.16/CarlaUE4.sh" "$HOME/CARLA_0.9.15/CarlaUE4.sh" "$HOME/CARLA_0.9.14/CarlaUE4.sh"
               "$HOME/CARLA/CarlaUE4.sh" "$HOME/carla/CarlaUE4.sh" "/opt/carla-simulator/CarlaUE4.sh")
  for c in "${candidates[@]}"; do
    if [[ -x "$c" ]]; then
      printf '"%s" -RenderOffScreen -quality-level=Low -nosound -carla-rpc-port=%s\n' "$c" "$CARLA_PORT"
      return 0
    fi
  done
  return 1
}

LAUNCHED_CARLA_BY_BATCH="0"
SLOTS_SINCE_CARLA_LAUNCH=0

launch_carla_if_needed() {
  if carla_alive; then
    log "CARLA is reachable at ${CARLA_HOST}:${CARLA_PORT}"
    return 0
  fi
  [[ "$LAUNCH_CARLA" == "1" ]] || { echo "[BATCH] CARLA is not reachable at ${CARLA_HOST}:${CARLA_PORT} and auto-launch is disabled." >&2; exit 2; }
  local cmd
  cmd="$(resolve_carla_cmd)" || { echo "[BATCH] Unable to resolve a headless CARLA command. Set CARLA_CMD or CARLA_ROOT." >&2; exit 2; }
  log "Launching headless CARLA: $cmd"
  # setsid: CARLA gets its own process group, so we can stop exactly the
  # server we started (and never touch other CARLA instances on this host).
  setsid nohup bash -c "$cmd" >"$CARLA_STDOUT_LOG" 2>&1 </dev/null &
  echo "$!" >"$CARLA_PID_FILE"
  LAUNCHED_CARLA_BY_BATCH="1"
  local waited=0
  until carla_alive; do
    sleep 2; waited=$((waited + 2))
    (( waited >= 120 )) && { echo "[BATCH] CARLA did not become reachable within 120s. See $CARLA_STDOUT_LOG" >&2; exit 2; }
  done
  log "CARLA is ready after ${waited}s"
  SLOTS_SINCE_CARLA_LAUNCH=0
}

kill_launched_carla() {
  [[ -f "$CARLA_PID_FILE" ]] || return 0
  local pgid
  pgid="$(cat "$CARLA_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pgid" ]]; then
    kill -TERM -- "-$pgid" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL -- "-$pgid" >/dev/null 2>&1 || true
  fi
  rm -f "$CARLA_PID_FILE"
}

restart_carla() {
  local reason="$1"
  if [[ "$LAUNCH_CARLA" != "1" ]]; then
    log "skipping CARLA restart (reason=$reason); auto-launch disabled"
    return 0
  fi
  log "restarting CARLA (reason=$reason, slots_since_launch=$SLOTS_SINCE_CARLA_LAUNCH)"
  kill_launched_carla
  # Stray episodes bound to this port would keep sensors alive on the new server.
  pkill -f "followbench\.scenes\..*--port[ =]${CARLA_PORT}\b" >/dev/null 2>&1 || true
  sleep 3
  LAUNCHED_CARLA_BY_BATCH="0"
  launch_carla_if_needed
}

maybe_restart_carla() {
  if (( RESTART_CARLA_EVERY_N_SLOTS > 0 && SLOTS_SINCE_CARLA_LAUNCH >= RESTART_CARLA_EVERY_N_SLOTS )); then
    restart_carla periodic
  elif ! carla_alive; then
    restart_carla rpc_unreachable
  fi
}

cleanup() {
  if [[ "$LAUNCHED_CARLA_BY_BATCH" == "1" && "$SHUTDOWN_CARLA_ON_EXIT" == "1" ]]; then
    kill_launched_carla
    log "Stopped CARLA launched by this batch"
  fi
}
trap cleanup EXIT

launch_carla_if_needed

# ── per-slot helpers ──────────────────────────────────────────────────────────
count_complete_trials() {
  [[ -d "$1" ]] || { echo 0; return 0; }
  find "$1" -mindepth 2 -maxdepth 2 -type f -name eval_result.json 2>/dev/null | wc -l | tr -d ' '
}

latest_trial_dir() {
  [[ -d "$1" ]] || return 1
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'trial_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

json_eval_fields() {
  python3 - "$1" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.isfile(path):
    print(","); raise SystemExit(0)
data = json.load(open(path, encoding="utf-8"))
succ = data.get("task_success")
print(f"{data.get('termination_reason') or ''},{'' if succ is None else str(bool(succ)).lower()}")
PY
}

eval_root_for() {  # difficulty flow
  printf '%s/%s%s' "$EVAL_BATCH_ROOT" "$1" "${2:+/$2}"
}

config_dir_for() {  # planner follow difficulty flow
  local walkers
  walkers="$(scene_walkers_for "$3")"
  printf '%s/%s/%s/H%d_%s_D%s' "$(eval_root_for "$3" "$4")" "$SCENE" "$1" "$((walkers > 0 ? walkers - 1 : 0))" "$2" "$DIST_LABEL"
}

append_manifest() {  # planner difficulty flow follow slot perc existing status config_dir run_dir start end dur exit term succ
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$BATCH_ID" "$SCENE" "$1" "$2" "$3" "$4" "$DESIRED_DISTANCE" "$(scene_walkers_for "$2")" \
    "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" >>"$MANIFEST_CSV"
}

append_failure() {  # planner difficulty flow follow slot status run_dir exit term
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$BATCH_ID" "$SCENE" "$@" >>"$FAILURES_CSV"
}

LAST_STATUS=""
run_one_slot() {  # planner difficulty flow follow slot needs_perception existing_before
  local planner="$1" difficulty="$2" flow="$3" follow="$4" slot="$5" perc="$6" before="$7"
  local config_dir start_iso start_epoch exit_code status run_dir term="" succ=""
  config_dir="$(config_dir_for "$planner" "$follow" "$difficulty" "$flow")"
  start_iso="$(date --iso-8601=seconds)"; start_epoch="$(date +%s)"
  log "[${GLOBAL_SLOT_INDEX}/${TOTAL_SLOTS}] planner=$planner difficulty=$difficulty${flow:+ flow=$flow} follow=$follow slot=${slot}/${REPEATS} perception=$perc"

  # Deterministic NPC seed so the same (difficulty, slot) always draws the
  # same crowd layout (reproducible), independent of the flow mode.
  local dcode
  case "$difficulty" in
    easy) dcode=0 ;; hard) dcode=1 ;; boss) dcode=2 ;; legend) dcode=3 ;;
    *) dcode=0 ;;
  esac
  export FB_NPC_SEED=$((1000 + dcode * 1000 + slot))

  # "random" flow mode is sampled uniformly (not randomly) across a cell:
  # slots are bucketed parallel -> perpendicular -> chaotic, with any
  # remainder going to chaotic, so the three flows stay balanced.
  local actual_flow="$flow"
  if [[ "$flow" == "random" ]]; then
    local per=$(( REPEATS / 3 ))
    if (( slot <= per )); then
      actual_flow="parallel"
    elif (( slot <= 2 * per )); then
      actual_flow="perpendicular"
    else
      actual_flow="chaotic"
    fi
  fi

  SLOT_ARGS=()
  scene_batch_slot_args "$difficulty" "$actual_flow" "$perc"
  local cmd=(bash "$RUN_EPISODE" --scene "$SCENE"
    --host "$CARLA_HOST" --port "$CARLA_PORT"
    --enable-evaluation
    --eval-log-root "$(eval_root_for "$difficulty" "$flow")"
    --auto-spawn-sec "$AUTO_SPAWN_SEC"
    --auto-follow-sec "$AUTO_FOLLOW_SEC"
    "${SLOT_ARGS[@]+"${SLOT_ARGS[@]}"}")
  if [[ "$perc" == "1" ]]; then
    cmd+=(--use-perception --reid-mode "$REID_MODE" --reid-kpr-config "$REID_KPR_CONFIG" --reid-device "$REID_DEVICE")
  fi
  if [[ -n "$EXTRA_EPISODE_ARGS" ]]; then
    local extra; read -r -a extra <<<"$EXTRA_EPISODE_ARGS"
    cmd+=("${extra[@]}")
  fi

  EPISODE_PRESET=batch PLANNER="$planner" FOLLOW_POSITION="$follow" \
  DESIRED_DISTANCE="$DESIRED_DISTANCE" SIM_MODE="$SIM_MODE" "${cmd[@]}"
  exit_code=$?

  local end_iso duration
  end_iso="$(date --iso-8601=seconds)"; duration=$(( $(date +%s) - start_epoch ))
  run_dir="$(latest_trial_dir "$config_dir" || true)"
  if (( exit_code != 0 )); then
    status="failed"
  elif [[ -n "$run_dir" && -f "$run_dir/eval_result.json" ]]; then
    status="completed"
    IFS=, read -r term succ < <(json_eval_fields "$run_dir/eval_result.json")
  elif [[ -n "$run_dir" ]]; then
    status="incomplete"
  else
    status="failed"
  fi
  LAST_STATUS="$status"

  append_manifest "$planner" "$difficulty" "$flow" "$follow" "$slot" "$perc" "$before" "$status" \
    "$config_dir" "$run_dir" "$start_iso" "$end_iso" "$duration" "$exit_code" "$term" "$succ"
  if [[ "$status" == "completed" ]]; then
    log "status=completed planner=$planner difficulty=$difficulty${flow:+ flow=$flow} follow=$follow slot=$slot termination=${term:-none} success=${succ:-}"
  else
    append_failure "$planner" "$difficulty" "$flow" "$follow" "$slot" "$status" "$run_dir" "$exit_code" "$term"
    log "status=$status planner=$planner difficulty=$difficulty${flow:+ flow=$flow} follow=$follow slot=$slot exit=$exit_code"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      echo "[BATCH] stop-on-failure triggered" >&2
      exit 1
    fi
  fi
}

# ── main loop ─────────────────────────────────────────────────────────────────
TOTAL_SLOTS=$(( (${#planners_perc[@]} * ${#positions_main[@]} + ${#planners_back[@]} * ${#positions_back[@]}) \
               * ${#difficulties[@]} * ${#flow_modes[@]} * REPEATS ))
GLOBAL_SLOT_INDEX=0
COMPLETED_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0
NON_SKIPPED_THIS_RUN=0
CONSECUTIVE_FAILURES=0

run_setting() {  # planner difficulty flow follow needs_perception
  local planner="$1" difficulty="$2" flow="$3" follow="$4" perc="$5"
  local config_dir existing s attempts=0 max_attempts=$((REPEATS * MAX_ATTEMPT_FACTOR))
  config_dir="$(config_dir_for "$planner" "$follow" "$difficulty" "$flow")"
  existing="$(count_complete_trials "$config_dir")"
  if [[ "$RESUME" == "1" && "$existing" -gt 0 ]]; then
    for ((s=1; s<=existing && s<=REPEATS; s++)); do
      GLOBAL_SLOT_INDEX=$((GLOBAL_SLOT_INDEX + 1))
      append_manifest "$planner" "$difficulty" "$flow" "$follow" "$s" "$perc" "$existing" "skipped_existing" \
        "$config_dir" "" "" "" "" "0" "" ""
      SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    done
    log "[resume] planner=$planner difficulty=$difficulty${flow:+ flow=$flow} follow=$follow: ${existing}/${REPEATS} on disk"
  fi
  while (( existing < REPEATS )); do
    if (( max_attempts > 0 && attempts >= max_attempts )); then
      log "[retry-cap] giving up planner=$planner difficulty=$difficulty${flow:+ flow=$flow} follow=$follow: ${existing}/${REPEATS} after ${attempts} attempts"
      break
    fi
    attempts=$((attempts + 1))
    GLOBAL_SLOT_INDEX=$((GLOBAL_SLOT_INDEX + 1))
    maybe_restart_carla
    run_one_slot "$planner" "$difficulty" "$flow" "$follow" "$((existing + 1))" "$perc" "$existing"
    SLOTS_SINCE_CARLA_LAUNCH=$((SLOTS_SINCE_CARLA_LAUNCH + 1))
    NON_SKIPPED_THIS_RUN=$((NON_SKIPPED_THIS_RUN + 1))
    if [[ "$LAST_STATUS" == "completed" ]]; then
      COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
      existing=$((existing + 1))
      CONSECUTIVE_FAILURES=0
    else
      FAILED_COUNT=$((FAILED_COUNT + 1))
      CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
      if [[ "$LAUNCH_CARLA" != "1" ]]; then
        if ! carla_alive; then
          log "[carla-watchdog] slot failed and CARLA unreachable; exiting 99 so the outer wrapper can recycle CARLA"
          exit 99
        fi
        if (( FAIL_FAST_AFTER > 0 && CONSECUTIVE_FAILURES >= FAIL_FAST_AFTER )); then
          log "[carla-watchdog] $CONSECUTIVE_FAILURES consecutive failed slots; exiting 99 so the outer wrapper can recycle CARLA"
          exit 99
        fi
      fi
    fi
    if (( MAX_SLOTS_THIS_RUN > 0 && NON_SKIPPED_THIS_RUN >= MAX_SLOTS_THIS_RUN )); then
      log "[max-slots] cap reached ($NON_SKIPPED_THIS_RUN); exiting 99"
      exit 99
    fi
  done
}

run_group() {  # needs_perception planners_array_name positions_array_name
  local perc="$1"
  local -n g_planners="$2" g_positions="$3"
  local planner difficulty flow follow
  for planner in "${g_planners[@]}"; do
    for difficulty in "${difficulties[@]}"; do
      for flow in "${flow_modes[@]}"; do
        for follow in "${g_positions[@]}"; do
          run_setting "$planner" "$difficulty" "$flow" "$follow" "$perc"
        done
      done
    done
  done
}

log "scene=$SCENE batch_id=$BATCH_ID"
log "config=$CONFIG_PATH"
log "eval_batch_root=$EVAL_BATCH_ROOT"
log "planners_with_perception=${PLANNERS_WITH_PERCEPTION:-} (positions: ${FOLLOW_POSITIONS_MAIN:-})"
log "planners_back_only=${PLANNERS_BACK_ONLY:-} (positions: ${FOLLOW_POSITIONS_BACK_ONLY:-})"
log "difficulties=${DIFFICULTIES:-}${SCENE_FLOW_MODES:+ flow_modes=${FLOW_MODES:-}} repeats=$REPEATS distance=$DESIRED_DISTANCE"
log "sim_mode=$SIM_MODE auto_spawn_sec=$AUTO_SPAWN_SEC auto_follow_sec=$AUTO_FOLLOW_SEC restart_every=$RESTART_CARLA_EVERY_N_SLOTS"
log "total_slots=$TOTAL_SLOTS"

run_group 1 planners_perc positions_main
run_group 0 planners_back positions_back

{
  echo "batch_id=$BATCH_ID"
  echo "scene=$SCENE"
  echo "total_slots=$TOTAL_SLOTS"
  echo "completed=$COMPLETED_COUNT"
  echo "failed=$FAILED_COUNT"
  echo "skipped_existing=$SKIPPED_COUNT"
  echo "manifest=$MANIFEST_CSV"
  echo "failures=$FAILURES_CSV"
  echo "progress_log=$PROGRESS_LOG"
  echo "eval_batch_root=$EVAL_BATCH_ROOT"
} >"$SUMMARY_TXT"
log "done completed=$COMPLETED_COUNT failed=$FAILED_COUNT skipped_existing=$SKIPPED_COUNT"
log "summary=$SUMMARY_TXT"
