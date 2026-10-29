#!/usr/bin/env bash
# Fan a batch config out over several GPUs. The (planner, difficulty[, flow],
# follow-position) matrix is split into tasks that are round-robined across N
# workers; worker i uses GPU (gpu-base + i) and CARLA port (port-base + i*stride)
# and runs its tasks sequentially through scripts/multi_gpu/run_worker.sh.
#
# Usage:
#   scripts/multi_gpu/eval_multi_gpu.sh --scene SCENE --config NAME|PATH --num-workers N [options]
#
# Options:
#   --gpu-base G               First GPU index (default 0)
#   --port-base P              First CARLA port (default 2000)
#   --port-stride S            Port spacing between workers (default 100)
#   --start-stagger-s N        Seconds between worker spawns (default 25)
#   --batch-id NAME            Base id; worker i writes batch <NAME>_w<i> (default <scene>_multi_<ts>)
#   --restart-every N          Recycle each worker's CARLA every N slots (default 1 = fresh server per episode)
#   --carla-bin PATH | --carla-graphics-adapter N | --container NAME |
#   --docker-host-ip IP | --no-sudo        forwarded to run_worker.sh
#
# Outputs: $FB_OUTPUT_ROOT/multi_gpu/<scene>/<batch-id>/ (task configs, worker logs,
# merged manifest.csv). Per-worker results live under logs_batch/<scene>/<batch-id>_w<i>/.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKER_SH="$REPO_ROOT/scripts/multi_gpu/run_worker.sh"
# shellcheck source=../lib/carla.sh
source "$REPO_ROOT/scripts/lib/carla.sh"

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

SCENE="" CONFIG_ARG="" NUM_WORKERS="" BATCH_ID_BASE=""
GPU_BASE="${GPU_BASE:-0}" PORT_BASE="${PORT_BASE:-2000}" PORT_STRIDE="${PORT_STRIDE:-100}"
WORKER_START_STAGGER_S="${WORKER_START_STAGGER_S:-25}" RESTART_EVERY="${RESTART_EVERY:-1}"
WORKER_PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)           SCENE="${2:?}"; shift 2 ;;
    --config)          CONFIG_ARG="${2:?}"; shift 2 ;;
    --num-workers)     NUM_WORKERS="${2:?}"; shift 2 ;;
    --gpu-base)        GPU_BASE="${2:?}"; shift 2 ;;
    --port-base)       PORT_BASE="${2:?}"; shift 2 ;;
    --port-stride)     PORT_STRIDE="${2:?}"; shift 2 ;;
    --start-stagger-s) WORKER_START_STAGGER_S="${2:?}"; shift 2 ;;
    --batch-id)        BATCH_ID_BASE="${2:?}"; shift 2 ;;
    --restart-every)   RESTART_EVERY="${2:?}"; shift 2 ;;
    --carla-bin|--carla-graphics-adapter|--container|--docker-host-ip)
                       WORKER_PASSTHROUGH+=("$1" "${2:?}"); shift 2 ;;
    --no-sudo)         WORKER_PASSTHROUGH+=("$1"); shift ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "[multi] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$REPO_ROOT/configs/scenes/$SCENE/episode.sh" ]] || { echo "[multi] --scene must be one of: passage corner park plaza" >&2; exit 2; }
[[ "$NUM_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "[multi] --num-workers must be a positive integer" >&2; exit 2; }
[[ -n "$CONFIG_ARG" ]] || { echo "[multi] --config is required" >&2; exit 2; }
if [[ -f "$CONFIG_ARG" ]]; then
  CONFIG="$(cd "$(dirname "$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
else
  CONFIG="$REPO_ROOT/configs/scenes/$SCENE/batch/${CONFIG_ARG%.sh}.sh"
fi
[[ -f "$CONFIG" ]] || { echo "[multi] config not found: $CONFIG" >&2; exit 2; }
[[ -n "$BATCH_ID_BASE" ]] || BATCH_ID_BASE="${SCENE}_multi_$(date +%Y%m%d_%H%M%S)"

# ── task matrix ───────────────────────────────────────────────────────────────
read_cfg() { bash -c "source $(printf %q "$CONFIG") >/dev/null 2>&1; printf '%s' \"\${$1:-}\""; }
read -r -a PERC   <<<"$(read_cfg PLANNERS_WITH_PERCEPTION)"
read -r -a BACK   <<<"$(read_cfg PLANNERS_BACK_ONLY)"
read -r -a POS_M  <<<"$(read_cfg FOLLOW_POSITIONS_MAIN)"
read -r -a POS_B  <<<"$(read_cfg FOLLOW_POSITIONS_BACK_ONLY)"
read -r -a DIFFS  <<<"$(read_cfg DIFFICULTIES)"
read -r -a FLOWS  <<<"$(read_cfg FLOW_MODES)"
(( ${#FLOWS[@]} > 0 )) || FLOWS=("")
(( ${#DIFFS[@]} > 0 )) || { echo "[multi] DIFFICULTIES is empty in $CONFIG" >&2; exit 2; }

T_GROUP=() T_PLANNER=() T_DIFF=() T_FLOW=() T_POS=()
emit() {  # group planners_name positions_name
  local -n e_planners="$2" e_positions="$3"
  local p d f pos
  for p in "${e_planners[@]}"; do for d in "${DIFFS[@]}"; do for f in "${FLOWS[@]}"; do for pos in "${e_positions[@]}"; do
    T_GROUP+=("$1"); T_PLANNER+=("$p"); T_DIFF+=("$d"); T_FLOW+=("$f"); T_POS+=("$pos")
  done; done; done; done
}
emit perception PERC POS_M
emit back_only BACK POS_B
NUM_TASKS=${#T_PLANNER[@]}
(( NUM_TASKS > 0 )) || { echo "[multi] no tasks derived from $CONFIG" >&2; exit 2; }

# ── pre-flight ────────────────────────────────────────────────────────────────
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
(( GPU_BASE + NUM_WORKERS <= GPU_COUNT )) || { echo "[multi] GPUs ${GPU_BASE}..$((GPU_BASE + NUM_WORKERS - 1)) exceed visible GPUs ($GPU_COUNT)" >&2; exit 2; }
for ((i=0; i<NUM_WORKERS; i++)); do
  port=$((PORT_BASE + i * PORT_STRIDE))
  if carla_port_in_use "$port"; then
    echo "[multi] ERROR: port $port (worker $i) is already in use" >&2; exit 2
  fi
done

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export FB_OUTPUT_ROOT
WORK_DIR="$FB_OUTPUT_ROOT/multi_gpu/$SCENE/$BATCH_ID_BASE"
mkdir -p "$WORK_DIR"

# Each task config sources the original config and narrows the matrix, so every
# other knob (REPEATS, REID_*, EXTRA_EPISODE_ARGS, ...) is inherited unchanged.
WORKER_NTASKS=()
for ((w=0; w<NUM_WORKERS; w++)); do WORKER_NTASKS+=(0); : >"$WORK_DIR/w${w}.tasks"; done
for ((t=0; t<NUM_TASKS; t++)); do
  w=$((t % NUM_WORKERS)); n="${WORKER_NTASKS[$w]}"
  cfg="$WORK_DIR/w${w}/t$(printf '%03d' "$n").sh"
  mkdir -p "$(dirname "$cfg")"
  {
    echo "source $(printf %q "$CONFIG")"
    if [[ "${T_GROUP[$t]}" == "perception" ]]; then
      echo "PLANNERS_WITH_PERCEPTION=\"${T_PLANNER[$t]}\""; echo 'PLANNERS_BACK_ONLY=""'
      echo "FOLLOW_POSITIONS_MAIN=\"${T_POS[$t]}\"";        echo 'FOLLOW_POSITIONS_BACK_ONLY=""'
    else
      echo 'PLANNERS_WITH_PERCEPTION=""';                  echo "PLANNERS_BACK_ONLY=\"${T_PLANNER[$t]}\""
      echo 'FOLLOW_POSITIONS_MAIN=""';                     echo "FOLLOW_POSITIONS_BACK_ONLY=\"${T_POS[$t]}\""
    fi
    echo "DIFFICULTIES=\"${T_DIFF[$t]}\""
    [[ -n "${T_FLOW[$t]}" ]] && echo "FLOW_MODES=\"${T_FLOW[$t]}\""
  } >"$cfg"
  printf '%s|%s/%s/%s%s/%s\n' "$cfg" "${T_GROUP[$t]}" "${T_PLANNER[$t]}" "${T_DIFF[$t]}" "${T_FLOW[$t]:+/${T_FLOW[$t]}}" "${T_POS[$t]}" >>"$WORK_DIR/w${w}.tasks"
  WORKER_NTASKS[$w]=$((n + 1))
done

{
  echo "scene=$SCENE"; echo "config=$CONFIG"; echo "batch_id_base=$BATCH_ID_BASE"
  echo "num_workers=$NUM_WORKERS num_tasks=$NUM_TASKS restart_every=$RESTART_EVERY"
  for ((w=0; w<NUM_WORKERS; w++)); do
    echo "worker $w: gpu=$((GPU_BASE + w)) port=$((PORT_BASE + w * PORT_STRIDE)) tasks=${WORKER_NTASKS[$w]}"
  done
} | tee "$WORK_DIR/meta.txt"

log() { printf '[multi] %s\n' "$*"; }

worker_loop() {  # widx gpu port
  local w="$1" gpu="$2" port="$3" logf="$WORK_DIR/w$1.log" rc_any=0 k=0 total cfg label rc
  total=$(wc -l <"$WORK_DIR/w$w.tasks")
  echo "[w$w] start gpu=$gpu port=$port tasks=$total $(date -Iseconds)" >>"$logf"
  while IFS='|' read -r cfg label; do
    k=$((k + 1))
    echo "[w$w][$k/$total] >>> $label $(date -Iseconds)" >>"$logf"
    bash "$WORKER_SH" --scene "$SCENE" --config "$cfg" --gpu "$gpu" --port "$port" \
      --batch-id "${BATCH_ID_BASE}_w$w" --restart-every "$RESTART_EVERY" \
      "${WORKER_PASSTHROUGH[@]+"${WORKER_PASSTHROUGH[@]}"}" >>"$logf" 2>&1
    rc=$?
    echo "[w$w][$k/$total] <<< rc=$rc $(date -Iseconds)" >>"$logf"
    (( rc != 0 )) && { rc_any=$rc; log "worker $w task $k/$total ($label) rc=$rc"; }
  done <"$WORK_DIR/w$w.tasks"
  return "$rc_any"
}

set -m   # each background worker gets its own process group
WORKER_PIDS=()
cleanup_workers() {
  local rc=$? pid i
  for pid in "${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 4
  for pid in "${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for ((i=0; i<NUM_WORKERS; i++)); do carla_stop "$((PORT_BASE + i * PORT_STRIDE))" >/dev/null 2>&1; done
  exit $rc
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((w=0; w<NUM_WORKERS; w++)); do
  (( WORKER_NTASKS[w] > 0 )) || { log "worker $w has no tasks; skipping"; continue; }
  log "spawning worker $w gpu=$((GPU_BASE + w)) port=$((PORT_BASE + w * PORT_STRIDE)) tasks=${WORKER_NTASKS[$w]} (tail -f $WORK_DIR/w$w.log)"
  worker_loop "$w" "$((GPU_BASE + w))" "$((PORT_BASE + w * PORT_STRIDE))" &
  WORKER_PIDS+=("$!")
  (( w + 1 < NUM_WORKERS )) && sleep "$WORKER_START_STAGGER_S"
done

overall_rc=0
for pid in "${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"}"; do wait "$pid" || overall_rc=1; done

merged="$WORK_DIR/manifest.csv"; : >"$merged"; rows=0
for ((w=0; w<NUM_WORKERS; w++)); do
  m="$FB_OUTPUT_ROOT/batch_manifests/$SCENE/${BATCH_ID_BASE}_w$w/manifest.csv"
  [[ -r "$m" ]] || continue
  [[ -s "$merged" ]] || head -n 1 "$m" >"$merged"
  tail -n +2 "$m" >>"$merged"; rows=$((rows + $(tail -n +2 "$m" | wc -l)))
done
log "merged manifest: $merged ($rows rows); overall rc=$overall_rc"
exit "$overall_rc"
