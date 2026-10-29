#!/usr/bin/env bash
# One batch-evaluation worker pinned to a GPU: owns its CARLA server and runs
# scripts/run_batch_eval.sh against it, recycling CARLA every N slots.
#
# Usage:
#   scripts/multi_gpu/run_worker.sh --scene SCENE --config NAME|PATH --gpu N --port P [options]
#
# Options:
#   --batch-id ID              Batch id (default: <scene>_worker_<timestamp>); reuse to resume.
#   --restart-every N          Recycle CARLA every N executed slots (default 1 = fresh server per
#                              episode; 0 = only when it crashes).
#   --carla-bin PATH           CarlaUE4.sh (default: $CARLA_BIN, $CARLA_ROOT/CarlaUE4.sh, <repo>/CARLA_multi_robots).
#   --carla-graphics-adapter N Override the auto-detected Vulkan adapter.
#   --container NAME           Run the evaluation inside this Docker container while CARLA
#                              stays on the host (default: run locally).
#   --docker-host-ip IP        Host address seen from the container (default 172.17.0.1).
#   --no-sudo                  Do not prefix docker with sudo.
#
# Container mode expects the repo at the same path inside the container and a
# conda env given by CONTAINER_CONDA_SH / CONTAINER_CONDA_ENV
# (defaults: /root/miniconda3/etc/profile.d/conda.sh, multiRobots).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/carla.sh
source "$REPO_ROOT/scripts/lib/carla.sh"

usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

SCENE="" CONFIG="" GPU="" PORT="" BATCH_ID="" RESTART_EVERY="${RESTART_EVERY:-1}"
# client.load_world() intermittently hangs this CARLA build's render thread.
# A fresh server already holds Town10HD_Opt, so episodes skip the reload when
# the world is clean (followbench/sim/carla_world.py). With --restart-every 1
# (default) no batch episode ever calls load_world.
export FB_LOAD_MAP_POLICY="${FB_LOAD_MAP_POLICY:-if_needed}"
CARLA_BIN="${CARLA_BIN:-${CARLA_ROOT:+$CARLA_ROOT/CarlaUE4.sh}}"
CARLA_BIN="${CARLA_BIN:-$REPO_ROOT/CARLA_multi_robots/CarlaUE4.sh}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-}"
CONTAINER="${CONTAINER:-}"
DOCKER_HOST_IP="${DOCKER_HOST_IP:-172.17.0.1}"
SUDO_BIN="${SUDO_BIN:-sudo}"
CONTAINER_CONDA_SH="${CONTAINER_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONTAINER_CONDA_ENV="${CONTAINER_CONDA_ENV:-multiRobots}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)                  SCENE="${2:?}"; shift 2 ;;
    --config)                 CONFIG="${2:?}"; shift 2 ;;
    --gpu)                    GPU="${2:?}"; shift 2 ;;
    --port)                   PORT="${2:?}"; shift 2 ;;
    --batch-id)               BATCH_ID="${2:?}"; shift 2 ;;
    --restart-every)          RESTART_EVERY="${2:?}"; shift 2 ;;
    --carla-bin)              CARLA_BIN="${2:?}"; shift 2 ;;
    --carla-graphics-adapter) CARLA_GRAPHICS_ADAPTER="${2:?}"; shift 2 ;;
    --container)              CONTAINER="${2:?}"; shift 2 ;;
    --docker-host-ip)         DOCKER_HOST_IP="${2:?}"; shift 2 ;;
    --no-sudo)                SUDO_BIN=""; shift ;;
    -h|--help)                usage; exit 0 ;;
    *) echo "[worker] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for v in SCENE CONFIG GPU PORT; do
  [[ -n "${!v}" ]] || { echo "[worker] --${v,,} is required" >&2; usage >&2; exit 2; }
done
[[ -n "$CARLA_BIN" ]] || { echo "[worker] set --carla-bin, CARLA_BIN or CARLA_ROOT" >&2; exit 2; }
[[ -n "$BATCH_ID" ]] || BATCH_ID="${SCENE}_worker_$(date +%Y%m%d_%H%M%S)"
[[ -f "$CONFIG" ]] && CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
if carla_port_in_use "$PORT"; then
  echo "[worker] ERROR: port $PORT is already in use; stop that server first (tools/carla/kill_carla.sh --port $PORT)" >&2
  exit 2
fi

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
LOG_DIR="$FB_OUTPUT_ROOT/batch_manifests/$SCENE/$BATCH_ID"
mkdir -p "$LOG_DIR"
CARLA_LOG="$LOG_DIR/carla_gpu${GPU}_port${PORT}.log"
log() { printf '[worker] %s\n' "$*"; }

trap 'rc=$?; log "cleanup (rc=$rc)"; carla_stop "$PORT"; exit $rc' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_inner_batch() {  # max_slots
  local inner=(bash scripts/run_batch_eval.sh --scene "$SCENE" --config "$CONFIG"
               --batch-id "$BATCH_ID" --no-launch-carla)
  (( $1 > 0 )) && inner+=(--max-slots "$1")
  if [[ -z "$CONTAINER" ]]; then
    log "local -> ${inner[*]}"
    (cd "$REPO_ROOT" && env CARLA_HOST=127.0.0.1 CARLA_PORT="$PORT" CUDA_VISIBLE_DEVICES="$GPU" \
       SDL_VIDEODRIVER=dummy FB_OUTPUT_ROOT="$FB_OUTPUT_ROOT" "${inner[@]}")
  else
    log "docker exec $CONTAINER -> ${inner[*]}"
    $SUDO_BIN docker exec \
      -e CARLA_HOST="$DOCKER_HOST_IP" -e CARLA_PORT="$PORT" -e CUDA_VISIBLE_DEVICES="$GPU" \
      -e SDL_VIDEODRIVER=dummy -e FB_OUTPUT_ROOT="$FB_OUTPUT_ROOT" \
      "$CONTAINER" bash -lc "source $(printf %q "$CONTAINER_CONDA_SH") && conda activate $(printf %q "$CONTAINER_CONDA_ENV") \
        && cd $(printf %q "$REPO_ROOT") && $(printf '%q ' "${inner[@]}")"
  fi
}

log "scene=$SCENE config=$CONFIG batch_id=$BATCH_ID gpu=$GPU port=$PORT restart_every=$RESTART_EVERY container=${CONTAINER:-<local>}"

# Exit 99 from the inner batch means either "--max-slots reached" or "CARLA
# died" (watchdog); both are handled by recycling CARLA and resuming the same
# batch id. Give up after MAX_STALLED_CHUNKS consecutive chunks without a new
# completed trial, so a permanently broken server cannot loop forever.
MAX_STALLED_CHUNKS="${MAX_STALLED_CHUNKS:-3}"
EVAL_DIR="$FB_OUTPUT_ROOT/logs_batch/$SCENE/$BATCH_ID"
n_done() { find "$EVAL_DIR" -name eval_result.json 2>/dev/null | wc -l; }

chunk=0 stalled=0
while true; do
  chunk=$((chunk + 1))
  log "===== chunk #$chunk (max_slots=${RESTART_EVERY}) ====="
  before=$(n_done)
  carla_launch "$CARLA_BIN" "$GPU" "$PORT" "$CARLA_LOG" "$CARLA_GRAPHICS_ADAPTER" || exit 2
  run_inner_batch "$RESTART_EVERY"
  rc=$?
  carla_stop "$PORT"
  case $rc in
    0)  log "config exhausted after $chunk chunk(s)"; exit 0 ;;
    99)
      if (( $(n_done) > before )); then stalled=0; else stalled=$((stalled + 1)); fi
      if (( stalled >= MAX_STALLED_CHUNKS )); then
        log "no progress in $stalled consecutive chunk(s); aborting"; exit 99
      fi
      log "chunk $chunk ended (rc=99); recycling CARLA"; sleep 3 ;;
    *)  log "batch failed (rc=$rc); aborting"; exit $rc ;;
  esac
done
