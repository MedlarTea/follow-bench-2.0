#!/usr/bin/env bash
# Run the full baseline matrix: every baseline row on every scene, `back` follow,
# D=1.5, all difficulties. One CARLA server on one GPU; resume-safe (re-run to
# continue, completed trials are skipped). For several GPUs, start one instance
# per GPU with disjoint --scenes / --rows and distinct --gpu / --port.
#
# Usage:
#   scripts/baselines/run_all_baselines.sh [options]
#
# Options:
#   --trials 50|10           Trials per cell -> configs baselines_<row>_<trials>x (default 50)
#   --scenes "a b ..."       Subset of: corner passage park plaza (default: all)
#   --rows "a b ..."         Subset of: traj_r18 traj_kpr ds_r18 mpc_r18 e2e (default: all)
#   --gpu N --port P         GPU / CARLA port (default 0 / 2000)
#   --carla-bin PATH         CarlaUE4.sh (default: $CARLA_BIN, $CARLA_ROOT/CarlaUE4.sh, <repo>/CARLA_multi_robots)
#   --carla-graphics-adapter N   Vulkan adapter override (auto-detected otherwise)
#   --skip-gate              Skip the 1-episode TrackVLA / OA-VAT import gate
#   --keep-carla             Reuse a CARLA already listening on --port
#
# Batch ids: baselines<trials>_<scene>_<row>; logs under $FB_OUTPUT_ROOT/logs_batch/<scene>/.
# Driver log: $FB_OUTPUT_ROOT/baselines/gpu<N>.log
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/carla.sh
source "$REPO_ROOT/scripts/lib/carla.sh"
usage() { sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

TRIALS=50 SCENES="corner passage park plaza" ROWS="traj_r18 traj_kpr ds_r18 mpc_r18 e2e"
GPU="${GPU:-0}" PORT="${PORT:-2000}" SKIP_GATE=0 KEEP_CARLA=0
CARLA_BIN="${CARLA_BIN:-${CARLA_ROOT:+$CARLA_ROOT/CarlaUE4.sh}}"
CARLA_BIN="${CARLA_BIN:-$REPO_ROOT/CARLA_multi_robots/CarlaUE4.sh}"
CARLA_GRAPHICS_ADAPTER="${CARLA_GRAPHICS_ADAPTER:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trials)                 TRIALS="${2:?}"; shift 2 ;;
    --scenes)                 SCENES="${2:?}"; shift 2 ;;
    --rows)                   ROWS="${2:?}"; shift 2 ;;
    --gpu)                    GPU="${2:?}"; shift 2 ;;
    --port)                   PORT="${2:?}"; shift 2 ;;
    --carla-bin)              CARLA_BIN="${2:?}"; shift 2 ;;
    --carla-graphics-adapter) CARLA_GRAPHICS_ADAPTER="${2:?}"; shift 2 ;;
    --skip-gate)              SKIP_GATE=1; shift ;;
    --keep-carla)             KEEP_CARLA=1; shift ;;
    -h|--help)                usage; exit 0 ;;
    *) echo "[baselines] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$TRIALS" == 50 || "$TRIALS" == 10 ]] || { echo "[baselines] --trials must be 50 or 10" >&2; exit 2; }
[[ -n "$CARLA_BIN" ]] || { echo "[baselines] set --carla-bin, CARLA_BIN or CARLA_ROOT" >&2; exit 2; }

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export FB_OUTPUT_ROOT
WORK_DIR="$FB_OUTPUT_ROOT/baselines"
mkdir -p "$WORK_DIR"
LOG="$WORK_DIR/gpu${GPU}.log"
CARLA_LOG="$WORK_DIR/gpu${GPU}_carla.log"
exec > >(tee -a "$LOG") 2>&1

export CUDA_VISIBLE_DEVICES="$GPU" CARLA_HOST=127.0.0.1 CARLA_PORT="$PORT"
export SDL_VIDEODRIVER=dummy PYTHONNOUSERSITE=1
export CARLA_CLIENT_TIMEOUT_S="${CARLA_CLIENT_TIMEOUT_S:-180}"

start_carla() {
  carla_port_in_use "$PORT" && { echo "[baselines] CARLA already listening on $PORT"; return 0; }
  carla_launch "$CARLA_BIN" "$GPU" "$PORT" "$CARLA_LOG" "$CARLA_GRAPHICS_ADAPTER" || return 1
  # A wrong Vulkan adapter makes the render thread abort ~60 s after start.
  sleep 70
  carla_port_in_use "$PORT" || { echo "[baselines] CARLA died during startup; see $CARLA_LOG" >&2; return 1; }
  echo "[baselines] CARLA ready"
}

run_job() {  # scene config batch_id
  local attempt rc
  for attempt in 1 2 3 4 5 6; do
    start_carla || return 1
    echo "[baselines] $(date -Iseconds) start $3 (scene=$1 config=$2 attempt=$attempt)"
    bash "$REPO_ROOT/scripts/run_batch_eval.sh" --scene "$1" --config "$2" --batch-id "$3" --no-launch-carla
    rc=$?
    echo "[baselines] $(date -Iseconds) end $3 exit=$rc"
    (( rc == 0 )) && return 0
    echo "[baselines] recycling CARLA on port $PORT after exit $rc"
    carla_stop "$PORT"; sleep 3
  done
  return 1
}

trap 'carla_stop "$PORT" >/dev/null 2>&1' EXIT
(( KEEP_CARLA )) || carla_stop "$PORT" >/dev/null 2>&1
start_carla || exit 1

# Gate: one easy episode per end-to-end model so a broken install fails fast.
if (( ! SKIP_GATE )) && [[ " $ROWS " == *" e2e "* ]]; then
  for model in trackvla oa_vat; do
    cat >"$WORK_DIR/gate_$model.sh" <<EOF
PLANNERS_WITH_PERCEPTION=""
PLANNERS_BACK_ONLY="$model"
DIFFICULTIES="easy"
FOLLOW_POSITIONS_BACK_ONLY="back"
DESIRED_DISTANCE="1.5"
REPEATS="1"
RESTART_CARLA_EVERY_N_SLOTS="0"
RESUME="0"
STOP_ON_FAILURE="1"
EXTRA_EPISODE_ARGS="--no-ui --max-active-sec 45 --duration-sec 400"
EOF
    run_job corner "$WORK_DIR/gate_$model.sh" "baselines_gate_$model" || { echo "[baselines] gate failed for $model" >&2; exit 1; }
  done
fi

for scene in $SCENES; do
  for row in $ROWS; do
    cfg="baselines_${row}_${TRIALS}x"
    if [[ ! -f "$REPO_ROOT/configs/scenes/$scene/batch/$cfg.sh" ]]; then
      echo "[baselines] missing config configs/scenes/$scene/batch/$cfg.sh; skipping" | tee -a "$WORK_DIR/failed_jobs.txt"
      continue
    fi
    run_job "$scene" "$cfg" "baselines${TRIALS}_${scene}_${row}" \
      || echo "[baselines] job failed: $scene $cfg" | tee -a "$WORK_DIR/failed_jobs.txt"
  done
done
echo "[baselines] $(date -Iseconds) all jobs finished"
