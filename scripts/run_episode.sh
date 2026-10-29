#!/usr/bin/env bash
# Run one FollowBench episode (or a teleop session) in a given scene.
#
# Usage:
#   scripts/run_episode.sh --scene <passage|corner|park|plaza> [--humanoid] [--teleop] [runner args...]
#
# Scene defaults live in configs/scenes/<scene>/episode.sh and can be tuned
# through environment variables (PLANNER, FOLLOW_POSITION, DESIRED_DISTANCE,
# DIFFICULTY, NUM_WALKERS, SIM_MODE, ...). Any extra arguments are appended to
# the Python runner's command line, so they override the preset.
#
# Examples:
#   scripts/run_episode.sh --scene corner
#   PLANNER=rda_traj DIFFICULTY=hard scripts/run_episode.sh --scene corner --use-perception
#   scripts/run_episode.sh --scene park --humanoid
#   ROBOT_TYPE=g1 scripts/run_episode.sh --scene plaza --teleop
#   scripts/run_episode.sh --scene passage -- --help      # runner help
#   FB_DRY_RUN=1 scripts/run_episode.sh --scene park      # print the command only
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

SCENE=""
HUMANOID=0
TELEOP=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)    SCENE="${2:?missing value for --scene}"; shift 2 ;;
    --humanoid) HUMANOID=1; shift ;;
    --teleop)   TELEOP=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    --)         shift; PASSTHROUGH+=("$@"); break ;;
    *)          PASSTHROUGH+=("$1"); shift ;;
  esac
done

PRESET="$REPO_ROOT/configs/scenes/$SCENE/episode.sh"
if [[ -z "$SCENE" || ! -f "$PRESET" ]]; then
  echo "[run_episode] --scene must be one of: passage corner park plaza (got '${SCENE}')" >&2
  exit 2
fi

FB_ASSETS_DIR="$REPO_ROOT/assets/scenes/$SCENE"
FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
FB_RUNS_DIR="${FB_RUNS_DIR:-$FB_OUTPUT_ROOT/runs/$SCENE}"
EPISODE_PRESET="${EPISODE_PRESET:-interactive}"

require_asset() {
  if [[ ! -f "$FB_ASSETS_DIR/$1" ]]; then
    echo "[$SCENE] ERROR: missing asset $FB_ASSETS_DIR/$1" >&2
    exit 1
  fi
}

# shellcheck source=/dev/null
source "$PRESET"

MODULE="${SCENE_RUNNER_MODULE:-followbench.scenes.$SCENE.run_episode}"
if [[ "$HUMANOID" == "1" ]]; then
  if [[ "${SCENE_HUMANOID_SUPPORTED:-0}" != "1" ]]; then
    echo "[run_episode] scene '$SCENE' has no humanoid runner (available: corner, park)" >&2
    exit 2
  fi
  MODULE="${MODULE}_humanoid"
fi

TELEOP_ARGS=()
if [[ "$TELEOP" == "1" ]]; then
  if [[ "$HUMANOID" == "1" ]]; then
    echo "[run_episode] --teleop and --humanoid cannot be combined (use ROBOT_TYPE=g1 for a humanoid teleop)" >&2
    exit 2
  fi
  if [[ "${SIM_MODE:-sync}" != "sync" ]]; then
    echo "[teleop] SIM_MODE must be sync; async teleoperation is not supported." >&2
    exit 2
  fi
  export PLANNER="${PLANNER:-pid}"
  export DEBUG="${DEBUG:-0}"
  TELEOP_ARGS=(
    --teleop
    --num-episodes "${NUM_EPISODES:-5}"
    --robot-type "${ROBOT_TYPE:-g1}"
    --sim-speed "${SIM_SPEED:-1.0}"
    --duration-sec "${DURATION_SEC:-300}"
  )
  if [[ "$SCENE" == "corner" ]]; then
    TELEOP_ARGS+=(--enable-evaluation --max-active-sec 0)
  fi
fi

EPISODE_ARGS=()
scene_episode_args

# CARLA_HOST / CARLA_PORT apply unless --host / --port are passed explicitly.
CONN_ARGS=()
has_arg() { local a; for a in "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"; do [[ "$a" == "$1" || "$a" == "$1="* ]] && return 0; done; return 1; }
if [[ -n "${CARLA_HOST:-}" ]] && ! has_arg --host; then CONN_ARGS+=(--host "$CARLA_HOST"); fi
if [[ -n "${CARLA_PORT:-}" ]] && ! has_arg --port; then CONN_ARGS+=(--port "$CARLA_PORT"); fi

CMD=("$PYTHON" -m "$MODULE"
  "${EPISODE_ARGS[@]}"
  "${CONN_ARGS[@]+"${CONN_ARGS[@]}"}"
  "${TELEOP_ARGS[@]+"${TELEOP_ARGS[@]}"}"
  "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}")
if [[ "${FB_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# Clean up a leftover episode of this scene. When a --port is given, only the
# episode bound to that CARLA port is killed, so parallel workers are safe.
CARLA_PORT_FOR_PKILL=""
prev=""
for arg in "${CONN_ARGS[@]+"${CONN_ARGS[@]}"}" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"; do
  [[ "$prev" == "--port" ]] && { CARLA_PORT_FOR_PKILL="$arg"; break; }
  prev="$arg"
done
PKILL_PATTERN="${MODULE//./\\.}"
if [[ -n "$CARLA_PORT_FOR_PKILL" ]]; then
  pkill -f "${PKILL_PATTERN}.*--port[ =]${CARLA_PORT_FOR_PKILL}\b" >/dev/null 2>&1 || true
else
  pkill -f "${PKILL_PATTERN}" >/dev/null 2>&1 || true
fi

cd "$REPO_ROOT"
exec "${CMD[@]}"
