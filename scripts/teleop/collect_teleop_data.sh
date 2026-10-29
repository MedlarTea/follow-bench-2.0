#!/usr/bin/env bash
# Human-teleoperation data collection sessions.
#
# Usage:
#   scripts/teleop/collect_teleop_data.sh --scene SCENE [DIFFICULTY] [runner args...]
#   scripts/teleop/collect_teleop_data.sh --all [runner args...]      # 4 scenes x 4 difficulties
#
# Environment: NUM_EPISODES (5), ROBOT_TYPE (g1), SIM_SPEED (1.0), DURATION_SEC (300),
# OPERATOR_ID (anonymous), FOLLOW_POSITION (back), DESIRED_DISTANCE (1.5),
# WEATHER (random), NUM_WALKERS (scene preset), PRINT_CONFIG=1 (print and exit).
# --all also honours FOLLOW_POSITIONS="back left_side ..." and BACK_DISTANCE / SIDE_DISTANCE.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { usage; exit 0; }

if [[ "${1:-}" == "--all" ]]; then
  shift
  FOLLOW_POSITIONS="${FOLLOW_POSITIONS:-${FOLLOW_POSITION:-back}}"
  BACK_DISTANCE="${BACK_DISTANCE:-${DESIRED_DISTANCE:-1.5}}"
  SIDE_DISTANCE="${SIDE_DISTANCE:-${DESIRED_DISTANCE:-1.5}}"
  for position in $FOLLOW_POSITIONS; do
    case "$position" in back|left_side|right_side) ;; *) echo "[collect] invalid follow position: $position" >&2; exit 2 ;; esac
    distance="$SIDE_DISTANCE"; [[ "$position" == back ]] && distance="$BACK_DISTANCE"
    for scene in plaza corner park passage; do
      for difficulty in easy hard boss legend; do
        echo "================ [collect] $scene $difficulty position=$position distance=$distance ================"
        FOLLOW_POSITION="$position" DESIRED_DISTANCE="$distance" \
          bash "${BASH_SOURCE[0]}" --scene "$scene" "$difficulty" "$@"
      done
    done
  done
  echo "[collect] all teleoperation groups completed."
  exit 0
fi

[[ "${1:-}" == "--scene" ]] || { usage >&2; exit 2; }
SCENE="${2:?missing scene}"; shift 2
DIFFICULTY="${DIFFICULTY:-easy}"
if [[ $# -gt 0 && "$1" != --* ]]; then DIFFICULTY="$1"; shift; fi
case "$DIFFICULTY" in easy|hard|boss|legend) ;; *) echo "[collect] difficulty must be easy|hard|boss|legend" >&2; exit 2 ;; esac

export NUM_EPISODES="${NUM_EPISODES:-5}" ROBOT_TYPE="${ROBOT_TYPE:-g1}" SIM_SPEED="${SIM_SPEED:-1.0}"
export SIM_MODE="${SIM_MODE:-sync}" DURATION_SEC="${DURATION_SEC:-300}"
export FOLLOW_POSITION="${FOLLOW_POSITION:-back}" DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"
OPERATOR_ID="${OPERATOR_ID:-anonymous}"
WEATHER="${WEATHER:-random}"   # five-episode sessions cycle all five weather presets
[[ "$SIM_MODE" == sync ]] || { echo "Error: teleop requires SIM_MODE=sync." >&2; exit 2; }

ARGS=()
case "$SCENE" in
  corner)
    case "$DIFFICULTY" in
      easy)   W=8;  MIN=0.50; MAX=1.00; BAND=4.0; SEP=2.0 ;;
      hard)   W=12; MIN=0.70; MAX=1.30; BAND=3.0; SEP=1.4 ;;
      boss)   W=16; MIN=0.90; MAX=1.55; BAND=2.5; SEP=1.0 ;;
      legend) W=20; MIN=1.10; MAX=1.80; BAND=2.0; SEP=0.8 ;;
    esac
    export NUM_WALKERS="${NUM_WALKERS:-$W}" DIFFICULTY
    echo "[collect / Street Corner] walkers=$NUM_WALKERS speed=[$MIN,$MAX] lateral_half_width=$BAND min_spawn_separation=$SEP"
    ARGS=(--operator-id "$OPERATOR_ID" --num-walkers "$NUM_WALKERS" --npc-min-speed "$MIN" --npc-max-speed "$MAX"
          --npc-pair-target-band-half-width-m "$BAND" --npc-spawn-min-sep-m "$SEP")
    ;;
  passage)
    case "$DIFFICULTY" in
      easy)   CAP=2; MIN=0.80; MAX=1.04 ;;
      hard)   CAP=4; MIN=0.88; MAX=1.20 ;;
      boss)   CAP=6; MIN=0.96; MAX=1.36 ;;
      legend) CAP=8; MIN=1.12; MAX=1.52 ;;
    esac
    # max_concurrent counts every walker incl. the target, so 8 walkers make the legend cap reachable.
    export NUM_WALKERS="${NUM_WALKERS:-8}"
    (( NUM_WALKERS >= CAP )) || { echo "Error: NUM_WALKERS=$NUM_WALKERS < concurrency cap $CAP for $DIFFICULTY" >&2; exit 2; }
    echo "[collect / Narrow Passage] walkers=$NUM_WALKERS max_concurrent=$CAP speed=[$MIN,$MAX]"
    ARGS=(--difficulty "$DIFFICULTY" --num-walkers "$NUM_WALKERS" --npc-min-speed "$MIN" --npc-max-speed "$MAX"
          --weather "$WEATHER" --operator-id "$OPERATOR_ID")
    ;;
  park)
    export NUM_WALKERS="${NUM_WALKERS:-30}"
    echo "[collect / Cluttered Park] walkers=$NUM_WALKERS (crowd behaviour from the '$DIFFICULTY' NPC preset)"
    ARGS=(--difficulty "$DIFFICULTY" --weather "$WEATHER" --operator-id "$OPERATOR_ID")
    ;;
  plaza)
    case "$DIFFICULTY" in easy) W=10 ;; hard) W=15 ;; boss) W=20 ;; legend) W=25 ;; esac
    export NUM_WALKERS="${NUM_WALKERS:-$W}" DIFFICULTY FLOW_MODE="${FLOW_MODE:-random}"
    echo "[collect / Open Plaza] walkers=$NUM_WALKERS crowd_speed=[0.60,1.60] target_speed=[0.80,1.40] flow=$FLOW_MODE"
    ARGS=(--operator-id "$OPERATOR_ID" --weather "$WEATHER")
    ;;
  *) echo "[collect] --scene must be one of: corner passage park plaza" >&2; exit 2 ;;
esac
echo "[collect] scene=$SCENE difficulty=$DIFFICULTY episodes=$NUM_EPISODES follow=$FOLLOW_POSITION distance=$DESIRED_DISTANCE operator=$OPERATOR_ID weather=$WEATHER"
echo "[Teleop] robot=$ROBOT_TYPE sim_speed=$SIM_SPEED duration=${DURATION_SEC}s"
[[ "${PRINT_CONFIG:-0}" == 1 ]] && exit 0

exec bash "$REPO_ROOT/scripts/run_episode.sh" --scene "$SCENE" --teleop "${ARGS[@]}" "$@"
