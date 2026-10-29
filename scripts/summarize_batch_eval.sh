#!/usr/bin/env bash
# Evaluate and tabulate a finished batch: (re)computes missing eval_result.json,
# writes one summary table per setting and comparison plots per difficulty
# (and flow mode) / follow position.
#
# Usage:
#   scripts/summarize_batch_eval.sh --scene SCENE [--batch-id ID] [--config NAME|PATH] [--with-episode-plots]
#
# --batch-id defaults to the most recent batch of the scene. The config defaults
# to the batch's own config_snapshot.sh, so it normally does not need to be given.
# Outputs: $FB_OUTPUT_ROOT/results/{tables,compare}/<scene>_<batch_id>_*.{csv,png}
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
usage() { sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

SCENE="" BATCH_ID="" CONFIG_ARG="" WITH_EPISODE_PLOTS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)              SCENE="${2:?}"; shift 2 ;;
    --batch-id)           BATCH_ID="${2:?}"; shift 2 ;;
    --config)             CONFIG_ARG="${2:?}"; shift 2 ;;
    --with-episode-plots) WITH_EPISODE_PLOTS=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "[BATCH-SUM] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
SCENE_PRESET="$REPO_ROOT/configs/scenes/$SCENE/episode.sh"
[[ -f "$SCENE_PRESET" ]] || { echo "[BATCH-SUM] --scene must be one of: passage corner park plaza" >&2; exit 2; }

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
LOGS_BASE="$FB_OUTPUT_ROOT/logs_batch/$SCENE"
RESULTS="$FB_OUTPUT_ROOT/results"
if [[ -z "$BATCH_ID" ]]; then
  BATCH_ID="$(find "$LOGS_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
fi
BATCH_ROOT="$LOGS_BASE/$BATCH_ID"
[[ -n "$BATCH_ID" && -d "$BATCH_ROOT" ]] || { echo "[BATCH-SUM] batch not found under $LOGS_BASE: ${BATCH_ID:-<none>}" >&2; exit 2; }

if [[ -n "$CONFIG_ARG" ]]; then
  [[ -f "$CONFIG_ARG" ]] && CONFIG="$CONFIG_ARG" || CONFIG="$REPO_ROOT/configs/scenes/$SCENE/batch/${CONFIG_ARG%.sh}.sh"
else
  CONFIG="$FB_OUTPUT_ROOT/batch_manifests/$SCENE/$BATCH_ID/config_snapshot.sh"
fi
[[ -f "$CONFIG" ]] || { echo "[BATCH-SUM] config not found: $CONFIG (pass --config)" >&2; exit 2; }

require_asset() { :; }
# shellcheck source=/dev/null
source "$SCENE_PRESET"
# shellcheck source=/dev/null
source "$CONFIG"
scene_batch_defaults
DESIRED_DISTANCE="${DESIRED_DISTANCE:-1.5}"
read -r -a planners_perc  <<<"${PLANNERS_WITH_PERCEPTION:-}"
read -r -a planners_back  <<<"${PLANNERS_BACK_ONLY:-}"
read -r -a positions_main <<<"${FOLLOW_POSITIONS_MAIN:-}"
read -r -a positions_back <<<"${FOLLOW_POSITIONS_BACK_ONLY:-}"
read -r -a difficulties   <<<"${DIFFICULTIES:-}"
flow_modes=("")
[[ -n "${SCENE_FLOW_MODES:-}" ]] && read -r -a flow_modes <<<"${FLOW_MODES:-random}"
DIST_LABEL="$(python3 -c 'import sys; t=f"{float(sys.argv[1]):.2f}".rstrip("0").rstrip("."); print(t or "0")' "$DESIRED_DISTANCE")"
mkdir -p "$RESULTS/tables" "$RESULTS/compare"
log() { printf '[BATCH-SUM] %s\n' "$*"; }

cd "$REPO_ROOT"
summarize_setting() {  # planner follow difficulty flow
  local planner="$1" follow="$2" difficulty="$3" flow="$4" humans log_root config_dir run_dir tag
  humans=$(( $(scene_walkers_for "$difficulty") - 1 ))
  log_root="$BATCH_ROOT/$difficulty${flow:+/$flow}"
  config_dir="$log_root/$SCENE/$planner/H${humans}_${follow}_D${DIST_LABEL}"
  [[ -d "$config_dir" ]] || { log "skip missing $config_dir"; return 0; }
  for run_dir in "$config_dir"/trial_*/; do
    [[ -d "$run_dir" ]] || continue
    if [[ ! -f "$run_dir/eval_result.json" || ! -f "$run_dir/eval_summary.txt" ]]; then
      "$PYTHON" -m followbench.evaluation.evaluator --run-dir "$run_dir"
    fi
    if [[ "$WITH_EPISODE_PLOTS" == 1 ]]; then
      "$PYTHON" -m followbench.visualization.plot_episode --run-dir "$run_dir"
    fi
  done
  tag="${SCENE}_${BATCH_ID}_${planner}_${difficulty}${flow:+_$flow}_H${humans}_${follow}_D${DIST_LABEL}"
  "$PYTHON" -m followbench.visualization.export_table --log-root "$log_root" --scenario "$SCENE" \
    --planners "$planner" --humans "$humans" --follow-position "$follow" --distance "$DESIRED_DISTANCE" \
    --output "$RESULTS/tables/${tag}_summary.csv"
  log "table=$RESULTS/tables/${tag}_summary.csv"
}

for d in "${difficulties[@]}"; do
  for f in "${flow_modes[@]}"; do
    for p in "${planners_perc[@]}"; do for pos in "${positions_main[@]}"; do summarize_setting "$p" "$pos" "$d" "$f"; done; done
    for p in "${planners_back[@]}"; do for pos in "${positions_back[@]}"; do summarize_setting "$p" "$pos" "$d" "$f"; done; done

    humans=$(( $(scene_walkers_for "$d") - 1 ))
    for pos in back left_side right_side; do
      compare=()
      [[ " ${positions_main[*]} " == *" $pos "* ]] && compare+=("${planners_perc[@]}")
      [[ " ${positions_back[*]} " == *" $pos "* ]] && compare+=("${planners_back[@]}")
      (( ${#compare[@]} > 0 )) || continue
      out="$RESULTS/compare/${SCENE}_${BATCH_ID}_${d}${f:+_$f}_H${humans}_${pos}_D${DIST_LABEL}.png"
      "$PYTHON" -m followbench.visualization.plot_compare --log-root "$BATCH_ROOT/$d${f:+/$f}" --scenario "$SCENE" \
        --planners "${compare[@]}" --humans "$humans" --follow-position "$pos" --distance "$DESIRED_DISTANCE" --output "$out"
      log "compare=$out"
    done
  done
done
log "done batch_id=$BATCH_ID"
