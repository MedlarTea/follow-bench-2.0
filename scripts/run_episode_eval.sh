#!/usr/bin/env bash
# Run one evaluated episode, then score it and render its plots.
#
# Usage:
#   scripts/run_episode_eval.sh --scene SCENE [runner args...]
#
# Environment: PLANNER, FOLLOW_POSITION, DESIRED_DISTANCE, NUM_WALKERS, ... (see
# scripts/run_episode.sh), AUTO_FOLLOW=1 to auto-spawn and auto-follow
# (otherwise press G / F in the UI). Logs go to $FB_OUTPUT_ROOT/eval_logs/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
usage() { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
[[ "${1:-}" == -h || "${1:-}" == --help ]] && { usage; exit 0; }
[[ "${1:-}" == "--scene" && -n "${2:-}" ]] || { usage >&2; exit 2; }
SCENE="$2"; shift 2

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
EVAL_ROOT="$FB_OUTPUT_ROOT/eval_logs"
latest_trial() {  # empty (not an error) before the first episode of a scene
  [[ -d "$EVAL_ROOT/$SCENE" ]] || return 0
  find "$EVAL_ROOT/$SCENE" -mindepth 3 -maxdepth 3 -type d -name 'trial_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true
}

EXTRA=()
[[ "${AUTO_FOLLOW:-0}" == 1 ]] && EXTRA+=(--auto-spawn-sec 0.5 --auto-follow-sec 2.0)
PREV="$(latest_trial)"
bash "$REPO_ROOT/scripts/run_episode.sh" --scene "$SCENE" --enable-evaluation \
  --eval-log-root "$EVAL_ROOT" "${EXTRA[@]+"${EXTRA[@]}"}" "$@"
RUN_DIR="$(latest_trial)"
if [[ -z "$RUN_DIR" || "$RUN_DIR" == "$PREV" ]]; then
  echo "[episode-eval] no new trial was recorded (press F to start following, or set AUTO_FOLLOW=1)."
  exit 0
fi

cd "$REPO_ROOT"
echo "[episode-eval] run: $RUN_DIR"
"$PYTHON" -m followbench.evaluation.evaluator --run-dir "$RUN_DIR"
[[ -f "$RUN_DIR/eval_summary.txt" ]] && cat "$RUN_DIR/eval_summary.txt"
"$PYTHON" -m followbench.visualization.plot_episode --run-dir "$RUN_DIR"
echo "[episode-eval] plots: $RUN_DIR/plots"
