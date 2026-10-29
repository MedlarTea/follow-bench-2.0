#!/usr/bin/env bash
# Run several multi-GPU batch evaluations back to back on the same GPU pool.
#
# Usage:
#   scripts/multi_gpu/run_pipeline.sh --num-workers N (--preset NAME | --job SCENE:CONFIG ...) [options]
#
# Presets:
#   full            corner:default_50x passage:back_only_50x park:e2e_default_50x plaza:back_only_50x
#   rda_traj_kpr    <scene>:rda_traj_kpr_50x for corner passage park plaza
#   plaza_default   plaza:default_50x
#
# Options:
#   --job SCENE:CONFIG         Add a job (repeatable; appended after the preset).
#   --only SCENE ...           Keep only jobs of these scenes (until the next option).
#   --batch-id-base NAME       Job batch ids are <NAME>_<scene> (default pipeline_<ts>).
#   --stop-on-failure          Abort the remaining jobs after a failure.
#   Any other option (--gpu-base, --port-base, --port-stride, --start-stagger-s,
#   --restart-every, --carla-bin, --container, ...) is forwarded to eval_multi_gpu.sh.
#
# Pipeline log + summary: $FB_OUTPUT_ROOT/pipelines/<batch-id-base>/
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MULTI_SH="$REPO_ROOT/scripts/multi_gpu/eval_multi_gpu.sh"
usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

JOBS=() ONLY=() FORWARD=() BATCH_ID_BASE="" STOP_ON_FAILURE=0
preset_jobs() {
  case "$1" in
    full)          JOBS+=(corner:default_50x passage:back_only_50x park:e2e_default_50x plaza:back_only_50x) ;;
    rda_traj_kpr)  JOBS+=(corner:rda_traj_kpr_50x passage:rda_traj_kpr_50x park:rda_traj_kpr_50x plaza:rda_traj_kpr_50x) ;;
    plaza_default) JOBS+=(plaza:default_50x) ;;
    *) echo "[pipeline] unknown preset: $1" >&2; exit 2 ;;
  esac
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)          preset_jobs "${2:?}"; shift 2 ;;
    --job)             JOBS+=("${2:?}"); shift 2 ;;
    --only)            shift; while [[ $# -gt 0 && "$1" != --* ]]; do ONLY+=("$1"); shift; done ;;
    --batch-id-base)   BATCH_ID_BASE="${2:?}"; shift 2 ;;
    --stop-on-failure) STOP_ON_FAILURE=1; shift ;;
    --no-sudo)         FORWARD+=("$1"); shift ;;
    -h|--help)         usage; exit 0 ;;
    --*)               FORWARD+=("$1" "${2:?missing value for $1}"); shift 2 ;;
    *) echo "[pipeline] unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if (( ${#ONLY[@]} > 0 )); then
  kept=()
  for j in "${JOBS[@]}"; do [[ " ${ONLY[*]} " == *" ${j%%:*} "* ]] && kept+=("$j"); done
  JOBS=("${kept[@]+"${kept[@]}"}")
fi
(( ${#JOBS[@]} > 0 )) || { echo "[pipeline] no jobs (use --preset or --job)" >&2; usage >&2; exit 2; }
for j in "${JOBS[@]}"; do
  s="${j%%:*}" c="${j#*:}"
  [[ -f "$REPO_ROOT/configs/scenes/$s/batch/$c.sh" || -f "$c" ]] || { echo "[pipeline] bad job $j (config not found)" >&2; exit 2; }
done
[[ -n "$BATCH_ID_BASE" ]] || BATCH_ID_BASE="pipeline_$(date +%Y%m%d_%H%M%S)"

FB_OUTPUT_ROOT="${FB_OUTPUT_ROOT:-$REPO_ROOT/outputs}"
export FB_OUTPUT_ROOT
WORK_DIR="$FB_OUTPUT_ROOT/pipelines/$BATCH_ID_BASE"
mkdir -p "$WORK_DIR"
SUMMARY="$WORK_DIR/summary.txt"
printf '%-8s | %-26s | %-38s | %-4s | %s\n' scene config batch_id rc elapsed >"$SUMMARY"
log() { printf '[pipeline] %s\n' "$*"; }

set -m
CHILD=""
trap '[[ -n "$CHILD" ]] && { kill -TERM -- "-$CHILD" 2>/dev/null; sleep 4; kill -KILL -- "-$CHILD" 2>/dev/null; }' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

worst=0
for j in "${JOBS[@]}"; do
  s="${j%%:*}" c="${j#*:}" bid="${BATCH_ID_BASE}_${j%%:*}"
  log "=== scene=$s config=$c batch_id=$bid (log: $WORK_DIR/$s.log)"
  start=$(date +%s)
  bash "$MULTI_SH" --scene "$s" --config "$c" --batch-id "$bid" "${FORWARD[@]+"${FORWARD[@]}"}" >>"$WORK_DIR/$s.log" 2>&1 &
  CHILD=$!
  wait "$CHILD"; rc=$?; CHILD=""
  el=$(( $(date +%s) - start ))
  printf '%-8s | %-26s | %-38s | %-4d | %dh%02dm\n' "$s" "$c" "$bid" "$rc" $((el / 3600)) $((el % 3600 / 60)) >>"$SUMMARY"
  if (( rc != 0 )); then
    worst=$rc; log "scene=$s FAILED rc=$rc"
    (( STOP_ON_FAILURE )) && break
  fi
done
cat "$SUMMARY"
exit "$worst"
