#!/usr/bin/env bash
# Start a CARLA server (foreground) for interactive use.
#
# Usage:
#   tools/carla/run_carla.sh [CarlaUE4.sh args...]     e.g. -RenderOffScreen -carla-rpc-port=2000
#
# CARLA location: $CARLA_BIN, else $CARLA_ROOT/CarlaUE4.sh, else <repo>/CARLA_multi_robots/CarlaUE4.sh.
# If the RPC port (default 2000) is busy, only the server bound to that port is stopped.
set -euo pipefail

[[ "${1:-}" == -h || "${1:-}" == --help ]] && { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_BIN="${CARLA_BIN:-${CARLA_ROOT:+$CARLA_ROOT/CarlaUE4.sh}}"
CARLA_BIN="${CARLA_BIN:-$REPO_ROOT/CARLA_multi_robots/CarlaUE4.sh}"
[[ -x "$CARLA_BIN" ]] || { echo "[run_carla] CARLA not found at $CARLA_BIN (set CARLA_ROOT or CARLA_BIN)" >&2; exit 2; }

PORT=2000
for arg in "$@"; do
  [[ "$arg" == -carla-rpc-port=* ]] && PORT="${arg#-carla-rpc-port=}"
done
if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "(:|^)$PORT\$"; then
  echo "[run_carla] port $PORT in use - stopping the server bound to it"
  bash "$REPO_ROOT/tools/carla/kill_carla.sh" --port "$PORT"
fi

cd "$(dirname "$CARLA_BIN")"
exec "./$(basename "$CARLA_BIN")" "$@"
