#!/usr/bin/env bash
# Stop CARLA server(s).
#
# Usage:
#   tools/carla/kill_carla.sh --port 2000    # only the server bound to RPC port 2000
#   tools/carla/kill_carla.sh -p 2000        # short form
#   tools/carla/kill_carla.sh --all          # every CarlaUE4 process of this user (shared hosts: careful)
#
# Strategy: SIGTERM first (graceful), wait 3 s, then SIGKILL anything left.
# Catches both the bash wrapper (CarlaUE4.sh / run_carla.sh) and the actual UE4
# binary (CarlaUE4-Linux-Shipping).
set -uo pipefail

PORT=""
ALL=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--port) PORT="$2"; shift 2 ;;
        --all) ALL=1; shift ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
if [[ -z "$PORT" && "$ALL" != 1 ]]; then
    echo "[kill_carla] pass --port N (or --all to stop every CARLA server)" >&2
    exit 2
fi

collect_pids() {
    local pids=()
    if [[ -n "$PORT" ]]; then
        # PIDs holding the RPC port.
        if command -v ss >/dev/null 2>&1; then
            mapfile -t pids < <(ss -tlnpH "sport = :$PORT" 2>/dev/null \
                | grep -oP 'pid=\K[0-9]+' | sort -u)
        fi
        # Also include any matching cmdline that mentions this port.
        mapfile -t -O "${#pids[@]}" pids < <(pgrep -f "carla-rpc-port=$PORT" 2>/dev/null || true)
    else
        mapfile -t pids < <(pgrep -f "CarlaUE4|tools/carla/run_carla\.sh" 2>/dev/null || true)
    fi
    # Dedupe + drop self-PID and our parent shell.
    printf '%s\n' "${pids[@]}" | awk -v me=$$ -v pp=$PPID 'NF && $1!=me && $1!=pp' | sort -u
}

PIDS=$(collect_pids)
if [[ -z "$PIDS" ]]; then
    echo "[kill_carla] no CarlaUE4 process found${PORT:+ on port $PORT}."
    exit 0
fi

echo "[kill_carla] sending SIGTERM to:"
ps -o pid=,cmd= -p $PIDS 2>/dev/null | sed 's/^/  /'
kill $PIDS 2>/dev/null || true

# Wait up to 3 s for graceful exit.
for _ in 1 2 3; do
    sleep 1
    REMAIN=$(collect_pids)
    [[ -z "$REMAIN" ]] && break
done

REMAIN=$(collect_pids)
if [[ -n "$REMAIN" ]]; then
    echo "[kill_carla] still alive after SIGTERM, sending SIGKILL to: $REMAIN"
    kill -9 $REMAIN 2>/dev/null || true
    sleep 1
fi

REMAIN=$(collect_pids)
if [[ -n "$REMAIN" ]]; then
    echo "[kill_carla] WARNING: PIDs still alive: $REMAIN" >&2
    exit 1
fi
echo "[kill_carla] CarlaUE4 cleaned up."
