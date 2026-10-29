# shellcheck shell=bash
# Shared CARLA server helpers. Source this file; it defines functions only.
#
#   carla_port_in_use PORT
#   carla_detect_vk_adapter GPU          -> Vulkan adapter index matching nvidia-smi GPU
#   carla_check_client_version BIN       -> warn if the python `carla` API is from another build
#   carla_launch BIN GPU PORT LOG [ADAPTER]  -> sets CARLA_PGID; waits until the world RPC answers
#   carla_world_ready PORT TIMEOUT_S     -> 0 once get_world().get_map() succeeds
#   carla_stop PORT                      -> stops CARLA_PGID (if set) and any server on PORT
#
# Servers are started with setsid so they own a process group; stopping only
# ever targets that group or the server bound to PORT, never other CARLAs.
#
# Launch hardening (recipe proven on long-running multi-GPU CARLA deployments):
#   * every instance gets its own HOME / XDG_RUNTIME_DIR under
#     $CARLA_RUNTIME_ROOT/gpu<G>-p<PORT>/ so parallel servers never share UE's
#     ~/.config/Epic state; its Engine.ini carries GC-safe [SystemSettings];
#   * CUDA_VISIBLE_DEVICES is removed from the server environment; the GPU is
#     chosen only through the Vulkan adapter index;
#   * the Shipping binary is started directly; readiness means the world RPC
#     answers (not just a bound port); a failed boot is retried
#     CARLA_START_ATTEMPTS times; stop waits until the ports are released.
#
# Render-fence timeout: g.TimeoutForBlockOnRenderFence cannot be set per-instance.
# UE only reads it from Engine/Config/Linux/LinuxEngine.ini during early init
# (user-config Engine.ini, the command line and -ini: overrides are all ignored),
# so it has been raised once from 60000 -> 300000 ms in the CARLA install itself.
#
# Known failure mode: client.load_world() intermittently hangs the UE render
# thread (~1 in 50 reloads, more under load); UE then dies with Signal 11 once
# the render fence times out. Episodes avoid it with FB_LOAD_MAP_POLICY=if_needed
# on a freshly booted server (see followbench/sim/carla_world.py).
#
# Environment: CARLA_BOOT_TIMEOUT_S (120), CARLA_READY_TIMEOUT_S (120),
# CARLA_START_ATTEMPTS (3), CARLA_RUNTIME_ROOT (default $FB_OUTPUT_ROOT/carla_runtime),
# CARLA_QUALITY (Low|Epic, default Low), CARLA_RENDER_SETTINGS (extra console
# variables, ';'-separated), PYTHON (interpreter with the carla module).

CARLA_BOOT_TIMEOUT_S="${CARLA_BOOT_TIMEOUT_S:-120}"
CARLA_READY_TIMEOUT_S="${CARLA_READY_TIMEOUT_S:-120}"
CARLA_START_ATTEMPTS="${CARLA_START_ATTEMPTS:-3}"
CARLA_PGID=""

carla_port_in_use() {
  ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "(:|^)$1\$"
}

# Vulkan ignores CUDA_VISIBLE_DEVICES and its device order usually differs from
# nvidia-smi (this host also exposes llvmpipe), so match the device UUID, with
# the PCI bus id as a fallback for older vulkaninfo builds.
carla_detect_vk_adapter() {
  local gpu="$1" uuid pci_hex pci_dec idx
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  command -v vulkaninfo >/dev/null 2>&1 || return 1
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]' | sed 's/^GPU-//')"
  if [[ -n "$uuid" ]]; then
    idx="$(vulkaninfo --summary 2>/dev/null | awk -v target="$uuid" '
      /^GPU[0-9]+:/ { a = $0; sub(/^GPU/, "", a); sub(/:.*/, "", a) }
      /deviceUUID[ \t]*=/ { u = $0; sub(/^.*=[ \t]*/, "", u); if (u == target) { print a; exit } }')"
    [[ -n "$idx" ]] && { echo "$idx"; return 0; }
  fi
  pci_hex="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$gpu" 2>/dev/null | awk -F: '{print $(NF-1)}' | tr -d ' ')"
  [[ -n "$pci_hex" ]] || return 1
  pci_dec="$(printf '%d' "0x$pci_hex" 2>/dev/null)" || return 1
  vulkaninfo 2>/dev/null | awk -v target="$pci_dec" '
    /^GPU[0-9]+:/ { idx = $0; sub(/^GPU/, "", idx); sub(/:$/, "", idx); next }
    /^[ \t]+pciBus[ \t]*=/ { if ($NF == target) { print idx; exit } }'
}

# The python `carla` module must come from the same build as the server
# (<CARLA_ROOT>/PythonAPI/carla/dist/*.whl). A mismatch only prints a warning
# in libcarla but can break RPC calls in subtle ways.
carla_check_client_version() {
  local bin="$1" server client py="${PYTHON:-python}"
  server="$(tr -d '[:space:]' <"$(dirname "$bin")/VERSION" 2>/dev/null || true)"
  client="$("$py" -c 'import carla; print(carla.Client("127.0.0.1", 1).get_client_version())' 2>/dev/null | tail -n 1)"
  [[ -n "$server" && -n "$client" ]] || return 0
  if [[ "$server" != "$client" ]]; then
    echo "[carla] WARN: python carla API ($client, $($py -c 'import carla,os;print(os.path.dirname(carla.__file__))' 2>/dev/null)) does not match the server build ($server)." >&2
    echo "[carla]       install $(dirname "$bin")/PythonAPI/carla/dist/carla-*.whl into this env." >&2
    return 1
  fi
}

_carla_prepare_home() {  # gpu port -> prints the instance root
  local root="${CARLA_RUNTIME_ROOT:-${FB_OUTPUT_ROOT:-/tmp}/carla_runtime}/gpu$1-p$2"
  local cfg_dir="$root/home/.config/Epic/CarlaUE4/Saved/Config/LinuxNoEditor"
  local cfg="$cfg_dir/Engine.ini"
  mkdir -p "$cfg_dir" "$root/xdg"
  chmod 700 "$root/xdg"
  touch "$cfg"
  # A packaged UE4 reads user config from $HOME/.config/Epic/<Project>; keep the
  # sections UE regenerates itself and (re)write our own block on every launch.
  # GC-safe settings: CARLA leaves dangling UObjects behind destroyed actors and
  # parallel GC / multithreaded destruction then crash or stall the engine;
  # servers are recycled long before the 1-day purge timer fires.
  awk '/^\[(SystemSettings|ConsoleVariables)\]$/ { skip = 1; next } /^\[/ { skip = 0 } !skip { print }' \
    "$cfg" >"$cfg.tmp"
  {
    cat "$cfg.tmp"
    printf '\n[SystemSettings]\n'
    printf 'gc.TimeBetweenPurgingPendingKillObjects=86400\n'
    printf 'gc.AllowParallelGC=0\n'
    printf 'gc.MultithreadedDestructionEnabled=0\n'
    printf 'g.TimeoutForBlockOnRenderFence=300000\n'
    if [[ -n "${CARLA_RENDER_SETTINGS:-}" ]]; then
      printf '%s\n' "$CARLA_RENDER_SETTINGS" | tr ';' '\n' | sed '/^[[:space:]]*$/d'
    fi
  } >"$cfg"
  rm -f "$cfg.tmp"
  echo "$root"
}

# World-level readiness: the RPC port is bound long before UE finishes loading
# the default map, and a server can also hang during that first load.
carla_world_ready() {  # port timeout_s
  local py="${PYTHON:-python}"
  if ! "$py" -c 'import carla' >/dev/null 2>&1; then
    echo "[carla] WARN: '$py' cannot import carla; falling back to port-only readiness (set PYTHON)" >&2
    return 0
  fi
  timeout "$(( $2 + 10 ))" "$py" - "$1" "$2" <<'PY' >/dev/null 2>&1
import sys, time, carla
port, budget = int(sys.argv[1]), float(sys.argv[2])
deadline = time.time() + budget
while time.time() < deadline:
    try:
        c = carla.Client("127.0.0.1", port)
        c.set_timeout(min(20.0, max(2.0, deadline - time.time())))
        c.get_world().get_map()
        sys.exit(0)
    except Exception:
        time.sleep(2.0)
sys.exit(1)
PY
}

_carla_wait_ports_free() {  # port timeout_s
  local waited=0
  while (( waited < $2 )); do
    carla_port_in_use "$1" || carla_port_in_use "$(( $1 + 1 ))" || return 0
    sleep 1; waited=$((waited + 1))
  done
  return 1
}

_carla_launch_once() {  # bin gpu port log adapter root
  local bin="$1" gpu="$2" port="$3" log="$4" adapter="$5" root="$6" exe prefix=()
  # Launch the Shipping binary directly (as CarlaUE4.sh would, minus its sh
  # wrapper and the chmod it cannot perform on a read-only install).
  exe="$(dirname "$bin")/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
  if [[ -x "$exe" ]]; then prefix=(CarlaUE4); else exe="$bin"; fi
  ( cd "$(dirname "$bin")" && exec setsid env -u CUDA_VISIBLE_DEVICES \
      HOME="$root/home" XDG_RUNTIME_DIR="$root/xdg" SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
      "$exe" ${prefix[@]+"${prefix[@]}"} -RenderOffScreen -nosound "-quality-level=${CARLA_QUALITY:-Low}" \
      "-ini:[/Script/Engine.RendererSettings]:r.GraphicsAdapter=$adapter" \
      "-carla-rpc-port=$port" -stdout -FullStdOutLogOutput \
      </dev/null >>"$log" 2>&1 ) &
  CARLA_PGID=$!
  # UE segfaults on its own SIGTERM shutdown path; that is harmless, so keep
  # bash from reporting the job's "Segmentation fault" into the worker log.
  disown "$CARLA_PGID" 2>/dev/null || true
  local waited=0
  while (( waited < CARLA_BOOT_TIMEOUT_S )); do
    if carla_port_in_use "$port"; then break; fi
    if ! kill -0 "$CARLA_PGID" 2>/dev/null; then
      echo "[carla] server exited before binding port $port" >&2
      return 1
    fi
    sleep 1; waited=$((waited + 1))
  done
  if ! carla_port_in_use "$port"; then
    echo "[carla] port $port not bound within ${CARLA_BOOT_TIMEOUT_S}s" >&2
    return 1
  fi
  if ! carla_world_ready "$port" "$CARLA_READY_TIMEOUT_S"; then
    echo "[carla] port $port bound after ${waited}s but the world RPC was not ready within ${CARLA_READY_TIMEOUT_S}s" >&2
    return 1
  fi
  if ! kill -0 "$CARLA_PGID" 2>/dev/null; then
    echo "[carla] server exited right after becoming ready" >&2
    return 1
  fi
  echo "[carla] up after ${waited}s, world ready (pgid=$CARLA_PGID)"
}

carla_launch() {
  local bin="$1" gpu="$2" port="$3" log="$4" adapter="${5:-}" root attempt
  if [[ ! -x "$bin" ]]; then
    echo "[carla] CARLA launcher not executable: $bin (set CARLA_BIN / --carla-bin)" >&2
    return 1
  fi
  if [[ -z "$adapter" ]]; then
    adapter="$(carla_detect_vk_adapter "$gpu" 2>/dev/null || true)"
    if [[ -z "$adapter" ]]; then
      echo "[carla] WARN: cannot derive the Vulkan adapter for GPU $gpu; using $gpu (install vulkan-tools or pass an adapter)" >&2
      adapter="$gpu"
    fi
  fi
  carla_check_client_version "$bin" || true
  root="$(_carla_prepare_home "$gpu" "$port")"
  mkdir -p "$(dirname "$log")"
  # Resolve to an absolute path: the server is launched after `cd` into the
  # CARLA directory, so a relative log path would be re-resolved from there.
  log="$(cd "$(dirname "$log")" && pwd)/$(basename "$log")"
  : >"$log"
  for ((attempt = 1; attempt <= CARLA_START_ATTEMPTS; attempt++)); do
    _carla_wait_ports_free "$port" 30 || echo "[carla] WARN: port $port/$((port + 1)) still busy after 30s" >&2
    echo "[carla] launching gpu=$gpu vulkan_adapter=$adapter port=$port quality=${CARLA_QUALITY:-Low} attempt=$attempt/$CARLA_START_ATTEMPTS home=$root log=$log"
    printf '\n===== [carla.sh] launch attempt %d %s =====\n' "$attempt" "$(date -Iseconds)" >>"$log"
    _carla_launch_once "$bin" "$gpu" "$port" "$log" "$adapter" "$root" && return 0
    echo "[carla] attempt $attempt failed; tail of $log:" >&2
    tail -n 15 "$log" >&2 || true
    carla_stop "$port"
    sleep 5
  done
  echo "[carla] ERROR: CARLA did not come up after $CARLA_START_ATTEMPTS attempts" >&2
  return 1
}

carla_stop() {
  local port="$1"
  if [[ -n "$CARLA_PGID" ]] && kill -0 "$CARLA_PGID" 2>/dev/null; then
    kill -TERM -- "-$CARLA_PGID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$CARLA_PGID" 2>/dev/null || true
  fi
  CARLA_PGID=""
  pkill -f "CarlaUE4.*-carla-rpc-port=${port}\b" 2>/dev/null || true
  sleep 1
  pkill -9 -f "CarlaUE4.*-carla-rpc-port=${port}\b" 2>/dev/null || true
  # UE releases its RPC / streaming sockets asynchronously; do not let the next
  # launch race the teardown.
  _carla_wait_ports_free "$port" 20 || true
}
