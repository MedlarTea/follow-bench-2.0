# 07 — Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Address already in use` :2000 | Old CARLA | `bash tools/carla/kill_carla.sh --port 2000` |
| Client/simulator version mismatch | Wrong CARLA build vs `carla` wheel | Install the wheel shipped with your CARLA build — see [Known issues](#known-issues) |
| Vulkan on wrong GPU | Adapter ≠ nvidia-smi index | Install `vulkan-tools` (auto-mapping in `scripts/lib/carla.sh`) or pass `--carla-graphics-adapter` |
| `time-out ... waiting for the simulator` in `load_world`; CARLA log ends in `RequestExit` / `Signal 11` | `client.load_world()` intermittently hangs this UE build's render thread (~1 in 50 reloads, more often under multi-GPU load); UE dies once the render fence times out | Do not reload the map: `run_worker.sh` defaults to a fresh CARLA per episode (`--restart-every 1`) with `FB_LOAD_MAP_POLICY=if_needed`, so episodes reuse the already loaded Town10HD_Opt. `carla_launch` waits for the world RPC and retries a failed boot (`CARLA_START_ATTEMPTS`) |
| `--teleop requires the pygame UI` | `--no-ui` passed to a teleop session | Teleop needs a display; drop `--no-ui` |
| `No module named followbench` / `baselines` | Package not installed in the active env | `pip install -e .` from the repo root |
| `No module named torchvision/ultralytics/filterpy/lap` | Incomplete env | Install perception extras (`docs/01_install.md`) |
| Permission denied under `outputs/` | Root-owned output dir | `export FB_OUTPUT_ROOT=/writable/path` (or pass `--output-dir`) |
| `--visualize` ignored | Eval off | Add `--enable-evaluation` |
| Pedestrians freeze mid-episode | Collision / task terminate called `set_motion_active(False)` | Check `collision_events.jsonl` / `termination_reason` |
| Batch resumes into a `manifest.legacy.csv` | Batch recorded by the pre-restructure runners | Expected: old rows are kept aside, trials on disk are still counted |
| Cannot match baseline SR | Wrong ReID thresholds / scenario mix | See warning in `docs/04_baselines.md` |

## Known issues

### Python `carla` client and CARLA server from different builds

**Symptom.** Every episode prints

```text
WARNING: Version mismatch detected ... Client API version = <A> / Simulator API version = <B>
```

and `carla_launch` prints `[carla] WARN: python carla API ... does not match the
server build`.

**Cause.** The `carla` module in the active Python env was installed from a
different CARLA build than the server in `$CARLA_ROOT`. All CARLA 0.9.16 wheels
share the file name `carla-0.9.16-cp310-cp310-linux_x86_64.whl`, so a wheel
taken from another installation is easy to pick up by mistake.

**Impact.** Libcarla only prints the warning; it does not refuse to connect.
Benchmark runs have completed under a mismatch without any failure traced to
it. The `load_world` hang above also occurs with a matched client, so it is
unrelated. An RPC incompatibility would still only surface as subtle errors, so
use a matched client for results you intend to report.

**Fix.** Install the wheel that ships with the server:

```bash
pip install --force-reinstall --no-deps \
  "$CARLA_ROOT"/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-linux_x86_64.whl
python -c 'import carla; print(carla.Client("127.0.0.1", 1).get_client_version())'
cat "$CARLA_ROOT"/VERSION   # the two builds should match
```

If the env is shared with another project that needs a different client,
install the wheel into a separate directory (`pip install --no-deps --target
<dir> ...`) and prepend `<dir>` to `PYTHONPATH` for FollowBench only.
