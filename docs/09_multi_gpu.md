# 09 — Multi-GPU evaluation, pipelines, baseline matrix

All three tools pin CARLA and the evaluation process to the **same physical
GPU**: `CUDA_VISIBLE_DEVICES=N` for the Python side and `-graphicsadapter=K` for
CARLA. Vulkan ignores `CUDA_VISIBLE_DEVICES` and usually enumerates GPUs in a
different order, so `K` is derived by matching PCI bus ids between `nvidia-smi`
and `vulkaninfo` (install `vulkan-tools`, or pass `--carla-graphics-adapter`).
CARLA is located via `--carla-bin`, `CARLA_BIN` or `CARLA_ROOT`.

## One worker — `scripts/multi_gpu/run_worker.sh`

```bash
bash scripts/multi_gpu/run_worker.sh --scene corner --config smoke_all_planners \
  --gpu 0 --port 2000 --restart-every 25 --batch-id smoke_corner
```

Starts CARLA on the GPU, runs `scripts/run_batch_eval.sh --no-launch-carla`
against it, and with `--restart-every N` recycles CARLA every N slots (the batch
exits `99`, the worker restarts CARLA and resumes the same batch id). The default
is `--restart-every 1`: every episode gets a fresh server and, with
`FB_LOAD_MAP_POLICY=if_needed` (exported by the worker), reuses the map the
server booted with instead of calling `load_world`, which intermittently hangs
this CARLA build. A boot counts as ready only once `get_world()` answers; failed
boots are retried `CARLA_START_ATTEMPTS` (3) times. The same
recovery applies when CARLA crashes mid-batch (the batch watchdog also exits
`99`); after `MAX_STALLED_CHUNKS` (default 3) consecutive restarts without a new
completed trial the worker gives up with `99`. CARLA is always stopped on exit.

Container mode (CARLA on the host, evaluation inside Docker): add
`--container NAME [--docker-host-ip 172.17.0.1] [--no-sudo]`. The repo must be
mounted at the same path in the container; the conda env is activated from
`CONTAINER_CONDA_SH` / `CONTAINER_CONDA_ENV`.

## Fan-out — `scripts/multi_gpu/eval_multi_gpu.sh`

```bash
bash scripts/multi_gpu/eval_multi_gpu.sh --scene plaza --config baselines_traj_kpr_50x \
  --num-workers 4 --gpu-base 0 --port-base 2000 --restart-every 25
```

The config's (planner × difficulty × flow × follow position) matrix is split into
tasks and round-robined over the workers (GPU `gpu-base+i`, port
`port-base+i*port-stride`). Each task config sources the original config and
only narrows the matrix, so every other knob is inherited. Worker `i` writes
batch `<batch-id>_w<i>`; task configs, worker logs and the merged
`manifest.csv` go to `outputs/multi_gpu/<scene>/<batch-id>/`.

## Several scenes in a row — `scripts/multi_gpu/run_pipeline.sh`

```bash
bash scripts/multi_gpu/run_pipeline.sh --preset full --num-workers 8 --restart-every 25
bash scripts/multi_gpu/run_pipeline.sh --job corner:baselines_e2e_50x --job park:baselines_e2e_50x --num-workers 4
```

Presets: `full`, `rda_traj_kpr`, `plaza_default`. Summary table:
`outputs/pipelines/<batch-id-base>/summary.txt`.

## Baseline table — `scripts/baselines/run_all_baselines.sh`

```bash
# everything on one GPU (resume-safe; re-run to continue)
bash scripts/baselines/run_all_baselines.sh --trials 50 --gpu 0 --port 2000

# split across GPUs: one instance per GPU with disjoint scenes / rows
bash scripts/baselines/run_all_baselines.sh --gpu 0 --port 2320 --scenes "corner passage" &
bash scripts/baselines/run_all_baselines.sh --gpu 1 --port 2330 --scenes "park plaza" --skip-gate &
```

Batch ids are `baselines<trials>_<scene>_<row>`. Before the matrix it runs one
TrackVLA and one OA-VAT episode as an import gate (skip with `--skip-gate`).

## Stopping servers

`tools/carla/kill_carla.sh --port P` stops only the server on port `P`; the
scripts above never kill CARLA instances they did not start.
