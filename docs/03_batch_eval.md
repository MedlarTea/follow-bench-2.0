# 03 — Batch evaluation

One driver for every scene:

```bash
bash scripts/run_batch_eval.sh --scene <passage|corner|park|plaza> --config <name> [--batch-id ID]
```

`--config` is either a path or a name in `configs/scenes/<scene>/batch/`
(default `default_50x`). Re-running with the same `--batch-id` resumes: settings
that already have `REPEATS` completed trials on disk are skipped.

## Config schema (shared by all scenes)

| Variable | Meaning |
|---|---|
| `PLANNERS_WITH_PERCEPTION` | Planners run with `--use-perception` (+ ReID), over `FOLLOW_POSITIONS_MAIN` |
| `PLANNERS_BACK_ONLY` | Planners run without perception (e.g. TrackVLA, OA-VAT), over `FOLLOW_POSITIONS_BACK_ONLY` |
| `DIFFICULTIES` | `easy hard boss legend` |
| `FLOW_MODES` | plaza only: `random parallel perpendicular chaotic` |
| `REPEATS` | trials per setting |
| `DESIRED_DISTANCE`, `SIM_MODE`, `AUTO_SPAWN_SEC`, `AUTO_FOLLOW_SEC` | episode knobs |
| `REID_MODE`, `REID_KPR_CONFIG`, `REID_DEVICE` | perception group only |
| `EXTRA_EPISODE_ARGS` | appended to every episode (e.g. `--rda-traj-disable-lidar-obstacles`) |
| `RESUME`, `STOP_ON_FAILURE`, `RESTART_CARLA_EVERY_N_SLOTS` | driver behaviour |
| `NUM_WALKERS_TOTAL` | passage / park crowd size (corner / plaza derive it from the difficulty) |
| `VISUALIZE` | corner: write `episode.mp4` / `episode_3view.mp4` |
| `WEATHER`, `ENABLE_UI`, `ENABLE_DEBUG_VIS`, `ENABLE_WORLD_DEBUG`, `ENABLE_LASER`, `DURATION_SEC` | park |

Park's former `traditional/` and `e2e/` batch trees are now plain configs of the
same scene: `baselines_*` / `default_50x` (perception planners) and `baselines_e2e_*` /
`e2e_*` (TrackVLA, OA-VAT).

## Outputs

```text
outputs/batch_manifests/<scene>/<batch_id>/   manifest.csv, failures.csv, progress.log,
                                              summary.txt, config_snapshot.sh
outputs/logs_batch/<scene>/<batch_id>/<difficulty>[/<flow>]/<scene>/<planner>/H<n>_<follow>_D<dist>/trial_*/
```

`manifest.csv` has one row per executed or resumed slot (`status` =
`completed | failed | incomplete | skipped_existing`).

## Summaries

```bash
bash scripts/summarize_batch_eval.sh --scene corner                 # latest batch
bash scripts/summarize_batch_eval.sh --scene plaza --batch-id baselines50_plaza_traj_kpr
```

It re-runs the evaluator where `eval_result.json` is missing and writes tables
and comparison plots to `outputs/results/{tables,compare}/`. The batch's own
`config_snapshot.sh` defines the matrix; pass `--config` for batches recorded
before this layout.

## CARLA handling

The driver connects to `CARLA_HOST:CARLA_PORT` (default `127.0.0.1:2000`). If
nothing listens there it launches `CARLA_CMD` or `$CARLA_ROOT/CarlaUE4.sh`
headless (in its own process group, so only that server is ever restarted or
stopped), restarts it every `RESTART_CARLA_EVERY_N_SLOTS` slots and when the RPC
port goes silent. With `--no-launch-carla` an outer wrapper owns CARLA; the
driver then exits `99` when CARLA dies or `--max-slots` is reached so the wrapper
can recycle the server and resume.

## Baseline matrix (back, D=1.5, all difficulties)

| Config (`configs/scenes/<scene>/batch/`) | Method |
|---|---|
| `baselines_traj_r18_{10,50}x` | MPC w/ Traj. + R18 (`--rda-traj-disable-lidar-obstacles`) |
| `baselines_traj_kpr_{10,50}x` | MPC w/ Traj. + KPR |
| `baselines_ds_r18_{10,50}x` | MPC w/ DS. + R18 (`rda_search`) |
| `baselines_mpc_r18_{10,50}x` | MPC + R18 (`rda_lidar`) |
| `baselines_e2e_{10,50}x` | TrackVLA + OA-VAT |

Run all of them with `scripts/baselines/run_all_baselines.sh` (see
[09_multi_gpu.md](09_multi_gpu.md)).
