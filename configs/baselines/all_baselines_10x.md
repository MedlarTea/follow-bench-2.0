# Baseline matrix (10× back-only)

Locked for the current baseline run:

| Knob | Value |
|---|---|
| `FOLLOW_POSITION` | `back` |
| `DESIRED_DISTANCE` | `1.5` |
| Difficulties | all (`easy` / `hard` / `boss` / `legend`) |
| `REPEATS` / cell | `10` |
| Scenes | plaza, corner, park, passage |

Per-method extra flags:

| Method | Extra |
|---|---|
| Traj. + R18 / KPR | `--rda-traj-disable-lidar-obstacles` + matching `--reid-mode` |
| DS. + R18 | `PLANNER=rda_search --reid-mode basic` |
| MPC + R18 | `PLANNER=rda_lidar --reid-mode basic` |
| TrackVLA / OA-VAT | no `--use-perception` |

Batch configs (one per row and scene):
`configs/scenes/{corner,passage,park,plaza}/batch/baselines_*_10x.sh`
(plaza also sweeps the flow modes).

```bash
bash scripts/baselines/run_all_baselines.sh --trials 10 --gpu 0 --port 2000
```

`EXTRA_EPISODE_ARGS` is forwarded by `scripts/run_batch_eval.sh` and inherited
by the multi-GPU task configs. Traj. configs set it to
`--rda-traj-disable-lidar-obstacles`.
