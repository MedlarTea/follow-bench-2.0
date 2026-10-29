# Baseline configs

One single-episode smoke command per baseline row (corner / back / easy). Run them
from the repo root, e.g. `bash configs/baselines/rda_traj_kpr.sh`; extra
arguments are forwarded to the episode runner.

| File | Baseline row |
|---|---|
| `rda_traj_r18.sh` | MPC w/ Traj. + R18 |
| `rda_traj_kpr.sh` | MPC w/ Traj. + KPR |
| `rda_search_r18.sh` | MPC w/ DS. + R18 |
| `rda_lidar_r18.sh` | MPC + R18 |
| `trackvla.sh` | TrackVLA |
| `oa_vat.sh` | OA-VAT |

ReID thresholds are the `PerceptionConfig` defaults (ridge `0.35` / cosine `0.55` / ID-switch `0.05`).

The full four-scene table uses the batch configs
`configs/scenes/<scene>/batch/baselines_<row>_{10,50}x.sh`, driven by
`scripts/baselines/run_all_baselines.sh`. Traj. rows pass
`--rda-traj-disable-lidar-obstacles` so numbers stay comparable to the published
rows; omit that flag to use the newer lidar-obstacle default.
