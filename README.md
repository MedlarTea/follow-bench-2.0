# FollowBench 2.0

A CARLA benchmark for **robotic human following in crowds**: a robot (AgileX
Scout, or Unitree G1 with `--humanoid`) keeps a designated pedestrian at a
desired distance without collisions.

| Scene | Setting | Humanoid |
|---|---|:---:|
| `passage` | two crowds cross a doorway in both directions | – |
| `corner` | corridor corner with paired counter-flows | ✓ |
| `park` | dense crowd, target loops between three nodes | ✓ |
| `plaza` | open plaza, parallel / perpendicular / chaotic flows | – |

Design notes: [`docs/scenes/`](docs/scenes). The CARLA build and model weights
are hosted on Hugging Face
([`hjyeee/Follow-Bench-2.0-packages`](https://huggingface.co/datasets/hjyeee/Follow-Bench-2.0-packages));
see [Install](#install).

## News

- **[2026-10-29]** FollowBench 2.0 is released.

## TODO
- [ ] More diverse benchmark scenes except the four scenes.
- [ ] More baselines.
- [ ] Data infrastructure for curating VLA training data.

## Highlights

- **One package, one set of launchers.** `pip install -e .` installs
  `followbench` (benchmark) and `baselines` (evaluated methods); no `sys.path`
  hacks.
- **Bring your own method.** `--planner custom --policy-factory ...` loads any
  `FollowerPolicyAdapter`; modular methods implement the perception -> planning
  interface in `followbench/interfaces/`.
- **Every script takes `--scene passage|corner|park|plaza`.** Scene defaults
  live in `configs/scenes/<scene>/episode.sh`; `--humanoid` / `--teleop`
  switch robot / control mode.
- **Resumable batches.** Re-running a batch with the same `--batch-id` skips
  completed trials; `--max-slots` lets an outer worker recycle CARLA.
- **Multi-GPU.** Fan a batch matrix over N GPUs, chain scenes into pipelines,
  or run the whole baseline table with one script.
- **Safe CARLA handling.** Scripts only stop the CARLA they launched; the
  Vulkan adapter is auto-mapped from the CUDA GPU.
- **Everything a run produces goes to `outputs/`** (override with
  `FB_OUTPUT_ROOT`). `FB_DRY_RUN=1` prints the final command without running.

## Install

Requirements: Linux x86-64, an NVIDIA GPU whose driver supports Vulkan, conda,
`curl` and `unzip`. Disk: about 32 GB for `core`, 75 GB for `all` (zips + unpacked;
add `--delete-zips` to drop each zip after it is unpacked).

```bash
# 1) Python env and the two packages of this repo
conda env create -f environment.multiRobots.yml
conda activate multiRobots
pip install -e .

# 2) CARLA build + weights from Hugging Face, checked file by file.
#    core = CARLA + perception weights + YOLO11 (enough for every non-end-to-end baseline)
#    all  = core + OA-VAT + TrackVLA weights
bash tools/install/install_assets.sh --download --install-wheel core

# 3) Verify (exit code 0 = ready). Add `--deep` to hash every file.
python tools/install/check_install.py core
```

- Downloads resume if interrupted; just run the same command again.
- If `huggingface.co` is not reachable, use a mirror:
  `export HF_ENDPOINT=https://hf-mirror.com`.
- `--install-wheel` installs the CARLA client wheel shipped with the build, so
  client and server versions match. `check_install.py` reports a mismatch.
- OA-VAT and TrackVLA also need DINOv3 checkpoints that Meta does not allow us
  to redistribute. Request access and download them yourself:
  [`docs/01_install.md`](docs/01_install.md#weights-you-download-yourself).
  `python tools/install/check_install.py all` tells you which files are still missing.

Unpacked layout:

```text
CARLA_multi_robots/  CarlaUE4.sh, ...  (or --carla-dir DIR, then export CARLA_ROOT=DIR)
data/                (may be a symlink to a larger disk; create it before installing)
  weights/
    yolo/            yolo11n.pt, yolo11s.pt   fetched from Ultralytics (AGPL-3.0)
    reid_basic/      ckpt.t7
    reid_kpr/        kpr_*.pth.tar
    traj_predictor/  sgan.pt, csgan.pt, csgan2.pt
  oa-vat/            ortrack/, mobileclip_blt.ts
                     yolo/yoloe-11l-seg.pt     fetched from the YOLOe authors (AGPL-3.0)
                     dinov3/*.pth              you download (gated)
  trackvla/          TrackVLA checkpoint + Qwen3-4B + SigLIP
                     facebook/dinov3-vits16-pretrain-lvd1689m/   you download (gated)
```

Package list, sizes, licenses and the manual steps: [`docs/01_install.md`](docs/01_install.md).

## Quick start

```bash
# 1) Start the FollowBench CARLA build (headless, port 2000) and wait for it
bash tools/carla/run_carla.sh -RenderOffScreen -nosound -quality-level=Low > carla.log 2>&1 &
until ss -tln | grep -q ':2000 '; do sleep 2; done

# 2) One episode: corner, RDA-Traj + basic ReID (baseline Traj. flags), at most 45 s
PLANNER=rda_traj DIFFICULTY=easy \
  bash scripts/run_episode.sh --scene corner \
    --use-perception --reid-mode basic --kinematics-mode kf \
    --rda-traj-disable-lidar-obstacles \
    --auto-spawn-sec 1 --auto-follow-sec 4 -- --no-ui --max-active-sec 45

# 3) One episode, then score it and render its plots
AUTO_FOLLOW=1 bash scripts/run_episode_eval.sh --scene park

# 4) A small batch (2 trials, ~3 min, core install only) and its tables / plots
bash scripts/run_batch_eval.sh --scene corner --config quickstart --batch-id my_first_batch
bash scripts/summarize_batch_eval.sh --scene corner --batch-id my_first_batch
#    -> outputs/results/tables/*_summary.csv, outputs/results/compare/*.png

# Stop the CARLA from step 1
bash tools/carla/kill_carla.sh --port 2000
```

`smoke_test` configs (`--config smoke_test`) cover more planners per scene; some
of them include TrackVLA / OA-VAT and need the `all` install plus DINOv3.

More launch variants:

```bash
bash scripts/run_episode.sh --scene park --humanoid                # G1 humanoid runner
ROBOT_TYPE=g1 bash scripts/run_episode.sh --scene plaza --teleop   # keyboard teleop
FB_DRY_RUN=1 bash scripts/run_episode.sh --scene passage           # print command only
bash scripts/run_episode.sh --scene passage -- --help              # runner CLI help
```

Environment knobs for `run_episode.sh`: `PLANNER`, `DIFFICULTY`,
`FOLLOW_POSITION`, `DESIRED_DISTANCE`, `NUM_WALKERS`, `SIM_MODE`,
`CARLA_HOST`/`CARLA_PORT`; extra arguments are appended to the Python runner
and override the preset.

### Multi-GPU and running all baselines

```bash
# Split one batch config over 4 GPUs (GPU 0..3, CARLA ports 2000, 2100, ...)
bash scripts/multi_gpu/eval_multi_gpu.sh --scene plaza --config baselines_traj_r18_10x --num-workers 4

# Chain several scenes / configs (presets: full, rda_traj_kpr, plaza_default)
bash scripts/multi_gpu/run_pipeline.sh --num-workers 4 --preset rda_traj_kpr
bash scripts/multi_gpu/run_pipeline.sh --num-workers 2 --job corner:smoke_test --job park:smoke_test

# Full baseline table: all rows x all scenes, back follow, D = 1.5 m (resume-safe)
bash scripts/baselines/run_all_baselines.sh --trials 10 --gpu 0 --port 2000
```

`scripts/multi_gpu/run_worker.sh` runs locally by default, or inside Docker
with `--container NAME`. See [`docs/09_multi_gpu.md`](docs/09_multi_gpu.md).

### Teleoperation data collection

```bash
OPERATOR_ID=alice bash scripts/teleop/collect_teleop_data.sh --scene corner hard
bash scripts/teleop/collect_teleop_data.sh --all          # 4 scenes x 4 difficulties
```

Standalone keyboard drivers (Scout / G1 / Dog): [`tools/teleop/`](tools/teleop/README.md).

### Offline tests (no CARLA)

```bash
python tests/test_target_reid.py
python tests/test_perception_states.py
python tests/test_debug_visualizer.py
```

## Baselines

| Method (baseline) | Planner | Perception / ReID | Batch config row |
|---|---|---|---|
| MPC w/ Traj. + R18 | `rda_traj` | `--use-perception --reid-mode basic --rda-traj-disable-lidar-obstacles` | `baselines_traj_r18_*` |
| MPC w/ Traj. + KPR | `rda_traj` | `--use-perception --reid-mode kpr --rda-traj-disable-lidar-obstacles` | `baselines_traj_kpr_*` |
| MPC w/ DS. + R18 | `rda_search` | `--use-perception --reid-mode basic` | `baselines_ds_r18_*` |
| MPC + R18 | `rda_lidar` | `--use-perception --reid-mode basic` | `baselines_mpc_r18_*` |
| TrackVLA | `trackvla` | built-in VLA | `baselines_e2e_*` |
| OA-VAT | `oa_vat` | built-in YOLOe + ORTrack + DINOv3 | `baselines_e2e_*` |

Each row exists as `configs/scenes/<scene>/batch/baselines_<row>_{10x,50x}.sh`;
single-episode examples are in [`configs/baselines/`](configs/baselines/README.md).
`rda_traj` defaults to lidar/NPC obstacles in this release; pass
`--rda-traj-disable-lidar-obstacles` to match the reference Traj. rows.

### Plugging in your own method

- **End-to-end policy**: subclass `FollowerPolicyAdapter`
  (`followbench/sim/follow_policy_adapter.py`) and implement `reset()` /
  `act(obs) -> FollowAction`.
- **Modular policy**: implement a `PerceptionModule` and a `Planner` and combine
  them with `PerceptionPlanningPolicy` (`followbench/interfaces/`). The wrapper
  strips ground truth from the observation, gives perception the target
  designation until the first lock-on, and fills the debug fields used by the
  logger, the identity metrics and the overlays.

Expose a factory `build_policy(ctx: PolicyBuildContext)` and run it with
`--planner custom --policy-factory <module:callable | file.py:callable>`
(or `PLANNER=custom FB_POLICY_FACTORY=...` for batch configs);
`--policy-arg KEY=VALUE` is forwarded as `ctx.options`. A minimal example is in
[`examples/custom_policy/gt_pursuit.py`](examples/custom_policy/gt_pursuit.py);
details in [`docs/04_baselines.md`](docs/04_baselines.md#custom-methods---planner-custom).

Other planner × ReID combinations are also supported; see
[`baselines/README.md`](baselines/README.md) and
[`docs/run_manual.md`](docs/run_manual.md).

## Repository layout

```text
followbench/                 the benchmark (Python package)
  sim/                       shared CARLA runtime: robot, humanoid, pedestrians, visibility, weather
  scenes/<scene>/            episode runner (+ run_episode_humanoid.py) and NPC logic per scene
  perception/                detection, tracking, target ReID (basic / KPR)
  interfaces/                perception -> planning interface, custom policy loader
  evaluation/                metrics, episode logging, evaluator
  visualization/             offline tables / plots + live debug viewers
  paths.py, weights.py       repo / asset / weight / output locations
baselines/                   evaluated methods (Python package)
  adapters/                  one adapter per planner (the plug-in interface)
  planning/ maps/ prediction/ behavior/ perception/ common/
  third_party/               TrackVLA, OA-VAT (not packaged)
examples/custom_policy/      minimal --planner custom example
assets/scenes/<scene>/       gridmaps, routes, flow points
configs/
  scenes/<scene>/episode.sh  scene launch preset
  scenes/<scene>/batch/      batch matrices (baselines_*, smoke_*, default_50x, ...)
  baselines/                 single-episode baseline-row examples
scripts/
  run_episode.sh             one episode / teleop session (--scene, --humanoid, --teleop)
  run_episode_eval.sh        one episode + scoring + plots
  run_batch_eval.sh          resumable batch driver
  summarize_batch_eval.sh    tables + comparison plots for a batch
  lib/carla.sh               shared CARLA launch / stop / adapter mapping
  multi_gpu/                 eval_multi_gpu.sh, run_worker.sh, run_pipeline.sh
  baselines/run_all_baselines.sh  full baseline table
  teleop/collect_teleop_data.sh   Human-teleop data collection sessions
tools/
  carla/                     run_carla.sh, kill_carla.sh, ROI crowd demo
  map_annotator/             new-scene scaffold + map utilities
  teleop/                    standalone keyboard drivers (Scout / G1 / Dog)
  analysis/                  ad-hoc analysis scripts
tests/                       offline unit tests (no CARLA needed)
docs/                        guides, scene notes, references
data/          (git-ignored) weights + checkpoints (symlink allowed)
outputs/       (git-ignored) everything runs produce, see below
```

### Outputs (`$FB_OUTPUT_ROOT`, default `outputs/`)

```text
outputs/
  runs/<scene>/                       per-episode artefacts of single runs
  eval_logs/                          run_episode_eval.sh logs
  logs_batch/<scene>/<batch_id>/      per-trial evaluation logs of a batch
  batch_manifests/<scene>/<batch_id>/ manifest.csv, failures.csv, progress.log, summary.txt
  results/{tables,compare}/           summarize_batch_eval.sh outputs
  multi_gpu/<scene>/<batch_id>/       task configs, worker logs, merged manifest
  pipelines/<batch_id_base>/          run_pipeline.sh log + summary
  baselines/                          run_all_baselines.sh driver logs
```

## Adding a new scene

```bash
python tools/map_annotator/create_scenario_scaffold.py --scenario-name warehouse
```

This creates `assets/scenes/warehouse/`, `configs/scenes/warehouse/episode.sh`,
`configs/scenes/warehouse/batch/smoke_test.sh` and `docs/scenes/warehouse.md`;
afterwards `scripts/run_episode.sh --scene warehouse` and
`scripts/run_batch_eval.sh --scene warehouse --config smoke_test` work. See
[`tools/map_annotator/README.md`](tools/map_annotator/README.md).

## Documentation

| Doc | Contents |
|---|---|
| [`docs/01_install.md`](docs/01_install.md) | Conda env, asset download + checksum check, manual weights, licenses |
| [`docs/02_single_episode.md`](docs/02_single_episode.md) | One-episode launch |
| [`docs/03_batch_eval.md`](docs/03_batch_eval.md) | Batch evaluation, configs, resume, summaries |
| [`docs/04_baselines.md`](docs/04_baselines.md) | Baseline commands, batch runs, custom-method interface |
| [`docs/05_visualization.md`](docs/05_visualization.md) | Debug UI + first-person video |
| [`docs/06_evaluation.md`](docs/06_evaluation.md) | Metrics and log format |
| [`docs/07_troubleshooting.md`](docs/07_troubleshooting.md) | Common failures (Vulkan adapter, ports, crashes) |
| [`docs/08_teleop.md`](docs/08_teleop.md) | Scenario teleop + keyboard driving |
| [`docs/09_multi_gpu.md`](docs/09_multi_gpu.md) | Multi-GPU workers, pipelines, baseline matrix |
| [`docs/scenes/`](docs/scenes) | Per-scene design notes (passage, corner, park, plaza) |
| [`docs/run_manual.md`](docs/run_manual.md) | Detailed CLI reference |
| [`docs/evaluation_reference.md`](docs/evaluation_reference.md) | Metric definitions in depth |
| [`docs/episode_termination.md`](docs/episode_termination.md) | Episode termination rules |
| [`docs/sensor_config.md`](docs/sensor_config.md) | Robot sensor configuration |
| [`docs/visualization_reference.md`](docs/visualization_reference.md) | Visualization reference |
| [`docs/teleop_guide.md`](docs/teleop_guide.md), [`docs/teleop_data_collection.md`](docs/teleop_data_collection.md) | Teleop guide and data collection protocol |
| [`tools/teleop/README.md`](tools/teleop/README.md) | Standalone Scout / G1 / Dog teleop |
| [`baselines/README.md`](baselines/README.md) | Baseline implementations and adapters |

## Citation

```bibtex
@article{followbench2,
  title   = {FollowBench 2.0},
  author  = {FollowBench 2.0 Team},
  year    = {2026},
  note    = {Preprint / forthcoming. Please update once the camera-ready version is public.}
}
```

## License

The FollowBench 2.0 code is released under the [MIT License](LICENSE)
(Copyright (c) 2026 The FollowBench 2.0 Authors).

This repository bundles and redistributes source code, model weights, and
binary assets from third-party projects, each under its own license
(e.g. CARLA, Ultralytics, KPR, ORTrack, DINOv3, TrackVLA, OA-VAT). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete
attribution and the per-component license texts under `licenses/`.
Model weights and binaries are downloaded separately at install time; their
licenses are listed in `docs/01_install.md`.
