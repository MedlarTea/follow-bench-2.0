# 04 — Baselines

Each row matches the baseline table in `README.md`. These are the commands
provided for this release.

## Locked evaluation protocol (v0.1)

| Knob | Value |
|---|---|
| Follow pose | `back` only |
| Desired distance | `1.5` m |
| Difficulties | all of `easy` / `hard` / `boss` / `legend` enter the average |
| Trials per cell | `10` (`baselines_<row>_10x`); `baselines_<row>_50x` configs are also provided |
| Scenes | `plaza`, `corner`, `park`, `passage` |
| ReID thresholds | `ridge=0.35`, `cosine=0.55`, `id_switch=0.05` (`PerceptionConfig` defaults) |

Published back-follow SR (%) reference:

| Method | Plaza | Corner | Park | Passage | Avg |
|---|---:|---:|---:|---:|---:|
| MPC w/ Traj. + KPR | 23.5 | 96.5 | 0.0 | 83.0 | 50.8 |
| MPC w/ Traj. + R18 | 25.0 | 95.4 | 0.0 | 86.9 | 51.6 |
| MPC w/ DS. + R18 | 35.0 | 90.0 | 0.0 | 30.0 | 38.8 |
| MPC + R18 | 16.1 | 93.9 | 0.0 | 71.0 | 45.2 |

## Shared evaluation flags

```bash
--enable-evaluation \
--auto-spawn-sec 1 --auto-follow-sec 4 \
--kinematics-mode kf
```

## Commands

### MPC w/ Traj. + R18

Default `rda_traj` injects lidar/NPC obstacles into MPC. To match the
baseline reference numbers, disable that path:

```bash
PLANNER=rda_traj FOLLOW_POSITION=back DESIRED_DISTANCE=1.5 DIFFICULTY=easy \
  bash scripts/run_episode.sh --scene corner \
    --enable-evaluation --use-perception --reid-mode basic --kinematics-mode kf \
    --rda-traj-disable-lidar-obstacles \
    --auto-spawn-sec 1 --auto-follow-sec 4
```

### MPC w/ Traj. + KPR

```bash
PLANNER=rda_traj ... bash scripts/run_episode.sh --scene corner \
  --enable-evaluation --use-perception --reid-mode kpr --kinematics-mode kf \
  --rda-traj-disable-lidar-obstacles \
  --auto-spawn-sec 1 --auto-follow-sec 4
```

### MPC w/ DS. + R18

`rda_search` + basic ReID. Search when the target is lost; obstacles use
`build_rda_obstacles` (same family as `rda_lidar`).

```bash
PLANNER=rda_search ... bash scripts/run_episode.sh --scene corner \
  --enable-evaluation --use-perception --reid-mode basic --kinematics-mode kf \
  --auto-spawn-sec 1 --auto-follow-sec 4
```

### MPC + R18

`rda_lidar` + basic ReID (not plain `rda`).

```bash
PLANNER=rda_lidar ... bash scripts/run_episode.sh --scene corner \
  --enable-evaluation --use-perception --reid-mode basic --kinematics-mode kf \
  --auto-spawn-sec 1 --auto-follow-sec 4
```

### TrackVLA

```bash
PLANNER=trackvla ... bash scripts/run_episode.sh --scene corner \
  --enable-evaluation --auto-spawn-sec 1 --auto-follow-sec 4 \
  --trackvla-instruction "Follow the person"
# do not pass --use-perception
```

### OA-VAT

```bash
PLANNER=oa_vat ... bash scripts/run_episode.sh --scene corner \
  --enable-evaluation --auto-spawn-sec 1 --auto-follow-sec 4
# do not pass --use-perception
```

## Batch runs

Every scene has `configs/scenes/<scene>/batch/baselines_<row>_{10x,50x}.sh` with
`FOLLOW_POSITION=back`, `DESIRED_DISTANCE=1.5` and all difficulties; Traj. rows
forward `--rda-traj-disable-lidar-obstacles`. Run the whole table with
`scripts/baselines/run_all_baselines.sh --trials 10|50` (see `docs/03_batch_eval.md`
and `docs/09_multi_gpu.md`).

Single-episode shell stubs: `configs/baselines/*.sh`.

## Acceptance

- Smoke: one trial each, no crash, `eval_result.json` written.
- Full table: aggregate SR/ASR/… within a agreed tolerance of the baseline row
  (recommend ±3–5 abs points for rates; document GPU + UE build).

## Custom methods (`--planner custom`)

### End-to-end

Subclass `FollowerPolicyAdapter` (`followbench/sim/follow_policy_adapter.py`):

```python
class MyPolicy(FollowerPolicyAdapter):
    def reset(self) -> None: ...
    def act(self, obs: FollowObservation) -> FollowAction: ...
    def get_debug_info(self) -> dict: ...   # optional
```

`FollowObservation` (`followbench/sim/core_types.py`) carries the ego state,
front / left / right RGB + depth, LiDAR, calibration, **and** privileged ground
truth (`target`, `npcs`, `target_visible`). An end-to-end method must only use
the ground truth for the initial target designation.

### Modular: perception -> planning

`followbench/interfaces/perception_planning.py`:

| Piece | Role |
|---|---|
| `PerceptionModule.perceive(obs, hint) -> PerceptionOutput` | sensors -> target estimate + tracked people + boxes |
| `Planner.plan(obs, perception) -> FollowAction` | target estimate + sensors -> `(v_mps, w_radps)` |
| `PerceptionPlanningPolicy(perception, planner)` | the `FollowerPolicyAdapter` the runners call |

`PerceptionOutput` fields: `has_target` (must be False when lost / unsure),
`target_xy`, `target_vel`, `target_yaw_rad`, `target_id`, `state`,
`people: List[TrackedPerson]`, `bboxes: List[BBox2D]`, `timing_ms`, `extras`.

`PerceptionPlanningPolicy`:

- passes perception and planner a copy of the observation with ground truth
  removed (`sensor_only_observation`); a perception module may opt out with
  `requires_privileged_obs = True` (oracle / debugging only);
- gives perception a `TargetHint` (GT target position and front-camera bbox)
  until it first reports `has_target=True` (`hint_policy="until_first_lock"`;
  also `"always"` / `"never"`);
- fills the debug fields read by the evaluation logger, the target-identity
  metrics, the HUD and the bbox overlays. Planner debug keys such as
  `goal_point`, `traj_points`, `obstacles` are forwarded to the step log.

Pass `--use-perception` (or list the planner in `PLANNERS_WITH_PERCEPTION`)
to enable the target-identity metrics for a modular method.

### Registering a factory

```python
def build_policy(ctx: PolicyBuildContext) -> FollowerPolicyAdapter:
    # ctx.scene, ctx.dt, ctx.desired_distance, ctx.follow_position,
    # ctx.grid_npz, ctx.options (from --policy-arg), ctx.args
    return PerceptionPlanningPolicy(MyPerception(), MyPlanner())
```

```bash
PLANNER=custom bash scripts/run_episode.sh --scene corner \
  --policy-factory my_pkg.my_policy:build_policy --policy-arg max_speed=1.2
```

The spec is `module:callable` or `path/to/file.py:callable` (relative paths
resolve against the repo root). In batch configs set `PLANNERS_BACK_ONLY="custom"`
(or `PLANNERS_WITH_PERCEPTION="custom"`) and pass the factory through
`EXTRA_EPISODE_ARGS="--policy-factory ..."` or `FB_POLICY_FACTORY=...`
(`FB_POLICY_ARGS="K1=V1 K2=V2"` for options).

Minimal example (GT oracle + pure pursuit, not a baseline):
`examples/custom_policy/gt_pursuit.py`.
