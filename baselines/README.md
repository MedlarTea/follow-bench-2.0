# Follow-Bench Planners

This directory contains all follower policy implementations for the Scout diff-drive robot.
Each planner wraps a `FollowerPolicyAdapter` ABC (defined in `followbench/sim/follow_policy_adapter.py`)
and can be selected via `--planner <name>` when running an episode manager.

---

## Implemented Planners

### `pid` — PID (default)
**Adapter:** `adapters/pid_adapter.py` · `PIDFollowerPolicy`  
**Planning:** `planning/pid/pid.py` · `PIDFollowerController`

Port of the ROS MonoController PID follower. Uses two PID controllers:
- Angular: bearing error → angular velocity
- Linear: forward distance error → linear velocity (disabled when target is behind)

Simple, fast, reliable. Good default for most scenarios.

### `dwa_traj` — DWA-Traj (stateful human-following DWA)
**Adapter:** `adapters/dwa_traj_adapter.py` · `DwaTrajFollowerPolicy`  
**Planning:** `planning/dwa/dwa_traj.py` · `DwaTrajPlanner`

CARLA-native DWA follower for the Scout diff-drive robot. This planner is
designed for human following rather than generic point navigation: the current
filtered follow point is the strong tracking target, while the predicted
follow-goal trajectory is used only as a weak direction hint. This avoids
forcing the robot rollout endpoint to chase an uncertain future target point
that may move through obstacles or differ from the actual human motion.

Pipeline per control tick:
1. Reads robot, target, NPC, lidar, and GT state from `FollowObservation`.
2. Estimates the actual robot odometry velocity from consecutive robot poses
   and blends it with the previous estimate. This velocity is used by the
   dynamic window instead of assuming that the last command was perfectly
   executed.
3. Builds a target prediction through the shared prediction framework
   (`TrajectoryPredictionService`). If the configured predictor is unavailable
   or not ready, it falls back to constant-velocity prediction from target GT.
4. Generates a DWA-local follow-goal trajectory with
   `planning/dwa/follow_goal_strategy.py`. The strategy is intentionally
   separated from SFM so SFM tuning cannot change DWA behavior.
5. Smooths the follow-goal trajectory with
   `planning/dwa/target_reference.py`. The first point is the current tracking
   goal; later points provide a short direction hint.
6. Converts non-target pedestrians to dynamic circular obstacles
   `[x, y, vx, vy, radius]`.
7. Builds a lidar local occupancy map when lidar is available. The target
   person's connected occupancy blob is removed near the target GT position,
   then the map is inflated by robot radius and queried through the shared
   ESDF-backed `OccupancyGridMap`.
8. Samples `(v, w)` in the dynamic window, rolls each command over the horizon,
   scores each rollout, and returns the first command of the best rollout.

DWA target-goal design:
- The current target for the DWA cost is `current_goal = goal_traj[0]`.
- `hint_point` is selected a short distance along the filtered goal trajectory:
  `hint_distance = clamp(abs(v_robot) * 0.45 + 0.20, 0.15, 0.55)`.
  It only adds a weak direction preference; it is not a terminal trajectory
  tracking target.
- Back/side/front following are expressed as a stateful "slot" around the
  target:
  - `back`: desired slot is behind the filtered target heading
    (`target_yaw + pi`) unless `back_goal_mode="bearing"` is configured.
  - `front`: desired slot is along the filtered target heading.
  - `left_side` / `right_side`: the requested side sign is locked and the slot
    angle is rate-limited, so a target U-turn does not immediately jump the
    goal from one side of the human to the other.
- Target yaw is filtered from velocity direction when target speed is high
  enough, otherwise the previous filtered yaw is kept. Target yaw and slot
  angle both have rate limits. This prevents back-follow "tail whipping" and
  side-follow left/right jumps during turns.
- The final follow goal is filtered across ticks with separate longitudinal
  and lateral speed limits. Side following uses stricter lateral limits because
  small yaw changes can otherwise create large lateral target-point jumps.

Planner modes and special handling:
- `FOLLOW`: normal forward following. Forward speed is gated by heading error
  so the robot slows when it is badly misaligned.
- `ALIGN`: entered when `abs(angle_to_goal) >= 45 deg` and the goal is not very
  close; exited below `20 deg`. Linear velocity is clamped near zero and the
  angular command is biased toward a bounded align yaw rate. This prevents the
  robot from driving forward while facing away from the follow goal.
- `REVERSE`: entered when the target is too close, when the goal is behind the
  robot, or when the target is walking toward the robot at close distance. It
  allows bounded reverse speed and reverse yaw samples, so the robot can back
  away during human U-turns instead of always trying to rotate in place.
- `RECOVERY`: entered after repeated no-progress frames while in `FOLLOW`.
  Recovery samples a small reverse speed and a fixed turn direction based on
  the goal bearing. It is intentionally short and bounded so it does not become
  a permanent spin behavior.

Dynamic window and sampling:
- Velocity window:
  `v in [v_robot - a_v * dt, v_robot + a_v * dt]`,
  `w in [w_robot - a_w * dt, w_robot + a_w * dt]`, clipped by Scout limits.
- `FOLLOW` uses a larger linear acceleration allowance for faster startup.
  `ALIGN` increases angular acceleration allowance. `REVERSE` and `RECOVERY`
  clamp the velocity range to bounded reverse/near-zero forward values.
- Candidate samples always include `0`, current measured velocity, and the last
  command. DWA also explicitly includes `w=0` and mode-specific yaw samples, so
  straight-line candidates are not lost due to discretization.
- Rollouts use diff-drive kinematics:
  `x[k+1] = x[k] + v*cos(yaw)*dt`,
  `y[k+1] = y[k] + v*sin(yaw)*dt`,
  `yaw[k+1] = yaw[k] + w*dt`.

Reference speed:
- `ALIGN`: `v_ref = 0`.
- `RECOVERY`: `v_ref = recovery_reverse_speed`.
- `REVERSE`:
  ```text
  v_ref = -clamp(max(0.60*spacing_error, 0.35*goal_error, 0.15),
                 0.15, reverse_max_speed)
  ```
- `FOLLOW`:
  ```text
  distance_error = max(distance_to_goal - 0.20, 0)
  angle_gate = max(0, cos(min(abs(angle_to_goal), pi/2)))
  v_ref = clamp((max(target_speed, 0.25) + 0.45*distance_error)
                * angle_gate, 0, max_speed)
  ```
  This tracks a target-relative desired speed rather than rewarding absolute
  high speed.

Cost function:

```text
J = w_d*J_dist + w_p*J_progress + w_h*J_heading + 0.30*J_hint
  + w_o*J_obstacle + w_v*J_velocity + w_t*J_turn
  + w_s*J_smooth + w_r*J_reverse
```

Default weights are:
`w_h=0.7`, `w_d=0.8`, `w_o=1.5`, `w_v=1.2`, `w_p=0.45`,
`w_t=0.18`, `w_s=0.12`, `w_r=0.7`.

All major cost terms are normalized and clipped before weighting so one raw
term does not dominate simply because it has a larger numeric scale:
- `J_dist`: minimum distance to `current_goal` over the first second,
  normalized by `distance_scale=1.2`.
- `J_progress`: penalty when the rollout does not reduce distance to
  `current_goal`, normalized by `progress_scale=0.8`.
- `J_heading`: first-step bearing error to `current_goal`, normalized by `pi`.
  For reverse rollouts, effective yaw is flipped by `pi` so backing up is
  scored with the rear direction.
- `J_hint`: weak alignment between the rollout displacement at about `0.6 s`
  and the vector from `current_goal` to `hint_point`.
- `J_velocity`: squared error to `v_ref`,
  `((v_end - v_ref) / velocity_scale)^2`, with `velocity_scale=0.8`.
- `J_turn`: normalized absolute yaw rate, discouraging unnecessary spinning.
- `J_smooth`: normalized command jump from measured robot velocity to rollout
  end velocity, discouraging sudden acceleration/yaw-rate changes.
- `J_reverse`: penalizes reverse motion in normal modes and forward motion in
  `REVERSE`; this is reduced in `REVERSE` so backing away remains available.
- Mode-specific biases are added after the base cost:
  - `ALIGN` penalizes linear speed and biases `w` toward `_align_yawrate`.
  - `REVERSE` biases `w` toward `_reverse_yawrate`, where the goal is treated
    relative to the robot rear direction.
  - `RECOVERY` biases `w` toward the recovery turn direction.

Obstacle handling:
- Static obstacles come from the lidar local map ESDF when lidar is available.
  DWA queries `distance_to_obstacle_points(..., unknown_is_occupied=False)` so
  unknown cells do not freeze the robot by default. If no map is available,
  static obstacle cost is zero.
- Dynamic pedestrians come from GT NPC states except the target track. Each
  non-target pedestrian is predicted as constant velocity:
  `ped_xy(t) = ped_xy0 + ped_vel*t`.
- Dynamic clearance at each rollout point is:
  ```text
  clearance = distance(robot_xy, ped_xy(t))
            - robot_radius - ped_radius - 0.15
  ```
- Static clearance is the ESDF distance at rollout points after map inflation.
- Static and dynamic obstacle costs are computed separately and merged with:
  `J_obstacle = max(J_static, J_dynamic)`,
  `hard_invalid = hard_static OR hard_dynamic`,
  `min_clearance = min(clearance_static, clearance_dynamic)`.
- Hard filtering is only applied to near-term unavoidable collisions. The first
  collision time is compared against
  ```text
  hard_horizon = clamp(abs(v_robot)/max_acc_v + reaction_time,
                       hard_horizon_min, hard_horizon_max)
  ```
  Farther future collisions remain selectable but receive soft cost, because
  only the first command is executed and the next tick may select a different
  safe path.
- Soft obstacle risk is time-discounted:
  `risk = soft_collision_penalty` if clearance is already too small, otherwise
  `exp(-clearance / clearance_sigma)`,
  weighted by `exp(-t / obstacle_tau)` over the rollout.
- If every candidate is hard-invalid, DWA uses a conservative fallback score
  based on penetration depth, low speed, low yaw rate, and soft obstacle risk
  instead of returning an arbitrary max-turn command.

Local-map target clearing:
- The map builder first creates a raw local occupancy grid with no inflation.
- It finds the connected occupied component near the target GT position within
  `target_radius + target_clear_margin`.
- Only that bounded connected component is cleared. This prevents the target
  person from being treated as a static obstacle while avoiding broad erasure
  of nearby walls or unrelated objects.
- After target clearing, the map is inflated by robot radius and exported as an
  ESDF-backed `OccupancyGridMap`.

Useful adapter parameters:
- `predictor_name="cv"`, `use_prediction=True`
- `desired_distance=1.5`, `follow_position="back" | "front" |
  "left_side" | "right_side"`
- `predict_horizon=2.0`
- DWA weights: `weight_heading`, `weight_distance`, `weight_obstacle`,
  `weight_velocity`
- Obstacle/map parameters: `target_radius=0.35`,
  `local_map_resolution=0.2`, `target_clear_margin=0.15`,
  `lidar_range_max=LIDAR_RANGE_MAX_M`

Debug output:
- `traj_points`: selected rollout shown by `debug_vis`
- `goal_point`: filtered current follow goal
- `goal_traj`: filtered DWA goal trajectory
- `raw_goal_point`, `filtered_goal_point`, `goal_jump`: goal smoothing
  diagnostics
- `predicted_target_traj`: predicted target positions used for goal generation
- `current_goal_point`, `hint_point`, `hint_index`: DWA strong target and weak
  direction hint
- `mode`, `v_ref`, `distance_to_goal`, `target_distance`,
  `target_closing_speed`, `reverse_reason`
- `slot_angle`, `desired_slot_angle`, `slot_error`, `side_locked`: stateful
  follow-slot diagnostics for turn/U-turn behavior
- `cost_distance`, `cost_progress`, `cost_heading`, `cost_hint`,
  `cost_obstacle`, `cost_static_obstacle`, `cost_dynamic_obstacle`,
  `cost_velocity`, `cost_turn`, `cost_smooth`
- `min_static_clearance`, `min_dynamic_clearance`, `min_clearance`,
  `first_collision_time`, `hard_invalid_count`, `candidate_count`,
  `fallback_used`
- Map visualization/debug fields: occupied cells, observed free cells, map
  outline, occupancy/ESDF/hybrid RGBA exports, target-blob clearing status, and
  local map resolution.

### `sfm` — Kinematic Social Force Planner
**Adapter:** `adapters/sfm_adapter.py` · `SfmFollowerPolicy`  
**Planning:** `planning/sfm/` · `SfmPlanner`

Predictive SFM local planner that keeps the Social Force Model as the motion
intention layer while adding a small kinematic execution layer. The planner:
1. Builds a short-horizon constant-velocity prediction for the target and
   non-target pedestrians. The target prediction first passes through an SFM
   reference filter that smooths target position, velocity, and yaw, and limits
   unrealistic acceleration/yaw-rate jumps from dense pedestrian interactions.
2. Generates an SFM-specific follow-goal trajectory. Side-follow goals are
   fixed to the requested side inside SFM so this planner does not inherit the
   shared `behavior.follow_goal` side-switching behavior used by other
   algorithms.
3. Smooths the final follow-goal trajectory across ticks. This makes SFM track
   a stable moving reference instead of reacting directly to every target-body
   yaw or velocity jitter.
4. Computes SFM desired world-frame velocity from goal attraction, pedestrian
   repulsion, weak target personal-space repulsion, and ESDF static-obstacle
   repulsion through the shared `MapQuery` interface.
5. Rolls the desired velocity through a configurable kinematics adapter and
   returns the first command. The debug trajectory is therefore a curved
   kinematically feasible rollout rather than a constant-velocity straight
   line.

The target-reference filter is deliberately SFM-local: it does not change
`behavior.follow_goal` and therefore does not change the target-point behavior
of PID, DWA, RDA, or BSO-HFC.

SFM stability improvements:
- Static obstacles are handled through the shared local-map/ESDF `MapQuery`
  interface, so the SFM force field includes wall/obstacle repulsion instead of
  only pedestrian interaction.
- The target reference filter prevents dense-crowd target jitter from being
  amplified into robot yaw oscillation. It smooths target position, fuses
  observed and finite-difference velocity, prefers velocity-derived yaw when
  the target is moving, uses recent target history as a path tangent, and
  limits target acceleration/yaw-rate changes.
- The final follow goal is also filtered across ticks. This is especially
  important for `left_side`/`right_side`, where a small target-yaw jump can move
  the side-follow point by a large lateral distance. Goal smoothing is
  decomposed into longitudinal and lateral components, with stricter lateral
  limits for side following.
- Pedestrian repulsion is capped and temporally filtered, and dense-crowd
  scenes reduce the goal-attraction speed. This keeps alternating left/right
  pedestrian forces from immediately flipping the desired velocity direction.
- Goal speed is target-speed-aware: normal following stays near the filtered
  target speed plus a small margin, and only increases gradually when the
  follow error is large.
- The diff-drive kinematics adapter applies a small heading deadband, lower
  heading gain, turn-speed coupling, command low-pass filtering, acceleration
  limits, and first-command jerk limits before producing
  `FollowAction(v_mps, w_radps)`.
- Debug output exposes both raw and filtered target/goal values so oscillation
  can be traced to the target observation, goal generation, force field, or
  command conversion stage.

Current kinematics support:
- `diff_drive` (default): outputs `FollowAction(v_mps, w_radps)` for Scout.
- `omni`: internal interface is present for future omnidirectional actions.
- `ackermann`: internal rollout interface is present for future steering-based
  platforms.

Useful adapter parameters:
- `kinematics="diff_drive" | "ackermann"` at the adapter level. `omni`
  remains an internal planner interface until the platform action type exposes
  world-frame lateral velocity.
- `predict_horizon=2.0`, `plan_dt=0.1`
- `back_goal_mode="target_heading" | "bearing"`
- `debug_print=True`, `debug_print_interval=10`
- Scout limits follow the other planners by default:
  `robot_radius=0.5`, `max_v_mps=2.5`, `max_w_radps=2.5`,
  `max_acc_v=2.0`, `max_acc_w=4.0`
- Comfort limits default to `max_jerk_v=12.0`, `max_jerk_w=16.0`

Debug output:
- `traj_points`: selected kinematic rollout shown by `debug_vis`
- `goal_traj`: predicted follow-goal trajectory
- `predicted_target_traj`: filtered constant-velocity target prediction
- `force_traj`: SFM total force along the rollout
- `clearance_traj`: ESDF clearance samples along the rollout
- `target_raw_*`, `target_filtered_*`, `raw_goal_point`,
  `filtered_goal_point`, `goal_jump`: target-reference stability diagnostics
- `raw_human_force`, `human_force`, `crowd_count`, `target_speed_for_goal`:
  force and speed-damping diagnostics for dense-crowd tuning
- terminal prints low-frequency command/force/clearance summaries when
  `debug_print=True`

### `rda_traj` — RDA-MPC Trajectory Chasing
**Adapter:** `adapters/rda_traj_adapter.py` · `RdaTrajFollowerPolicy`  
**Planning:** `planning/rda/rda_traj.py` · `RdaTrajPlanner`

Trajectory-chasing variant of RDA-MPC. Each tick:
1. Predicts the target's future trajectory with a constant-velocity (CV) model.
2. Computes the robot's back-following goal trajectory from those predictions.
3. By default builds obstacles with the same `build_rda_obstacles` path as
   `rda_lidar` (lidar DBSCAN rectangles + nearby NPC boxes; target masked).
4. Runs `mpc_chasing_traj.MPC` to track the goal trajectory under those obstacles.

MPC output `[v, w]` is used directly. Supports reverse driving.
`enable_uncertainty_brake` (default `False`) disables the NIS-based braking so
the robot reacts immediately when the human walks toward it.

Legacy behavior: pass `--rda-traj-disable-lidar-obstacles`
(empty obstacle list except optional side-follow target protection).

With `--use-perception`, target/NPCs come from the camera pipeline; lidar
clusters still come from `obs.lidar_points` when lidar obstacles are enabled.

**Extra deps:** same as `rda_lidar` when lidar obstacles are on
(`cvxpy`, `pathos`, `scipy`, `osqp`, `sklearn`, `opencv`)

---

### `rda_search` — RDA-MPC + Target Search
**Adapter:** `adapters/rda_search_adapter.py` · `RdaSearchFollowerPolicy`  
**Planning:** `behavior/search/` + `planning/rda/rda_search.py`

CARLA-native migration of the RDA dynamic target-search follower. Uses
`FollowObservation` directly: visible targets are followed with point-goal RDA;
lost/occluded targets switch to a reusable search module that predicts recent
pedestrian motion, tries an occluder-overtake goal, then falls back to a
dynamic density/velocity-field search goal.

Shared first-stage modules now live in dedicated packages: `prediction/` wraps
the trajectory predictors, `behavior/search/` owns target-search behavior, and
`maps/` provides the reusable local occupancy-grid and ESDF query layer used by
`rda_search`.
This stage uses GT target/NPC states, but target history is not refreshed from
current GT while `target_visible=False`.

Exposes `get_debug_info()` with standard `obstacles` and `traj_points`, plus
search samples, selected search goal, search mode, occluder id, and predicted
target trajectory.

**Extra deps:** same as `rda_lidar` plus `shapely`

### `bso_hfc` — BSO-HFC Human Following
**Adapter:** `adapters/bso_hfc_adapter.py` · `BsoHfcFollowerPolicy`  
**Planning:** `planning/bso_hfc/` · `BSOHFCPlanner`

CARLA-native BSO-HFC migration. The planner uses the shared `behavior.follow_goal`
follow-position target calculation, then runs Hybrid A*, B-spline optimization,
and MPC tracking over the shared `maps` ESDF query layer. Target-loss search is
intentionally not included.

**Extra deps:** `scipy`, `osqp` for the full MPC QP solve. If `osqp` is absent,
the tracker falls back to feed-forward reference tracking.

### Package roles

The package-level entry points are intentionally minimal:

- `adapters`
  - Preferred namespace for concrete follower-policy entry points.
  - Root-level `*_adapter.py` modules have been removed; use this package for
    all platform policy imports.
- `common`
  - Exports the shared state/result dataclasses from `types.py`
    (`RobotState2D`, `AgentState2D`, `PredictionBundle`, `SearchResult`)
- `prediction`
  - Exports `TrajectoryHistory`, `TrajectoryPredictionService`, and `get_predictor`
  - Owns predictor backends under `prediction/backends/`
  - Owns SGAN helpers and model checkpoints under `prediction/utils/` and `prediction/checkpoints/`
  - The old `traj_predictor/` compatibility package has been removed.
- `perception`
  - Owns planner-side sensor / GT obstacle construction helpers.
  - Also owns people-tracking / relock pipelines and GT scene/visibility helpers.
- `behavior.search`
  - Exports `SearchConfig` and `TargetSearchPlanner`
- `maps`
  - Exports `OccupancyGridMap`, `NullMapQuery`,
    `build_local_occupancy_grid`, and ESDF helpers
- `planning`
  - Owns planner/control implementations under `planning/pid/`,
    `planning/dwa/`, `planning/rda/`, `planning/sfm/`, and
    `planning/bso_hfc/`
  - `planning/rda/core/` owns solver-backed RDA MPC internals
    (`mpc_chasing_point2.py`, `mpc_chasing_traj.py`, `rda_solver*.py`)
  - Small planners keep explicit implementation modules. SFM uses a small
    package split (`sfm.py`, `target_reference.py`, `force_field.py`,
    `kinematics.py`, `follow_goal_strategy.py`, `types.py`) because it
    separates target-reference filtering, social-force intent, follow-goal
    generation, and robot kinematics.
  - The previous migration-only wrapper layers have been removed
  - `adapters/*.py` own platform observation/action wiring and call planning
    entrypoints here
  - Current `*_tpt.py` modules intentionally remain adapter-side perception
    shells that wrap an inner planner plus the shared `PeopleTrackerPipeline`

This keeps imports short at adapter call sites while still making it clear
which layer owns which responsibility.

---

## Shared Map Layer (`maps`)

The reusable local-map utilities used by `rda_search` live under
`baselines/maps/`. This layer is intended for planner-side
local reasoning and visualization; it is deliberately kept separate from the
planner logic itself so future planners can reuse the same map/query API
without inheriting `rda_search`'s search behavior.

### Current modules

- `map_query.py`
  - Abstract planner-facing map interface.
  - Provides point queries (`is_free`, `is_observed`, `is_unknown`,
    `distance_to_obstacle`) plus batch point queries
    (`is_free_points`, `is_observed_points`, `is_unknown_points`,
    `distance_to_obstacle_points`) and line queries
    (`ray_clear`, `line_min_clearance`).
- `occupancy_grid.py`
  - Main local-map container: occupancy grid, observed mask, inflated obstacle
    mask, and ESDF distance field.
  - Uses vectorized `numpy` indexing for bulk queries.
  - Uses `cv2.dilate` for inflation and `cv2.distanceTransform` for ESDF.
  - Exposes two ESDF unknown-space policies:
    - conservative: unknown cells are treated as blocked
    - optimistic: unknown cells are treated as free unless occupied/inflated
  - Exposes debug exports:
    - `export_occupancy_rgba()` → occupancy-style local-map visualization
    - `export_esdf_rgba()` → ESDF heatmap
    - `export_hybrid_rgba()` → occupancy map with ESDF overlay
- `lidar_grid_builder.py`
  - Builds a local map from:
    - `obs.lidar_points`
    - `obs.lidar_extrinsics_robot_to_sensor`
    - robot pose
  - Produces:
    - `occupied_grid`
    - `observed_grid`
    - `inflated_grid`
    - ESDF-backed `distance_field`
- `esdf.py`
  - Pure utility functions for ESDF-related operations.
  - Current helpers:
    - `compute_unsigned_esdf(...)`
    - `sample_line_min_clearance(...)`
- `null_map.py`
  - Placeholder map implementation for planners that do not want a real local
    map yet.

### Local map occupancy classes

The local map distinguishes four occupancy classes:

- `unknown`
  - Cells that have not been confirmed by the current local map.
- `free`
  - Observed and traversable cells.
- `inflated`
  - Obstacle safety buffer after dilation.
- `occupied`
  - Obstacle hit cells.

### ESDF Unknown-Space Policies

The default `OccupancyGridMap.distance_field` is conservative: it computes
clearance only over cells that are both observed and not inflated. Unknown cells
therefore have zero clearance. This is appropriate for planners that should not
drive into unobserved space.

Some local planners, including BSO-HFC, assume unknown space is traversable and
only known obstacles should block the local plan. For those planners, use the
optimistic ESDF policy:

```python
# Conservative, default: unknown = occupied.
clearance = grid.distance_to_obstacle(x, y)
clearance = grid.distance_to_obstacle(x, y, unknown_is_occupied=True)

# Optimistic: unknown = free, occupied/inflated cells still block.
clearance = grid.distance_to_obstacle(x, y, unknown_is_occupied=False)
clearances = grid.distance_to_obstacle_points(points_xy, unknown_is_occupied=False)
line_clearance = grid.line_min_clearance(x0, y0, x1, y1, unknown_is_occupied=False)

# Direct access when a planner needs to sample the field itself.
field = grid.distance_field_for_unknown_policy(unknown_is_occupied=False)
```

Keep the policy choice explicit at planner boundaries. A conservative ESDF can
make a target outside the currently observed free region look unreachable,
whereas an optimistic ESDF can plan through unobserved space and should be used
only when that matches the planner's assumptions.

### Debug visualization modes

When running with `--debug`, the 2D debug visualizer can show the local map in
three modes. Use the clickable box in the upper-left corner of the debug window
or press `V` while the debug window has focus:

- `occupancy`
  - Shows occupancy classes directly:
    - light grey = `unknown`
    - blue = `free`
    - yellow = `inflated`
    - dark grey = `occupied`
- `esdf`
  - Shows the ESDF heatmap over observed free space.
  - Useful for checking whether distance-to-obstacle values and clearance
    gradients look reasonable.
- `hybrid`
  - Default mode.
  - Keeps occupancy classes visible and overlays ESDF coloring on observed
    free space.

The dashed ring around the robot in the debug window is the current
`lidar_range_max`, i.e. the local-map sensing radius used by the planner-side
LiDAR map builder.

### Implementation notes

- The map layer is intentionally library-backed and vectorized where possible:
  - `numpy` for batch coordinate transforms and grid indexing
  - `opencv-python` (`cv2`) for morphology, line rasterization, and ESDF
- We avoid handwritten low-level grid algorithms unless there is no practical
  library alternative.
- `rda_search` currently consumes this map for local occupancy reasoning and
  debug output, but the ESDF layer is not yet used to change the `rda_search`
  search policy itself.

---

### `rda_lidar` — RDA-MPC + Lidar Obstacle Avoidance
**Adapter:** `adapters/rda_lidar_adapter.py` · `RdaLidarFollowerPolicy`  
**Planning:** `planning/rda/rda_lidar.py` · `RdaLidarPlanner`

Same MPC core as `rda`, but builds obstacle list from a fused lidar + GT-NPC pipeline:
1. DBSCAN-clusters single-line lidar returns in world BEV frame → rectangular obstacles.
2. Injects GT non-target NPCs as additional box obstacles with velocity.
3. Masks the followed target from the lidar cluster list.

Exposes `get_debug_info()` returning `{"obstacles": [...], "traj_points": [...]}` for the 2D visualizer.

**Extra deps:** `sklearn`, `opencv-python` (plus RDA deps above)

### `gt_planner` — GT Expert RDA-MPC (oracle baseline)
**Adapter:** `adapters/gt_planner_adapter.py` · `GtPlannerFollowerPolicy`  
**Planning:** `planning/rda/rda_lidar.py` · `RdaLidarPlanner`

Oracle-style follower for benchmarking and upper-bound evaluation. Assumes
full simulator GT is available and does **not** use the YOLO/ReID perception
frontend (`--use-perception` is rejected).

Requires `grid_npz=` at construction (episode managers pass `--grid-npz`).

Per tick:
1. **Target tracking** — GT target pose/velocity via `follow_goal_pose_from_obs`.
2. **Dynamic obstacles** — all non-target NPCs within 12 m as velocity-aware boxes.
3. **Static obstacles** — `obstacle_inflated` from `gridmap_roi.npz`, cropped
   near the robot (8 m), clustered via connected-components → min-area rectangles
   (`maps/roi_static_map.py`). No lidar DBSCAN.
4. **Global path** — 8-neighbor A* on `free_grid` ∩ `roi_mask`, with wall-clearance
   cost and line-of-sight smoothing (`RoiStaticMap.plan_path_world`), replanned each
   tick from robot pose to the follow goal.
5. **MPC** — `RdaLidarPlanner` (`max_obs_num=5`).

Exposes `get_debug_info()` with `obstacles`, `traj_points`, `goal_point`,
`global_path` (green polyline in `--debug` visualizer).

---

### `rda_depth_tpt` — RDA-MPC + RGB+Depth Tracker (no lidar perception)
**Adapter:** `adapters/rda_depth_tpt_adapter.py` · `RdaDepthTptFollowerPolicy`
**Tracker:** `followbench/perception/ped_tracker_depth.py` · `PedTrackerDepth`
**Re-ID:** `followbench/perception/target_reid.py` · `TargetReID`

Perception-only follower that estimates pedestrian world-BEV positions from
the RGB camera and the co-mounted **depth** camera — no lidar in the perception
path:

1. **YOLO detection + tracking** on the RGB image → bbox + stable per-episode integer ID.
2. **Robust per-bbox depth**: sample a central 50%×50% sub-patch of the depth image,
   filter `[0.3 m, max_range]`, drop the far-tail (>75-th percentile, kills background
   bleed through silhouette gaps), then take the median.
3. **Back-project** `(u_center, v_center, depth_median)` with the camera intrinsics
   → camera frame → robot body frame → world BEV `(x, y)`.
4. **Initial target lock**: closest-to-GT lock at the start (≤ 2.5 m). GT is *never* consulted again after this single bootstrap call.
5. **Target re-identification on loss**: when ByteTrack drops the locked id (occlusion / re-entry / id switch), the Ridge classifier + 512-d deep embedding from `TargetReID` scores every fresh YOLO bbox and re-locks onto the best match after a 3-frame streak. A 6 m world-distance jump check prevents far-away false positives.
6. All other tracked peds become NPC obstacles fed to the inner RDA-MPC.

`get_debug_info()` returns the standard keys (`obstacles`, `traj_points`,
`tracked_peds`) plus `track_bboxes` (image-pixel bboxes with `is_target` and
per-bbox `depth`), a `timing` block (`yolo_ms`, `depth_ms`, `project_ms`,
`reid_crop_ms`, `reid_infer_ms`, `reid_fit_ms`, `track_ms`, `id_ms`, `reid_ms`,
`percep_ms`, `ran_yolo`), and a `reid` block (`trained`, `num_positive`,
`num_negative`, `consecutive`, `timing`).

**Validation (corner scenario, 2026-04-24):**
- 430 perception ticks (~43 s), 0 target losses, distance to target 1.7-1.85 m the whole time.
- Stress test (12 walkers, 80 s): 1 ByteTrack id switch (2 → 110), recovered by ReID within 3 frames (`ridge=0.20  cos=0.85`), no GT consulted.

**Extra deps:** `ultralytics`, `sklearn`, `opencv-python`, `torch`, `torchvision` (plus RDA deps above)

```bash
PLANNER=rda_depth_tpt bash scripts/run_episode.sh --scene <scene> --debug
```

---

### `rda_perception` — RDA-MPC + Unified PerceptionPipeline (FSM-managed)
**File:** `baselines/adapters/rda_perception_adapter.py` · `RdaPerceptionFollowerPolicy`
**Pipeline:** `followbench/perception/perception_pipeline.py` · `PerceptionPipeline`
**Re-ID modes:** `basic` (ResNet, ~5 MB) or `kpr` (SOLIDER-Swin, ~370 MB)

Same RGB+depth perception backbone as `rda_depth_tpt` but the bespoke
identification heuristics (`_identify_target` + GT bootstrap + ReID jump gate)
are replaced with a four-state machine ported from
`follow/follow_server/src/states_standalone`:

1. **InitialState** — picks the first target. If the planner provides a GT
   hint (only allowed while the FSM is in `initial`), pick the visible track
   nearest to it; otherwise the closest in-front pedestrian (lateral offset
   ≤ 0.8 m, distance ≤ 6 m).
2. **InitialTrainingState** — collects 10 positive samples into the Ridge
   classifier before transitioning to active tracking.
3. **TrackingState** — every tick: confirm the lock with
   `TargetReID.predict_one(target_feature)`; if the score drops below
   `id_switch_thresh` (default 0.05) or the track id disappears, fall back to
   `ReidState`. Otherwise, refit the classifier on this frame.
4. **ReidState** — calls `TargetReID.find_target(...)` on every visible track;
   re-locks once the streak gate (3 consecutive positives) fires.

The whole stack lives behind one call:
```python
result = perception.update(rgb, depth, robot_x, robot_y, robot_z, robot_yaw,
                           rgb_intrinsics, rgb_extrinsics,
                           gt_target_xy=gt_xy_or_none)
# result.target_id, result.target_xy_world, result.target_bbox,
# result.all_tracks, result.state_name, result.timing
```
This is the same single-entry API used by `following_node.py` in the source
project. Any other follower planner can drop `PerceptionPipeline` in without
re-implementing tracking + identification.

`get_debug_info()` returns the standard keys plus a full `perception` block
(state name, target id, bbox, world XY, per-track bboxes, all-track world XY,
per-stage timings, ReID gallery snapshot).

**Extra deps:** `ultralytics`, `sklearn`, `opencv-python`, `torch`,
`torchvision` (basic mode); add `timm`, `yacs`, `omegaconf`,
`albumentations==1.3.1`, `tabulate`, `h5py`, `torchmetrics`, `monai`,
`tensorboard`, `wandb`, `deepdiff`, `gdown`, plus the github-only
`segment_anything` and `cosine_annealing_warmup` for KPR mode (see
`environment.multiRobots.yml`).

```bash
PLANNER=rda_perception bash scripts/run_episode.sh --scene <scene> --debug
```

---

### `rda` — RDA-MPC (Regularized Dual ADMM)
**Adapter:** `adapters/rda_adapter.py` · `RdaFollowerPolicy`  
**Planning:** `planning/rda/rda.py` · `RdaPointPlanner`

ADMM-based MPC planner from the follow-bench RDA_planner package. Solves a constrained
quadratic program each tick to generate an optimal (v, w) trajectory over a receding horizon.
Key smoothness measures:
- Goal point uses bearing (not noisy NPC `yaw_deg`)
- EMA filter on goal (x, y, yaw) to suppress tick-to-tick jumps
- `ref_speed` scales with distance error — slows as robot reaches desired standoff
- Output rate limiting on v and w

**Extra deps:** `cvxpy`, `pathos`, `scipy`, `osqp` (see `environment.scout.yml`)

---

### `trackvla` — End-to-End Vision-Language-Action Follower
**Adapter:** `adapters/trackvla_adapter.py` · `TrackVlaFollowerPolicy`
**Vendored model:** `third_party/trackvla/socialRPF/` (4 .py from the social-rpf-vla repo, slimmed)
**Checkpoints:** `data/trackvla/{google,facebook,LLM_hf,ckpts}/...` (≈19 GB, gitignored)
**Source:** social-rpf-vla — `OpenTrackVLAUnifiedAlpha`, single-view "forward" Qwen3-4B unified-nav-qa checkpoint

Unlike every other planner above, this is a **single end-to-end model** instead of a perception → planner pipeline. Each tick:

1. Resize-with-padding `obs.rgb_image` (front camera only) to 384×384.
2. ViT (DINOv3-S/16 + SigLIP-SO/400M) → grid-pool → 4 coarse + 64 fine tokens.
3. Append coarse tokens to a rolling 31-step history.
4. Qwen3-4B VLM consumes (history coarse, current fine, instruction text, view yaws) and the planner head predicts `(n_waypoints=8, 3)` waypoints `(x, y, theta)` in the robot frame.
5. The second waypoint `(x₁, y₁, θ₁)` is divided by `dt=0.1 s` to give `[vx, vy, wz]`. We map `v_mps = vx`, `w_radps = wz` (drop `vy` for the diff-drive Scout).

The instruction is plumbed via CLI: `--trackvla-instruction "Follow the person on your left"` (default `"Follow the person"`). `--use-perception` is not compatible (the VLM is the perception).

`get_debug_info()` returns `obstacles=[]`, `traj_points=[(world_x, world_y), …]` (8 waypoints transformed from robot frame using `obs.robot.{x, y, yaw_rad}`), the current `instruction`, and (when `--debug`) a `rendered_front` image with the predicted trajectory drawn — same overlay as `social-rpf-vla/offline_video_inference.py`.

**Extra deps (added to `multiRobots`):** `transformers==5.7.0`, `peft==0.19.1`, `tokenizers==0.22.2`, `accelerate==1.13.0`, `regex` (auto-pulled).
torch 2.11+cu130 already present is forward-compatible with the ckpt saved under torch 2.7.

**Self-contained layout (after `rsync` from the source repo):**
```
data/trackvla/                                              # gitignored
├── google/siglip-so400m-patch14-384/                       # 3.3 GB
├── facebook/dinov3-vits16-pretrain-lvd1689m/               # 83 MB
├── LLM_hf/qwen3-4b/                                        # 7.6 GB
└── ckpts/260417-qwen3-4b-unified-nav-qa-view-fwd-.../
    ├── model_config.json
    └── model_epoch04_step024000.pt                         # 7.7 GB
```
The path is resolved via `socialRPF.constants.DATA_ROOT` (override with env var `TRACKVLA_DATA_ROOT`).

**Smoke test (multiRobots, no CARLA needed):**
```bash
conda run -n multiRobots python \
  baselines/third_party/trackvla/scripts/verify_offline.py \
  --video /path/to/some.mp4 --max-frames 10
```
Should print 8-waypoint predictions per frame and finish with `OK`.

**Corner run:**
```bash
PLANNER=trackvla bash scripts/run_episode.sh --scene corner --debug \
  --trackvla-instruction "Follow the person"
```

**Limitations:**
- Side cameras unused (the loaded ckpt is single-view "forward").
- Per-tick inference ≈90-140 ms on cu130 (about 7-10 Hz, bf16). Below the default `dt=0.1 s` so commands may repeat for 1-2 ticks. Expect coarser tracking than the modular planners.
- VRAM ≈9.3 GB peak (Qwen3-4B + SigLIP + DINOv3 all in bf16). With CARLA at ≈10 GB this stays within a 24 GB card.

**Memory tuning notes:**
- Ckpt is loaded on CPU first, then moved to GPU once (avoids the ~17 GB peak from holding state_dict + freshly-allocated fp32 model on GPU simultaneously).
- The whole stack runs in bf16 — small fp32 heads (planner / projector / TVI embedder) are downcast in `OfflineUnifiedAgent._init_unified_model` (`offline_agent_unified.py`); SigLIP and DINOv3 are loaded with `dtype=torch.bfloat16` in `cache_gridpool.VisionFeatureCacher`. Vision tokens and yaw tensors are cast to match the model dtype at the boundary.

---

## Running a Planner

### Corner scenario (recommended test)
```bash
PLANNER=rda bash scripts/run_episode.sh --scene corner
```

### Random scenario
```bash
PLANNER=rda bash scripts/run_episode.sh --scene plaza
```

Replace `rda` with any planner name above.

### Pass criteria
A planner is considered working when, in the corner scenario:
- Target stays within **3 m** of the robot
- Target remains visible (>400 pixels) for **>20 s** continuously
- No rescue/respawn triggered during the window

For perception-based variants (`rda_depth_tpt`, `rda_perception`), the additional
ReID acceptance criterion is:
- The very first lock may use GT; **all subsequent re-locks must come from
  `TargetReID.find_target` (ReID)**. Search the run log for the strings
  `initial GT lock` (allowed once) vs. `[ReID] recovered target` (allowed any
  number of times). The string `ReID searching` may appear briefly while the
  classifier is rebuilding a streak, but should not persist for more than a few
  ticks once at least one positive sample is in the gallery.

---

## Adding a New Planner

1. Put platform-neutral planner/controller code under `baselines/planning/<name>/`.
   - Small planners should prefer one clear implementation file.
   - Multi-module planners can use a package such as `planning/bso_hfc/`.
   - Learning-based / end-to-end models go under `baselines/third_party/<name>/`
     (vendored model code stays self-contained there). Heavy weights go to
     `data/<name>/` (gitignored).
2. Create `baselines/adapters/<name>_adapter.py` implementing `FollowerPolicyAdapter`.
3. Put reusable sensing, map, prediction, or behavior logic under `perception/`,
   `maps/`, `prediction/`, or `behavior/` instead of inside the adapter.
4. Register the new planner in every episode runner:
   - `followbench/scenes/{corner,park,passage,plaza}/run_episode.py`
   - `followbench/scenes/{corner,park}/run_episode_humanoid.py`

   Add the name to `--planner` choices and a lazy-import dispatch branch.
5. Update the per-scene notes in `docs/scenes/` if behaviour differs per scene.
6. If new pip dependencies are required, install them in the `multiRobots` conda env and add them to `environment.multiRobots.yml`.
7. Test in the corner scenario before marking done (see pass criteria above).
8. Update this README.

---

## Environment Setup

All planners run in the `multiRobots` conda environment. To create it:
```bash
conda env create -f environment.multiRobots.yml
```

`environment.multiRobots.yml` already pins everything needed by the modular
planners (RDA / DWA / SFM / BSO-HFC), the perception frontend (YOLO / ByteTrack /
ReID / KPR), and the TrackVLA end-to-end planner (transformers / peft /
tokenizers / accelerate).

To install the RDA deps in an existing env (legacy escape hatch):
```bash
conda activate multiRobots
pip install cvxpy==1.5.2 pathos==0.3.4 scipy==1.11.4 osqp==1.1.1
```

For the TrackVLA planner specifically, you also need to populate
`data/trackvla/` with the vision/LLM/trained checkpoints
(≈19 GB, gitignored). See the `trackvla` section above.
