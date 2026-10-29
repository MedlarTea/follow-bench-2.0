# FollowBench2.0 原生评估层

本目录是 FollowBench2.0 基于 CARLA 的原生评估层。它不复用 FollowBench 1.0/IRSim 的日志格式，但保留旧版指标命名，便于延续实验口径和论文表述。

评估层的目标是：episode 运行时实时记录必要数据，episode 结束后自动得到 `eval_result.json` 和 `eval_summary.txt`，并支持离线重算、批量统计和结果可视化。

## 快速使用

以 corner 场景为例，推荐使用一键脚本：

```bash

PLANNER=rda_search \
FOLLOW_POSITION=back \
DESIRED_DISTANCE=1.5 \
NUM_WALKERS=10 \
bash scripts/run_episode_eval.sh --scene corner
```

这个脚本会完成：

1. 调用 `scripts/run_episode.sh --scene <scene> --enable-evaluation` 跑一次 episode。
2. 自动启用 collision sensor 和几何碰撞兜底。
3. episode 结束后自动定位最新 `trial_*` 日志目录。
4. 生成或重算 `eval_result.json`。
5. 生成单 episode 诊断图。
6. 导出当前配置的汇总表和对比图。

如果不用一键脚本，也可以直接给 episode manager 加评估参数：

```bash
PLANNER=rda_search bash scripts/run_episode.sh --scene <scene> --enable-evaluation
```

目前三个场景都有一键脚本：

```text
scripts/run_episode_eval.sh --scene corner
scripts/run_episode_eval.sh --scene park
scripts/run_episode_eval.sh --scene passage
```

手动重算某次 run：

```bash
python -m followbench.evaluation.evaluator --run-dir <run_dir>
```

生成单次诊断图：

```bash
python -m followbench.visualization.plot_episode --run-dir <run_dir>
```

## 目录结构

```text
followbench/evaluation/
  __init__.py
  evaluator.py                 # 单次 run 离线评估 CLI 薄入口
  readme.md
  core/
    schemas.py                 # 默认参数、日志命名、状态序列化
    logging.py                 # runtime logger、clearance、碰撞事件写入
    collision.py               # CARLA collision sensor wrapper
    metrics.py                 # 离线指标计算
    evaluator.py               # evaluator CLI 实现
  visualization/
    readme.md                  # 可视化层说明
    constants.py
    data_loader.py
    aggregate.py               # 长表统计
    export_table.py            # 宽表统计
    plot_compare.py            # planner 对比图
    plot_sweep.py              # sweep 曲线
    plot_episode.py            # 单 episode 诊断图
  logs/                        # episode 原始评估日志
  results/                     # 表格和图片输出
```

已接入评估层的 episode manager：

```text
followbench/scenes/corner/run_episode.py
followbench/scenes/park/run_episode.py
followbench/scenes/passage/run_episode.py
```

## 核心评估口径

评估层统计的是 episode 进入算法接管后的完整过程，不只统计 planner 内部的 `following=true` 模式。

有效 tick 集合定义为：

```text
S = {t | episode_active(t)=true and paused(t)=false}
```

含义：

- `episode_active=true`：episode 已经进入正式评估阶段。通常由用户按 `F` 或自动 follow 触发。
- `paused=false`：暂停状态不计入指标。
- planner 内部进入 `follow`、`search`、`hold`、`lost`、`recovery`、`reacquire` 等状态，仍属于同一个 episode，只要 `episode_active=true` 就计入评估。

这样可以正确评估状态机 planner：例如目标丢失后进入 search，再重新找到目标继续跟随，这段搜索时间和路径都属于同一次任务表现。

## 运行参数

评估相关参数由 `core/schemas.py` 统一注册：

```text
--enable-evaluation
--eval-log-root outputs/eval_logs
--eval-scenario-type corner
--eval-trial-id 1
--max-search-time-s 5.0
--terminate-on-human-collision
--no-terminate-on-human-collision
--human-collision-margin 0.0
--disable-collision-sensor
```

默认行为：

- 不传 `--enable-evaluation` 时不写正式评估日志。
- 默认启用 CARLA `sensor.other.collision`。
- 几何碰撞兜底始终启用。
- 默认检测到 robot-human collision 后终止 episode。
- `max_search_time_s=5.0`，即目标连续不可见超过 5 秒后 `search_success=false`。
- `human_collision_margin=0.0`，即表面距离小于等于 0 判为几何碰撞。

## 日志命名

正式评估日志保存到：

```text
outputs/eval_logs/
  {scenario_type}/
    {planner}/
      H{num_pedestrians}_{follow_position}_D{desired_distance}/
        trial_{trial_id}_{timestamp}/
          episode_meta.json
          episode_step.jsonl
          collision_events.jsonl
          eval_result.json
          eval_result_schema.json
          episode_data_schema.json
          eval_summary.txt
```

示例：

```text
outputs/eval_logs/corner/rda_search/H9_back_D1.5/trial_001_20260430_150050/
```

字段含义：

- `scenario_type`：场景类型，如 `corner`、`park`、`passage`。
- `planner`：规划器名，如 `pid`、`dwa_traj`、`rda_search`、`bso_hfc`。
- `H{num_pedestrians}`：非目标行人数量，不包含目标人。
- `follow_position`：期望跟随位置，如 `back`、`left_side`、`right_side`。
- `D{desired_distance}`：期望跟随距离，单位 m。
- `trial_id`：同一配置下第几次重复实验。
- `timestamp`：运行开始时间，用于防止目录覆盖。

行人数量口径：

```text
num_pedestrians = max(num_walkers_total - 1, 0)
```

例如 `--num-walkers 10` 表示总行人数 10，其中 `N01` 是目标人，则日志目录使用 `H9`。

trial id 支持两种模式：

```bash
--eval-trial-id 7
```

会生成：

```text
trial_007_YYYYmmdd_HHMMSS
```

如果不传 `--eval-trial-id`，评估层会扫描同一配置目录下已有 `trial_*`，自动取最大编号加 1。

## 日志文件

### episode_meta.json

保存 episode 配置、场景标签、物理尺寸、阈值和终止原因。典型字段：

```json
{
  "benchmark_version": "followbench2.0",
  "scenario_type": "corner",
  "scenario_name": "corner_pair_12_34_astar",
  "scenario_variant": {
    "robot_spawn_mode": "behind_target",
    "npc_flow_mode": "pair_12_34",
    "npc_navigation_mode": "astar",
    "npc_min_speed": 0.6,
    "npc_max_speed": 1.1
  },
  "map_name": "Town10HD_Opt",
  "planner": "rda_search",
  "follow_position": "back",
  "desired_distance": 1.5,
  "num_walkers_total": 10,
  "num_pedestrians": 9,
  "target_track_id": "N01",
  "trial_id": 1,
  "dt": 0.05,
  "max_search_time_s": 5.0,
  "max_search_steps": 100,
  "zone_thresholds": [0.45, 1.2],
  "robot": {"radius": 0.31, "length": 0.62, "width": 0.58},
  "target": {"radius": 0.29, "length": 0.35, "width": 0.58},
  "human": {"radius": 0.35},
  "collision": {
    "terminate_on_human_collision": true,
    "human_collision_margin": 0.0,
    "use_collision_sensor": true,
    "use_geometry_fallback": true
  },
  "termination_reason": "human_collision"
}
```

重要字段：

- `scenario_name`：自动生成的场景配置名，用于区分同一 `scenario_type` 下的不同流量、导航或难度设置。
- `scenario_variant`：影响场景行为的关键参数，便于筛选和复现实验。
- `num_walkers_total`：episode manager 生成的总行人数，包含目标人。
- `num_pedestrians`：非目标行人数量，用于日志命名和可视化筛选。
- `robot`、`target`、`human`：评估距离和碰撞时使用的几何尺寸。

### episode_step.jsonl

主数据源。每 tick 一行 JSON。关键字段：

```text
tick                                  episode manager 内部 tick
time_s                                当前进程时间
episode_active                        是否进入正式评估阶段
paused                                是否暂停
planner_mode                          planner 内部状态，来自 get_debug_info()
robot                                 robot 位姿
target                                目标人状态
humans                                非目标行人状态
visibility.target_visible             目标是否可见
visibility.target_pixel_count         目标可见像素数
command.v_mps / command.w_radps       planner 输出控制命令
planner_cost_ms                       兼容旧字段，等同于 timing.policy_total_ms
timing.policy_total_ms                policy.act(obs) 总耗时
timing.perception_total_ms            感知层总耗时，若无感知则为 null
timing.planner_core_ms                纯 planner 执行耗时，不含感知
timing.wrapper_overhead_ms            wrapper/数据转换等未归因耗时
timing.perception.detection_ms        检测耗时，如 YOLO
timing.perception.tracking_ms         跟踪/关联耗时
timing.perception.reid_ms             ReID 特征提取/匹配耗时
timing.perception.mapping_ms          深度反投影、BEV/世界坐标映射、多视角融合耗时
timing.perception.fsm_ms              感知目标状态机耗时
clearance.robot_target_surface_dist   robot 到目标人的表面距离
clearance.min_human_surface_dist      robot 到最近非目标行人的表面距离
collision                             当前 tick 是否记录到行人碰撞
collision_source                      collision_sensor 或 geometry
termination_triggered                 当前 tick 是否触发 episode 终止
```

`clearance` 使用 `episode_meta.json` 中的 robot/target/human 半径计算，和离线指标复算使用同一套几何尺寸。

耗时字段口径：

```text
timing.policy_total_ms
= policy.act(obs) 的完整 wall-clock 时间

timing.planner_core_ms
= 纯 planner/controller/MPC/SFM/DWA/RDA 执行时间

timing.perception_total_ms
= 感知层完整耗时，包括检测、跟踪/ReID、建图/映射和状态机

timing.wrapper_overhead_ms
= policy_total_ms - planner_core_ms - perception_total_ms
```

无感知 planner 的默认口径：

```text
policy_total_ms = planner_core_ms
perception_total_ms = null
```

带 `--use-perception` 的 wrapper planner 会在 wrapper 内部分别统计感知层和内层 planner。`rda_depth_tpt` 这类自带感知路径的 adapter 也会单独记录感知总耗时和 planner core 耗时。

### collision_events.jsonl

只记录碰撞事件。来源有两类：

CARLA collision sensor：

```json
{
  "tick": 145,
  "time_s": 7.25,
  "episode_active": true,
  "source": "collision_sensor",
  "collision_type": "human",
  "robot_actor_id": 120,
  "other_actor_id": 331,
  "other_type_id": "walker.pedestrian.0001",
  "other_track_id": "N03",
  "is_target": false,
  "impulse_norm": 12.11
}
```

几何兜底：

```json
{
  "tick": 146,
  "time_s": 7.3,
  "episode_active": true,
  "source": "geometry",
  "collision_type": "human",
  "other_track_id": "N03",
  "is_target": false,
  "min_human_surface_dist": -0.03
}
```

离线评估只统计 `episode_active=true` 的碰撞事件。旧日志如果没有该字段，会按 `true` 处理，保持向后可读。

### eval_result.json

离线指标结果，episode 结束时自动生成，也可以手动重算：

```bash
python -m followbench.evaluation.evaluator --run-dir <run_dir>
```

典型字段：

```json
{
  "max_steps": 1994,
  "max_search_steps": 100,
  "obstacle_avoidance_success": false,
  "target_visibility_ratio": 1.0,
  "search_success": true,
  "search_path_length": 0.0,
  "avg_robot_target_dist_no_radius": 1.79,
  "path_length": 58.04,
  "avg_velocity": 0.58,
  "avg_acceleration": 0.75,
  "avg_jerk": 14.99,
  "time_in_target_personal_zone": 71.3,
  "time_in_human_private_zone": 2.8,
  "time_in_target_search": 0.0,
  "total_time": 99.7,
  "termination_reason": "human_collision",
  "collision_count": 1,
  "human_collision_count": 1,
  "geometry_collision_detected": false,
  "min_human_surface_dist": 0.35,
  "planner_cost_ms_mean": 59.67,
  "planner_cost_ms_p95": 83.32,
  "policy_total_ms_mean": 59.67,
  "policy_total_ms_p95": 83.32,
  "planner_core_ms_mean": 59.67,
  "planner_core_ms_p95": 83.32,
  "perception_total_ms_mean": null,
  "perception_total_ms_p95": null,
  "perception_detection_ms_mean": null,
  "perception_tracking_ms_mean": null,
  "perception_reid_ms_mean": null,
  "perception_mapping_ms_mean": null,
  "perception_fsm_ms_mean": null
}
```

### eval_result_schema.json / episode_data_schema.json

JSON 本身不支持注释，因此评估层不会在 `eval_result.json` 中写 `//` 或 `#`。每次生成 `eval_result.json` 时会同步生成两个说明文件：

```text
eval_result_schema.json
episode_data_schema.json
```

`eval_result_schema.json` 解释每个指标字段的显示名、单位、类型、含义、公式和数据来源。`episode_data_schema.json` 解释 `episode_meta.json`、`episode_step.jsonl`、`collision_events.jsonl` 中的关键字段。

### eval_summary.txt

面向快速查看的摘要文件，保存最常用的几个结果：

```text
run_dir: ...
obstacle_avoidance_success: False
search_success: True
target_visibility_ratio: 1.0
path_length: 58.04
avg_velocity: 0.58
avg_acceleration: 0.75
avg_jerk: 14.99
termination_reason: human_collision
```

## 碰撞策略

评估层使用双层 robot-human collision 检测：

1. CARLA `sensor.other.collision`。
2. robot 与行人的几何表面距离兜底。

CARLA collision sensor 挂在 robot actor 上，只记录 `walker.pedestrian.*` 事件。由于当前 Scout robot 使用 `set_transform` 驱动，且 robot physics 可能关闭，物理碰撞事件存在漏报风险，因此几何兜底是正式评估的一部分，不只是调试补充。

几何表面距离：

```text
d_surface(robot, human_i)
= ||p_robot - p_human_i||_2 - r_robot - r_human
```

变量：

- `p_robot`：robot 平面位置 `(x, y)`。
- `p_human_i`：第 `i` 个行人的平面位置。
- `r_robot`：robot 半径，来自 `episode_meta.json`。
- `r_human`：行人半径，来自 `episode_meta.json`。

碰撞判定：

```text
human_collision =
    collision_sensor_hit_human
    OR min_i d_surface(robot, human_i) <= human_collision_margin
```

默认：

```text
human_collision_margin = 0.0
```

目标人也是行人。撞到目标人同样记为 human collision，并在事件中记录：

```text
is_target = true
```

如果启用 `--terminate-on-human-collision`，检测到行人碰撞后立即结束 episode：

```text
termination_reason = "human_collision"
```

## 指标说明

所有指标默认在有效 tick 集合 `S` 上计算：

```text
S = {t | episode_active(t)=true and paused(t)=false}
```

时间步长：

```text
Δt = episode_meta["dt"]
```

### max_steps

```text
max_steps = |S|
```

有效评估 tick 数。

### max_search_steps

```text
max_search_steps = round(max_search_time_s / Δt)
```

目标允许连续不可见的最大 tick 数。

### obstacle_avoidance_success

```text
obstacle_avoidance_success =
    human_collision_count == 0
    AND no active step has collision=true
    AND no active step has min_human_surface_dist <= human_collision_margin
    AND no active step has robot_target_surface_dist <= human_collision_margin
```

含义：episode 有效区间内没有记录到 robot-human collision。目标人碰撞也算失败。

### target_visibility_ratio

```text
target_visibility_ratio = N_visible / max_steps
```

变量：

- `N_visible`：`S` 中 `visibility.target_visible=true` 的 tick 数。

### search_success

```text
search_success =
    max_consecutive_invisible_steps <= max_search_steps
```

变量：

- `max_consecutive_invisible_steps`：`S` 中目标连续不可见的最长 tick 数。

### search_path_length

```text
search_path_length =
    Σ ||p_robot(t) - p_robot(t-1)||_2
    for t in S and target_visible(t)=false
```

表示目标不可见期间 robot 走过的路径长度。

### avg_robot_target_dist_no_radius

```text
avg_robot_target_dist_no_radius =
    mean(||p_robot(t) - p_target(t)||_2), t in S
```

robot 与目标人的平均中心距离，不扣除半径。

### path_length

```text
path_length =
    Σ ||p_robot(t) - p_robot(t-1)||_2, t in S
```

robot 在有效评估区间内的总路径长度。

### avg_velocity

```text
v(t) = (p_robot(t) - p_robot(t-1)) / Δt
avg_velocity = mean(||v(t)||_2)
```

当前 robot 由 `set_transform` 驱动，CARLA `get_velocity()` 可能不稳定，因此评估层使用位置差分计算速度。

### avg_acceleration

```text
a(t) = (v(t) - v(t-1)) / Δt
avg_acceleration = mean(||a(t)||_2)
```

### avg_jerk

```text
j(t) = (a(t) - a(t-1)) / Δt
avg_jerk = mean(||j(t)||_2)
```

### time_in_target_personal_zone

```text
d_target_surface(t)
= ||p_robot(t) - p_target(t)||_2 - r_robot - r_target

time_in_target_personal_zone =
    Σ Δt for zone_min < d_target_surface(t) < zone_max
```

默认：

```text
zone_min = 0.45
zone_max = 1.20
```

变量：

- `r_robot`：robot 半径。
- `r_target`：目标人半径。

### time_in_human_private_zone

```text
d_human_surface_min(t)
= min_i(||p_robot(t) - p_human_i(t)||_2 - r_robot - r_human)

time_in_human_private_zone =
    Σ Δt for d_human_surface_min(t) < zone_min
```

变量：

- `p_human_i`：第 `i` 个非目标行人位置。
- `r_human`：行人半径。

### time_in_target_search

```text
time_in_target_search =
    Σ Δt for target_visible(t)=false
```

目标不可见的总时间。

### total_time

```text
total_time = max_steps * Δt
```

有效评估总时长。

### policy_total_ms_mean / policy_total_ms_p95

```text
policy_total_ms_mean = mean(timing.policy_total_ms)
policy_total_ms_p95  = percentile_95(timing.policy_total_ms)
```

用于衡量 `policy.act(obs)` 的完整单步耗时，包含感知、planner 和 wrapper 开销。

旧字段：

```text
planner_cost_ms_mean = policy_total_ms_mean
planner_cost_ms_p95  = policy_total_ms_p95
```

这两个旧字段保留是为了兼容已有脚本。新实验中，若需要纯 planner 耗时，应使用下面的 `planner_core_ms_*`。

### planner_core_ms_mean / planner_core_ms_p95

```text
planner_core_ms_mean = mean(timing.planner_core_ms)
planner_core_ms_p95  = percentile_95(timing.planner_core_ms)
```

纯 planner/controller 执行时间，不包含感知模块耗时。无感知 planner 中它通常等于 `policy_total_ms`。

### perception_total_ms_mean / perception_total_ms_p95

```text
perception_total_ms_mean = mean(timing.perception_total_ms)
perception_total_ms_p95  = percentile_95(timing.perception_total_ms)
```

感知层总耗时。无感知 planner 中为 `null`。

### perception_*_ms_mean / perception_*_ms_p95

```text
perception_detection_ms_* = 检测耗时统计
perception_tracking_ms_*  = 跟踪/关联耗时统计
perception_reid_ms_*      = ReID 耗时统计
perception_mapping_ms_*   = 深度反投影/建图/世界坐标映射耗时统计
perception_fsm_ms_*       = 感知状态机耗时统计
```

这些字段只有在对应感知模块提供分项 timing 时才有值，否则为 `null`。

## 可视化与统计

可视化脚本位于：

```text
followbench/visualization/
```

详细说明见：

```text
docs/visualization_reference.md
```

常用命令：

```bash
python -m followbench.visualization.plot_episode \
  --run-dir outputs/eval_logs/corner/rda_search/H9_back_D1.5/trial_001_20260430_150050
```

会在该 run 下生成：

```text
plots/
  trajectory.png
  distance_visibility.png
  motion_profile.png
  clearance_collision.png
  planner_cost.png
```

导出宽表：

```bash
python -m followbench.visualization.export_table \
  --scenario corner \
  --planners rda_search \
  --humans 9 \
  --follow-position back \
  --distance 1.5
```

生成 planner 对比图：

```bash
python -m followbench.visualization.plot_compare \
  --scenario corner \
  --planners rda_search dwa_traj bso_hfc \
  --humans 9 \
  --follow-position back \
  --distance 1.5
```

默认输出目录：

```text
outputs/results/
```

可视化层提供的短指标名：

```text
SR          = obstacle_avoidance_success AND search_success
ASR         = obstacle_avoidance_success
TVR         = target_visibility_ratio
TinTPerson  = time_in_target_personal_zone / total_time
TinPrivate  = time_in_human_private_zone / total_time
PL          = path_length
AvgVel      = avg_velocity
AvgAcc      = avg_acceleration
Jerk        = avg_jerk
CostMean    = planner_cost_ms_mean
CostP95     = planner_cost_ms_p95
PolicyMean  = policy_total_ms_mean
PolicyP95   = policy_total_ms_p95
PlanMean    = planner_core_ms_mean
PlanP95     = planner_core_ms_p95
PercMean    = perception_total_ms_mean
PercP95     = perception_total_ms_p95
DetMean     = perception_detection_ms_mean
TrackMean   = perception_tracking_ms_mean
ReIDMean    = perception_reid_ms_mean
MapMean     = perception_mapping_ms_mean
FSMMean     = perception_fsm_ms_mean
```

其中 `SR`、`ASR`、`TVR`、`TinTPerson`、`TinPrivate` 在表格和图中以百分比显示。

## 推荐实验流程

1. 固定场景、planner、`num_walkers_total`、`follow_position`、`desired_distance`。
2. 连续跑多次 episode。不要手动改日志目录，trial id 会自动递增。
3. 单次异常先看该 run 的 `eval_summary.txt` 和 `plots/`。
4. 多次重复实验后用 `export_table.py` 导出统计表。
5. 多 planner 或多参数对比时使用 `plot_compare.py` 和 `plot_sweep.py`。

## 已知限制

1. CARLA collision sensor 是物理事件传感器；在 robot 使用 `set_transform` 且 physics 关闭时可能漏报。
2. 几何兜底是正式碰撞判断的一部分，因此碰撞指标优先相信 sensor event 和几何距离的并集。
3. 当前重点实现 robot-human collision；静态障碍物碰撞尚未作为独立指标展开。
4. `planner_mode` 来自 planner 的 `get_debug_info()`，不是所有 planner 都一定提供。
5. 旧日志如果是在 clearance 半径口径修正前生成的，`eval_result.json` 可以按 meta 重新计算，但旧 `episode_step.jsonl` 中已经写入的 `clearance` 不会自动回填。
