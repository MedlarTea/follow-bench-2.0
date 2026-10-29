# FollowBench2.0 评估可视化

本目录提供基于 `outputs/eval_logs/` 的离线统计和画图脚本。可视化层只读取评估日志，不依赖 CARLA，也不改变 episode 数据。

## 脚本说明

```text
followbench/visualization/
  constants.py       指标显示名、颜色、线型、默认输出目录
  data_loader.py     扫描 logs 并合并 episode_meta.json 与 eval_result.json
  aggregate.py       输出长表，每行是一个 group + metric
  export_table.py    输出宽表，每行是一个实验配置，列为各指标 mean/std
  plot_compare.py    固定配置下对比不同 planner 的柱状图
  plot_sweep.py      距离或行人数量 sweep 曲线
  plot_episode.py    单个 episode 的轨迹、距离、碰撞、耗时诊断图
```

默认输入：

```text
outputs/eval_logs/
```

默认输出：

```text
outputs/results/
```

## 数据筛选口径

所有批量脚本都支持以下筛选参数：

```bash
--log-root outputs/eval_logs
--scenario corner
--scenario-name corner_two_way_astar
--planners rda_search dwa_traj bso_hfc
--humans 10
--follow-position back
--distance 1.5
```

其中 `--humans` 对应 `episode_meta.json` 中的 `num_pedestrians`，表示非目标行人数量，不包含目标人。日志目录中的 `H10` 也表示 10 个非目标行人。

## 指标显示

可视化层保留评估层原始指标，同时提供论文图表中更短的显示名：

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

`SR`、`ASR`、`TVR`、`TinTPerson`、`TinPrivate` 在图表和表格中以百分比显示。`TinTPerson` 和 `TinPrivate` 使用占 episode 有效评估时间的比例，避免不同 episode 长度导致绝对时间不可比。

`CostMean` / `CostP95` 是兼容旧字段，等同于完整 `policy.act(obs)` 耗时。新实验推荐使用：

```text
PolicyMean / PolicyP95  完整 policy.act(obs) 耗时
PlanMean / PlanP95      纯 planner 耗时
PercMean / PercP95      感知层耗时
```

## 导出统计表

长表适合后处理和自定义画图：

```bash
python -m followbench.visualization.aggregate \
  --scenario corner \
  --humans 10 \
  --follow-position back \
  --distance 1.5 \
  --output outputs/results/tables/corner_H10_back_D1.5_long.csv
```

宽表适合直接放进实验汇总：

```bash
python -m followbench.visualization.export_table \
  --scenario corner \
  --humans 10 \
  --follow-position back \
  --distance 1.5 \
  --output outputs/results/tables/corner_H10_back_D1.5_summary.csv
```

默认分组字段：

```text
scenario_type, scenario_name, planner, num_pedestrians, follow_position, desired_distance
```

可以用 `--group-by` 改变分组，例如只按场景和 planner 汇总：

```bash
python -m followbench.visualization.export_table \
  --scenario corner \
  --group-by scenario_type planner
```

## 对比柱状图

固定场景配置下对比多个 planner：

```bash
python -m followbench.visualization.plot_compare \
  --scenario corner \
  --planners rda_search dwa_traj bso_hfc \
  --humans 10 \
  --follow-position back \
  --distance 1.5
```

默认输出类似：

```text
outputs/results/compare/compare_corner_H10_back_D1.5.png
```

可以通过 `--metrics` 指定图中的指标：

```bash
python -m followbench.visualization.plot_compare \
  --scenario corner \
  --metrics SR ASR TVR TinTPerson TinPrivate Jerk CostP95
```

## Sweep 曲线

按期望跟随距离画 sweep：

```bash
python -m followbench.visualization.plot_sweep \
  --scenario corner \
  --x desired_distance \
  --humans 10
```

按非目标行人数量画 sweep：

```bash
python -m followbench.visualization.plot_sweep \
  --scenario corner \
  --x num_pedestrians \
  --distance 1.5
```

默认输出类似：

```text
outputs/results/sweep/sweep_corner_desired_distance.png
outputs/results/sweep/sweep_corner_num_pedestrians.png
```

## 单 episode 诊断图

对某次运行生成诊断图：

```bash
python -m followbench.visualization.plot_episode \
  --run-dir outputs/eval_logs/corner/rda_search/H10_back_D1.5/trial_003_20260430_153012
```

默认输出到该 run 目录下的 `plots/`：

```text
trajectory.png
distance_visibility.png
motion_profile.png
clearance_collision.png
planner_cost.png
perception_timing.png  # 仅当日志包含感知 timing 时生成
```

这些图主要用于排查单次 episode 的问题：目标是否长时间不可见、robot 是否进入目标私人空间、是否发生几何碰撞、planner 耗时是否异常等。

## 推荐流程

1. 跑 episode 时启用 `--enable-evaluation`，让每次运行自动写入 logs。
2. 单次异常先用 `plot_episode.py` 看轨迹、距离、碰撞和耗时。
3. 多次重复实验后用 `export_table.py` 导出宽表。
4. 论文图或汇报图使用 `plot_compare.py` 和 `plot_sweep.py`。

## 注意事项

- 可视化脚本只统计已有 `eval_result.json` 的 run。若某个 run 缺少结果文件，可先运行 `python -m followbench.evaluation.evaluator --run-dir <run_dir>` 补算。
- `scenario_name` 和 `scenario_variant` 来自 episode manager 的参数自动记录，用于区分同一 `scenario_type` 下不同流量、导航或难度设置。
- `passage` 当前不额外展开 passage 专属 follow-position/distance 口径，仍使用通用字段筛选。
