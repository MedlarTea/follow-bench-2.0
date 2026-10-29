# corner 场景（Episode 控制器）

本场景的 runner 位于 `followbench/scenes/corner/`（共享运行时在 `followbench/sim/`），并通过
`scripts/run_episode.sh --scene corner` 提供 corner 版本的一键运行。

## 关键约束

- 使用 `assets/scenes/corner/gridmap_roi.npz` 与 `assets/scenes/corner/gridmap_roi_flow_points.json`
- NPC 冲流模式：`pair_12_34`
  - 1<->2 对冲
  - 3<->4 对冲
  - 1-4、2-3 不对冲（不混流）
- NPC 导航模式：`astar`
  - 每个行人从当前位置到对应端点做 A* 路径搜索
  - 沿路径行走
  - 避让压力较大时触发快速重规划（replan-on-avoid）
- 默认目标人为 `N01`
- 默认跟随方式为 `back`，默认距离为 `1.5m`

## 启动

```bash
PLANNER=sfm bash scripts/run_episode.sh --scene corner
```

`scripts/run_episode.sh --scene corner` 支持通过环境变量选择 planner、跟随位置和距离：

```bash
PLANNER=sfm FOLLOW_POSITION=back DESIRED_DISTANCE=1.5 bash scripts/run_episode.sh --scene corner
```

## 一键评估与可视化

如果需要跑完 episode 后自动生成评估结果、单次诊断图和汇总表，使用：

```bash

PLANNER=rda_search \
FOLLOW_POSITION=back \
DESIRED_DISTANCE=1.5 \
NUM_WALKERS=10 \
bash scripts/run_episode_eval.sh --scene corner
```

脚本会自动调用 `scripts/run_episode.sh --scene corner --enable-evaluation`，并在 episode 结束后生成：

```text
outputs/eval_logs/corner/{planner}/H{non_target_pedestrians}_{follow_position}_D{desired_distance}/trial_*/
outputs/results/tables/
outputs/results/compare/
```

注意：`H` 表示非目标行人数量，不包含目标人。例如 `NUM_WALKERS_TOTAL=10` 时，日志目录为 `H9_...`。

## 侧跟随

侧跟随通过 `FOLLOW_POSITION=left_side` 或 `FOLLOW_POSITION=right_side` 开启：

```bash
PLANNER=sfm FOLLOW_POSITION=left_side DESIRED_DISTANCE=1.3 bash scripts/run_episode.sh --scene corner
PLANNER=sfm FOLLOW_POSITION=right_side DESIRED_DISTANCE=1.3 bash scripts/run_episode.sh --scene corner
```

`DESIRED_DISTANCE` 表示机器人相对目标人的侧向间距。corner 行人密度较高时，
侧跟随会比后跟随更容易受到目标人变向、避让和人体朝向抖动影响；SFM planner
内部会对目标人状态和最终侧跟随目标点做滤波，但距离过小仍可能导致机器人频繁
横向修正。建议侧跟随从 `1.3m` 到 `1.6m` 范围内调试。

不同算法对侧跟随目标点的处理不完全一致：

- `sfm`：使用 SFM 内部的固定侧目标点和目标参考滤波，不会自动切换到另一侧。
- 共享 `behavior.follow_goal` 的算法可能会根据机器人到左右侧目标点的距离做选择，
  具体行为以对应 planner/adapter 为准。

## 常用按键

- `G`：生成（先静止）
- `F`：开始/停止跟随
- `T`：切换目标
- `R`：重置
- `Q`：退出

## 常用调试

显示激光、局部地图和 planner debug：

```bash
PLANNER=sfm FOLLOW_POSITION=left_side bash scripts/run_episode.sh --scene corner --debug
```

SFM 默认会低频打印命令、力场、目标速度滤波和 `goal_jump` 等信息。若侧跟随出现
左右摆动，可优先观察：

- `goal_jump`：侧跟随目标点跨帧跳变量
- `target_speed raw/filt`：目标人原始速度与滤波速度
- `crowd`：机器人附近参与 SFM 降速/降敏的人群数量
- `herr`：机器人当前朝向和 SFM 期望速度方向的误差
- `w_raw` / `w`：角速度是否被目标点跳变持续拉大

## 修改默认参数

`scripts/run_episode.sh --scene corner` 底层调用：

- `followbench/scenes/corner/run_episode.py`

常用入口参数在脚本中通过环境变量暴露：

- `PLANNER`：默认 `pid`，可选包括所有模块化 planner（`pid` / `dwa_traj` / `sfm` / `rda` / `rda_lidar` / `rda_traj` / `rda_search` / `rda_depth_tpt` / `bso_hfc` 等）以及端到端 `trackvla`
- `FOLLOW_POSITION`：默认 `back`
- `DESIRED_DISTANCE`：默认 `1.5`

planner 侧实现位于：

- `baselines/adapters/`
- `baselines/planning/`
- `baselines/third_party/trackvla/` （端到端 VLA，权重在 `data/trackvla/`）

`trackvla` 用法（与 `--use-perception` 互斥）：

```bash
PLANNER=trackvla bash scripts/run_episode.sh --scene corner --debug --trackvla-instruction "Follow the person"
```
