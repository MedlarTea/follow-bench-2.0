# Park 场景

室内拥挤行人跟随场景，基于 CARLA Town10HD_Opt。目标人物在三个流量节点之间循环行走，机器人需在密集 NPC 干扰下持续跟随。

## 资产准备

`assets/scenes/park/` 下需要：

- `gridmap_roi.npz` — ROI 可行走地图
- `gridmap_roi_flow_points.json` — 三个流量节点坐标
- `extra_routes.json` — 用 `park_route_painter.py` 手绘的干线路由（可选，无则 trunk-walk 禁用）
- `bottleneck_zones.json` — 用 `park_zone_painter.py` 标注的窄路/路口区域（可选，无则 zone token 禁用）

生成流程：

```bash
# 1. 在 CARLA 中采集 ROI 地图（交互式采集器 carla_walk_roi_annotator.py 不在本发布中，
#    见 tools/map_annotator/README.md；park 的资产已随仓库提供）

# 2. 生成最小资产
python tools/map_annotator/convert_roi_to_minimal_assets.py

# 3. （可选）手绘干线路由
python tools/map_annotator/park_route_painter.py

# 4. （可选）标注瓶颈区域
python tools/map_annotator/park_zone_painter.py
```

## 启动

```bash
conda activate followbench
bash scripts/run_episode.sh --scene park
```

人形机器人（Unitree G1）跟随：

```bash
conda activate followbench
bash scripts/run_episode.sh --scene park --humanoid
```

与 Scout 版共用 park 场景与 NPC 调度，机器人换为 `G1RobotRuntime`（`walker.pedestrian.g1_unitree`）。常用环境变量与 Scout 版相同（`PLANNER`、`NUM_WALKERS`、`FOLLOW_POSITION`、`DESIRED_DISTANCE`）。

默认启用通用评估（`--enable-evaluation`），episode 结束后自动落 metrics 到 `outputs/eval_logs/park/<planner>/<config>/trial_*/`。撞到 NPC（`--terminate-on-human-collision` 默认 True）立刻终止并记录 `termination_reason=human_collision`。

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `PLANNER` | `pid` | 跟随算法，见下方列表 |
| `NUM_WALKERS` | `6` | NPC 总数 |
| `SIM_MODE` | `async` | `async` / `sync` |
| `FOLLOW_POSITION` | `back` | `back` / `left_side` / `right_side` |
| `DESIRED_DISTANCE` | `1.5` | 跟随目标距离（米） |

示例：

```bash
PLANNER=rda_lidar NUM_WALKERS=10 bash scripts/run_episode.sh --scene park
```

强制指定天气（默认 `random`）：

```bash
bash scripts/run_episode.sh --scene park --weather fog --difficulty hard
```

## 一键评估与可视化

如果需要跑完 episode 后自动生成评估结果、单次诊断图和汇总表，使用：

```bash

PLANNER=rda_search \
FOLLOW_POSITION=back \
DESIRED_DISTANCE=1.5 \
NUM_WALKERS=6 \
bash scripts/run_episode_eval.sh --scene park
```

脚本会自动调用 `scripts/run_episode.sh --scene park --enable-evaluation`，并在 episode 结束后生成：

```text
outputs/eval_logs/park/{planner}/H{non_target_pedestrians}_{follow_position}_D{desired_distance}/trial_*/
outputs/results/tables/
outputs/results/compare/
```

注意：`H` 表示非目标行人数量，不包含目标人。例如 `NUM_WALKERS_TOTAL=6` 时，日志目录为 `H5_...`。

`scripts/run_episode_eval.sh` 使用 `NUM_WALKERS_TOTAL` 表示总行人数；普通启动脚本 `scripts/run_episode.sh --scene park` 使用 `NUM_WALKERS`。一键评估脚本内部会把 `NUM_WALKERS_TOTAL` 传给 `--num-walkers`。

## 可用 Planner

| Planner | 避障 | 说明 |
|---|---|---|
| `pid` | 无 | 纯 PID 跟随，默认 |
| `rda_lidar` | LiDAR | RDA + LiDAR 避障，推荐 |
| `dwa_traj` | 有 | DWA 动态窗口 |
| `sfm` | SFM | 社会力模型 |
| `bso_hfc` | 有 | BSO-HFC 规划器 |
| `trackvla` | 端到端 | TrackVLA 视觉-语言-动作模型；用 `--trackvla-instruction "..."` 给 prompt，与 `--use-perception` 互斥 |

## NPC 系统

### 调度架构

- **distractor_orbit**：NPC 默认状态，在自动生成的三角形轨道上循环
- **corner_flow**：目标换段时触发，从轨道派出走完一段再回轨道
- **shadow_follower**：延迟跟在目标后面，模拟尾随行为
- **trunk_walk**：沿 `extra_routes.json` 手绘干线穿行，完成后回轨道
- **rescue**：区间测距检测到卡死时临时接管，走到开阔点后回轨道

### 难度预设（D 键切换）

| 难度 | 流量概率 | 速度范围 | 干线密度 |
|---|---|---|---|
| easy | 30% | 0.7–1.0 | 1/route |
| hard | 70% | 0.9–1.3 | 2/route |
| boss | 100% | 1.1–1.6 | 3/route |
| legend | 100% | 1.3–2.0 | 4/route，20% NPC 与目标同外观 |

每次选择难度（启动时初始化、运行中按 D）都会**重新随机派送一个天气**，记录到 `episode_meta.json` 的 `scenario_extras.weather` 字段。`--weather <name>` 显式指定后则锁定，不再随机。

### 天气预设

| key | 中文 | 视觉特征 |
|---|---|---|
| `clear` | 晴朗 | 高空太阳、对比度强、视野最佳 |
| `cloudy` | 阴天 | 80% 云、阴影柔和 |
| `dusk` | 下午傍晚 | 太阳低角度（8°）、暖色调 |
| `rain` | 雨天 | 80 强降雨 + 60 沉积水 + 80 wetness、地面反光 |
| `fog` | 大雾 | fog_density=80、能见度 ~8m |

预览图：[`weather_preview.png`](weather_preview.png)（5 个天气拼图）、[`difficulty_preview.png`](difficulty_preview.png)（4 难度 × 随机天气示例）。

### NPC ↔ Robot 交互

NPC 的 SFM 排斥力把机器人**当作另一名行人**（`sfm_robot_mult`，默认 1.0）：NPC 主动避让机器人，机器人本身不受 SFM 影响（避障策略由 planner 决定）。Target 仍然不受任何力。改 `npc.params["sfm_robot_mult"]` 调整权重，0 即关闭。

### Zone Token 系统

`bottleneck_zones.json` 中标注的区域限制同时通过的 dispatched NPC 数量：
- **narrow**（窄路）：最多 1 个
- **junction**（路口）：最多 2 个

Orbit NPC 不受约束，依赖 SFM 轻度避让。

### 全局脱困机制

每 5 秒统计每个 NPC 的累积路程，若 < 1.5m 判定为卡死：
1. 以 `hash(track_id)` 为角度偏移从卡死点向外扫描，找 LoS 可达的开阔点
2. NPC 步行前往（不传送），到达后回归轨道

## 键位

| 键 | 功能 |
|---|---|
| G | 生成 NPC + 机器人 |
| F | 开始/停止跟随 |
| D | 循环切换难度 |
| V | 开启/关闭 NPC 实时 debug 视图 |
| M | 手动控制目标行人 |
| B | 手动控制机器人 |
| T | 切换跟随目标 |
| R | 重置场景 |
| SPACE | 暂停/继续 |

## 评估输出

跑完后 `outputs/eval_logs/park/<planner>/H<人数>_<位置>_D<距离>/trial_NNN_<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `episode_meta.json` | 启动配置、actor footprint、`scenario_extras.{difficulty,weather,difficulty_history,weather_history}`、`termination_reason` |
| `episode_step.jsonl` | 每 tick：robot/target/humans 状态、可见性、clearance、command、collision flag |
| `collision_events.jsonl` | 碰撞事件流（来源：CARLA 碰撞传感器 + 几何 fallback） |
| `eval_result.json` | 终局指标：`target_visibility_ratio`、`collision_count`、`human_collision_count`、`obstacle_avoidance_success`、`search_success`、`tracking_completion`、`geometry_collision_detected` |
| `eval_summary.txt` | 可读摘要 |

## Debug 视图

按 V 键启动 live_viewer：

```bash
conda run -n followbench python followbench/visualization/live_viewer.py \
  --npz assets/scenes/park/gridmap_roi.npz --half 30
```
