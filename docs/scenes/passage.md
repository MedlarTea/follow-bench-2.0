# Passage 场景

双向门口人群穿越场景：两侧人群排队、依序通过门口，机器人跟随目标人物穿越门口区域。

## 场景设计

```
Flow Point (North)          Flow Point (North)
   FP1 / FP2                  FP1 / FP2
      ↓                           ↓
  Queue A                     Queue B
  (北侧排队)                   (北侧→南侧)
      ↓                           ↓
      └──────── Passage ──────────┘
      ↑                           ↑
  Queue B                     Queue A
  (南/西侧)                   (南/西→北侧)
      ↑                           ↑
   FP0 / FP3                  FP0 / FP3
Flow Point (South/West)    Flow Point (South/West)
```

- Agent 在 flow points 附近生成，走向各自队列位置（APPROACH）
- 到位后排队等候（QUEUE），到队首时出发过门（PASS）
- 过门后加入对侧队尾，形成持续双向对流
- 全局 `_passing_count` 控制同时过门人数上限（由难度决定）

## 快速启动

```bash
conda activate followbench
bash scripts/run_episode.sh --scene passage
```

默认启用通用评估（`--enable-evaluation`），episode 结束后自动落 metrics 到 `outputs/eval_logs/passage/<planner>/<config>/trial_*/`。撞到 NPC（`--terminate-on-human-collision` 默认 True）立刻终止并记录 `termination_reason=human_collision`。

指定 planner：
```bash
PLANNER=pid bash scripts/run_episode.sh --scene passage       # 默认
PLANNER=dwa_traj bash scripts/run_episode.sh --scene passage
PLANNER=sfm bash scripts/run_episode.sh --scene passage
PLANNER=trackvla bash scripts/run_episode.sh --scene passage --trackvla-instruction "Follow the person"
```

可通过环境变量指定跟随位置和距离：

```bash
PLANNER=rda_search FOLLOW_POSITION=left_side DESIRED_DISTANCE=1.3 bash scripts/run_episode.sh --scene passage
```

强制指定难度 / 天气（默认难度 `easy`、天气 `random`）：

```bash
bash scripts/run_episode.sh --scene passage --difficulty boss --weather fog
```

## 一键评估与可视化

如果需要跑完 episode 后自动生成评估结果、单次诊断图和汇总表，使用：

```bash

PLANNER=rda_search \
FOLLOW_POSITION=back \
DESIRED_DISTANCE=1.5 \
NUM_WALKERS=6 \
bash scripts/run_episode_eval.sh --scene passage
```

脚本会自动调用 `scripts/run_episode.sh --scene passage --enable-evaluation`，并在 episode 结束后生成：

```text
outputs/eval_logs/passage/{planner}/H{non_target_pedestrians}_{follow_position}_D{desired_distance}/trial_*/
outputs/results/tables/
outputs/results/compare/
```

注意：`H` 表示非目标行人数量，不包含目标人。例如 `NUM_WALKERS_TOTAL=6`、`FOLLOW_POSITION=back`、`DESIRED_DISTANCE=1.5` 时，日志目录为 `H5_back_D1.5`。

## 按键

| 键 | 功能 |
|----|------|
| `G` | 生成所有 actor |
| `F` | 开始 / 停止跟随 |
| `D` | 循环切换难度（任意时刻可用） |
| `T` | 切换目标 track ID |
| `H` | 手动控制目标 NPC |
| `J` | 手动控制机器人 |
| `R` | 重置场景 |
| `SPACE` | 暂停 / 继续 |
| `V` | 打开 / 关闭 NPC Live Viewer |
| `Q` | 退出 |

## 难度档位

| 档位 | 同时过门人数上限 | 速度范围 |
|------|----------------|---------|
| easy   | 2 | 0.80 ~ 1.04 m/s |
| hard   | 4 | 0.88 ~ 1.20 m/s |
| boss   | 6 | 0.96 ~ 1.36 m/s |
| legend | 8 | 1.12 ~ 1.52 m/s |

每次选择难度（启动时初始化、运行中按 D）都会**重新随机派送一个天气**，记录到 `episode_meta.json` 的 `scenario_extras.weather` 字段。`--weather <name>` 显式指定后则锁定，不再随机。

## 天气预设

| key | 中文 | 视觉特征 |
|---|---|---|
| `clear` | 晴朗 | 高空太阳、对比度强、视野最佳 |
| `cloudy` | 阴天 | 80% 云、阴影柔和 |
| `dusk` | 下午傍晚 | 太阳低角度（8°）、暖色调 |
| `rain` | 雨天 | 80 强降雨 + 60 沉积水 + 80 wetness、地面反光 |
| `fog` | 大雾 | fog_density=80、能见度 ~8m |

预览图：见 `docs/figures/park_weather_preview.png`（5 个天气拼图）。

## 评估输出

跑完后 `outputs/eval_logs/passage/<planner>/H<人数>_<位置>_D<距离>/trial_NNN_<时间戳>/`：

| 文件 | 内容 |
|---|---|
| `episode_meta.json` | 启动配置、actor footprint、`scenario_extras.{difficulty,weather,difficulty_history,weather_history}`、`termination_reason` |
| `episode_step.jsonl` | 每 tick：robot/target/humans 状态、可见性、clearance、command、collision flag |
| `collision_events.jsonl` | 碰撞事件流（来源：CARLA 碰撞传感器 + 几何 fallback） |
| `eval_result.json` | 终局指标：`target_visibility_ratio`、`collision_count`、`human_collision_count`、`obstacle_avoidance_success`、`search_success`、`tracking_completion`、`geometry_collision_detected` |
| `eval_summary.txt` | 可读摘要 |

## 核心文件

| 文件 | 说明 |
|------|------|
| `followbench/scenes/passage/npc_runtime.py` | NPC 状态机（IDLE→APPROACH→QUEUE→PASS）、SFM 避障、难度系统 |
| `followbench/scenes/passage/run_episode.py` | 主循环、pygame UI、传感器、planner 接入 |
| `scripts/run_episode.sh --scene passage` | 启动脚本，设置默认参数 |
| `assets/scenes/passage/gridmap_roi.npz` | 可行走区域占用栅格 |
| `assets/scenes/passage/routes.json` | 4 条路由（Queue A/B、Pass A→B/B→A） |
| `assets/scenes/passage/gridmap_roi_flow_points.json` | 4 个 spawn 区域锚点 |

## NPC 状态机

```
IDLE ──(release_tick到达)──→ APPROACH ──(到达队列位置)──→ QUEUE
                                 ↑                             │
                          (depart_front                  (到队首且
                           重新分配位置)                 passing_count < max)
                                 │                             │
                                 └──────── PASS ←─────────────┘
                                     (过门后→ _on_passage_done
                                      → 加入对侧队尾 → APPROACH)
```

## SFM 参数

| 参数 | 值 | 说明 |
|------|----|------|
| `_SFM_PED_R` | 1.25 m | 行人感知半径 |
| `_SFM_PED_A` | 2.0 | 排斥力强度 |
| `_SFM_PED_B` | 0.8 m | 衰减距离 |
| `_SFM_LATERAL_BIAS` | 0.6 | 正面相遇时右偏权重 |
| `_SFM_ROBOT_R` | 2.0 m | 机器人感知半径 |

死区（dead zone）：agent 距 waypoint < 0.2 m 时 goal force 归零，自然静止；
被推开超过 0.2 m 后 goal force 恢复，自动回位。
