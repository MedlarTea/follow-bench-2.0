# 遥控数据采集脚本

通用遥控参数和操作流程以 [`teleop_guide.md`](teleop_guide.md) 为准。

本文档说明如何用遥控方式为四个场景采集数据（人群、成功条件、评测日志）：

| 场景 | 仓库场景 | 启动脚本 |
|---|---|---|
| Open Plaza | `--scene plaza` | `collect_teleop_data.sh --scene plaza` |
| Street Corner | `--scene corner` | `collect_teleop_data.sh --scene corner` |
| Parked Park | `--scene park` | `collect_teleop_data.sh --scene park` |
| Narrow Passage | `--scene passage` | `collect_teleop_data.sh --scene passage` |

## 使用方法

先启动一次 CARLA：

```bash
cd /path/to/followbench2.0-release
bash tools/carla/run_carla.sh -RenderOffScreen -nosound
```

在另一个终端激活环境：

```bash
conda activate <your-env>
```

运行单个场景、单个难度，例如 Open Plaza Hard 连续 5 次：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 bash scripts/teleop/collect_teleop_data.sh --scene plaza hard
```

其他场景同理：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 bash scripts/teleop/collect_teleop_data.sh --scene corner boss
NUM_EPISODES=5 OPERATOR_ID=op01 bash scripts/teleop/collect_teleop_data.sh --scene park legend
NUM_EPISODES=5 OPERATOR_ID=op01 bash scripts/teleop/collect_teleop_data.sh --scene passage easy
```

一次顺序运行全部 16 组：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 bash scripts/teleop/collect_teleop_data.sh --all
```

这会运行 `4 场景 x 4 难度 x 5 次 = 80` 个 episode。每组完成后会自动启动下一组；新窗口出现后继续按 `G`、`F`。

## 侧跟随

单个场景左侧跟随：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 \
FOLLOW_POSITION=left_side DESIRED_DISTANCE=1.5 \
bash scripts/teleop/collect_teleop_data.sh --scene park hard
```

右侧跟随只需改成：

```bash
FOLLOW_POSITION=right_side
```

全部四个场景、四种难度跑左侧跟随：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 \
FOLLOW_POSITION=left_side DESIRED_DISTANCE=1.5 \
bash scripts/teleop/collect_teleop_data.sh --all
```

一次跑左右两侧：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 \
FOLLOW_POSITIONS="left_side right_side" SIDE_DISTANCE=1.5 \
bash scripts/teleop/collect_teleop_data.sh --all
```

一次跑后跟、左侧跟随和右侧跟随：

```bash
NUM_EPISODES=5 OPERATOR_ID=op01 \
FOLLOW_POSITIONS="back left_side right_side" \
BACK_DISTANCE=1.5 SIDE_DISTANCE=1.5 \
bash scripts/teleop/collect_teleop_data.sh --all
```

最后一条命令会运行 `4 场景 x 4 难度 x 3 跟随位置 x 5 次 = 240` 个 episode。侧跟随中的 `DESIRED_DISTANCE` / `SIDE_DISTANCE` 表示机器人相对目标人的侧向期望距离。

通用可选环境变量：

```text
NUM_EPISODES       每组 episode 数，默认 5
NUM_WALKERS        行人总数（包含目标 N01）；不设置时使用场景/难度默认值
ROBOT_TYPE         scout / g1，默认 g1
SIM_SPEED          仿真速度倍率，默认 1.0
DURATION_SEC       每个 episode 最长仿真时间（秒），默认 300
OPERATOR_ID        操作员编号，默认 anonymous
FOLLOW_POSITION    back / left_side / right_side，默认 back
FOLLOW_POSITIONS   总矩阵要顺序运行的位置列表，例如 "left_side right_side"
DESIRED_DISTANCE   跟随距离，默认 1.5
BACK_DISTANCE      总矩阵中的后跟距离，默认取 DESIRED_DISTANCE
SIDE_DISTANCE      总矩阵中的左右侧跟距离，默认取 DESIRED_DISTANCE
WEATHER            天气，默认 random；每 5 次将五种天气随机排序后各使用一次
SIM_MODE           仅支持 sync；teleop 使用 async 会拒绝启动
PRINT_CONFIG=1     只打印最终参数，不连接 CARLA
```

行人数均为总人数，包含目标行人 `N01`。Narrow Passage 默认生成 8 人，因此 Legend 的并发上限 8 可以实际生效；Parked Park 的 `max_per_segment` 也统计目标、环绕、主干和调度行人等所有类型。

正式采集时难度和调参按键会由 teleop 模式锁定。每个 episode 的操作为 `G` 生成、`F` 开始、`WASD` 控制；达到终止条件后会自动保存并准备下一次。

评估结果位于：

```text
outputs/eval_logs/<scenario>/teleop/H*_<follow_position>_D<distance>/trial_*/
```

`park` 和 `passage` 的难度保存在每个 trial 的 `episode_meta.json` 中。
