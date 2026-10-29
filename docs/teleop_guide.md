# FollowBench 2.0 遥控数据采集手册

本文档说明如何使用遥控数据采集套件运行 OpenPlaza、Corner、
Passage 和 Park 四个场景，并配置机器人类型、场景人数、仿真速度和
连续 episode 数量。

## 1. 环境准备

进入项目并激活环境：

```bash
conda activate <your-env>
cd /path/to/followbench2.0-release
```

遥控采集需要：

- CARLA Server 已运行，默认地址为 `127.0.0.1:2000`。
- 当前 Python 环境能够导入 `carla` 和 `pygame`。
- 使用前台图形窗口，不能添加 `--no-ui`。
- 使用同步仿真模式。teleop 不支持 `SIM_MODE=async`。

## 2. 常用参数

推荐在启动命令前通过环境变量设置参数，不必修改 Python 代码。

| 参数 | 含义 | 推荐值 |
|---|---|---|
| `NUM_EPISODES` | 同一进程连续运行的 episode 数 | `5` |
| `NUM_WALKERS` | 行人总数，包含目标人物 | 按实验设置 |
| `ROBOT_TYPE` | 机器人类型：`scout` 或 `g1` | `g1` |
| `SIM_SPEED` | 目标仿真速度倍率 | `1.0` |
| `--operator-id` | 匿名操作者编号 | 例如 `operator01` |
| `--duration-sec` | 每个 episode 最长仿真时间，单位秒 | 例如 `300` |

人数口径：

```text
NUM_WALKERS=12
= 1 个目标人物 + 11 个其他行人
= 日志目录中的 H11
```

仿真倍率：

- `SIM_SPEED=1.0`：目标实时速度。
- `SIM_SPEED=0.5`：半速，适合熟悉控制。
- `SIM_SPEED=2.0`：目标二倍速。

倍率是目标上限。如果渲染、传感器或计算负载较高，实际速度可能低于
设置值，但不会为了追赶时间而跳过仿真 tick。

## 3. OpenPlaza

OpenPlaza 在代码中对应场景 `plaza`（`--scene plaza`）。

```bash
NUM_EPISODES=5 \
NUM_WALKERS=12 \
ROBOT_TYPE=g1 \
SIM_SPEED=1.0 \
scripts/run_episode.sh --scene plaza --teleop \
  --operator-id operator01 \
  --duration-sec 300
```

命令末尾的 `\` 表示下一行仍属于同一条命令，可以整段粘贴执行。也可以
写成完全等价的一行：

```bash
NUM_EPISODES=5 NUM_WALKERS=12 ROBOT_TYPE=g1 SIM_SPEED=1.0 scripts/run_episode.sh --scene plaza --teleop --operator-id operator01 --duration-sec 300
```

如果没有设置 `NUM_WALKERS`，OpenPlaza 会使用难度默认人数：

| 难度 | 总人数 |
|---|---:|
| `easy` | 10 |
| `hard` | 15 |
| `boss` | 20 |
| `legend` | 25 |

四种难度的 OpenPlaza 行人速度范围均为 `0.60–1.60 m/s`。

设置 `NUM_WALKERS` 后只覆盖人数，难度对应的行人速度配置仍然生效：

```bash
DIFFICULTY=hard NUM_WALKERS=12 \
ROBOT_TYPE=g1 SIM_SPEED=1.0 \
scripts/run_episode.sh --scene plaza --teleop \
  --operator-id operator01
```

## 4. Corner

```bash
NUM_EPISODES=5 \
NUM_WALKERS=12 \
ROBOT_TYPE=g1 \
SIM_SPEED=1.0 \
scripts/run_episode.sh --scene corner --teleop \
  --operator-id operator01 \
  --duration-sec 300
```

`DIFFICULTY` 仍控制行人速度、通道分布等参数；`NUM_WALKERS` 可以单独
覆盖该难度的默认人数。

## 5. Passage

```bash
NUM_EPISODES=5 \
NUM_WALKERS=6 \
ROBOT_TYPE=g1 \
SIM_SPEED=1.0 \
scripts/run_episode.sh --scene passage --teleop \
  --operator-id operator01 \
  --duration-sec 300
```

## 6. Park

```bash
NUM_EPISODES=5 \
NUM_WALKERS=12 \
ROBOT_TYPE=g1 \
SIM_SPEED=1.0 \
scripts/run_episode.sh --scene park --teleop \
  --operator-id operator01 \
  --duration-sec 300
```

## 7. 键盘操作

| 按键 | 功能 |
|---|---|
| `G` | 生成当前 episode 的行人和机器人 |
| `F` | 开始或暂停当前 episode |
| `W` | 前进 |
| `S` | 后退，G1 后退时相机朝向保持不变 |
| `A` / `D` | 左转 / 右转 |
| `Shift` | 加速 |
| `Space` | 暂停或恢复整个场景 |
| `R` | 手动重置当前 episode |
| `Q` | 退出 |

标准操作流程：

1. 启动 teleop 脚本。
2. 等待 Pygame 窗口和 CARLA 场景加载完成。
3. 按 `G` 生成场景。
4. 确认机器人和目标人物位置正常。
5. 按 `F` 开始计时和记录。
6. 使用 `W/A/S/D` 控制机器人。
7. episode 自动结束后，程序准备下一次运行。
8. 再次按 `G`、`F` 开始下一个 episode。

等待生成场景以及按下第一次 `F` 之前的时间不计入 episode 时长。

## 8. Scout 与 G1

使用 G1：

```bash
ROBOT_TYPE=g1
```

使用 Scout：

```bash
ROBOT_TYPE=scout
```

两种机器人使用相同的键盘控制、场景逻辑、指标和连续 episode
流程。元数据中的 `scenario_extras.robot_type` 会记录实际选择。

## 9. 采集结果

默认采集结果位于各场景的 evaluation 日志目录中，每个 episode 使用
独立的 `trial_*` 目录。主要文件包括：

| 文件 | 内容 |
|---|---|
| `episode_meta.json` | 场景、机器人、人数、操作者和参数 |
| `episode_step.jsonl` | 每 tick 状态和人工速度命令 |
| `collision_events.jsonl` | 碰撞事件 |
| `eval_result.json` | 最终指标 |
| `eval_summary.txt` | 文本格式指标摘要 |

采集记录应满足：

```text
planner = teleop
control_mode = human_keyboard
operator_id = 启动时传入的编号
session_episode = 当前是第几个 episode
session_num_episodes = 本次计划运行的总 episode 数
robot_type = scout 或 g1
sim_speed = 启动时设置的倍率
```

## 10. 修改永久默认值

通常应在命令行设置参数。确实需要修改默认值时，可编辑对应脚本：

```text
scripts/run_episode.sh --scene plaza --teleop
scripts/run_episode.sh --scene corner --teleop
scripts/run_episode.sh --scene passage --teleop
scripts/run_episode.sh --scene park --teleop
```

例如：

```bash
NUM_EPISODES="${NUM_EPISODES:-5}"
ROBOT_TYPE="${ROBOT_TYPE:-g1}"
SIM_SPEED="${SIM_SPEED:-1.0}"
```

各场景默认人数位于对应的 `scripts/run_episode.sh`。为了实验可复现，
仍建议保留脚本默认值，并在每次实验命令中显式传入参数。

## 11. 常见问题

### 无法连接 CARLA

确认 CARLA Server 已启动且端口正确：

```bash
ss -ltn | grep 2000
```

如使用其他端口：

```bash
scripts/run_episode.sh --scene plaza --teleop --port 2002
```

### G1 blueprint 不存在

当前套件需要包含以下自定义 blueprint 的 CARLA：

```text
walker.pedestrian.g1_unitree
```

应使用 FollowBench 项目配套的自定义 CARLA Server。

### 设置二倍速但没有达到二倍

`SIM_SPEED` 只能限制最大速度，不能突破机器的渲染、传感器和 Python
处理能力。降低分辨率、人数或关闭额外 debug 可视化可以提升实际速度。

### 人数与 H 编号不同

这是正常现象。`NUM_WALKERS` 包含目标人物，而 `H` 只统计目标以外的
其他行人：

```text
H = NUM_WALKERS - 1
```
