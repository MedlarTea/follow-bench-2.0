# Plaza 场景（NPC/Robot 解耦）

相关入口：

1. **crowd 验收脚本**：`tools/carla/roi_crowd_runner.py`（`tools/carla/run_roi_crowd_demo.sh`）
2. **episode 管理器**：`followbench/scenes/plaza/run_episode.py`

重点是第 2 个：支持按键触发、NPC GT 流、机器人传感器与标定导出、instance 像素可见性判定、以及可插拔跟随策略。

## 目录结构（核心）

- `followbench/scenes/plaza/run_episode.py`：综合控制器（Episode 管理）
- `followbench/scenes/plaza/npc_runtime.py`：NPC 生成/对向流动/编号显示/GT 接口
- `followbench/sim/robot_runtime.py`：Scout 机器人生成、控制、传感器、标定导出（所有场景共享）
- `followbench/sim/follow_policy_adapter.py`：机器人控制策略插件接口 + GT PID 默认实现
- `followbench/sim/visibility_instance.py`：目标可见性判定（instance 像素阈值）
- `followbench/sim/core_types.py`：统一数据结构（Observation/Action/State）
- `scripts/run_episode.sh --scene plaza`：快速启动（工程版）
- `assets/scenes/plaza/`：当前场景资产（ROI、四点锚点）

## 工程原理（简要）

### 1) 解耦架构

- **NPC Runtime** 只管 NPC：生成、走动、避让、编号、GT 输出
- **Robot Runtime** 只管机器人：Scout 控制、传感器数据、内外参
- **Policy Adapter** 只管算法：输入 observation，输出 `(v,w)`
- **Episode Manager** 只做调度：键盘状态机、日志、可见性判定

这样你合作者只需要替换 `follow_policy_adapter.py` 里的策略类，不需要动 NPC 场景逻辑。

### 2) 对向冲流逻辑

- `A,B` 定义 Side A
- `C,D` 定义 Side B
- NPC 在 A/B 两侧往返（不是 `A->B->C->D` 环路）

### 3) 可见性判定（instance）

使用 `sensor.camera.instance_segmentation`：

- 每帧计算目标行人像素数 `target_pixel_count`
- 若 `target_pixel_count > threshold`，则 `visible=True`

输出到 `visibility_stream.jsonl`。

## 快速启动（工程版）

前提：

- CARLA 服务已启动（默认 `127.0.0.1:2000`）
- Python 环境可导入 `carla`、`numpy`（建议 `carla38`）
- 若使用键盘窗口，需可导入 `pygame`（可选）

在本目录执行：

```bash
bash scripts/run_episode.sh --scene plaza
```

## 键盘操作（工程版）

- `G`：生成全部 actor（NPC + Robot），初始静止
- `F`：开始/停止跟随（同时启动/暂停 NPC 运动）
- `T`：切换目标 NPC（N01/N02/...）
- `R`：重置当前 episode（销毁 actor，保留参数）
- `Q`：退出

参数调节（建议在 `G` 前调，或 `R` 后重新 `G` 生效）：

- `Up/Down`：调整行人数（每次 ±2）
- `[` `]`：调整人群最小速度 `min_speed`
- `-` `=`：调整人群最大速度 `max_speed`

## 传感器宫格与可见性验收

`G` 后窗口会显示 2x2 宫格：

- 左上：RGB
- 右上：Depth
- 左下：Instance
- 右下：LiDAR BEV

顶部会实时显示：

- `Visibility(instance): pix=<count> thr=<threshold> => visible=<0/1>`

其中可见性判定规则为：`pix > threshold` 即可见。

默认启动脚本已开启：

- `--draw-laser`：机器人跟随时绘制红色“激光线”指向目标
- `--robot-rescue-lift-z 0.25`：机器人生成后执行一次抬升复位，降低卡地风险
- `--robot-spawn-mode near_target`：机器人默认在 ROI 内、目标 NPC 附近生成（不是固定世界坐标）
- 启动脚本默认把机器人初始距离限制在目标 `0.8m ~ 2.0m`

## 运行输出

每次运行会在 `runs/<timestamp>/` 下产生：

- `npc_gt_stream.jsonl`：每帧所有 NPC GT（位置、速度、ID）
- `visibility_stream.jsonl`：目标可见性流（pixel_count 与阈值）
- `robot_sensor_calibration.json`：传感器内参与 robot->sensor 外参

## 旧脚本

若你只想继续跑纯 crowd 可视化验收：

```bash
bash tools/carla/run_roi_crowd_demo.sh
```
