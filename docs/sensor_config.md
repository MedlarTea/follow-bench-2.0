# 传感器参数配置说明

本文档说明 Follow-Bench v2 当前场景（random / passage / corner）中，机器人传感器参数在哪里修改、每个参数的作用，以及如何验证生效。

---

## 1. 参数修改入口总览

当前有两层参数来源：

1. **运行脚本层（推荐优先）**  
   在各场景 `scripts/run_episode.sh` 里通过命令行参数传入，例如：
   - `--sensor-image-w`
   - `--sensor-image-h`
   - `--sensor-fov-deg`
   - `--sensor-lidar-range`
   - `--sensor-lidar-height-agl`

2. **代码默认层（兜底）**  
   在 `followbench/sim/robot_runtime.py` 的 `spawn_sensors()` 里定义默认值和挂载参数：
   - 相机/雷达挂载位置（外参）
   - LiDAR 通道数、上下视场、每秒点数等

---

## 2. 具体文件位置

### 2.1 场景启动脚本（脚本层）

- `scripts/run_episode.sh --scene plaza`
- `scripts/run_episode.sh --scene passage`
- `scripts/run_episode.sh --scene corner`

你通常应该先改这些脚本里的 CLI 参数。

### 2.2 通用管理器（参数转发层）

- `followbench/scenes/plaza/run_episode.py`

负责解析参数并传给 robot runtime。

### 2.3 传感器实现（默认值与底层行为）

- `followbench/sim/robot_runtime.py`

`ScoutRobotRuntime.spawn_sensors()` 是核心入口。

---

## 3. 目前可调参数与作用

### 3.1 相机参数

在启动脚本中通过以下参数控制：

- `--sensor-image-w`：图像宽度（像素）
- `--sensor-image-h`：图像高度（像素）
- `--sensor-fov-deg`：相机水平视场角（度）

作用：

- 分辨率越高，画面更细，但计算和传输开销更大；
- FOV 越大，看到范围更广，但目标像素通常更小。

### 3.2 LiDAR 参数（脚本层）

- `--sensor-lidar-range`：雷达量程（米）
- `--sensor-lidar-height-agl`：单线扫描平面离地高度（米，默认 `0.50`）

作用：

- 量程越大，看到越远，但点云分布更稀、噪声区域更大；
- `height-agl` 控制 2D 激光扫描高度（相对地面），与相机高度解耦。

### 3.3 LiDAR 参数（代码层，当前默认）

在 `robot_runtime.py` 中已固定为单线配置：

- `channels = 1`
- `upper_fov = 0`
- `lower_fov = 0`
- `points_per_second = 12000`
- `rotation_frequency = 1/dt`

这表示单层扫描线，不是多线雷达。

### 3.4 传感器安装外参（代码层）

在 `robot_runtime.py` 的 `spawn_sensors()` 内：

- 相机挂载高度：`camera_mount_z`（当前约 1.05 m，相对 actor 原点）
- 单线雷达：`lidar_mount_z = scan_height_agl - origin_height_agl`
  - Scout 默认 `origin_height_agl = 0.35` → `lidar_mount_z ≈ 0.15`
  - G1 人形（`humanoid_runtime.py`）`origin_height_agl = 0.65` → `lidar_mount_z ≈ -0.15`

作用：

- 改 `origin_height_agl` 会修正不同机器人 actor 原点到地面的偏差；
- 改 `--sensor-lidar-height-agl` 会整体抬高/降低扫描平面。

---

## 4. 如何改参数（推荐方式）

推荐优先改某个场景的 `scripts/run_episode.sh`，例如 `corner`：

```bash
python "$BASE_DIR/`run_episode.py`" \
  ... \
  --sensor-image-w 1024 \
  --sensor-image-h 768 \
  --sensor-fov-deg 80 \
  --sensor-lidar-range 25 \
  --sensor-lidar-height-agl 0.50 \
  ...
```

如果要改“单线/多线”，改 `robot_runtime.py` 的 `channels` / `upper_fov` / `lower_fov`。
扫描高度优先用 `--sensor-lidar-height-agl`；若某机器人型号仍偏高/偏低，再微调
`SCOUT_ORIGIN_HEIGHT_AGL` 或 `G1_ORIGIN_HEIGHT_AGL`。

---

## 5. 修改后如何验证生效

### 5.1 UI 可视化验证

运行 manager 后，2x2 面板包含：

- RGB
- Depth
- Instance
- LiDAR BEV

观察项：

- 分辨率/FOV 改动是否明显影响画面视角与像素密度；
- LiDAR 是否呈单线扫描特征（稀疏、单层）；BEV 仅显示 **range < 10 m** 命中点。
- `--debug` 俯视图叠加 **青色** LiDAR 点（世界坐标，同样 < 10 m）。

### 5.2 标定文件验证

每轮运行输出：

- `runs/<timestamp>/robot_sensor_calibration.json`

检查：

- 相机内参：`width/height/fov_deg/fx/fy/cx/cy`
- 雷达参数：`channels/range/upper_fov/lower_fov/points_per_second/scan_height_agl`
- 外参：`extrinsics_robot_to_sensor`

---

## 6. 常见需求与改法

### 需求 A：把相机再降低一点

改 `robot_runtime.py`：

- `camera_mount_z` 调小（例如 `1.05 -> 0.95`）

### 需求 B：把单线雷达改回 32 线

改 `robot_runtime.py`：

- `channels: 1 -> 32`
- `upper_fov/lower_fov` 改为如 `10/-30`
- `points_per_second` 相应提高

### 需求 C：仅某个场景改分辨率，不影响其他场景

只改该场景的 `scripts/run_episode.sh` 参数，不改 `robot_runtime.py` 默认值。

---

## 7. 设计建议

为避免多人协作时频繁改代码，后续可把传感器参数抽成独立配置文件（JSON/YAML），由 `run_episode.py` 读取并注入 runtime。

