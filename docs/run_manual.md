# docs/run_manual.md — 运行手册

这份文档记录 FollowBench 2.0 **实验脚本** 的启动方式与可选参数：如何选 planner、开关感知前端、切换跟随姿态（后向 / 侧向）、调整 ReID / 丢失策略等。

> 需要先启动 CARLA Server 与对应的 Conda 环境（`followBench`）。

---

## 0. 首次 clone 后的准备工作

项目使用 `data/weights/` 统一管理权重文件，**权重本身不进 git**（`.gitignore` 屏蔽了 `*.pt / *.pth.tar / *.t7`）。Clone 后需要准备以下文件：

```
data/weights/
├── yolo/               ← YOLO 检测模型（首次运行自动下载）
│   ├── yolo11n.pt
│   └── yolo11s.pt
├── reid_basic/         ← 基础 ReID（手动放置）
│   └── ckpt.t7
├── reid_kpr/           ← KPR ReID（可选，--reid-mode kpr 时需要，手动放置）
│   └── kpr_*.pth.tar
└── traj_predictor/     ← 轨迹预测（rda_traj / rda_search 需要，手动放置）
    ├── sgan.pt
    ├── csgan.pt
    └── csgan2.pt
```

### YOLO 权重（自动）

发布包**不包含**这两个文件（Ultralytics AGPL-3.0）。首次启动感知模式时，`ultralytics` 会自动将 `yolo11s.pt`（~19 MB）和 `yolo11n.pt`（~5.4 MB）下载到 `data/weights/yolo/`。

也可以事先放好：

```bash
mkdir -p data/weights/yolo
curl -L -o data/weights/yolo/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
curl -L -o data/weights/yolo/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
```

### KPR 符号链接（已进 git，自动还原）

```
followbench/perception/reid_kpr/deep_person_reid/pretrained_models
→ ../../../../data/weights/reid_kpr   (symlink)
```

这条 symlink 已经进入版本库，clone 后自动还原。当你把 `data/weights/reid_kpr/*.pth.tar` 放好之后，KPR 代码通过这条 symlink 就能找到文件，**不需要任何代码改动**。

> **注意**：`--reid-mode kpr` 还额外需要 `wandb` 和 `tensorboard` 依赖：
> ```bash
> pip install wandb tensorboard
> ```

### 其他权重（手动放置）

按 [`docs/01_install.md`](01_install.md) 中的下载说明获取后，按上方目录结构放好即可。

---

## 1. 快速开始

### 1.1 Corner 场景
```bash
bash scripts/run_episode.sh --scene corner
```

### 1.2 Park 场景
```bash
bash scripts/run_episode.sh --scene park
```

两个脚本默认跑 `PLANNER=pid` + `FOLLOW_POSITION=back`，不启用感知前端（走 GT baseline）。

### 1.3 遥控数据采集

四个场景使用相同的环境变量。`plaza` 即 OpenPlaza，`NUM_WALKERS`
表示包含目标人物在内的总人数，因此评测目录中的 `H` 等于
`NUM_WALKERS - 1`。

```bash
NUM_EPISODES=5 \
NUM_WALKERS=12 \
ROBOT_TYPE=g1 \
SIM_SPEED=1.0 \
scripts/run_episode.sh --scene plaza --teleop --operator-id operator01
```

- `ROBOT_TYPE=scout|g1`：选择 Scout 或 Unitree G1。
- `SIM_SPEED=1.0`：实时；`0.5`：半速；`2.0`：目标二倍速。实际速度上限受渲染和传感器计算吞吐限制。
- `NUM_WALKERS`：总行人数，四个场景均可设置。
- `NUM_EPISODES`：同一 CARLA/UI 进程连续运行的 episode 数。

把 `--scene plaza` 换成 `park`、`corner` 或 `passage` 即可运行另外三个场景。

---

## 2. 启动参数的三种传递方式

### 方式 A：环境变量（场景预设 `configs/scenes/<scene>/episode.sh` 读取）
场景预设暴露了几个常用开关，用 `KEY=value` 前缀即可覆盖：

| 变量 | 含义 | 默认 |
|---|---|---|
| `PLANNER` | 选择 planner | `pid` |
| `FOLLOW_POSITION` | 跟随姿态（`back` / `left_side` / `right_side`） | `back` |
| `DESIRED_DISTANCE` | 期望跟随距离（米） | `1.5`（corner）/ `3.0`（park） |
| `NUM_WALKERS`（park） | 行人数量 | `6` |
| `SIM_MODE`（park） | 仿真模式（`async` / `sync`） | `async` |

示例：
```bash
PLANNER=rda_lidar bash scripts/run_episode.sh --scene <scene>
PLANNER=gt_planner bash scripts/run_episode.sh --scene <scene>
PLANNER=bso_hfc FOLLOW_POSITION=left_side bash scripts/run_episode.sh --scene <scene>
PLANNER=sfm DESIRED_DISTANCE=2.0 bash scripts/run_episode.sh --scene <scene>
```

### 方式 B：透传任意 CLI 参数
`scripts/run_episode.sh` 会把你追加的参数原样转给 Python runner（排在预设参数之后，因此可覆盖预设）：
`FB_DRY_RUN=1` 只打印最终命令。

```bash
# 打开感知前端
bash scripts/run_episode.sh --scene <scene> --use-perception

# 调整仿真时长（默认 300 秒）
bash scripts/run_episode.sh --scene <scene> --duration-sec 60

# 打开 2D debug 可视化
bash scripts/run_episode.sh --scene <scene> --debug

# 组合
PLANNER=rda_lidar bash scripts/run_episode.sh --scene <scene> \
    --use-perception --reid-mode kpr --debug --duration-sec 120
```

### 方式 C：直接跑 Python（不走场景预设）
```bash
python -m followbench.scenes.corner.run_episode \
    --planner rda_lidar --use-perception \
    --load-map --map-name Town10HD_Opt \
    --duration-sec 60
```
绕过场景预设后所有参数都要手写，一般不推荐。

---

## 3. 可用的 Planner 一览

### 3.1 纯 GT baseline（不加 `--use-perception`）

| `--planner` | 控制器 | 后向跟随 | 侧向跟随 | 目标丢失时 | 障碍避让 |
|---|---|:-:|:-:|---|---|
| `pid` | PID | ✅ | ✅ | 崩（不处理） | 无（只跟点） |
| `dwa_traj` | DWA + 轨迹规划 | ✅ | ✅ | 刹停 `(v=0, w=0)` | lidar BEV |
| `sfm` | 社会力模型 | ✅ | ✅ | 刹停 | lidar |
| `rda` | 基础 RDA-MPC | ✅ | ✅ | 崩 | 仅目标 |
| `rda_lidar` | RDA-MPC + lidar 聚类障碍 | ✅ | ✅ | 崩 | lidar 聚类 |
| `gt_planner` | GT 专家 RDA-MPC（12 m GT 行人 + ROI 静态图） | ✅ | ✅ | 崩（GT 目标） | GT 动态 + `obstacle_inflated` |
| `rda_traj` | RDA-MPC + 轨迹预测 | ✅ | ✅ | 崩 | 默认 lidar 聚类 + NPC；`--rda-traj-disable-lidar-obstacles` 可关 |
| `rda_search` | RDA-MPC + 主动搜索（overtake / fluid / occlusion） | ✅ | ✅ | **主动搜索** | lidar 局部地图 + 预测 |
| `bso_hfc` | BSO + Hybrid-A* + Bspline + MPC | ✅ | ✅ | 刹停 | lidar + 局部占据图 |
| `rda_depth_tpt` | RDA-MPC + RGB+深度 YOLO 跟踪（自带感知） | ✅ | ✅ | 感知内部 | lidar |
| `trackvla` | 端到端 VLA（DINOv3 + SigLIP + Qwen3-4B）| ✅ | ❌ | 端到端内部 | 无 |
| `oa_vat` | OA-VAT（YOLOe + ORTrack + DINOv3 ReID + PID）| ✅ | ❌ | 端到端内部 | 无 |

> "崩（不处理）" 指这些 planner 假设 `obs.target` 永远存在（GT 一定有目标）；一旦接入感知链路再丢失就会抛 `AttributeError`。加 `--use-perception` 后 wrapper 会自动兜底。

### 3.2 感知前端版本（加 `--use-perception`）
加上 flag 后，**除 `rda_depth_tpt` 以外的所有 planner** 都会被包装成感知版本：
- `obs.target` / `obs.npcs` 从 GT 切换为 YOLO+深度+ReID 的跟踪输出
- 支持多视角 tracker（front / left / right 相机联合 MOT + 跨视角 merge）
- `--lost-policy` 生效：`brake` / `last_known` / `gt_fallback`
- 对 `rda_search`：内部 `target_visible` 由 FSM lock 状态决定，感知丢失时自动进搜索模式

`rda_depth_tpt` **不支持** `--use-perception`（它自己就是感知 planner，baseline 无意义）。

---

## 4. 跟随姿态（Follow Position）

所有上述 planner 都支持通过 `--follow-position`（或环境变量 `FOLLOW_POSITION`）切换姿态：

| 值 | 含义 | 常见用例 |
|---|---|---|
| `back`（默认） | 机器人跟在目标**后方** `DESIRED_DISTANCE` 米 | 跟随员 / 观察员 |
| `left_side` | 机器人在目标**左侧** `DESIRED_DISTANCE` 米平行跟随 | 并肩护送 |
| `right_side` | 机器人在目标**右侧** `DESIRED_DISTANCE` 米平行跟随 | 并肩护送 |

示例：
```bash
# 后向跟随（默认）
PLANNER=rda_lidar bash scripts/run_episode.sh --scene <scene>

# 左侧跟随
PLANNER=rda_lidar FOLLOW_POSITION=left_side bash scripts/run_episode.sh --scene <scene>

# 右侧跟随 + 感知
PLANNER=rda_lidar FOLLOW_POSITION=right_side bash scripts/run_episode.sh --scene <scene> --use-perception
```

> `rda_traj` 默认把 lidar/NPC 障碍送进 MPC；若要复现旧版行为（空障碍列表），加 `--rda-traj-disable-lidar-obstacles`。

> `rda_search` 在侧向跟随时默认会给目标注入一个"保护区"禁止机器人穿过。可用 `--rda-search-disable-target-protection` 关闭。

---

## 5. 感知前端的可选参数

仅在 `--use-perception` 时生效：

| 参数 | 选项 | 默认 | 含义 |
|---|---|---|---|
| `--reid-mode` | `basic` / `kpr` | `basic` | 外观 ReID 骨干：`basic` 轻量 ResNet；`kpr` 是 SOLIDER-Swin，抗遮挡但吃显存 |
| `--reid-kpr-config` | 配置名 | `kpr_occ_duke_test` | 仅 `--reid-mode=kpr` 时用 |
| `--reid-device` | `auto` / `cuda` / `cpu` | `auto` | ReID 推理设备 |
| `--lost-policy` | `brake` / `last_known` / `gt_fallback` | `last_known` | 感知丢失目标时的兜底策略 |

三种 lost policy：
- **`brake`**：立即刹停 `(v=0, w=0)` 等 ReID 找回。最安全。
- **`last_known`**（默认）：把最后一次跟踪到的位置当静止目标，让 MPC 自然滑行过去后减速到零。最自然。
- **`gt_fallback`**：感知丢失时直接用 GT target（仅调试用，**破坏感知闭环**，不建议跑正式实验）。

`rda_search` 强制 `last_known`：其他两个选项会短路搜索逻辑。

---

## 6. 常用命令速查

```bash

# ── 1. GT baseline（最快，无 YOLO 开销）
PLANNER=rda_lidar bash scripts/run_episode.sh --scene corner
PLANNER=bso_hfc FOLLOW_POSITION=left_side bash scripts/run_episode.sh --scene corner

# ── 2. 感知前端 + 默认 ReID（basic ResNet）
PLANNER=rda_lidar bash scripts/run_episode.sh --scene corner --use-perception

# ── 3. 感知前端 + KPR ReID（抗遮挡）
PLANNER=rda_lidar bash scripts/run_episode.sh --scene corner --use-perception --reid-mode kpr

# ── 4. 主动搜索 + 感知（目标丢失会主动寻找）
PLANNER=rda_search bash scripts/run_episode.sh --scene corner --use-perception

# ── 5. 侧向跟随 + 感知
PLANNER=rda_lidar FOLLOW_POSITION=right_side bash scripts/run_episode.sh --scene corner \
    --use-perception

# ── 6. CPU-only 感知（无 GPU 或显存不够）
PLANNER=pid bash scripts/run_episode.sh --scene corner --use-perception --reid-device cpu

# ── 7. 感知 + 丢失即刹停（安全优先）
PLANNER=rda_lidar bash scripts/run_episode.sh --scene corner \
    --use-perception --lost-policy brake

# ── 8. 打开 2D debug 可视化
PLANNER=rda_lidar bash scripts/run_episode.sh --scene corner --use-perception --debug

# ── 9. 短 episode（30 秒，快速验证）
bash scripts/run_episode.sh --scene corner --duration-sec 30

# ── 10. 自带感知的 rda_depth_tpt（不用 --use-perception）
PLANNER=rda_depth_tpt bash scripts/run_episode.sh --scene corner

# ── 11. 录制机器人第一视角视频到评估目录
PLANNER=rda_traj bash scripts/run_episode.sh --scene corner \
    --enable-evaluation --eval-scenario-type corner \
    --eval-log-root /tmp/vis_demo \
    --use-perception --kinematics-mode kf \
    --visualize --duration-sec 30
# 后向跟随 → trial_*/episode.mp4
# 侧向跟随 → trial_*/episode_3view.mp4（Left | Front | Right 拼接）
```

### 6.1 录制 episode 视频（`--visualize`）

加 `--visualize` 后，每个 episode 结束时会在对应的 `eval_logger.run_dir`
（即 `trial_*` 目录，与 `eval_result.json` 同级）写入一个 mp4：

| `--follow-position` | 文件 | 布局 |
|---|---|---|
| `back`（默认） | `episode.mp4` | 前视相机单图 + 40 px HUD |
| `left_side` / `right_side` | `episode_3view.mp4` | `Left \| Front \| Right` 三视图水平拼接 + 40 px HUD |

- 帧率 = `1 / --dt`（默认 `dt=0.1` → **10 Hz**）。
- 编码器 mp4v（OpenCV `VideoWriter`，无新增依赖）；初始化失败时回退到 MJPG/.avi。
- `--use-perception` 时自动叠加 YOLO 跟踪框：绿色=目标，黄色=其他 track；
  跟踪信息陈旧（`track_bboxes_age > 0`）时颜色变暗并在 HUD 右上角显示
  红色 **STALE N** 标签。
- HUD 含：planner / follow_position / target_id / visible / dist / tick / t。
- **必须搭配 `--enable-evaluation`**（视频写入 trial 目录，没有 trial 就没地方写）。
- 批量评估：在 `configs/scenes/corner/batch/*.sh` 里设
  `VISUALIZE="1"` 即可对每个 slot 自动开启录制；smoke_test 默认已开。
  详见 `docs/03_batch_eval.md` 与 `docs/09_multi_gpu.md`。

### 6.2 单卡 worker（`scripts/multi_gpu/run_worker.sh`）

`run_worker.sh` 把一个 batch 绑定到一张 GPU：自己启动 CARLA、跑
`scripts/run_batch_eval.sh`，每 `--restart-every N` 个 slot 回收一次 CARLA
（内层以 `--max-slots` 退出 99，外层重启 CARLA 后用同一个 batch id 续跑）。
默认 `--restart-every 1`：每个 episode 都用新启动的 CARLA，并导出
`FB_LOAD_MAP_POLICY=if_needed`，episode 直接复用 CARLA 启动时已加载的
Town10HD_Opt，不再调用 `load_world`（该 CARLA 版本反复 `load_world` 会间歇性
卡死渲染线程，随后 Signal 11）。CARLA 以 `get_world()` 可应答作为就绪标准，
启动失败最多重试 `CARLA_START_ATTEMPTS`（默认 3）次。
默认在本机运行；加 `--container NAME` 则 CARLA 留在宿主、算法在容器里跑。

```bash
# smoke：GPU 0 / 端口 2000，跑完自动停掉自己启动的 CARLA
scripts/multi_gpu/run_worker.sh --scene corner --config smoke_test --gpu 0 --port 2000

# full 50x，每 50 个 slot 重启一次 CARLA（防显存泄漏）
scripts/multi_gpu/run_worker.sh --scene corner --config default_50x \
    --gpu 0 --port 2000 --restart-every 50 --batch-id corner_full_$(date +%Y%m%d)

# 宿主 CARLA + 容器算法
scripts/multi_gpu/run_worker.sh --scene corner --config smoke_test \
    --gpu 0 --port 2000 --container <container_name>
```

`--gpu N` 是 `nvidia-smi` 编号；CARLA 的 Vulkan adapter 编号通常与之不同，
脚本通过比对 PCI bus id 自动换算 `-graphicsadapter`（需要宿主装
`vulkan-tools`），也可用 `--carla-graphics-adapter K` 手动指定。

### 6.3 多 GPU 并行（`scripts/multi_gpu/eval_multi_gpu.sh`）

把 batch 矩阵按 `(planner_group, planner, difficulty[, flow_mode], follow_position)`
展开成 task，round-robin 分给 N 个 worker；worker i 使用 GPU `gpu_base+i`、
端口 `port_base+i*port_stride`，内部逐个 task 调用 `run_worker.sh`。所有场景
入口一致（plaza 多一个 `FLOW_MODES` 维度）：

```bash
# 8 卡并行，端口 2000 / 2100 / ... / 2700
scripts/multi_gpu/eval_multi_gpu.sh --scene corner --config default_50x --num-workers 8

# 4 卡并行 + 每 25 个 slot 重启 CARLA
scripts/multi_gpu/eval_multi_gpu.sh --scene plaza --config default_50x \
    --num-workers 4 --restart-every 25 --batch-id plaza_full_$(date +%Y%m%d)

# 验证多 GPU 路径：每个场景都有 REPEATS=1 的 smoke_all_planners
scripts/multi_gpu/eval_multi_gpu.sh --scene passage --config smoke_all_planners --num-workers 2
```

- CARLA 占用 3 个连续端口，`--port-stride` 必须 ≥ 3（默认 100）。
- `--start-stagger-s`（默认 25 秒）错峰启动 CARLA，避免多个 UE4 同时读盘。
- worker i 的 batch id 为 `<base>_w<i>`，结果在
  `$FB_OUTPUT_ROOT/logs_batch/<scene>/<base>_w<i>/`；task 配置、`w<i>.log`
  和合并后的 `manifest.csv` 在 `$FB_OUTPUT_ROOT/multi_gpu/<scene>/<base>/`。
- Ctrl+C / SIGTERM 会终止所有 worker 及其启动的 CARLA，不影响其他 CARLA。

`run_batch_eval.sh` 的容错行为（所有入口共享）：

- 每个 setting 会补跑到磁盘上有 `REPEATS` 份 `eval_result.json` 为止，单次
  运行内最多尝试 `REPEATS * MAX_ATTEMPT_FACTOR` 次（默认 2）。
- watchdog：slot 失败后若 CARLA 端口不通，或连续 `FAIL_FAST_AFTER`（默认 2）
  次失败，以 99 退出，由外层 worker 重启 CARLA 后续跑。
- Python 客户端超时可用 `CARLA_CLIENT_TIMEOUT_S` 调整。

### 6.4 多场景串联（`scripts/multi_gpu/run_pipeline.sh`）

依次对每个 `SCENE:CONFIG` job 调用 `eval_multi_gpu.sh`（同一时间只有一个
job 占用 GPU 池）：

```bash
scripts/multi_gpu/run_pipeline.sh --preset full --num-workers 8 \
    --restart-every 25 --batch-id-base full_$(date +%Y%m%d)

# 自定义 job 列表 / 只跑部分场景
scripts/multi_gpu/run_pipeline.sh --num-workers 4 \
    --job corner:smoke_test --job park:e2e_smoke_test
scripts/multi_gpu/run_pipeline.sh --preset rda_traj_kpr --only corner plaza --num-workers 4
```

| Preset | Jobs |
|---|---|
| `full` | `corner:default_50x`, `passage:back_only_50x`, `park:e2e_default_50x`, `plaza:back_only_50x` |
| `rda_traj_kpr` | `<scene>:rda_traj_kpr_50x`，四个场景 |
| `plaza_default` | `plaza:default_50x` |

- job 的 batch id 为 `<batch-id-base>_<scene>`。
- pipeline 日志和汇总在 `$FB_OUTPUT_ROOT/pipelines/<batch-id-base>/`。
- 某个 job 失败默认继续后面的 job，退出码取最差者；`--stop-on-failure` 遇错即停。
- 其余参数（`--gpu-base`、`--port-base`、`--container` 等）转发给 `eval_multi_gpu.sh`。

### 6.5 基线矩阵一键运行（`scripts/baselines/run_all_baselines.sh`）

```bash
scripts/baselines/run_all_baselines.sh --trials 10 --gpu 0 --port 2000
scripts/baselines/run_all_baselines.sh --trials 10 --scenes "corner park" --rows "traj_r18 e2e" --gpu 1 --port 2100
```

batch id 为 `baselines<trials>_<scene>_<row>`，可断点续跑。完整说明见 `docs/09_multi_gpu.md`。

---

## 7. TrackVLA（端到端视觉-语言-动作模型）

TrackVLA 是本项目的端到端 VLA planner。它**不使用任何手写控制器**：每帧 RGB 图像直接送入 ViT 视觉编码器 + Qwen3-4B VLM，由语言模型回归出 8 步轨迹航点，再转换成速度命令。

### 7.1 模型架构

```
RGB (384×384) → DINOv3-ViT-S16 (facebook) + SigLIP-So400M (google)
                     ↓  vision tokens (2048-d)
                 Qwen3-4B LLM (data/trackvla/LLM_hf/qwen3-4b)
                     ↓  8 个 (x, y) 航点（机器人坐标系）
                 取第 2 个航点 → (v, w) 速度命令
```

| 子模型 | 本地路径 | 大小 |
|---|---|---|
| TrackVLA checkpoint | `data/trackvla/ckpts/<run_id>/model_epoch04_step024000.pt` | 7.7 GB |
| Qwen3-4B LLM | `data/trackvla/LLM_hf/qwen3-4b` | 7.6 GB |
| SigLIP-So400M | `data/trackvla/google/siglip-so400m-patch14-384` | 3.3 GB |
| DINOv3-ViT-S16 | `data/trackvla/facebook/dinov3-vits16-pretrain-lvd1689m` | 83 MB，**不在发布包里** |
| **Qwen3-VL-4B-Instruct**（用于 auto-caption） | `data/trackvla/LLM_hf/qwen3-vl-4b-instruct` | ~8 GB，不在发布包里 |

DINOv3 在 Hugging Face 上是 gated 仓库。先在 <https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m> 接受协议，再执行：

```bash
huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir data/trackvla/facebook/dinov3-vits16-pretrain-lvd1689m
```

路径由 `constants.py` 中 `DATA_ROOT` 自动拼接，默认指向 `<repo>/data/trackvla/`。可通过环境变量 `TRACKVLA_DATA_ROOT` 覆盖（例如指向共享网络挂载盘）：

```bash
TRACKVLA_DATA_ROOT=/path/to/trackvla PLANNER=trackvla bash scripts/run_episode.sh --scene <scene>
```

### 7.2 推理频率与 dt

模型以 **10 Hz**（`dt=0.1s`）训练。如果 CARLA 仿真以更高频率跑（如 20 Hz），adapter 会自动每 2 帧才真正调用 VLM 一次，中间帧复用上一帧的命令，保证模型看到的时间窗口与训练时一致。

### 7.3 与其他 planner 的区别

| | TrackVLA | 其他 planner（如 rda_lidar） |
|---|---|---|
| 输入 | 前视 RGB 图像 + 文字指令 | GT NPC 位置（或感知） |
| 可解释性 | 端到端黑盒 | 显式控制逻辑 |
| `--use-perception` | ❌ 不兼容（它本身就是感知策略） | ✅ 支持 |
| `--follow-position` | ❌ 不支持 | ✅ back/left_side/right_side |

### 7.4 启动命令

```bash

# ── 基本启动（默认开启 auto-caption：Qwen3-VL 描述目标外观后跟随）
PLANNER=trackvla bash scripts/run_episode.sh --scene corner

# ── 关闭 auto-caption，回到朴素 "Follow the person"
PLANNER=trackvla bash scripts/run_episode.sh --scene corner --trackvla-no-auto-caption

# ── 自定义跟随指令（显式指令会自动禁用 auto-caption）
PLANNER=trackvla bash scripts/run_episode.sh --scene corner \
    --trackvla-instruction "Stay beside the person on their left"

# ── 把 captioner 跑在 CPU（显存不够时）
PLANNER=trackvla bash scripts/run_episode.sh --scene corner \
    --trackvla-caption-device cpu

# ── 指定 checkpoint 文件（覆盖默认 epoch04_step024000.pt）
PLANNER=trackvla bash scripts/run_episode.sh --scene corner \
    --trackvla-ckpt /path/to/my_checkpoint.pt

# ── 打开可视化（在 debug 面板上渲染预测轨迹）
PLANNER=trackvla bash scripts/run_episode.sh --scene corner --debug

# ── 短 episode 快速验证
PLANNER=trackvla bash scripts/run_episode.sh --scene corner --duration-sec 30

# ── 指定共享权重目录（多机共享时不需要拷贝 data/）
TRACKVLA_DATA_ROOT=/path/to/trackvla PLANNER=trackvla bash scripts/run_episode.sh --scene corner
```

### 7.5 TrackVLA 专属 CLI 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--trackvla-instruction` | `"Follow the person"` | 每 tick 喂给 VLM 的自然语言指令 |
| `--trackvla-ckpt` | `None`（自动找默认 epoch04） | 手动指定 `.pt` checkpoint 路径 |
| `--trackvla-auto-caption` / `--trackvla-no-auto-caption` | **开启** | 首帧用 Qwen3-VL 描述目标人外观，自动改写 instruction 为 `"Follow the person wearing ..."` |
| `--trackvla-caption-ckpt` | `data/trackvla/LLM_hf/qwen3-vl-4b-instruct` | Qwen3-VL 权重目录 |
| `--trackvla-caption-device` | `cuda`（可用时） | Captioner 推理设备；首帧 caption 后自动释放 VRAM |

### 7.6 自动外观描述（Auto-caption）

**工作原理**：当 `--trackvla-auto-caption` 开启（默认）且 `--trackvla-instruction` 保持默认值 `"Follow the person"` 时：

1. Episode 开始后，manager 每帧把目标的 **GT bbox** 投影到前视相机（通过 `visibility_instance._target_bbox_roi`），塞进 `obs.extras["target_bbox_xyxy"]`
2. Adapter 在 **第一帧**检测到 bbox 可用时，裁剪出目标的 image patch（带 10% padding）
3. 加载 `Qwen3-VL-4B-Instruct`，用 prompt `"Describe the clothing of the person in this image in one short phrase. Start your answer with 'wearing'..."` 生成描述
4. 拼接成 `"Follow the person wearing a red jacket and blue jeans."` 覆盖 `self._instruction`
5. 释放 Qwen3-VL 的 VRAM（首帧后不再需要），让 TrackVLA 自己的 navigation VLM 独占 GPU
6. 整个 episode 复用这条指令；`reset()` 时清空，下个 episode 重新 caption

**用户显式传 `--trackvla-instruction "..."`（非默认值）会跳过 auto-caption**，直接使用用户指令。

### 7.7 Clone 后的准备

TrackVLA 需要的 4 个子模型都存在 `data/trackvla/` 下，均不进 git（文件太大）。Clone 后按 [`docs/01_install.md`](01_install.md) 下载，放到 `data/trackvla/`：

```bash
# 方式 A：解压到仓库内
mkdir -p data/trackvla && tar -xf <downloaded-archive> -C data/trackvla

# 方式 B：放在任意位置，用 symlink 或环境变量指过去
ln -s /path/to/trackvla data/trackvla
# 或直接设置环境变量，不需要移动文件：
export TRACKVLA_DATA_ROOT=/path/to/trackvla
```

Python 依赖（如未安装）：
```bash
pip install transformers peft accelerate
```

---

## 8. OA-VAT（外观感知视觉跟随基线）

OA-VAT（Occlusion-Aware Vision-Appearance Tracker）是基于 DINO 特征的人员辨别 + PID 控制的视觉跟随基线。与 TrackVLA 的端到端 VLM 不同，OA-VAT 是模块化设计：

```
第一帧 RGB → YOLOe（文本提示"person"）→ 候选框 + GT 辅助选目标
                      ↓  初始化目标框
后续每帧 RGB → ORTrack（短时跟踪）→ bbox
                      ↓  bbox crop
               DINOv3-ViT-B/16（外观匹配）→ 与参考特征比对
                      ↓  匹配判定（+KF 遮挡补全）
              P 控制器（bbox center/area → v, w）
```

第一帧用 GT 目标世界坐标做候选框筛选（YOLOe 检测所有人，取距离 GT 最近者）。此后纯视觉闭环。

### 8.1 子模型与权重

| 子模型 | 本地路径 | 大小 |
|---|---|---|
| YOLOe 检测 | `data/oa-vat/yolo/yoloe-11l-seg.pt` | 68 MB，**不在发布包里**（AGPL-3.0，不会自动下载） |
| ORTrack 跟踪 | `data/oa-vat/ortrack/ORTrack_ep0300.pth.tar` | 103 MB |
| DINOv3-ViT-B16 | `data/oa-vat/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | 330 MB，**不在发布包里** |
| DINOv3-ViT-S16 | `data/oa-vat/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | 80 MB，**不在发布包里** |
| MobileCLIP（YOLOe text encoder）| `data/oa-vat/mobileclip_blt.ts` | 572 MB，仅限非商业科学研究 |

YOLOe：

```bash
mkdir -p data/oa-vat/yolo
curl -L -o data/oa-vat/yolo/yoloe-11l-seg.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt
```

DINOv3 的 `.pth` 需要先在 <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/> 申请权限，批准邮件里的链接用 `wget` 下载后，按上表文件名放到 `data/oa-vat/dinov3/`。默认骨干是 ViT-B/16。Hugging Face 上的 `model.safetensors` 不能直接给 OA-VAT 用。

MobileCLIP 随包附带 `data/oa-vat/LICENSE_MODELS`。Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement. 只能用于非商业科学研究。

路径由 `constants.py` 中 `DATA_ROOT` 自动拼接，默认指向 `<repo>/data/oa-vat/`。可用 `OAVAT_DATA_ROOT` 环境变量覆盖。

### 8.2 启动命令

```bash

# ── 基本启动（默认参数）
PLANNER=oa_vat bash scripts/run_episode.sh --scene corner

# ── 使用更轻的 ViT-S 骨干（速度更快）
PLANNER=oa_vat bash scripts/run_episode.sh --scene corner --oa-vat-dinov3 dinov3_vits16

# ── 关闭 Kalman 滤波（遮挡时不做轨迹补全）
PLANNER=oa_vat bash scripts/run_episode.sh --scene corner --oa-vat-kf-mode none

# ── 关闭在线特征更新（固定第一帧参考特征）
PLANNER=oa_vat bash scripts/run_episode.sh --scene corner --oa-vat-no-online-enhance

# ── 使用共享权重目录
OAVAT_DATA_ROOT=/path/to/oa-vat PLANNER=oa_vat bash scripts/run_episode.sh --scene corner

# ── debug 可视化（显示 bbox overlay）
PLANNER=oa_vat bash scripts/run_episode.sh --scene corner --debug
```

### 8.3 OA-VAT 专属 CLI 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--oa-vat-text-query` | `"person"` | YOLOe 的文本提示（第一帧 + 重获取） |
| `--oa-vat-kf-mode` | `confaware` | KF 遮挡补全：`none` / `standard` / `confaware` |
| `--oa-vat-online-enhance` | `True` | 是否每 N 帧 EMA 更新参考 DINOv3 特征 |
| `--oa-vat-dinov3` | `dinov3_vitb16` | DINOv3 骨干：`dinov3_vitb16` / `dinov3_vits16` |
| `--oa-vat-match-thresh` | `0.4` | 第一帧匹配的 cosine 阈值 |
| `--oa-vat-reacq-thresh` | `0.5` | 重获取（目标丢失后）的 cosine 阈值 |
| `--oa-vat-yolo-conf` | `0.1` | YOLOe 检测置信度阈值 |
| `--oa-vat-target-area-ratio` | `0.12` | 期望 bbox 面积 / 帧面积（距离控制目标） |

### 8.4 Clone 后的准备

按 [`docs/01_install.md`](01_install.md) 下载 `data/oa-vat/` 目录后，按目录结构放好即可：

```bash
mkdir -p data/oa-vat && tar -xf <downloaded-archive> -C data/oa-vat
# 或放在任意位置，用环境变量指过去
OAVAT_DATA_ROOT=/path/to/oa-vat PLANNER=oa_vat bash scripts/run_episode.sh --scene <scene>
```

Python 依赖：`timm`、`tensorboardX`、`tensorboard`、`jpeg4py` 已通过 followBench 环境安装，不影响其他 planner。

---

## 9. 完整参数清单在哪里

| 途径 | 命令 |
|---|---|
| **CLI `--help`**（最权威） | `python -m followbench.scenes.corner.run_episode --help`（需要 CARLA 能 `import carla`） |
| **源码 `build_parser()`** | `followbench/scenes/corner/run_episode.py` 中的 `build_parser()` 函数 |
| 同上 | `followbench/scenes/park/run_episode.py` 的 `build_parser()`（约 line 70–240） |
| **场景预设固定的参数** | 查看 `configs/scenes/<scene>/episode.sh`（或 `FB_DRY_RUN=1 bash scripts/run_episode.sh --scene <scene>`） |

想看 planner 自身的参数（如 `receding / iter_num`），直接看 `baselines/adapters/*_adapter.py` 的 `__init__`。

---

## 10. 文件布局速查

```text
configs/scenes/<scene>/episode.sh        # 场景启动预设（corner / park / passage / plaza）
followbench/scenes/<scene>/run_episode.py          # Python 入口
followbench/scenes/{corner,park}/run_episode_humanoid.py
followbench/sim/                          # 机器人 / 行人 / 可见性 / 天气 等共享运行时
followbench/perception/                   # 检测 + 跟踪 + ReID
followbench/evaluation/                   # 指标 + 日志
baselines/adapters/                       # planner 适配层（*_adapter.py / *_perception_adapter.py）
baselines/{planning,maps,prediction,behavior,perception,common}/
scripts/run_episode.sh                    # 单局入口（--scene / --humanoid / --teleop）
```

每个 `*_adapter.py` 是 GT-reading baseline；对应的 `*_perception_adapter.py` 是"GT → 感知"的 wrapper。运行时由 `run_episode.py` 根据 `--use-perception` 决定加载哪一个。

---

## 11. 已知限制

1. **不支持 `--use-perception` 的 planner**：`rda_depth_tpt`（它自己就是感知版本）、`trackvla`（端到端 VLA）和 `oa_vat`（自带视觉跟踪）—— 这三者都有内置感知，传入感知前端无意义，互斥。
2. **`rda_search` 的 `--lost-policy`**：只接受 `last_known`，其他会在构造时 `ValueError`。
3. **多视角相机默认启用**：`ScoutRobotRuntime.spawn_sensors()` 会强制为机器人装三个相机（front / left / right，各自带 RGB + depth + instance）。目前没有 CLI 开关跳过侧视相机。
4. **CARLA 启动顺序**：先开 CARLA Server → 再开脚本。否则 `import carla` 会报 `ModuleNotFoundError`。
5. **GPU 兼容性**：RTX 5090 (sm_120) 目前不被系统 PyTorch 支持；感知前端可用 `--reid-device cpu` 跑，YOLO 用 `ultralytics` 自己的 CPU 推理。
