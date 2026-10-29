# Corner 人流仿真策略说明

## 总体概述

corner 场景的行人仿真目标是构造一个更接近真实走廊人流的动态环境：目标人 `N01` 从走廊一端出发，穿过 L 形 corner 到另一端，再完成一次返回；普通行人不再集中生成在两端，而是随机分布在整个 ROI 区域内，并根据所在位置选择去往另一端的方向。行人的全局路径由方向 lane 和 L 形 corner 几何共同约束，局部运动由 SFM 处理行人、机器人和墙体交互，从而减少相向对冲、贴脸卡住和列队行走。

当前设计把问题拆成三层：

1. 初始生成层：决定每个人从哪里出现、属于哪条 flow lane、目标端点是哪一端、初始朝向是什么。
2. 路径生成层：为每个人生成带右侧通行偏置、横向分散和拐角平滑的路线。
3. 局部交互层：用 SFM 在每个 tick 根据路径局部目标、邻居、机器人和墙体计算实际速度与避让方向。

目标人和普通行人采用分离策略。目标人是评估主体，固定使用 `N01`，生成位置靠近某个端点的中线区域，不能太贴墙，并要求完成一次来回；普通行人则随机生成在整个 corner ROI 内，围绕方向 lane 做横向和纵向扰动，形成更自然的人流背景。

## 代码链路

主要文件：

- `scripts/run_episode.sh --scene corner`
  - corner 场景默认启动脚本。
  - 设置 flow 模式、spawn 策略、lane 参数、速度范围、目标人任务次数等。

- `followbench/scenes/corner/run_episode.py`
  - 负责加载 CARLA world、创建 `NpcRuntime`、创建机器人、启动传感器和跟随策略。
  - 将命令行参数写入 `npc.params`。
  - 按 `G` 生成行人和机器人，按 `F` 启动跟随与行人运动。
  - 评估日志只在开始跟随后才启动。

- `followbench/scenes/corner/npc_runtime.py`
  - corner 人流仿真的核心实现。
  - 负责 ROI 地图、行人 profile 生成、路线生成、目标切换、SFM 输入构造和 WalkerControl 输出。

- `followbench/sim/pedestrian_sfm.py`
  - 通用 SFM planner。
  - 根据期望速度、邻居交互、预测冲突风险、passing-side bias 和墙体斥力输出局部速度。

默认 corner 启动脚本使用的关键配置：

```bash
--npc-flow-mode pair_12_34
--npc-pair-spawn-mode paired_endpoint
--target-spawn-policy endpoint_center
--crowd-spawn-policy roi_random
--npc-pair-goal-policy fixed_until_arrival
--npc-pair-target-band-half-width-m 3.0
--npc-spawn-min-sep-m 1.4
--target-centerline-half-width-m 0.8
--target-wall-clearance-min-m 1.2
--target-endpoint-sample-radius-m 5.0
--right-hand-lane-sep-m 0.72
--npc-navigation-mode astar
--npc-replan-on-block
--npc-min-speed 0.55
--npc-max-speed 1.15
--target-track-id N01
--target-required-legs 2
```

## ROI 与 flow points

`ROIMap` 从 `gridmap_roi.npz` 加载：

- `free_grid`：可行走栅格。
- `roi_mask`：corner 有效区域。
- `world_min/world_max`：栅格到世界坐标转换。
- `clearance_m`：每个可行走 cell 到障碍/墙体的距离。
- `clearance_grad_x/y`：平滑后的 clearance 梯度，用于估计墙体法向。

`flow_points_json` 提供 corner 端点。当前 `pair_12_34` 下使用两组 pair：

- lane 0：point 1 <-> point 2
- lane 1：point 3 <-> point 4

每个 pair 表示一条 L 形 corner 流线。`NpcRuntime` 会根据端点构造 L 形中心线，并在转弯处自动选择 elbow 点。

## 初始生成策略

### Profile-based spawn

当前 corner 默认满足下面条件时使用 profile 生成：

```python
flow_mode 是 pair_*
target_spawn_policy != "legacy"
crowd_spawn_policy != "legacy"
```

对应入口是：

- `_build_spawn_profiles()`
- `_build_target_spawn_profile()`
- `_build_crowd_spawn_profile()`
- `_spawn_one_from_profile()`

每个 `SpawnProfile` 会提前确定：

- `track_id`
- 是否目标人
- 初始位置
- 目标端点
- 初始朝向
- 基础速度
- lane 编号
- 当前 pair endpoint
- 横向个人偏移
- route waypoint spacing
- route lookahead
- route longitudinal phase
- lane-aware route

这样生成 actor 前，位置、路线、速度和朝向已经是一致的。

### 目标人 N01 生成

目标人由 `target_track_id` 指定，默认是 `N01`。

目标人生成逻辑：

1. 随机选择一条 pair lane。
2. 随机选择起始端点 `start_ep`。
3. 目标端点为另一端 `target_ep = 1 - start_ep`。
4. 在起始端点附近采样，但约束更严格：
   - 横向位置靠近端点中线：`target_centerline_half_width_m`。
   - 离墙距离不低于：`target_wall_clearance_min_m`。
   - 采样半径：`target_endpoint_sample_radius_m`。
   - 与已生成行人保持：`npc_spawn_min_sep_m`。
5. 用路线的第一个有效 waypoint 计算初始 yaw，使目标人面向将要移动的方向。

目标人这样设计的原因：

- 目标人是机器人跟随对象，起点太靠墙会影响机器人初始跟随几何。
- 目标人不应和普通行人的随机分布策略混在一起，否则容易一开始就出现贴脸、遮挡或贴墙。
- 初始 yaw 使用规划路线方向，而不是固定 yaw，避免启动时先原地大角度转身。

### 普通行人生成

普通行人使用 `roi_random`，即在整个 corner ROI 内随机采样，而不是集中在两端。

普通行人生成逻辑：

1. 在 navmesh/ROI 中随机采样位置。
2. 用 `npc_spawn_min_sep_m` 保证和已经生成的人有初始间距。
3. 根据该位置投影到哪条 pair polyline 更近，选择 `pair_lane`。
4. 根据当前位置在该 lane 上的进度选择目标端点：
   - 靠近 endpoint 0 的人倾向去 endpoint 1。
   - 靠近 endpoint 1 的人倾向去 endpoint 0。
   - 中间区域根据 flow counts 做平衡，避免单方向过多。
5. 分配个人横向偏移、spacing、lookahead 和纵向 phase。
6. 生成 lane-aware route，并用路线方向设置初始 yaw。

这种策略相比端点集中生成的好处：

- 背景行人覆盖整个 corner，而不是全部从两端涌入。
- 中间和拐角区域天然存在行人，仿真更接近真实人流。
- 通过初始间距和 route personality 减少开局重叠、排队和同步行走。

## 速度策略

速度不再所有人相同，而是每个人生成时采样一个 `base_speed`。

普通行人：

- 均值：`npc_speed_mean = 0.85`
- 标准差：`npc_speed_std = 0.18`
- 下限：`npc_speed_min = 0.55`
- 上限：`npc_speed_max = 1.15`
- 同时受命令行 `--npc-min-speed` / `--npc-max-speed` 限制。

目标人：

- 均值：`target_speed_mean = 0.82`
- 标准差：`target_speed_std = 0.10`
- 下限：`target_speed_min = 0.70`
- 上限：`target_speed_max = 0.95`

每次到达端点重新设定目标时，会调用 `_retarget_speed()`，在 `base_speed` 附近加入 `speed_retarget_jitter_mps` 小扰动。这样可以避免所有行人长期同速移动，也避免速度过慢像静止，或过快破坏跟随评估。

## 路径生成策略

### 为什么不用纯 A*

早期纯 A* 会倾向走全局最短路。在 L 形 corner 中，这会导致：

- 起点分散但路线重新收敛到中心线。
- 多个行人在拐角附近集中。
- 同方向行人容易前后排队。
- 相向行人会沿近似同一条路径对冲。

因此当前 pair flow 默认使用 lane-aware L-shape route。A* 仍保留作为 fallback 和非 lane-aware 模式的规划器。

### L 形中心线

`_directed_l_shape_polyline()` 根据起点端和目标端构造：

```text
start endpoint -> elbow -> target endpoint
```

`_pair_elbow_anchor()` 会在两个可能 elbow 中选择 clearance 更好、偏移更小的点。

### 拐角平滑

`_build_rounded_l_shape_centerline()` 不直接使用硬折线，而是在 elbow 附近做二次 Bezier 采样：

1. 根据 `corner_turn_radius_m` 从入弯段和出弯段截取 entry/exit。
2. 使用 `entry -> elbow -> exit` 的二次曲线生成转弯点。
3. 直线段按 `corner_route_waypoint_spacing_m` 采样。

关键参数：

- `corner_turn_radius_m = 4.0`
- `corner_turn_samples = 8`
- `corner_route_waypoint_spacing_m = 1.8`

这样可以避免行人在拐角必须走到一个硬 waypoint 后突然转向。

### 靠右通行 lane

`_right_hand_route_offset()` 负责生成 lane offset。

普通行人的 offset：

```text
offset = -right_hand_lane_sep_m + preferred_lateral_offset_m
```

并限制在：

- `min_lane_offset_from_center_m = 0.45`
- `max_lane_offset_from_center_m = 1.35`
- `pair_target_band_half_width_m` 范围内

目标人的 offset 更保守：

```text
offset = -right_hand_lane_sep_m * target_lane_bias_scale + target_lateral_jitter
```

并限制在：

- `target_lane_min_offset_m = 0.25`
- `target_lane_max_offset_m = 0.75`

这样做的意图：

- 普通行人按方向靠右通行，降低相向对冲概率。
- 目标人保持偏中间一些，避免太靠墙影响机器人跟随。
- 不让 lane 离墙太近，也不让所有人挤在中心线。

### 横向和纵向分散

普通行人的 route personality 由 `_sample_route_personality()` 生成。

横向分散：

- `_assign_pair_lateral_offset()` 使用 golden-ratio slot 分配横向位置。
- 再叠加 `npc_lateral_jitter_m` 随机扰动。
- 个人偏移范围由 `npc_lateral_personal_span_m` 控制。

纵向分散：

- `corner_route_spacing_jitter_m` 随机扰动 waypoint 间距。
- `corner_route_lookahead_jitter_m` 随机扰动起步 lookahead。
- `corner_route_longitudinal_phase_min_m/max_m` 为每个人加入不同的路线相位。

这些设计共同减少“同方向排队走”的视觉效果。即使多人方向相同，也会因为横向 offset、waypoint spacing、lookahead 和速度不同而形成轻微错位。

### 路线安全修正

`_offset_centerline_samples()` 对中心线做横向 offset 时，不是盲目平移：

1. 根据局部切线计算法向。
2. 从完整 offset 开始尝试。
3. 如果点不可走或离墙太近，就逐步缩小 offset。
4. 要求 clearance 尽量满足 `lane_wall_clearance_m`。
5. 失败时退回到最近可行走点。

这样保证 lane 分散不会把人推到墙外或贴墙。

## 局部目标选择

每个 tick 中，`_get_local_goal_xy()` 从 route 中选择当前 waypoint：

- 到达当前 waypoint 小于 `astar_waypoint_arrive_dist` 时，推进到下一个 waypoint。
- 如果 route 用完但还没到最终目标，会触发 replan。
- 对 lane-aware route，局部目标来自预生成 L-shape route。

初始朝向和 SFM reference forward 都尽量从 route 方向获得：

- spawn yaw 使用 `_yaw_from_route_or_target()`。
- SFM reference forward 使用 `_get_reference_forward_xy()`。
- lane-aware 情况优先使用 route segment 方向。

因此行人启动后会朝路线前方走，而不是先向 corner 中心线收敛。

## SFM 交互策略

### SFM 输入

每个行人的局部运动由 `_compute_local_motion()` 计算。

传入 SFM 的 `AgentKinematics` 包括：

- 当前 xy 位置。
- 当前 xy 速度。
- 行人半径 `sfm_ped_radius`。
- 当前期望速度 `ag.speed`。
- 当前局部 goal。
- reference forward。
- passing-side preference。

动态障碍来自 `_collect_dynamic_obstacles()`：

- 其他行人都会作为 `kind="pedestrian"`。
- 机器人会作为 `kind="robot"` 加入普通行人的 obstacle list。
- 目标人 `N01` 不把 robot 当 SFM obstacle。

目标人不避让机器人是有意设计：目标人是被跟随对象，如果目标人主动躲机器人，会改变评估任务本身，使机器人跟随策略和目标运动相互耦合过强。

### 行人-行人避让

`pedestrian_sfm.py` 中的 SFM 包含：

- driving force：让行人朝局部目标速度收敛。
- pedestrian interaction force：行人之间的法向斥力。
- tangential force：侧向绕行力。
- prediction risk：基于相对速度预测未来最小距离。
- close risk：近距离时即使预测速度不明显也会增加风险。
- anisotropy：前方障碍影响更强，后方障碍影响更弱。

预测冲突使用：

```text
relative position + relative velocity
预测 horizon 内最近距离 d_min
根据 d_min、当前距离、closing speed 和 time-to-conflict 计算 risk
```

关键参数：

- `sfm_prediction_enabled = True`
- `sfm_prediction_horizon = 1.8`
- `sfm_close_risk_margin = 0.65`
- `sfm_neighbor_radius = 4.5`
- `sfm_max_neighbors = 12`

### passing-side bias

相向行人有时会同时朝同一侧避让，导致贴脸卡住。为降低这种情况，SFM 加了 passing-side bias。

head-on 判断条件：

- obstacle 是 pedestrian。
- 两人运动方向接近相反：`sfm_headon_cos_threshold`。
- 对方在自己前方：`sfm_headon_front_dot_threshold`。
- 横向距离小于：`sfm_headon_lateral_trigger_m`。

一旦判断为 head-on：

- risk 至少提升到 `sfm_headon_min_risk`。
- tangential force 使用更强的 `sfm_headon_bias_weight`。
- 默认 passing side 为右侧。

`_update_pass_side_commitment()` 会把 SFM 建议的 passing side 保持一段时间：

- 触发阈值：`sfm_pass_side_trigger_risk = 0.22`
- 保持时间：`sfm_pass_side_hold_sec = 3.2`

这样避免每个 tick 左右摇摆，也降低双方同时换边的概率。

### 对机器人避让

普通行人会把机器人作为 SFM obstacle：

- `sfm_robot_A = 4.2`
- `sfm_robot_B = 0.75`
- `sfm_robot_radius = 1.05`

机器人 obstacle 排序优先级高于普通行人，因此普通行人会更优先考虑机器人附近的安全距离。

目标人不加入 robot obstacle，原因见前文。

### 墙体避让

墙体不是单独 actor，而是由 ROI clearance field 推出：

1. `_query_wall_state()` 在当前位置插值读取 `clearance_m`。
2. 读取 clearance 梯度作为远离墙体的 normal。
3. 对 normal 做时间平滑。
4. SFM 根据 wall distance 和 normal 生成 wall force。

关键参数：

- `sfm_wall_A = 4.6`
- `sfm_wall_B = 0.28`
- `sfm_wall_influence_dist = 1.25`
- `sfm_wall_normal_smooth_alpha = 0.35`

这用于减少行人贴墙、穿墙或在边界处抖动。

### 速度后处理

SFM 输出速度后，`_compute_local_motion()` 还有一层速度修正：

- 如果冲突风险低但 SFM 输出太慢，会给一个向前速度下限，避免无故停住。
- 如果表面距离太近，会降低速度下限，让行人更容易停下/让行。
- 如果风险超过 `sfm_yield_trigger_risk`，按风险降低速度，上限收缩到 `sfm_yield_min_speed_scale`。
- 如果卡住且风险高，会触发 route replan。

这样 SFM 不只是“推开”，还会表现出减速让行。

## 目标人往返任务

目标人 `N01` 不再到达一端就直接评估成功，而是要求完成 `target_required_legs` 条端到端 leg。

当前默认：

```text
target_required_legs = 2
```

含义：

1. 从起点端到另一端，完成第一段。
2. 从另一端返回起点端，完成第二段。
3. 第二段到达后标记 `route_task_done=True`。
4. `followbench/scenes/corner/run_episode.py` 检测 `npc.is_target_route_task_done(N01)` 后，以 `target_round_trip_done` 结束评估。

### 目标人平滑掉头

目标人到达端点后，不直接原地切换目标，而是先进入 `route_phase="turnaround"`。

`_build_target_turnaround_route()` 使用三次 Bezier 生成短掉头路线：

- 起点：目标人当前位置。
- 控制点 1：沿当前前进方向继续一点。
- 控制点 2：靠近返回路线 merge 点。
- 终点：返回 lane 上的 merge 点。

关键参数：

- `target_turnaround_forward_m = 1.0`
- `target_turnaround_merge_m = 2.2`
- `target_turnaround_side_m = 0.75`
- `target_turnaround_samples = 8`
- `target_turnaround_speed_mps = 0.55`

掉头阶段会降低速度，并禁用 block replan，避免目标人在端点附近因为局部冲突不断重新规划。

普通行人到达端点后的转向仍使用普通 pair retarget 逻辑，因为视觉效果上已经足够自然。

## 到达、重规划与卡住处理

行人到达目标条件：

```text
distance(loc, target) < arrive_threshold
```

默认 `arrive_threshold = 1.6`。

普通 pair 行人到达后：

1. 翻转 pair endpoint。
2. 重新采样目标。
3. 重新采样 retarget speed。
4. 重新生成 route。

如果 `pair_goal_policy=fixed_until_arrival`：

- 行人不会因为短时间堵塞就改变目标端点。
- 但仍允许在 stuck/timed retarget 时重新规划当前目标路线。

如果 `pair_goal_policy=retarget_on_block`：

- 卡住或超时会直接翻转目标端点。

当前 corner 默认使用 `fixed_until_arrival`，因为它更稳定，能保持明确的人流方向。

block replan 条件：

- 开启 `replan_on_block`。
- 当前处于 astar/lane-aware route。
- 距离目标还较远。
- SFM conflict risk 高于阈值。
- 在窗口时间内前向进展不足。

关键参数：

- `block_replan_window_sec = 2.0`
- `block_replan_progress_thresh = 0.25`
- `block_replan_risk_thresh = 0.5`
- `astar_replan_cooldown_sec = 0.6`

## 每 tick 运动更新顺序

`NpcRuntime.step()` 每个 tick 对每个 agent 执行：

1. 如果未启动 motion，给 walker 速度 0。
2. 如果 route task done，给 walker 速度 0。
3. 更新 stuck 状态。
4. 调用 `_update_agent_target()` 判断是否到达、掉头、换端点或重规划。
5. 调用 `_compute_local_motion()`：
   - 选择局部目标。
   - 计算 reference forward。
   - 收集行人/机器人 obstacles。
   - 查询墙体状态。
   - 调用 SFM。
   - 更新 passing-side commitment。
   - 做速度后处理。
   - 必要时 replan。
6. 将 SFM 输出转换为 `carla.WalkerControl`：
   - `direction = move`
   - `speed = move_speed`
   - `z_dir` 根据目标 z 做小范围调整
7. 如果检测到 curb/stuck，做轻微 step-up 和 forward nudge。

## 评估启动策略

当前 corner manager 的评估日志是延迟启动的：

- `--enable-evaluation` 只表示 armed。
- 按 `F` 开始 following 后才创建 evaluation trial。
- 如果只是运行可视化、不按 `F`，不会生成新的评估 trial。
- `scripts/run_episode_eval.sh` 会检测是否真的有新 trial，没有则跳过 evaluator 和画图。

这保证“看可视化效果”和“正式评估”不会混在一起。

## 当前策略的设计取舍

### 已解决的主要问题

- 目标人和普通行人初始生成策略分离。
- 目标人起点更居中，不贴墙。
- 普通行人覆盖整个 corner ROI，而不是集中在两端。
- 初始 yaw 对齐 route 方向。
- 方向 lane 采用靠右通行，降低相向对冲。
- 拐角 route 使用平滑 L 形曲线。
- 普通行人有横向、纵向、速度三种随机差异，减少列队。
- SFM 加入预测风险、近距离风险、passing-side bias 和墙体力。
- 普通行人避让机器人，目标人不避让机器人。
- 目标人完成一次往返后再成功结束评估。

### 仍然需要关注的点

- corner 很窄或行人数过多时，SFM 仍可能出现短时拥堵。
- `right_hand_lane_sep_m` 过大会贴墙，过小会重新挤到中间。
- `npc_lateral_personal_span_m` 过大会导致路线过散，过小会重新排队。
- `sfm_headon_bias_weight` 过大可能导致避让动作突兀，过小则相向对冲改善不明显。
- `target_turnaround_*` 参数需要和具体端点空间匹配，端点太窄时掉头曲线会被投影回可行走区域。

## 调参建议

如果同方向行人仍然像列队：

- 增大 `npc_lateral_personal_span_m`。
- 适当增大 `npc_lateral_jitter_m`。
- 增大 `corner_route_longitudinal_phase` 范围。
- 增大 `corner_route_spacing_jitter_m`。
- 适当扩大普通行人速度分布。

如果相向对冲仍明显：

- 略增大 `right_hand_lane_sep_m`，但需要检查离墙距离。
- 增大 `sfm_headon_lateral_trigger_m`。
- 增大 `sfm_headon_bias_weight`。
- 增大 `sfm_pass_side_hold_sec`，减少左右摇摆。

如果行人离墙太近：

- 增大 `lane_wall_clearance_m`。
- 减小 `max_lane_offset_from_center_m`。
- 减小 `right_hand_lane_sep_m`。
- 增大 `sfm_wall_A` 或 `sfm_wall_influence_dist`。

如果行人过慢或像卡住：

- 提高 `npc_speed_min` 或命令行 `--npc-min-speed`。
- 降低 `sfm_yield_trigger_risk` 会更容易减速，不适合这种情况。
- 适当提高 `sfm_yield_min_speed_scale`。
- 检查 `sfm_close_surface_slowdown_m` 是否过大。

如果目标人掉头仍不自然：

- 增大 `target_turnaround_merge_m`，让掉头更早并入返回路线。
- 增大 `target_turnaround_forward_m`，减少原地掉头感。
- 调整 `target_turnaround_side_m`，改变曲线侧向控制点。
- 增加 `target_turnaround_samples`，让局部 waypoint 更密。

## 推荐默认参数记录

当前较稳定的一组 corner 参数：

```text
num_walkers_total = 10
npc_spawn_min_sep_m = 1.4
pair_target_band_half_width_m = 3.0
right_hand_lane_sep_m = 0.72
lane_wall_clearance_m = 0.80
npc_lateral_personal_span_m = 0.65
npc_lateral_jitter_m = 0.20
target_lateral_jitter_m = 0.15
npc_speed_range = [0.55, 1.15]
target_speed_range = [0.70, 0.95]
corner_turn_radius_m = 4.0
corner_turn_samples = 8
target_required_legs = 2
```

这组参数偏向“有一定随机性但不混乱”的人流效果。后续如果增加行人数，优先调小速度方差或加大 route 横向分散，而不是简单增大 SFM 斥力。
