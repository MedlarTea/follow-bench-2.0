# Episode 终止逻辑说明

本文档汇总 followbench2.0 三个场景（Doorway / Random-Corridor / Clutter）下 episode 的终止判定条件，以及相关可配置超参数的默认值。

> 仿真步长 `dt` 全局默认 **0.1s**（10Hz）。下文所有“秒”与“步”的换算均基于此值。

## 0. 成功判定（`eval_result.json` 中的 `task_success`）

对齐 Follow-Bench 1.0 / EVT-Bench：

- **目标丢失**：robot–target 中心距离 > 3m（`--robot-target-far-fail-dist-m`）。
- **长时目标丢失**：目标丢失连续超过 N_lost = 100 步（`--robot-target-far-fail-steps`，10s）。
- **成功** = 无任何碰撞（行人 / 目标 / 静态障碍物）且无长时目标丢失，**且终止时刻** robot–target 距离 ≤ 3m。短时丢失和暂时的队形变化不影响成功。
- 按终止时刻的状态判定，与 `termination_reason` 无关：`max_active_time` / `duration_timeout` / `manual_reset` 等提前结束的 episode 也按上述三条判定。
- 判定由 `followbench/evaluation/metrics.py` 根据 `episode_step.jsonl` 与 `collision_events.jsonl` 离线计算，`episode_meta.json` 中的 `episode_success` 仅表示运行时的结束状态，不参与 `task_success` 判定。
- 相关输出字段：`collision_free`、`long_term_target_loss`、`max_consecutive_target_loss_steps`、`final_robot_target_dist`、`final_within_loss_dist`、`target_route_completed`。

---

## 1. Doorway 场景

### 1.1 成功终止

| termination_reason | 触发条件 | 代码位置 |
|---|---|---|
| `target_completed_a_to_b` | 目标 NPC 完成从 A 端点经门槛通过到 B 端点 | `followbench/scenes/passage/npc_runtime.py:624` |

### 1.2 失败终止（当前启用）

判定顺序（一旦命中即终止，更高优先级先判）：

| 优先级 | termination_reason | 触发条件 | 代码位置 |
|---|---|---|---|
| 1 | `target_collision` | 机器人与目标发生碰撞（`is_target=True`） | `followbench/scenes/passage/run_episode.py:423` |
| 2 | `pedestrian_collision` | 与行人碰撞（需 `--terminate-on-human-collision`） | `followbench/scenes/passage/run_episode.py:426` |
| 3 | `obstacle_collision` | 与静态障碍物碰撞（**总是终止**，无开关） | `followbench/scenes/passage/run_episode.py:429` |
| 4 | `target_too_far` | 机器人到目标中心距离 > `fail_dist_m`，且**连续步数 > `fail_steps`**（长时目标丢失） | `followbench/scenes/passage/run_episode.py:294` |

### 1.3 失败终止（当前禁用，监控仍写入 telemetry）

| termination_reason | 现状 | 重新开启方式 |
|---|---|---|
| `target_identity_lost` | 默认关闭 | `--terminate-on-target-identity-loss` |
| `robot_stuck_static_obstacle` / `robot_stuck_far_from_target` | 默认关闭 | `--terminate-on-robot-stuck` |

### 1.4 超参数（Doorway）

| CLI 参数 | 默认值 | 作用 |
|---|---|---|
| `--terminate-on-human-collision` / `--no-terminate-on-human-collision` | **True**（启用） | 行人碰撞是否终止 |
| `--terminate-on-target-identity-loss` / `--no-terminate-on-target-identity-loss` | **False**（关闭） | 身份识别丢失是否终止（Doorway 场景通过 `p.set_defaults` 覆盖 schemas 默认值） |
| `--terminate-on-robot-stuck` / `--no-terminate-on-robot-stuck` | **False**（关闭） | 机器人卡住是否终止 |
| `--robot-target-far-fail-dist-m` | **3.0**（米） | 触发"目标过远"计数的距离阈值 |
| `--robot-target-far-fail-steps` | **100**（步，= 10s @10Hz，即 N_lost） | 连续步数阈值；>0 时优先于秒数判定 |
| `--robot-target-far-fail-sec` | 2.0（秒） | 秒数阈值（仅当 `fail_steps<=0` 时生效，向后兼容） |
| `--robot-target-far-warmup-sec` | 0.0（秒） | 启动热身期（默认关闭，从第一步开始计数） |
| `--human-collision-margin` | 见 `schemas.py` | 几何碰撞判定边距 |

> **`target_too_far` 当前逻辑**：robot-target 中心距离 > 3m 连续超过 100 步（N_lost，10s）→ 长时目标丢失，触发失败。仅在 `following=True` 时计数（warmup 默认 0）；距离回落到 ≤3m 则立即清零。

机器人卡住监控（默认已关闭终止，但监控仍运行，值写入 `debug_info`）：

| CLI 参数 | 默认值 |
|---|---|
| `--robot-stuck-window-sec` | 3.0 |
| `--robot-stuck-grace-sec` | 3.0 |
| `--robot-stuck-min-progress-m` | 0.25 |
| `--robot-stuck-target-wait-speed-mps` | 0.15 |
| `--robot-stuck-target-escape-distance-m` | 6.0 |
| `--robot-stuck-static-obstacle-clearance-m` | 0.35 |
| `--map-obstacle-overlap-confirm-steps` | 2 |
| `--map-obstacle-clearance-probe-m` | 2.0 |

身份识别监控（默认已关闭终止）：

| CLI 参数 | 默认值（Doorway 覆盖） |
|---|---|
| `--target-identity-startup-grace-sec` | 1.5 |
| `--target-identity-confirm-sec` | 1.5 |
| `--target-identity-fail-sec` | 2.5 |
| `--target-identity-wrong-dist-m` | 4.0 |
| `--target-identity-associate-gate-m` | 1.5（来自 `schemas.py`） |

---

## 2. Random / Corridor 场景

### 2.1 成功终止

| termination_reason | 触发条件 | 代码位置 |
|---|---|---|
| `target_round_trip_done` | 目标 NPC 完成 `target_required_legs` 条路段 | `followbench/scenes/corner/npc_runtime.py:2242` |

### 2.2 失败终止（当前启用）

| termination_reason | 触发条件 | 开关 |
|---|---|---|
| `human_collision` | 与行人碰撞 | `--terminate-on-human-collision`（默认 True） |
| `target_too_far` | 机器人到目标中心距离 > `fail_dist_m`，且**连续步数 > `fail_steps`**（长时目标丢失） | 无独立开关（长按 `fail_steps<=0` 回退到秒数） |
| `max_active_time` | `active_steps · dt ≥ max_active_sec` | `max_active_sec > 0` 时生效 |
| `manual_reset` | 用户 UI 手动触发 | — |

### 2.3 失败终止（当前禁用）

| termination_reason | 现状 | 重新开启方式 |
|---|---|---|
| `target_identity_lost` | 默认关闭（通过 `p.set_defaults` 覆盖 schemas 默认 True） | `--terminate-on-target-identity-loss` |

### 2.4 超参数（Corridor）

| CLI 参数 | 默认值 | 作用 |
|---|---|---|
| `--robot-target-far-fail-dist-m` | **3.0**（米） | 触发"目标过远"计数的距离阈值 |
| `--robot-target-far-fail-steps` | **100**（步，= 10s @10Hz，即 N_lost） | 连续步数阈值；>0 时优先于秒数 |
| `--robot-target-far-fail-sec` | 2.0（秒） | 秒数阈值（仅当 `fail_steps<=0` 时生效） |
| `--robot-target-far-warmup-sec` | 0.0（秒） | 启动热身期（默认关闭） |
| `--terminate-on-human-collision` / `--no-terminate-on-human-collision` | **True**（启用） | 行人碰撞是否终止 |
| `--terminate-on-target-identity-loss` / `--no-terminate-on-target-identity-loss` | **False**（关闭，Corridor 覆盖） | 身份识别丢失是否终止 |
| `--max-active-sec` | 见 schemas.py | 最大活跃时长（>0 启用超时判定） |

> Corridor 场景**不启用** `robot_stuck` 监控。

---

## 3. Clutter 场景

### 3.1 成功终止

| termination_reason | 触发条件 | 代码位置 |
|---|---|---|
| `target_completed_route` | 目标 NPC 完成环形路线 (P0→P1→P2→P0) | `followbench/scenes/park/npc_runtime.py:1417` |

### 3.2 失败终止（当前启用）

| termination_reason | 触发条件 | 开关 |
|---|---|---|
| `human_collision` | 与行人碰撞 | `--terminate-on-human-collision`（默认 True） |
| `target_too_far` | 机器人到目标中心距离 > `fail_dist_m`，且**连续步数 > `fail_steps`**（长时目标丢失） | 无独立开关（`fail_steps<=0` 回退到秒数） |

### 3.3 超参数（Clutter）

| CLI 参数 | 默认值 | 作用 |
|---|---|---|
| `--robot-target-far-fail-dist-m` | **3.0**（米） | 触发"目标过远"计数的距离阈值 |
| `--robot-target-far-fail-steps` | **100**（步，= 10s @10Hz，即 N_lost） | 连续步数阈值；>0 时优先于秒数 |
| `--robot-target-far-fail-sec` | 2.0（秒） | 秒数阈值（仅当 `fail_steps<=0` 时生效） |
| `--robot-target-far-warmup-sec` | 0.0（秒） | 启动热身期（默认关闭） |
| `--terminate-on-human-collision` / `--no-terminate-on-human-collision` | **True**（启用） | 行人碰撞是否终止 |

> Clutter 场景**不启用** `robot_stuck` / `target_identity_lost` 监控（本场景代码中也无 `TargetIdentityMonitor` 定义）。

---

## 4. 跨场景共享参数（`followbench/evaluation/schemas.py`）

| CLI 参数 | 默认值 |
|---|---|
| `--terminate-on-human-collision` / `--no-terminate-on-human-collision` | True |
| `--terminate-on-target-identity-loss` / `--no-terminate-on-target-identity-loss` | True（Doorway 场景被覆盖为 False） |
| `--target-identity-associate-gate-m` | 1.5 |
| `--target-identity-wrong-dist-m` | 7.0 |
| `--target-identity-startup-grace-sec` | 3.0 |
| `--target-identity-confirm-sec` | 3.0 |
| `--target-identity-fail-sec` | 5.0 |

---

## 5. 变更记录

- **2026-05-04** — Doorway 场景失败终止简化：
  - 关闭 `terminate_on_target_identity_loss`（默认 True → False）。
  - 关闭 `terminate_on_robot_stuck`（默认 True → False）。
  - `TargetDistanceMonitor` 新增步数阈值，`--robot-target-far-fail-steps` 默认 150。
  - `--robot-target-far-fail-dist-m` 默认 8.0 → 3.0。
  - 当前 Doorway 失败条件：**碰撞** 或 **目标过远（≥3m 持续 150 步）**。

- **2026-05-04** — Corridor 场景对齐 Doorway 终止策略：
  - 新增 `TargetDistanceMonitor`（本地类定义，与 Doorway 逻辑一致）。
  - 新增 CLI 参数 `--robot-target-far-fail-dist-m`（默认 3.0）、`--robot-target-far-fail-steps`（默认 150）、`--robot-target-far-fail-sec`（默认 2.0）、`--robot-target-far-warmup-sec`（默认 2.0）。
  - 关闭 `terminate_on_target_identity_loss`（通过 `p.set_defaults(...)` 覆盖 schemas 默认 True → False）。
  - 监控值写入 `policy_debug_info["target_far"]`，便于调试与日志回放。

- **2026-05-04** — Clutter 场景对齐 Doorway 终止策略：
  - 新增 `TargetDistanceMonitor`（本地类定义，与 Doorway/Corridor 一致）。
  - 新增 CLI 参数 `--robot-target-far-fail-dist-m`（默认 3.0）、`--robot-target-far-fail-steps`（默认 150）、`--robot-target-far-fail-sec`（默认 2.0）、`--robot-target-far-warmup-sec`（默认 2.0）。
  - 失败终止打印由固定字符串改为动态 `termination_reason`，支持 `target_too_far` / `human_collision`。
  - 监控值写入 `policy_debug_info["target_far"]`。

- **2026-10-26** — 成功判定对齐 Follow-Bench 1.0 / EVT-Bench，仿真统一为 10Hz：
  - `--dt` 默认 0.05 → 0.1（所有场景）。
  - `--robot-target-far-fail-steps`（N_lost）默认 150 → 100（10s）；`--robot-target-far-warmup-sec` 默认 2.0 → 0.0。
  - 目标丢失判定改为距离 **> 3m**，长时丢失改为连续步数 **> N_lost**。
  - `task_success` = 无碰撞 && 无长时丢失 && 终止时距离 ≤ 3m，不再依赖 `termination_reason`；可视化 `SR` 改为直接使用 `task_success`。
