# Map Annotator 标准化流程

为新场景（例如 `warehouse`）生成资产和启动预设的工具。所有命令都在仓库根目录执行。

## 目录内容

- `create_scenario_scaffold.py`：一键创建新场景骨架（资产目录 + 启动预设 + 冒烟批量配置 + 文档）
- `convert_roi_to_minimal_assets.py`：把 ROI 标注结果转换为 episode 最小可用资产集
- `park_route_painter.py` / `park_zone_painter.py`：在 park 的 gridmap 上手绘干线路由 / 窄路区域，
  结果写入 `assets/scenes/park/extra_routes.json`、`bottleneck_zones.json`

> CARLA 内交互式 ROI 采集器（`carla_walk_roi_annotator.py`）的实现不在本发布中；
> 可用任意方式得到 ROI 标注 npz，再用下面的转换脚本处理。

## 一、创建新场景骨架

```bash
python tools/map_annotator/create_scenario_scaffold.py \
  --scenario-name warehouse \
  --map-name Town10HD_Opt
```

生成内容：

- `assets/scenes/warehouse/`（放资产）
- `configs/scenes/warehouse/episode.sh`（启动预设，默认复用 plaza 的 runner，日志标记为 `warehouse`）
- `configs/scenes/warehouse/batch/smoke_test.sh`
- `docs/scenes/warehouse.md`

## 二、转换为最小可用资产集

```bash
python tools/map_annotator/convert_roi_to_minimal_assets.py \
  --annotated-npz assets/scenes/warehouse/gridmap_roi_raw.npz \
  --base-npz /path/to/base/gridmap.npz \
  --output-dir assets/scenes/warehouse \
  --polygon-json assets/scenes/warehouse/gridmap_roi_polygon.json \
  --preserve-walked-path \
  --walked-path-radius-cells 1 \
  --overwrite
```

输出固定为：

- `gridmap_roi.npz`
- `gridmap_roi_flow_points.json`
- `gridmap_roi.png`（matplotlib 可用时）
- `gridmap_roi_polygon.json`（传入 polygon 时）

窄通道建议开启“保通行轨迹”：

- `--preserve-walked-path`：保证采集时走过的轨迹保持可走
- `--walked-path-radius-cells`：轨迹保留宽度（网格半径，建议 1~2）
- `--walked-path-close-loop`：需要闭环连通时可加

## 三、与 Episode 对接

```bash
bash scripts/run_episode.sh --scene warehouse
bash scripts/run_batch_eval.sh --scene warehouse --config smoke_test
```

或手动指定资产：

```bash
python -m followbench.scenes.plaza.run_episode \
  --load-map --map-name Town10HD_Opt \
  --grid-npz assets/scenes/warehouse/gridmap_roi.npz \
  --flow-points-json assets/scenes/warehouse/gridmap_roi_flow_points.json \
  --eval-scenario-type warehouse
```

## 四、质量检查（强烈建议）

- `gridmap_roi.npz` 的 `walkable > 0`
- `gridmap_roi_flow_points.json` 有 4 个点
- base npz 与 CARLA 当前地图一致

若 `gridmap_roi.png` 全黑，通常代表 ROI 与 base free 区没有重叠。
