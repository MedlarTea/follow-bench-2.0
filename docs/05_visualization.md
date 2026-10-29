# 05 — Visualization

## Runtime UI

- Default pygame window: RGB left/front/right + depth / instance / LiDAR (when UI enabled).
- `--no-ui`: headless batch / remote.

## Top-down planner debug

```bash
bash scripts/run_episode.sh --scene <scene> --debug
```

Separate process (`followbench/visualization/`); never blocks the sim loop. Shows ROI,
local occupancy / ESDF, planned trajectory, obstacles.

## First-person evaluation video

```bash
scripts/run_episode.sh --scene corner \
  --enable-evaluation --visualize \
  --auto-spawn-sec 1 --auto-follow-sec 4
```

| Follow position | File |
|---|---|
| `back` | `episode.mp4` (front + HUD) |
| `left_side` / `right_side` | `episode_3view.mp4` (L\|F\|R + HUD) |

With `--use-perception`, overlays green target / yellow other tracks; stale
tracks dim + `STALE N` HUD tag.

## Offline plots

```bash
python -m followbench.visualization.plot_episode --run-dir <trial_dir>
python -m followbench.visualization.plot_compare --scenario corner ...
```
