# Robot keyboard teleop (standalone)

Standalone pygame teleop for the FollowBench custom robots. Use this to drive
Scout / Unitree G1 / Dog in a running CARLA world without starting a follow
episode. For **scenario teleop data collection** (crowd + metrics), see
[`docs/teleop_guide.md`](../../docs/teleop_guide.md) and
[`docs/teleop_data_collection.md`](../../docs/teleop_data_collection.md).

## Scripts

| Script | Robot |
|---|---|
| `keyboard_Scout.py` | `vehicle.scout.scout` |
| `keyboard_G1.py` | `walker.pedestrian.g1_unitree` |
| `keyboard_Dog.py` | Dog blueprint (see script header) |

## Requirements

- FollowBench UE / CARLA already running
- Conda env with `carla`, `pygame`, `numpy`
- A display (these scripts open a pygame window)

## Scout example

```bash
# terminal 1: CARLA
bash tools/carla/run_carla.sh -RenderOffScreen -nosound -quality-level=Low

# terminal 2
conda activate <your-env>
python tools/teleop/keyboard_Scout.py --host 127.0.0.1 --port 2000
```

Useful Scout flags: `--blueprint`, `--role-name`, `--spawn-index`.

## Controls (Scout)

| Key | Action |
|---|---|
| `W` / `S` | throttle / reverse |
| `A` / `D` | steer |
| `Space` | brake |
| `Tab` | toggle main / surround camera |
| `R` | reset pose |
| `Esc` | quit |

G1 / Dog use their own on-screen HUD keys (speed step, turn, camera). Open the
script or the HUD overlay for the live binding list.

## Tips

- Prefer sync CARLA for predictable control feel.
- If the actor already exists with the same `--role-name`, Scout reuses it.
- This path does **not** write FollowBench eval logs. Use `--teleop` on a
  scenario launcher when you need metric-compatible human-collected data.
