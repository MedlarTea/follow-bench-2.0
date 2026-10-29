# 08 — Human teleoperation data collection

FollowBench ships two teleop paths.

## A. Scenario teleop (crowd + metrics, for human data collection)

Drive the robot inside a full scene (crowd, success criteria, eval logs):

```bash
ROBOT_TYPE=g1 NUM_EPISODES=5 bash scripts/run_episode.sh --scene corner --teleop
```

Per-difficulty crowd presets are in
`scripts/teleop/collect_teleop_data.sh`:

```bash
NUM_EPISODES=5 OPERATOR_ID=operator01 bash scripts/teleop/collect_teleop_data.sh --scene corner easy
FOLLOW_POSITIONS="back left_side" bash scripts/teleop/collect_teleop_data.sh --all   # 4 scenes x 4 difficulties
PRINT_CONFIG=1 bash scripts/teleop/collect_teleop_data.sh --scene passage legend     # show the preset only
```

- Guide: [`docs/teleop_guide.md`](teleop_guide.md)
- Data collection protocol: [`docs/teleop_data_collection.md`](teleop_data_collection.md)

Requires a pygame UI (do not pass `--no-ui`) and `SIM_MODE=sync`. Logs land
under the evaluation root with `planner=teleop`.

## B. Standalone robot keyboard scripts

Drive Scout / G1 / Dog in a bare CARLA world (no episode metrics):

```bash
python tools/teleop/keyboard_Scout.py --host 127.0.0.1 --port 2000
```

See [`tools/teleop/README.md`](../tools/teleop/README.md).
