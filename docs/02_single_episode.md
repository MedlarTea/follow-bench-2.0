# 02 — Single episode

One launcher for every scene:

```bash
bash scripts/run_episode.sh --scene corner
bash scripts/run_episode.sh --scene passage
bash scripts/run_episode.sh --scene park
bash scripts/run_episode.sh --scene plaza
bash scripts/run_episode.sh --scene corner --humanoid   # Unitree G1 follower (corner, park)
bash scripts/run_episode.sh --scene plaza  --teleop     # human teleop data collection, see 08_teleop.md
```

Each scene's defaults (map, crowd preset, spawn rules, sensors) live in
`configs/scenes/<scene>/episode.sh`. Tune them with environment variables:
`PLANNER`, `FOLLOW_POSITION`, `DESIRED_DISTANCE`, `DIFFICULTY`, `NUM_WALKERS`,
`SIM_MODE` (`sync`|`async`), plus scene-specific ones (`FLOW_MODE` for plaza,
`DEBUG` for park / plaza). Any extra argument is appended to the Python runner
and overrides the preset:

```bash
PLANNER=rda_traj DIFFICULTY=hard bash scripts/run_episode.sh --scene corner \
  --use-perception --reid-mode kpr --auto-spawn-sec 1 --auto-follow-sec 4

FB_DRY_RUN=1 bash scripts/run_episode.sh --scene park   # print the command only
bash scripts/run_episode.sh --scene passage -- --help   # all runner flags
```

The runners can also be called directly once the package is installed
(`python -m followbench.scenes.<scene>.run_episode --help`), but then no scene
preset is applied.

Useful flags:

```bash
--enable-evaluation
--use-perception --reid-mode basic|kpr
--auto-spawn-sec 1 --auto-follow-sec 4
--visualize          # corner: episode.mp4 / episode_3view.mp4
--debug              # top-down planner view
--no-ui              # headless
--npc-rng-seed N     # reproducible crowd sample
--host/--port        # CARLA endpoint
```

Keys (pygame UI): `G` spawn, `F` follow, `R` reset, `Q` quit.

Outputs: run directories in `outputs/runs/<scene>/`, evaluation logs in
`outputs/eval_logs/<scene>/<planner>/H<n>_<follow>_D<dist>/trial_*/`.
`bash scripts/run_episode_eval.sh --scene <scene>` runs one evaluated episode and
renders its plots.

See `docs/run_manual.md` for the full CLI matrix.
