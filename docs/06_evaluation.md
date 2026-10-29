# 06 — Evaluation

Enable with `--enable-evaluation`. Trial directory:

```text
outputs/eval_logs/<scenario>/<planner>/H{non_target}_{follow}_D{dist}/trial_*/
  episode_meta.json
  episode_step.jsonl
  collision_events.jsonl
  eval_result.json
  eval_summary.txt
```

`H` = non-target pedestrians (`num_walkers_total - 1`).

Active ticks: `episode_active && !paused` (search / recover still counted).

Key fields in `eval_result.json`: `task_success`, `obstacle_avoidance_success`,
`search_success`, `target_visibility_ratio`, zone times, jerk, timing splits
(`policy_total_ms_*`, `perception_*`, `planner_core_*`).

Baseline display names (viz layer):

- SR = OAS ∧ search_success  
- ASR = OAS  
- TinTPerson / TinPrivate = zone_time / total_time  

Recompute:

```bash
python -m followbench.evaluation.evaluator --run-dir <trial_dir>
```

On human collision (non-teleop), the episode runner exits the main loop
after writing the collision step/events so `termination_reason` stays
`human_collision` instead of being overwritten by `duration_timeout`.
