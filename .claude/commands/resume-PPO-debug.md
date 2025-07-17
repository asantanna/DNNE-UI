# Resume PPO Debug Session

You are resuming work on debugging the DNNE PPO implementation. In DNNE, the workflow is "Cartpole_PPO". In IsaacGymEnvs, the environment is "cartpole". Read these critical context files to understand the current state:

## Required Reading

```
@docs-dnne/code-quality-checklist.md
@docs-dnne/for_claude/debug_strategy_for_ppo.md  
@docs-dnne/for_claude/isaac_gym_reorganization_plan.md
```

## Useful Tools

1. **DNNE vs IsaacGymEnvs Performance Profiler**: @claude_scripts/profiling/performance_profiler.py [debug flags]
2. **DNNE workflow exporter**: @claude_scripts/programmatic_export.py
3. **NOTE**: Due to a bug in CLAUDE CODE, files sometimes disappear mysteriously. You can check that it happened by doing git status and verifying it now appears as deleted. You can restore from git but often this loses your current work. One thing you can do is make one change at a time to that file and backup the file to /tmp so that you don't lose all your work again.

## Current Status

We decided to change our approach to use rl_games code more directly. See the file @docs-dnne/for_claude/isaac_gym_reorganization_plan.md
