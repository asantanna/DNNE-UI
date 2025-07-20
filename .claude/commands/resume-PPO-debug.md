# Resume PPO Debug Session

You are resuming work on debugging the DNNE PPO implementation. In DNNE, the workflow is "Cartpole_PPO". In IsaacGymEnvs, the environment is "cartpole". Read these critical context files to understand the current state:

## Required Reading

```
@docs-dnne/for_claude/dnne_debugging_guide.md
@docs-dnne/for_claude/performance_analysis_overview.md
@docs-dnne/code-quality-checklist.md
```

## Useful Tools

1. **DNNE vs IsaacGymEnvs Performance Profiler**: @claude_scripts/profiling/performance_profiler.py [debug flags]
2. **DNNE workflow exporter**: @claude_scripts/programmatic_export.py
3. **NOTE**: Due to a bug in CLAUDE CODE, files sometimes disappear mysteriously. You can check that it happened by doing git status and verifying it now appears as deleted. You can restore from git but often this loses your current work. One thing you can do is make one change at a time to that file and backup the file to /tmp so that you don't lose all your work again.

## Current Status & Achievements

**Major Progress**: DNNE has been successfully refactored to use IsaacGymEnvs' cartpole infrastructure and multiple critical debugging issues have been resolved.

**Current Phase**: Correctness verification - ensuring DNNE matches IGE behavior exactly before performance optimization.

**Debugging Methodology**: Use matching debug prints (`PPO_CYCLE_DEBUG=1`) to compare DNNE vs IGE execution step-by-step. Run only one PPO cycle because debug prints are very verbose (`PPO_STOP_AFTER_CYCLE=1`)

**Log Comparison Tools**:
```bash
# Run 1-cycle comparison with automatic log comparison
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/profiling/ppo_comparison/run_1cycle_comparison.py

# Compare logs with diff-based alignment and line numbers
python claude_scripts/profiling/ppo_comparison/compare_ppo_logs.py

# Optional flags for comparison tools:
# --check-shared-attrib : Check D/I/B differences in shared code (by default they are ignored)
```

**Next Priority**: Verify that PPO training completes successfully and produces learning behavior identical to IGE.

## Quick Start Commands

```bash
# Export and run DNNE with debug
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/programmatic_export.py
cd export_system/exports/Cartpole_PPO
PPO_CYCLE_DEBUG=1 PPO_STOP_AFTER_CYCLE=1 python runner.p

# Run IGE for comparison (separate terminal)
cd /home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs
PPO_CYCLE_DEBUG=1 PPO_STOP_AFTER_CYCLE=1 python isaacgymenvs/train.py task=Cartpole --timeout 30s
```
