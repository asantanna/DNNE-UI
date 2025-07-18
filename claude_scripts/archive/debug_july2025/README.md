# Debug Scripts Archive - July 2025

This directory contains scripts created during debugging sessions in July 2025, primarily focused on resolving PPO training issues between DNNE and IsaacGymEnvs.

## Context

These scripts were created while debugging:
- PPO training not stopping after specified epochs
- Episode completion tracking issues
- Fixed seed determinism problems
- Initial state differences between DNNE and IsaacGymEnvs
- CUDA and GPU-related issues
- rl_games integration and observation normalization
- Cartpole-specific implementation differences

## Status

**All issues addressed by these scripts have been resolved.**

The main issue was that exported DNNE code needed to be re-exported with updated templates from git. Once re-exported, DNNE properly:
- Stops after the specified number of epochs
- Completes PPO training cycles correctly
- Matches IsaacGymEnvs behavior

## Current Working Tools

The production-ready comparison tools are now located in:
`/claude_scripts/profiling/ppo_comparison/`

Use those tools for any future PPO comparisons or debugging.

## Archived Scripts

### Debug Scripts
- `debug_compare_single_cycle.py` - Single PPO cycle comparison debugging
- `debug_dnne_fixed_seed.py` - Fixed seed reproducibility testing
- `debug_episode_completion.py` - Episode completion tracking issues
- `debug_episode_tracking.py` - Episode tracking debugging
- `debug_isaacgym_single_cycle.py` - IsaacGymEnvs single cycle debugging
- `debug_isaacgymenvs_fixed_seed.py` - IsaacGymEnvs fixed seed testing

### Test Scripts
- `test_basic_isaac.py` - Basic Isaac Gym functionality test
- `test_clean_gpu.py` - GPU memory cleanup testing
- `test_dnne_cpu.py` - CPU mode testing
- `test_dnne_cuda_fix.py` - CUDA issues debugging
- `test_dnne_direct.py` - Direct DNNE execution testing
- `test_dnne_logging.py` - Logging functionality testing
- `test_dnne_minimal.py` - Minimal DNNE test case
- `test_ige_debug.py` - IsaacGymEnvs debug test
- `test_initial_state.py` - Initial state comparison
- `test_isaac_minimal.py` - Minimal Isaac Gym test
- `analyze_initial_state_differences.py` - Initial state difference analysis

### RL Games Debug Scripts
- `run_with_rl_games_debug.sh` - Script to run IsaacGymEnvs with debug version of rl_games
- `setup_rl_games_debug.py` - Setup script for rl_games debug environment
- `cartpole_debug.py` - Cartpole-specific debugging script
- `isaacgymenvs_debug_runner.py` - Debug runner for IsaacGymEnvs

### Analysis Documents
- `observation_normalization_analysis.md` - Analysis of PPO observation normalization bug

### Other Scripts
(none)

## Historical Reference

These scripts are preserved for historical reference and may contain useful debugging patterns or approaches for future issues. However, they should not be used for current testing as:
1. The issues they address have been resolved
2. They may contain outdated assumptions
3. Better tools exist in the profiling directory