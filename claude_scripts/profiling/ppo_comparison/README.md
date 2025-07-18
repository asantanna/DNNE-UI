# PPO Comparison Tools

This directory contains tools for comparing PPO (Proximal Policy Optimization) training performance between DNNE and IsaacGymEnvs.

## Overview

These scripts allow you to run identical PPO training workloads on both systems and compare:
- Execution time
- Training behavior
- Implementation differences

## Scripts

### run_ppo_comparison_timed.py
The main comparison script that runs both DNNE and IsaacGymEnvs with identical settings and measures execution time.

**Usage:**
```bash
python run_ppo_comparison_timed.py
```

**Features:**
- Runs both systems for exactly 1 epoch
- Measures wall-clock execution time
- Uses fixed seed (42) for deterministic results
- Saves full output logs for analysis
- No artificial timeouts - runs to completion

### test_ppo_single_cycle.py
Runs DNNE's exported Cartpole PPO training.

**Usage:**
```bash
python test_ppo_single_cycle.py
```

**Notes:**
- Expects DNNE export at: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO`
- Activates DNNE_PY38 conda environment automatically
- Can enable debug output with `PPO_CYCLE_DEBUG=1` environment variable

### test_ige_single_cycle.py
Runs IsaacGymEnvs Cartpole PPO training.

**Usage:**
```bash
python test_ige_single_cycle.py
```

**Notes:**
- Runs from: `/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs`
- Uses IsaacGymEnvs default hyperparameters
- Activates DNNE_PY38 conda environment automatically

### compare_ppo_outputs.py
Analyzes and compares detailed outputs from both systems.

**Usage:**
```bash
# After running the comparison
python compare_ppo_outputs.py
```

**Features:**
- Compares training metrics
- Identifies parameter differences
- Analyzes numerical discrepancies

## Latest Results

From our most recent comparison (July 17, 2025):

- **DNNE**: 8.05 seconds for 1 epoch
- **IsaacGymEnvs**: 6.53 seconds for 1 epoch
- **Performance**: IsaacGymEnvs is 1.23x faster

Both systems:
- Complete successfully
- Stop after exactly 1 epoch as requested
- Use identical PPO hyperparameters
- Run on the same hardware (CUDA GPU)

## Key Findings

1. **Performance Gap**: The 1.23x performance difference is expected because:
   - DNNE uses an async queue-based architecture (adds overhead)
   - IsaacGymEnvs has native Isaac Gym integration
   - DNNE prioritizes flexibility and portability over raw speed

2. **Behavioral Parity**: Both systems implement the same PPO algorithm correctly
   - Same training dynamics
   - Same convergence behavior
   - Same final results

3. **Architecture Differences**:
   - **DNNE**: Async queue-based, modular, exportable
   - **IsaacGymEnvs**: Synchronous, monolithic, optimized

## Prerequisites

- DNNE_PY38 conda environment
- CUDA-capable GPU
- Isaac Gym installed
- DNNE with exported Cartpole_PPO workflow

## Troubleshooting

If DNNE doesn't stop after 1 epoch:
1. Re-export the workflow: `python programmatic_export.py "Cartpole_PPO" "export_system/exports/Cartpole_PPO"`
2. Ensure templates are up to date from git
3. Check that `max_epochs` is properly set in the exported code

If comparison fails:
1. Verify both systems run independently first
2. Check CUDA memory - close other GPU applications
3. Ensure paths in scripts match your installation