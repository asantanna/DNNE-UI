# PPO Comparison Tools

This directory contains tools for comparing PPO (Proximal Policy Optimization) training performance between DNNE and IsaacGymEnvs.

## Overview

These scripts allow you to run identical PPO training workloads on both systems and compare:
- Execution time
- Training behavior
- Implementation differences

## Scripts

### run_1cycle_comparison.py
Runs both DNNE and IGE for exactly 1 PPO cycle with debug logging and automatically compares the outputs.

**Usage:**
```bash
python run_1cycle_comparison.py
```

**Features:**
- Runs with PPO_CYCLE_DEBUG=1 and PPO_STOP_AFTER_CYCLE=1
- Automatically exports DNNE workflow before running
- Saves logs to `/tmp` with timestamps
- Runs compare_ppo_logs.py automatically after both complete
- Uses fixed seed (42) for deterministic results

### compare_ppo_logs.py
Compares DNNE and IGE debug logs with intelligent preprocessing to handle expected differences.

**Usage:**
```bash
# Compare specific files
python compare_ppo_logs.py dnne.log ige.log

# Compare latest logs in /tmp (default)
python compare_ppo_logs.py
```

**Features:**
- Preprocesses logs to normalize paths, timestamps, and numeric values
- Ignores expected differences (paths, addresses, etc.)
- Highlights actual behavioral differences
- Saves preprocessed logs to `/tmp` for further analysis

### diff_based_compare.py
Advanced comparison tool using diff algorithm for accurate line-by-line alignment.

**Usage:**
```bash
# Compare latest preprocessed logs (default)
python diff_based_compare.py

# With options
python diff_based_compare.py --ignore-shared-differences
```

**Features:**
- Uses `diff --minimal -U 0` for accurate alignment
- Color-coded output (white=match, yellow=similar, red/green=different)
- Handles zero-length hunks correctly
- Option to ignore D/I/B shared attribute differences
- Side-by-side comparison with proper alignment

### run_ppo_comparison_timed.py
The main performance comparison script that runs both DNNE and IsaacGymEnvs with identical settings and measures execution time.

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

2. **RNG State Divergence** (July 2025 Discovery):
   - Despite using the same seed (42), DNNE and IGE have different torch RNG states
   - DNNE torch seed hash: -7310990532694816709
   - IGE torch seed hash: -8705526928541509108
   - This causes different neural network initialization and subsequent behavior divergence
   - Root cause still under investigation

3. **Behavioral Differences**:
   - Initial observations: Fixed in CartpoleDNNE by properly triggering reset
   - Network weights: Different due to RNG state divergence
   - Training dynamics: Similar patterns but different values due to initialization

4. **Architecture Differences**:
   - **DNNE**: Async queue-based, modular, exportable
   - **IsaacGymEnvs**: Synchronous, monolithic, optimized

## Debug Logging System

The comparison tools rely on a unified debug logging system using DNNE_print:

```python
DNNE_print(shared, category, message)
```

Where:
- `shared`: "D" (DNNE only), "I" (IGE only), "B" (both systems)
- `category`: Log category like "PPO_CYCLE", "PPO_INITIAL", "PPO_GRAD"
- `message`: The debug message

Enable debug logging with:
- `PPO_CYCLE_DEBUG=1`: Verbose PPO training debug logs
- `PPO_STOP_AFTER_CYCLE=1`: Stop after 1 PPO cycle (for manageable log sizes)

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