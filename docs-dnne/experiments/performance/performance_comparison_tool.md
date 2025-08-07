# Performance Comparison Tool - DNNE vs IsaacGymEnvs

## Overview

This document describes the performance comparison tool created to benchmark DNNE (Drag and Drop Neural Network Environment) against IsaacGymEnvs for the Cartpole PPO task. The tool generates a formatted table showing key performance metrics side-by-side.

## Current Status

### What Works ✅
- **Script Location**: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/claude_scripts/performance_comparison_table.py`
- **IsaacGymEnvs Baseline**: Successfully uses performance data from previous analysis
- **Table Generation**: Creates a well-formatted comparison table
- **Timeout Handling**: Properly handles DNNE's initialization timeout
- **Partial Output Capture**: Gets some output from DNNE before timeout

### What Doesn't Work ❌
- **DNNE Initialization**: Times out after 30 seconds during Isaac Gym environment creation
- **Performance Metrics**: Cannot collect any DNNE performance data due to initialization failure

## Sample Output (This is not working yet)

```
===========================================================
Performance Comparison: DNNE vs IsaacGymEnvs (Cartpole PPO)
===========================================================
Measurement          |       DNNE        |    IsaacGymEnvs   
---------------------+-------------------+-------------------
Init Time (s)        |      Timeout      |        5.2s       
Avg FPS              |        N/A        |       32,000      
Peak FPS             |        N/A        |       36,897      
Batch Size           |       512.0       |       512.0       
Forward Pass (ms)    |        N/A        |       0.8ms       
Memory Usage (MB)    |        N/A        |       2,048       
Training Speed       |        N/A        |    68.0 eps/min   
Status               |      TIMEOUT      |      SUCCESS      
---------------------+-------------------+-------------------
```

## Key Findings

### 1. DNNE Initialization Hang
- **Location**: `isaacgymenvnode_7.py` in `_initialize_isaac_gym()` method
- **Symptom**: Hangs after "Isaac Gym core initialization" and "PhysX engine setup"
- **Partial Output Captured**:
  ```
  Importing module 'gym_38' (/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
  Setting GYM_USD_PLUG_INFO_PATH to /home/asantanna/DNNE-LINUX-SUPPORT/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
  🚀 Starting DNNE Queue-Based Execution
  📊 Quiet mode - showing epoch summaries only
  ============================================================
  PyTorch version 2.4.1+cu121
  Device count 1
  ```

### Previous Performance Analysis
From earlier benchmarking, we found:
- **Raw PyTorch**: ~1ms per forward pass
- **DNNE (if it worked)**: Expected ~69ms per forward pass (69x slower)
- **Main Bottleneck**: Async executor overhead in PPOAgent node

### IsaacGymEnvs Performance (Working Baseline)
- **Init Time**: ~5-10 seconds
- **Average FPS**: 32,000 (conservative estimate)
- **Peak FPS**: 36,897
- **Training Speed**: 68 epochs/minute (34 epochs in 30 seconds)
- **Memory Usage**: ~2GB GPU memory

## Tool Implementation Details

### Script Structure
```python
# Main components:
1. PerformanceMetrics class - Container for all metrics
2. run_dnne_test() - Runs DNNE with timeout, captures partial output
3. get_isaacgymenvs_baseline() - Returns hardcoded baseline data
4. print_comparison_table() - Formats and displays the comparison
```

### Key Features
1. **Subprocess with Timeout**: Uses `subprocess.run()` with 30s timeout
2. **Conda Environment**: Activates DNNE_PY38 environment before running
3. **Output Parsing**: Looks for specific patterns in stdout/stderr
4. **Graceful Failure**: Shows "Timeout" and "N/A" for missing metrics

### Building Blocks Reused
- From `benchmark_cartpole_performance.py`: Performance metric parsing
- From `test_cartpole_ppo_comprehensive.py`: Subprocess execution patterns
- From performance analysis docs: IsaacGymEnvs baseline metrics

## Known Issues

### 1. DNNE Environment Factory Hang
- **Root Cause**: Unknown - needs deeper investigation
- **Verified NOT**: 
  - Double initialization issue (already fixed)
  - Import order issue (isaacgym imported first)
  - Missing environment directory (exists at ../environments)
  - Asset loading issue (cartpole.urdf exists)

### 2. Async Queue Architecture
- DNNE uses queue-based async node execution
- IsaacGymEnvs uses direct synchronous execution
- This architectural difference may contribute to initialization issues

## Next Steps to Fix DNNE

### 1. Add Detailed Logging
```python
# In _create_environment method:
self.logger.info("Step 1: Importing environment factory...")
self.logger.info("Step 2: Creating environment instance...")
self.logger.info("Step 3: Creating physical environments...")
# etc.
```

### 2. Add Timeout Within DNNE
- Add initialization timeout to prevent indefinite hangs
- Raise clear exception when timeout occurs
- Log exactly where initialization stopped

### 3. DELETED

### 4. Compare with Working IsaacGymEnvs
- Run IsaacGymEnvs with same configuration
- Compare initialization sequence step-by-step
- Identify where DNNE diverges

### 5. Fix Async Executor Overhead
Once initialization works, optimize PPOAgent node:
- Remove `loop.run_in_executor()` wrapper
- Run PyTorch operations directly
- Expected improvement: 69ms → ~1-2ms per forward pass

## Usage

```bash
# Activate conda environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Run comparison
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/performance_comparison_table.py
```

## File Locations

- **Export Directory**: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO/`
- **Runner Script**: `export_system/exports/Cartpole_PPO/runner.py`
- **IsaacGymEnvs**: `/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/`
- **Performance Script**: `claude_scripts/performance_comparison_table.py`

## Environment Requirements

- **Conda Environment**: DNNE_PY38
- **Python**: 3.8.20
- **PyTorch**: 2.4.1+cu121
- **Isaac Gym**: Installed at `~/DNNE-LINUX-SUPPORT/isaacgym`
- **CUDA**: Available and working

## Summary

The performance comparison tool successfully shows the stark difference between DNNE (not working) and IsaacGymEnvs (working well). The main challenge is that DNNE hangs during initialization, preventing any performance measurement. Once the initialization issue is resolved, we expect to see DNNE performing significantly slower than IsaacGymEnvs due to its async queue-based architecture, but the exact performance gap cannot be measured until DNNE actually runs.