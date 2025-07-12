# DNNE Performance Analysis Quick Start

## Current State (January 2025)

**Performance Gap**: DNNE runs at **166 FPS** vs IsaacGymEnvs baseline of **32,000 FPS** (192x slower)

**Previous Bottleneck Fixed**: Debug logging was causing 95% overhead (60ms → 1.73ms) ✅

**Current Bottleneck**: Queue coordination and async overhead
- PPO forward pass: 1.73ms (matches raw PyTorch!)
- Queue coordination: ~4ms per step
- Total system throughput: ~6ms per iteration

## Architecture Overview

### Export System Flow
```
Visual Workflow (JSON) → Graph Exporter → Node Templates → Generated Python Code
                                               ↓
                                    Async Queue-Based Execution
```

### Key Components
1. **Graph Exporter** (`export_system/graph_exporter.py`) - Converts workflows to code
2. **Node Templates** (`export_system/templates/nodes/`) - Code generation templates
3. **Queue Framework** (`export_system/templates/base/`) - Async execution engine
4. **Exported Code** (`export_system/exports/Cartpole_PPO/`) - Generated training script

### Critical Performance Paths

#### PPO Agent Node (Fixed! Was 60ms, now 1.73ms)
- **Template**: `export_system/templates/nodes/ppo_agent_queue.py`
- **Exported**: `export_system/exports/Cartpole_PPO/nodes/ppoagentnode_3.py`
- **Fixed**: Removed debug logging that was causing tensor string formatting overhead

#### PPO Trainer Node
- **Template**: `export_system/templates/nodes/ppo_trainer_queue.py`
- **Exported**: `export_system/exports/Cartpole_PPO/nodes/ppotrainernode_6.py`
- **Issue**: Multiple redundant `.to(device)` calls

#### Isaac Gym Step Node
- **Template**: `export_system/templates/nodes/isaac_gym_step_queue.py`
- **Exported**: `export_system/exports/Cartpole_PPO/nodes/isaacgymstepnode_9.py`
- **Issue**: Template missing smart throttling (fixed in export but not template)

## Running Performance Tests

### 1. Export Workflow
```bash
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/programmatic_export.py
```

### 2. Run Performance Comparison
```bash
python claude_scripts/performance_comparison_table.py
```

### 3. Run Direct Benchmark
```bash
cd export_system/exports/Cartpole_PPO
python runner.py --headless --timeout 15s --profile
```

## Key Metrics to Monitor

1. **FPS (Frames Per Second)**: Main throughput metric (currently 166 FPS)
2. **Forward Pass Time**: PPO Agent computation time (now 1.73ms)
3. **Node Computations**: Per-node execution counts
4. **Queue Wait Times**: Async coordination overhead

## Performance Targets

- **Short-term**: ✅ 100+ FPS (achieved 166 FPS!)
- **Medium-term**: 1,000+ FPS (6x improvement needed)
- **Long-term**: 10,000+ FPS (approaching IsaacGymEnvs performance)

## Quick Debugging Commands

```bash
# Check if conda environment is active
echo $CONDA_DEFAULT_ENV  # Should show DNNE_PY38

# Activate if needed
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Quick performance test
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/performance_comparison_table.py | grep "Avg FPS"
```

## Known Issues & Solutions

1. **Import Order**: Isaac Gym must be imported before PyTorch
2. **Debug Logging**: ✅ FIXED - Tensor string formatting was causing 95% overhead
3. **Queue Sizes**: Default size of 2 may cause blocking
4. **Async Overhead**: Queue coordination adds ~4ms per step

## Next Optimization Targets

1. **Queue Batching**: Process multiple environment steps before PPO updates
2. **Increase Queue Sizes**: Reduce blocking on queue operations
3. **Sync vs Async**: Consider synchronous execution for tight loops
4. **Multi-Environment Batching**: Better utilize GPU with larger batches