# DNNE Performance Analysis Quick Start

## Current State

**Refactoring Complete**: DNNE has been completely refactored to use IsaacGymEnvs' (IGE) cartpole code and infrastructure as much as possible. This represents a major architectural shift from custom implementations to proven IGE components.

**Current Focus - Performance Optimization**: DNNE learning has been fixed and now performs effectively. The focus has shifted to performance optimization to approach IsaacGymEnvs speeds.

**Development Phase**: 
1. ✅ **Refactoring**: Complete - DNNE now inherits from IGE cartpole
2. ✅ **Correctness**: Complete - DNNE learning matches IGE behavior  
3. 🔄 **Performance**: Current phase - optimize to reach target FPS

**Note**: PPO training with both DNNE and IsaacGymEnvs is fully functional. Learning performance issues have been resolved.

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

## Correctness Verification

**Methodology**: Use matching debug prints to compare DNNE and IGE execution step-by-step. When outputs diverge, instrument the divergence point until the root cause is identified.

**Debug Environment Variables**:
- `PPO_CYCLE_DEBUG=1` - Enables detailed PPO cycle logging
- `USE_STANDARD_RL_GAMES=1` - Makes IGE use standard rl_games instead of rl_games_dnne (which is now default)
- `FIXED_SEED=42` - Forces deterministic execution

**Verification Commands**:
```bash
# Run IGE with debug output (rl_games_dnne is default)
cd /home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs
PPO_CYCLE_DEBUG=1 python isaacgymenvs/train.py task=Cartpole > /tmp/ige_debug.txt

# Run DNNE with debug output  
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO
PPO_CYCLE_DEBUG=1 python runner.py > /tmp/dnne_debug.txt

# Compare outputs
diff /tmp/ige_debug.txt /tmp/dnne_debug.txt
```

**Success Criteria**: Debug outputs should match exactly for:
- Initialization sequences
- Action generation patterns
- PPO training cycles
- Loss computation values

**Success Criteria**: Debug outputs should match exactly for all key metrics listed above.

## Running Tests

### 1. Correctness Verification (Run First)
```bash
# Export workflow
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/programmatic_export.py

# Run DNNE with debug output
cd export_system/exports/Cartpole_PPO
PPO_CYCLE_DEBUG=1 python runner.py --timeout 30s

# Compare with IGE (in separate terminal)
cd /home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs
PPO_CYCLE_DEBUG=1 python isaacgymenvs/train.py task=Cartpole
```

### 2. Performance Testing (After Correctness Verified)
```bash
# Performance comparison
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/performance_comparison_table.py

# Direct benchmark
cd export_system/exports/Cartpole_PPO
python runner.py --headless --timeout 15s --profile
```

## Key Metrics to Monitor

1. **FPS (Frames Per Second)**: Main throughput metric
2. **Forward Pass Time**: PPO Agent computation time
3. **Node Computations**: Per-node execution counts
4. **Queue Wait Times**: Async coordination overhead

## Performance Targets

**Current Status**: With learning issues resolved, these performance targets are now the primary focus.

- **Short-term**: 100+ FPS
- **Medium-term**: 1,000+ FPS  
- **Long-term**: 10,000+ FPS (approaching IsaacGymEnvs performance)

## Quick Debugging Commands

```bash
# Check if conda environment is active
echo $CONDA_DEFAULT_ENV  # Should show DNNE_PY38

# Activate if needed
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Run correctness comparison
PPO_CYCLE_DEBUG=1 python runner.py --timeout 30s | grep -E "(PPO_CYCLE|PPO_BATCH|PPO_GRAD)"

# Quick performance test (after correctness verified)
cd /mnt/e/ALS-Projects/DNNE/DNNE-UI
python claude_scripts/performance_comparison_table.py | grep "Avg FPS"
```

## Known Issues

For debugging techniques and common issues, check the archived debug documents for historical reference.