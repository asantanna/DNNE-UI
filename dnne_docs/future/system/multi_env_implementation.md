# Multi-Environment Batch Processing Implementation Plan

## Overview
This document outlines the plan to enable batch processing with multiple parallel environments in Franka_Coop_V2 and similar workflows. Currently, DNNE is limited to `num_envs=1` to avoid complexity, but supporting multiple environments would provide significant training benefits.

## Current State Analysis

### Franka_Coop_V2 Architecture
The Franka_Coop_V2 workflow implements cooperative control with:
- **3 independent networks** controlling joints 0, 1, and 2 of the Franka robot
- **Shadow environment network** for gradient bridging across non-differentiable simulator
- **TrainingSequencer** to orchestrate multiple optimizers and prevent gradient conflicts
- **Temporal alignment** using Eat_N and Barrier nodes for proper (s, a, s') transitions
- **Currently limited to num_envs=1** (single environment)

### Key Issues with Batch Processing
1. **IsaacGymSim forces num_envs=1** - Explicit check prevents multiple environments
2. **TrainingSequencer processes losses sequentially** - Not designed for batch-aware operations
3. **Loss computation expects single environments** - No batch dimension handling
4. **Network nodes handle batches but workflow isn't designed for it** - Infrastructure exists but unused

## Proposed Solution: Full Batch Support

### Phase 1: Enable Multi-Environment Support in IsaacGymSim
- Remove the `num_envs=1` restriction in `isaac_gym_sim_queue.tpl`
- Handle batched observations with shape: `[num_envs, obs_dim]`
- Handle batched actions with shape: `[num_envs, action_dim]`
- Ensure null_action broadcasting for all environments
- Update reset logic to handle per-environment done flags

### Phase 2: Update TrainingSequencer for Batch Processing
- Modify `backward_only()` to handle batched loss tensors
- Ensure gradient isolation works correctly with batched parameters
- Update step_complete signals for batch synchronization
- Verify retain_graph logic works with batched backward passes

### Phase 3: Fix Loss Computations
- Update `franka_coop_nodes_loss.py` to handle batch dimensions properly
- Update `franka_coop_nodes_shadow_loss.py` for batch-aware processing
- Ensure proper reduction (mean/sum) across batch dimension
- Maintain per-environment loss tracking for debugging

### Phase 4: Verify Split/Concat Operations
- Ensure Split nodes preserve batch dimension (dim 0)
- Verify Concat operations maintain batch structure
- Test tensor shapes through entire pipeline
- Update node documentation to clarify batch handling

### Phase 5: Testing Strategy
1. **Start with num_envs=2** - Minimal batch to verify basic functionality
2. **Test with num_envs=16** - Typical batch size for RL training
3. **Verify gradient flow** - Ensure gradients reach all networks
4. **Monitor training convergence** - Compare single vs batch training
5. **Performance benchmarking** - Measure speedup from parallelization

## Implementation Order

### 1. Isaac Gym Simulator Updates
```python
# Remove restriction in isaac_gym_sim_queue.tpl
# Old:
if config["num_envs"] != 1:
    raise ValueError(f"Only num_envs=1 supported, got {config['num_envs']}")

# New:
self.num_envs = config["num_envs"]
self.node_logger.info(f"Initializing with {self.num_envs} parallel environments")
```

### 2. TrainingSequencer Batch Handling
```python
# Update backward_only to handle batched losses
def backward_only(self, loss, retain_graph=False):
    """Perform backward with batch-aware loss"""
    # Loss shape: [num_envs] or scalar
    if loss.dim() > 0:
        # Reduce batch dimension for backward
        loss = loss.mean()
    loss.backward(retain_graph=retain_graph)
```

### 3. Loss Function Updates
```python
# Add batch dimension handling
def compute_loss(obs):
    # obs shape: [num_envs, obs_dim]
    eef_pos = obs[..., 3:6]  # [num_envs, 3]
    target_pos = obs[..., 0:3]  # [num_envs, 3]
    
    # Compute per-environment loss
    distances = torch.norm(eef_pos - target_pos, dim=-1)  # [num_envs]
    
    # Return per-env losses for debugging, mean for training
    return distances  # TrainingSequencer will handle reduction
```

### 4. Test Export and Verification
- Export workflow with different num_envs settings
- Verify generated code structure
- Check tensor shapes at each node
- Monitor memory usage with larger batches

### 5. Training and Monitoring
- Run training with various batch sizes
- Compare convergence rates
- Monitor GPU utilization
- Track per-environment performance

## Expected Benefits

### Performance Improvements
- **Faster training** - Process multiple environments in parallel on GPU
- **Better GPU utilization** - Batch operations are more efficient
- **Reduced wall-clock time** - More samples per second

### Training Quality
- **Better gradient estimates** - Average over multiple environment samples
- **Improved stability** - Less noise in gradient updates
- **Faster convergence** - More diverse experiences per update

### Standard RL Practice
- **Industry standard** - Most RL algorithms expect batched data
- **Algorithm compatibility** - Easier to implement advanced algorithms
- **Benchmarking** - Can compare with standard implementations

## Risk Mitigation

### Gradual Rollout
1. Keep `num_envs=1` as default initially
2. Add `--num-envs` command line argument for testing
3. Test progressively: 1 → 2 → 4 → 8 → 16 → 32 environments
4. Document any batch-size-dependent behaviors

### Debugging Support
- Add extensive logging for batch operations
- Create tensor shape validation throughout pipeline
- Implement per-environment metric tracking
- Add visualization for batch statistics

### Fallback Options
- Maintain single-environment mode as option
- Create compatibility layer for non-batch-aware nodes
- Document migration path for existing workflows

## Diagnostic Tools

### Batch Processing Validator
Create a diagnostic script to verify:
- Tensor shapes at each node
- Gradient flow through batched operations
- Memory usage patterns
- Performance metrics

### Example Diagnostic Output
```
=== Batch Processing Diagnostic ===
Num Environments: 16
Observation Shape: [16, 20] ✓
Action Shape: [16, 7] ✓
Loss Shape: [16] ✓
Gradient Flow: All networks receiving gradients ✓
Memory Usage: 2.3 GB / 8.0 GB
Throughput: 1250 steps/second
```

## Future Extensions

### Dynamic Batching
- Support variable num_envs during training
- Implement adaptive batch sizing based on GPU memory
- Handle environment resets asynchronously

### Distributed Training
- Support multi-GPU training with larger batches
- Implement distributed data parallel training
- Enable cloud-based training with massive parallelism

### Advanced Features
- Implement prioritized experience replay with batches
- Support heterogeneous environment configurations
- Enable curriculum learning with progressive batch complexity

## Implementation Timeline

### Week 1: Core Infrastructure
- Update IsaacGymSim for multi-environment support
- Modify TrainingSequencer for batch processing
- Update loss functions

### Week 2: Testing and Validation
- Create test workflows with various batch sizes
- Implement diagnostic tools
- Fix any batch-related issues

### Week 3: Performance Optimization
- Profile and optimize batch operations
- Implement memory-efficient solutions
- Add performance monitoring

### Week 4: Documentation and Release
- Update user documentation
- Create migration guide
- Release as experimental feature

## Success Criteria

1. **Functional** - Workflows run correctly with num_envs > 1
2. **Performance** - At least 5x speedup with num_envs=16 vs num_envs=1
3. **Stable** - Training convergence comparable or better than single environment
4. **Documented** - Clear documentation and examples for users
5. **Tested** - Comprehensive test coverage for batch operations

## Notes

This plan assumes:
- Isaac Gym supports multiple environments (it does)
- GPU has sufficient memory for batched operations
- Users understand batch processing concepts
- Existing single-environment workflows remain functional