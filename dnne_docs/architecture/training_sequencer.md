# Training Sequencer Architecture

## Problem Statement

When multiple SGDOptimizer nodes share networks (e.g., control networks feeding into a shadow network), PyTorch gradient computation fails with:
- "One of the variables needed for gradient computation has been modified by an inplace operation"
- Gradient accumulation conflicts when networks contribute to multiple losses

## Root Cause

PyTorch gradient mechanics cause conflicts:

1. **Gradient Accumulation**: `loss.backward()` adds gradients to existing `.grad` attributes rather than replacing them
2. **Shared Parameters**: Networks contributing to multiple losses accumulate gradients from all backward passes
3. **Concurrent Backward**: Multiple optimizers calling backward() simultaneously create race conditions
4. **Graph Retention**: First backward pass destroys computation graph unless `retain_graph=True`, but this enables unwanted gradient accumulation

Example failure scenario:
```
Control Networks (62, 33, 54) → Shadow Network (78)
                ↓                        ↓
         Control Loss              Shadow Loss
                ↓                        ↓
        SGDOptimizer_40           SGDOptimizer_81

Both optimizers backward() → Control networks get gradients from BOTH losses → Conflict
```

## Solution: Training Sequencer Node

A centralized orchestrator that controls gradient computation order and prevents conflicts.

### Design

**Inputs**: `loss1`, `loss2`, `loss3`, `loss4` - Loss tensors from different sources  
**Outputs**: `to_opt1`, `to_opt2`, `to_opt3`, `to_opt4` - Signals to SGDOptimizer nodes  
**Widgets**: 
- `order`: Comma-separated execution order (e.g., "1,2,3")
- `retain_graph`: Boolean for graph retention strategy

### Operation Sequence

1. **Initialization**:
   - Detect which loss inputs are actually connected
   - Set only connected inputs as required (e.g., if only loss1 and loss2 connected: `self.setup_inputs(required=["loss1", "loss2"])`)
   - Resolve optimizer node references from exported IDs

2. **Compute (all losses arrive together)**:
   - QueueNode framework guarantees all required inputs arrive simultaneously
   - No state tracking needed

3. **Sequential Backward Passes**:
   ```python
   async def compute(self, **kwargs):
       # All required losses arrive together
       losses = {i: kwargs[f"loss{i}"] for i in self.connected_indices}
       
       for i, opt_idx in enumerate(self.order):
           # Disable gradients for non-target networks
           for other_idx, other_opt in enumerate(self.optimizers):
               if other_idx != opt_idx - 1:  # Convert 1-based to 0-based
                   for param in other_opt.get_parameters():
                       param.requires_grad_(False)
           
           # Backward pass for this optimizer's loss
           self.optimizers[opt_idx - 1].zero_grad()
           retain = (i < len(self.order) - 1)  # All but last
           losses[opt_idx].backward(retain_graph=retain)
           
           # Re-enable gradients
           for other_idx, other_opt in enumerate(self.optimizers):
               if other_idx != opt_idx - 1:
                   for param in other_opt.get_parameters():
                       param.requires_grad_(True)
   ```

4. **Optimizer Steps**:
   - After all backward passes complete
   - Call `step()` on all optimizers
   - Return signals to connected optimizer nodes

### SGDOptimizer Modifications

Add methods for external control:
```python
def zero_grad_only(self):
    """Zero gradients without backward"""
    for optimizer in self.optimizers:
        optimizer.zero_grad()

def backward_only(self, loss, retain_graph=False):
    """Perform backward without step"""
    loss.backward(retain_graph=retain_graph)

def step_only(self):
    """Step optimizer without backward"""
    for optimizer in self.optimizers:
        optimizer.step()

def get_parameters(self):
    """Return all managed parameters for requires_grad toggling"""
    params = []
    for model_node in self.model_nodes:
        params.extend(model_node.get_parameters())
    return params
```

## Key Benefits

1. **Single Control Point**: One entity manages all backward passes
2. **Explicit Ordering**: Training sequence visible in UI widget
3. **Automatic retain_graph**: Handled correctly without user intervention
4. **Prevents Accumulation**: requires_grad toggling ensures each loss only affects intended networks
5. **No Race Conditions**: Sequential execution prevents concurrent gradient computation

## Implementation Notes

- Sequencer sets only connected inputs as required, ensuring all arrive together
- Export system must detect connected inputs and substitute optimizer node IDs correctly
- SGDOptimizer nodes detect sequencer connection and disable normal backward path
- Backwards compatibility: SGDOptimizer works standalone when no sequencer connected
- Order widget specifies 1-based indices matching loss input numbers (e.g., "2,1,3" for loss2→loss1→loss3)