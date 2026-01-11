# Utility Nodes Documentation

*Last Updated: 2026-01-11*

Workflow control, synchronization, and data manipulation nodes.

## Available Nodes

### Synchronization
- **[Barrier](barrier_node.md)** - Hold data until triggered (FIFO queue)
- **[Eat_N](eat_n_node.md)** - Consume first N inputs, then passthrough + trigger

### Data Flow
- **Tensor** - Create/manipulate tensors
- **Concat** - Concatenate tensors along dim=1
- **Split** - Split tensors along dim=1
- **DataStreamer** - Stream data from external files

### Balancing
- **Balancer** - Measure throughput and frequency
- **BalancerConfig** - Configuration for balancer nodes

### Custom
- **CustomComputation** - User-defined Python code execution

## Overview

Utility nodes provide essential workflow control and configuration capabilities:

- **Synchronization**: Temporal alignment and pipeline coordination
- **Data Flow**: Tensor manipulation (concat, split, streaming)
- **Configuration Management**: Task-specific parameter settings
- **Control Flow**: Workflow orchestration

## Key Concepts

### Data Dimension Convention
All tensor operations follow DNNE's dimension convention:
- **Dim 0**: Batch/environment dimension
- **Dim 1**: Feature dimension (Concat/Split operate here)

### Temporal Alignment for RL
Synchronization nodes enable proper temporal relationships:
- Align obs(t) with obs(t+1) for loss computation
- Bootstrap initial observations with Eat_N
- Synchronize gradient updates with Barrier nodes

## Common Use Cases

### Conditional Execution
With multiple input connections:
- Trigger actions from multiple sources directly
- No need for separate OR nodes
- Cleaner, more intuitive workflows

### Task Configuration
Configuration nodes:
- Centralize hyperparameters
- Enable easy experimentation
- Support configuration reuse

## Integration Patterns

### With ML Nodes
```
BalancerConfig → parameters
        ↓
     Balancer → metrics
        ↓
   Environment
```

### With RL Nodes
```
PPOConfig + BalancerConfig → combined_config
              ↓
          PPOAgent
```

### Multiple Input Connections
```
condition_1 ↘
            → trigger (multiple connections supported)
condition_2 ↗
```

### Temporal Synchronization for RL
```
obs(t) → Split → [Barrier1, Barrier2, Barrier3] (hold)
   ↓
   └→ Split → Eat_N → Loss → [SGD1, SGD2, SGD3]
                ↓                    ↓
           trigger            step_complete
                └──────┬──────────────┘
                       ↓
                  [Barrier.release]
                       ↓
            [Network1, Network2, Network3]
                       ↓
                Concat → action → Simulator → obs(t+1)
```

## Export Behavior

Utility nodes export to:
- Python control logic
- Configuration dictionaries
- Conditional execution paths
- Helper functions and utilities

## Best Practices

1. **Configuration Reuse**: Share configs across similar tasks
2. **Logic Simplification**: Use utility nodes to reduce complexity
3. **Parameter Validation**: Validate configurations early
4. **Documentation**: Document custom utility nodes thoroughly
5. **Modularity**: Keep utilities focused and composable

## Creating Custom Utility Nodes

When creating new utility nodes:
1. Inherit from `RoboticsNodeBase`
2. Define clear input/output types
3. Implement both UI and export behavior
4. Add to appropriate category
5. Document thoroughly

## Common Patterns

### Configuration Management
```python
class ConfigNode(RoboticsNodeBase):
    CATEGORY = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param1": ("FLOAT", {"default": 1.0}),
                "param2": ("INT", {"default": 10})
            }
        }

    RETURN_TYPES = ("CONFIG",)
```

### Logic Operations
```python
class LogicNode(RoboticsNodeBase):
    CATEGORY = "utility"

    def process(self, input_a, input_b):
        return (input_a or input_b,)
```

## Implementation

- **Location**: `custom_nodes/*_visnode.py`
- **Templates**: `export_system/templates/nodes/*_queue.tpl`
- **Base Class**: All utility nodes inherit from `RoboticsNodeBase`
