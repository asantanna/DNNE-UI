# Utility Nodes Documentation

General utility nodes for workflow control and configuration in DNNE.

## Available Nodes

### [ORNode](or_node.md)
Logical OR operation for conditional workflow control.

### [BalancerConfig](balancing_config.md)
Configuration node for balancing task parameters.

### [BalancerNode](balancing_node.md)
Control logic for balancing tasks and simulations.

### [Eat_N](eat_n_node.md)
Consume first N inputs then become passthrough - essential for temporal alignment.

### [Barrier](barrier_node.md)
Hold and release data based on triggers - enables synchronization between pipeline stages.

## Overview

Utility nodes provide essential workflow control and configuration capabilities that support the ML and robotics nodes. These nodes handle:

- **Logic Operations**: Boolean logic for conditional execution
- **Configuration Management**: Task-specific parameter settings
- **Control Flow**: Workflow orchestration and synchronization
- **Task-Specific Utilities**: Specialized helpers for common tasks

## Node Categories

### Logic Nodes
- **ORNode**: Logical OR for combining conditions
- Future: AND, NOT, XOR nodes

### Configuration Nodes
- **BalancerConfig**: Parameters for balance control tasks
- Task-specific configuration management

### Control Nodes
- **BalancerNode**: Implements balancing control algorithms
- Task execution and monitoring

### Synchronization Nodes
- **Eat_N**: Consume and pass through data with trigger generation
- **Barrier**: FIFO-based data holding with triggered release
- Enable temporal alignment and pipeline synchronization

## Common Use Cases

### Conditional Execution
Use OR nodes to:
- Trigger actions based on multiple conditions
- Combine different trigger sources
- Implement fallback logic

### Task Configuration
Configuration nodes:
- Centralize hyperparameters
- Enable easy experimentation
- Support configuration reuse

### Control Algorithms
Control nodes like BalancerNode:
- Implement domain-specific algorithms
- Bridge between RL agents and environments
- Provide reference implementations

### Temporal Alignment for RL
Synchronization nodes enable proper temporal relationships:
- Align obs(t) with obs(t+1) for loss computation
- Bootstrap initial observations with Eat_N
- Synchronize gradient updates with Barrier nodes
- See [Temporal Alignment Pattern](../../patterns/temporal_alignment_rl.md) for detailed example

## Integration Patterns

### With ML Nodes
```
BalancerConfig → parameters
        ↓
  BalancerNode → control_signal
        ↓
   Environment
```

### With RL Nodes
```
PPOConfig + BalancerConfig → combined_config
              ↓
          PPOAgent
```

### Logic Flow
```
condition_1 ↘
            OR → trigger
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

## Implementation Details

- **Base Class**: All utility nodes inherit from `RoboticsNodeBase`
- **Location**: `/home/asantanna/DNNE/DNNE-UI/custom_nodes/*_visnode.py`
- **Templates**: `/home/asantanna/DNNE/DNNE-UI/export_system/templates/nodes/*_queue.py`
- **Export**: Generates helper functions and control logic