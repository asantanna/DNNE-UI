# Utility Nodes Documentation

General utility nodes for workflow control and configuration in DNNE.

## Available Nodes

### [ORNode](or_node.md)
Logical OR operation for conditional workflow control.

### [BalancingConfig](balancing_config.md)
Configuration node for balancing task parameters.

### [BalancingNode](balancing_node.md)
Control logic for balancing tasks and simulations.

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
- **BalancingConfig**: Parameters for balance control tasks
- Task-specific configuration management

### Control Nodes
- **BalancingNode**: Implements balancing control algorithms
- Task execution and monitoring

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
Control nodes like BalancingNode:
- Implement domain-specific algorithms
- Bridge between RL agents and environments
- Provide reference implementations

## Integration Patterns

### With ML Nodes
```
BalancingConfig → parameters
        ↓
  BalancingNode → control_signal
        ↓
   Environment
```

### With RL Nodes
```
PPOConfig + BalancingConfig → combined_config
              ↓
          PPOAgent
```

### Logic Flow
```
condition_1 ↘
            OR → trigger
condition_2 ↗
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