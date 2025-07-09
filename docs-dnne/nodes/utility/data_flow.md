# Utility Data Flow Nodes

## Overview

Data flow nodes manage the routing, merging, and organization of connections in DNNE workflows. These utility nodes are essential for creating complex workflows with multiple data paths.

---

## ORNode

### Purpose
Routes and merges multiple inputs into a single output, solving initialization and feedback loop challenges in cyclic workflows.

### Category
`Utility/Data Flow` (registered as "OR/ANY Router")

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input_0 | Any | First input option |
| input_1 | Any | Second input option |
| ... | Any | Additional inputs (dynamic) |

### Parameters
None - automatically handles any number of inputs

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Any | The first non-None input received |

### Usage Example
```
InitialData → ORNode → Processing
StepData ──────┘
```

Common uses:
- **RL Loops**: Combine initial observations with step observations
- **Fallback Logic**: Use first available data source
- **Data Merging**: Combine multiple data streams

### Export Support
✅ Full support with queue template

### Notes
- Essential for reinforcement learning loops
- Outputs the first non-None input
- Maintains data type of input
- Thread-safe for async execution

---

## RerouteNode

### Purpose
Visual organization tool that creates a waypoint for connections without modifying data.

### Category
`Utility/Data Flow`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| input | Any | Data to pass through |

### Parameters
None

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| output | Any | Unchanged input data |

### Usage Example
```
Source → RerouteNode → RerouteNode → Destination
              ↓
         OtherNode
```

Benefits:
- Cleaner visual layouts
- Avoid crossing connections
- Group related connections
- No performance overhead

### Export Support
✅ Full support (optimized out during export)

### Notes
- Pure visual organization
- No computational cost
- Can chain multiple reroutes
- Helpful for complex workflows

---

## Data Flow Patterns

### Initialization Pattern
```
Environment → InitialObs → ORNode → Policy
                             ↑
Step → Observations ─────────┘
```

### Branching Pattern
```
Source → RerouteNode ─→ ProcessA
              ↓
              └─→ ProcessB
```

### Feedback Loop
```
Input → Process → Output
  ↑                 ↓
  └─── ORNode ←─────┘
```

### Multi-Source Merge
```
Source1 → ORNode → Processing
Source2 ───┘
Source3 ───┘
```

## Best Practices

### ORNode Usage
- Place at convergence points
- Use for initialization handling
- Implement fallback logic
- Keep input types consistent

### RerouteNode Organization
- Use to prevent line crossings
- Group by functionality
- Create visual hierarchy
- Don't overuse (can clutter)

### Workflow Layout
- Top-to-bottom flow
- Left-to-right for sequences
- Minimize connection length
- Use consistent spacing

## Common Issues

### Issue: ORNode not outputting expected data
**Solution**: Check input priorities, ensure at least one input is connected

### Issue: Workflow looks cluttered
**Solution**: Add RerouteNodes to organize connections

### Issue: Circular dependency errors
**Solution**: Use ORNode to break initialization cycles

## Advanced Patterns

### Conditional Routing
```
Condition → Switch → ORNode1 → PathA
                  └→ ORNode2 → PathB
```

### Priority Selection
```
HighPriority → ORNode → Output
MedPriority ────┘
LowPriority ────┘
```

### State Machine
```
State1 → ORNode → Process → State2
  ↑        ↑                   ↓
  └────────┴───────────────────┘
```

## Integration Examples

### With RL Training
```
IsaacGymEnv → ORNode → PPOAgent
      ↑          ↑          ↓
      └──────────┴─── IsaacGymStep
```

### With Data Pipeline
```
Dataset1 → ORNode → BatchSampler
Dataset2 ───┘
```

### With Multi-Modal Processing
```
Camera → ORNode → FeatureExtractor
Lidar ────┘
```

## Performance Considerations

- ORNode: Minimal overhead, just passes references
- RerouteNode: Zero computational cost
- No data copying in either node
- Efficient for real-time applications

## Future Enhancements

- SwitchNode for conditional routing
- MergeNode with custom merge strategies
- PriorityNode for weighted selection
- DelayNode for timing control
- BufferNode for data accumulation