# Utility Constants and Debug Nodes

## Overview

Constants and debug nodes provide fixed values and debugging capabilities for DNNE workflows. These utility nodes are essential for testing, configuration, and troubleshooting.

---

## ConstantFloatNode

### Purpose
Provides a constant floating-point value to the workflow.

### Category
`Utility/Constants`

### Inputs
None (Source node)

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| value | float | 0.0 | The constant value to output |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| value | float | The constant float value |

### Usage Example
```
ConstantFloat(0.01) → LearningRate input
ConstantFloat(0.9) → Momentum input
```

Common uses:
- Hyperparameter values
- Threshold settings
- Scale factors
- Fixed references

### Export Support
✅ Full support

### Notes
- Value is embedded in exported code
- No runtime computation
- Useful for workflow parameterization

---

## ConstantIntNode

### Purpose
Provides a constant integer value to the workflow.

### Category
`Utility/Constants`

### Inputs
None (Source node)

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| value | int | 0 | The constant value to output |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| value | int | The constant integer value |

### Usage Example
```
ConstantInt(64) → BatchSize input
ConstantInt(10) → NumClasses input
```

Common uses:
- Array dimensions
- Loop counts
- Category counts
- Index values

### Export Support
✅ Full support

### Notes
- Type-safe integer values
- Prevents float/int confusion
- Clear workflow documentation

---

## DebugPrintNode

### Purpose
Prints values to console for debugging and monitoring.

### Category
`Utility/Debug`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| value | Any | Value to print |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| prefix | string | "" | Text to print before value |
| enabled | boolean | true | Enable/disable printing |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| value | Any | Pass-through of input |

### Usage Example
```
Network → DebugPrint("Loss: ") → Optimizer
```

Features:
- Non-intrusive (passes data through)
- Can be disabled without removing
- Supports tensor shapes and stats
- Useful for debugging data flow

### Export Support
✅ Full support with conditional execution

### Notes
- Minimal performance impact
- Can print tensor statistics
- Helps identify data flow issues
- Should be disabled in production

---

## APICallNode *(Partially Implemented)*

### Purpose
Makes API calls via WebSocket for external communication.

### Category
`Utility/API`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| data | Any | Data to send |
| trigger | Any | Optional execution trigger |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| endpoint | string | "" | WebSocket endpoint URL |
| timeout | float | 30.0 | Request timeout in seconds |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| response | Any | API response data |
| status | int | Response status code |

### Usage Example
```
Model → APICall(endpoint="/predict") → ProcessResponse
```

### Export Support
⚠️ Limited support (requires runtime configuration)

### Notes
- Useful for cloud integration
- Supports async operations
- Handle network errors gracefully

---

## Debug Patterns

### Value Inspection
```
Tensor → DebugPrint("Shape: ") → NextNode
```

### Conditional Debugging
```
Loss → DebugPrint(enabled=debug_mode) → Optimizer
```

### Multi-Point Debugging
```
Input → DebugPrint("Input: ") → 
Process → DebugPrint("After Process: ") → 
Output
```

### Performance Monitoring
```
StartTime → Process → DebugPrint("Duration: ") → EndTime
```

## Best Practices

### Using Constants
- Document constant purposes
- Group related constants
- Use meaningful values
- Consider configuration files

### Debug Strategy
- Add prints at key points
- Disable in production
- Print shapes not full tensors
- Use descriptive prefixes

### Workflow Testing
- Start with constants for testing
- Replace with real data later
- Use debug nodes liberally
- Remove before final export

## Common Issues

### Issue: Too much debug output
**Solution**: Use prefix to filter, disable selectively

### Issue: Constants need frequent changes
**Solution**: Consider configuration node (future)

### Issue: Debug affects performance
**Solution**: Disable debug nodes, use sampling

## Advanced Usage

### Parameterized Workflows
```
[ConstantFloat(lr), ConstantInt(epochs)] → TrainingConfig
```

### Debug Sampling
```
Every_N_Steps → DebugPrint(enabled=sample) → Log
```

### API Integration
```
LocalCompute → APICall → CloudProcess → APICall → Result
```

## Integration Examples

### With Training
```
ConstantFloat(0.001) → SGDOptimizer.learning_rate
Epoch → DebugPrint("Epoch: ") → Continue
```

### With Testing
```
ConstantInt(1) → BatchSize (for debugging)
Loss → DebugPrint("Test Loss: ") → Metrics
```

### With Production
```
Config → Constants → Model
Stats → APICall(metrics_endpoint) → Monitor
```

## Future Enhancements

- ConfigurationNode for external configs
- AssertNode for validation
- ProfilerNode for performance
- LoggerNode with levels
- WatchNode for breakpoints