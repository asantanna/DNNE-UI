# Eat_N Node

## Overview
The Eat_N node is a synchronization primitive that consumes the first N inputs and then becomes a passthrough for all subsequent inputs. It's essential for temporal alignment in reinforcement learning workflows where the first few observations need special handling.

## Purpose
- Bootstrap reinforcement learning pipelines by consuming initial observations
- Generate triggers to release held data in downstream Barrier nodes
- Create temporal shifts in data streams for proper t vs t+1 alignment

## Node Configuration

### Category
`utility`

### Inputs
| Name | Type | Description |
|------|------|-------------|
| input | *TENSOR | Any tensor input to consume or pass through |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| output | TENSOR | Passthrough output (only after N inputs consumed) |
| trigger | EAT_N_TRIGGER | Trigger signal sent based on trigger_mode |

### Widgets
| Name | Type | Default | Description |
|------|------|---------|-------------|
| num_to_eat | INT | 1 | Number of inputs to consume before becoming passthrough |
| trigger_mode | ENUM | "every_eat" | When to send triggers: "every_eat" or "last_only" |

## Behavior

### State Machine
The node maintains an internal counter tracking consumed inputs:

1. **Consuming State** (counter < num_to_eat):
   - Receives input data
   - Does NOT output to `output` port
   - Sends trigger based on trigger_mode
   - Increments counter

2. **Passthrough State** (counter >= num_to_eat):
   - Receives input data
   - Immediately outputs to `output` port
   - No triggers sent

### Trigger Modes
- **"every_eat"**: Sends a trigger for each consumed input (up to num_to_eat triggers total)
- **"last_only"**: Sends a single trigger only when the last input is consumed (when counter reaches num_to_eat)

## Use Cases

### 1. RL Bootstrap (num_to_eat=1)
In reinforcement learning, the first observation obs(0) needs special handling:
```
obs(0) → Eat_N → (consumed, trigger sent)
obs(1) → Eat_N → output:obs(1) (passthrough begins)
obs(2) → Eat_N → output:obs(2) (continues passthrough)
```

The trigger from consuming obs(0) can release held data in Barrier nodes to start the training loop.

### 2. Warm-up Period (num_to_eat=N)
Skip the first N samples during warm-up:
```
data[0..N-1] → Eat_N → (consumed, triggers sent if configured)
data[N..∞] → Eat_N → output:data[N..∞]
```

### 3. Temporal Shift
Create a shifted data stream by eating the first observation:
- Original stream: obs(0), obs(1), obs(2), ...
- After Eat_N(1): obs(1), obs(2), obs(3), ...

## Integration with Barrier Nodes

Eat_N and Barrier nodes work together for synchronization:

```
obs → Eat_N ──trigger──→ Barrier.release
       ↓                       ↑
    output                  (held data)
       ↓                       ↑
    Loss ←─────────────── Network
```

The trigger from Eat_N releases data held in Barrier nodes, ensuring proper temporal alignment between observations and loss computation.

## Export Behavior

When exported, the Eat_N node generates:
- Stateful counter tracking consumed inputs
- Conditional logic for consume vs passthrough behavior
- Trigger emission based on configured mode
- Async queue-based implementation for real-time performance

## Example Configuration

### Basic RL Bootstrap
```python
{
    "class_type": "Eat_N",
    "inputs": {
        "input": ["node_25", "observation"],
        "num_to_eat": 1,
        "trigger_mode": "every_eat"
    }
}
```

### Multi-step Warm-up
```python
{
    "class_type": "Eat_N",
    "inputs": {
        "input": ["node_10", "data"],
        "num_to_eat": 10,
        "trigger_mode": "last_only"
    }
}
```

## Implementation Notes

- The node maintains state across the entire workflow execution
- Counter is never reset once num_to_eat is reached
- Triggers are fire-and-forget signals (no acknowledgment required)
- Compatible with batch processing (processes entire batch as one unit)
- Thread-safe for async queue-based execution

## Related Nodes
- [Barrier](barrier_node.md) - Works with Eat_N for synchronization
- [Split](../data/split_node.md) - Often used before Eat_N to distribute data
- [Concat](../data/concat_node.md) - Aggregates data after synchronization