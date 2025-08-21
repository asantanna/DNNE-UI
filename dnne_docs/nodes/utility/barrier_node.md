# Barrier Node

## Overview
The Barrier node is a synchronization primitive that holds incoming data in a queue until triggered to release it. It enables precise control over data flow timing, essential for coordinating multiple pipeline stages in reinforcement learning and other temporal workflows.

## Purpose
- Hold observations until gradients are applied from previous timestep
- Synchronize data flow between asynchronous pipeline stages
- Maintain temporal ordering with FIFO queue semantics
- Enable controlled release of data based on external triggers

## Node Configuration

### Category
`utility`

### Inputs
| Name | Type | Description |
|------|------|-------------|
| input | *TENSOR | Any tensor data to hold in the queue |
| release | *TRIGGER | Trigger signal to release held data |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| output | HELD_TENSOR | Released data from the queue |

### Widgets
| Name | Type | Default | Description |
|------|------|---------|-------------|
| hold_mode | ENUM | "FIFO" | Queue management strategy (currently only FIFO supported) |

## Behavior

### Internal State
- **FIFO Queue**: Holds incoming data in arrival order
- **release_count**: Integer counter tracking pending releases (starts at 0)

### Processing Algorithm

#### On Data Arrival (input port):
```python
1. Add data to FIFO queue
2. While release_count > 0 and queue not empty:
   a. Remove oldest item from queue
   b. Send item to output port
   c. Decrement release_count
```

#### On Trigger Arrival (release port):
```python
1. Increment release_count
2. While release_count > 0 and queue not empty:
   a. Remove oldest item from queue
   b. Send item to output port
   c. Decrement release_count
```

### Key Properties
- **Triggers are counted**: If a trigger arrives when queue is empty, it's remembered for future data
- **Order preserved**: FIFO mode ensures data exits in the same order it entered
- **No data loss**: All data is eventually released (assuming sufficient triggers)
- **Backpressure**: Can accumulate data if triggers arrive slower than data

## Use Cases

### 1. RL Gradient Synchronization
Ensure networks don't process new observations until gradients from previous step are applied:
```
obs(t) → Barrier (hold) ← SGD.step_complete (trigger)
            ↓
        Network → action
```

### 2. Multi-Network Coordination
Synchronize multiple networks to process data in lockstep:
```
obs → Split → [Barrier1, Barrier2, Barrier3]
                   ↓ (all triggered together)
              [Network1, Network2, Network3]
```

### 3. Pipeline Stage Synchronization
Control when data flows between pipeline stages:
```
Stage1 → Barrier ← Controller.ready
           ↓
        Stage2
```

## Integration with Eat_N Node

Barrier and Eat_N nodes form a powerful synchronization pattern:

```
Initial State:
obs(0) → Barrier (holds) ← Eat_N.trigger (releases)
           ↓
        Network

Steady State:
obs(t) → Barrier (holds) ← SGD.step_complete (releases)
           ↓
        Network
```

The Eat_N trigger bootstraps the system, then SGD.step_complete maintains synchronization.

## Hold Modes

### FIFO (First In, First Out)
Current implementation - maintains strict ordering:
- Data released in order of arrival
- Suitable for temporal sequences
- Preserves causality in RL workflows

### Future Modes (Planned)
- **LIFO**: Last In, First Out for stack-based processing
- **Priority**: Release based on data priority values
- **Random**: Random selection for sampling scenarios

## Export Behavior

When exported, the Barrier node generates:
- Queue data structure (collections.deque for FIFO)
- State management for release_count
- Async handlers for both input and release ports
- Thread-safe queue operations
- Proper memory management for held data

## Example Configuration

### Basic RL Synchronization
```python
{
    "class_type": "Barrier",
    "inputs": {
        "input": ["node_45", "split_output_1"],
        "release": ["node_80", "trigger"],
        "hold_mode": "FIFO"
    }
}
```

### Multi-trigger Setup
```python
{
    "class_type": "Barrier",
    "inputs": {
        "input": ["node_25", "observation"],
        "release": [
            ["node_80", "trigger"],      # Eat_N trigger
            ["node_40", "step_complete"]  # SGD completion
        ],
        "hold_mode": "FIFO"
    }
}
```

## Implementation Notes

- Queue has no size limit (bounded by available memory)
- Release triggers are additive (each trigger releases one item)
- Empty queue with pending releases is valid (triggers are remembered)
- Compatible with batch processing (entire batch treated as one queue item)
- Thread-safe for concurrent access in async execution
- Maintains gradient flow for backpropagation

## Edge Cases

### Multiple Triggers, No Data
```
Trigger → release_count = 1
Trigger → release_count = 2
Data → immediately released, release_count = 1
Data → immediately released, release_count = 0
```

### Data Accumulation
```
Data1 → queue = [Data1]
Data2 → queue = [Data1, Data2]
Trigger → output:Data1, queue = [Data2]
```

### Continuous Flow
```
Data + Trigger (simultaneous) → immediate passthrough
```

## Performance Considerations

- FIFO queue implemented with collections.deque for O(1) operations
- Minimal memory overhead per queued item
- No data copying (references maintained)
- Efficient trigger counting (simple integer increment)
- Scales linearly with queue depth

## Related Nodes
- [Eat_N](eat_n_node.md) - Provides initial triggers for Barrier
- [SGDOptimizer](../ml/sgd_optimizer.md) - Sends step_complete triggers
- [Split](../data/split_node.md) - Distributes data to multiple Barriers
- [Concat](../data/concat_node.md) - Aggregates outputs from multiple Barriers