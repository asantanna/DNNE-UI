# Temporal Alignment Pattern for Reinforcement Learning

## Problem Statement

In reinforcement learning workflows, computing loss or rewards requires access to both:
- The observation that led to an action: obs(t)
- The resulting observation after the action: obs(t+1)

However, in a reactive dataflow system, these observations arrive at different times, creating a temporal alignment challenge. Additionally, gradient updates must be synchronized to prevent networks from processing new observations before learning from previous experiences.

## Solution Architecture

The temporal alignment pattern uses two key synchronization nodes:
- **Eat_N Node**: Consumes initial observation and provides bootstrap trigger
- **Barrier Node**: Holds observations until triggered to release

Together, these nodes create a synchronized pipeline where loss computation receives properly aligned temporal pairs.

## Workflow Structure

### Components
1. **Simulator**: Produces observations based on actions
2. **Networks** (3 independent): Process observations to generate actions
3. **Eat_N**: Handles bootstrap and creates temporal shift
4. **Barriers** (3, one per network): Hold observations until triggered
5. **Loss Computation**: Calculates loss from obs(t+1) after action
6. **SGD Optimizers** (3): Apply gradients and trigger next cycle

### Data Flow Diagram

```
                    ┌─────────────────────────────────┐
                    │       Isaac Gym Simulator        │
                    │    obs(t) = state at time t     │
                    └─────────────┬───────────────────┘
                                  │
                        ┌─────────┴─────────┐
                        │      Split        │
                        └──┬──────────────┬─┘
                           │              │
                    (To Barriers)    (To Eat_N)
                           │              │
                ┌──────────┼──────────┐   │
                ↓          ↓          ↓   ↓
         [Barrier1] [Barrier2] [Barrier3] [Eat_N]
            (hold)    (hold)     (hold)     │
                                            │ (eat obs(0), then pass)
                                            ↓
                                    [Loss Computation]
                                            │
                              ┌─────────────┼─────────────┐
                              ↓             ↓             ↓
                          [SGD1]        [SGD2]        [SGD3]
                              │             │             │
                        step_complete step_complete step_complete
                              │             │             │
                              └─────────────┴─────────────┘
                                            │
                    (Triggers + initial Eat_N trigger release Barriers)
                              │             │             │
                              ↓             ↓             ↓
                        [Barrier1]    [Barrier2]    [Barrier3]
                          release      release      release
                              ↓             ↓             ↓
                        [Network1]    [Network2]    [Network3]
                              ↓             ↓             ↓
                              └─────────────┴─────────────┘
                                            │
                                        [Concat]
                                            │
                                        action(t)
                                            │
                                      [Simulator]
                                            │
                                        obs(t+1)
```

## Execution Timeline

### Initial Bootstrap (t=0)

1. **Simulator** produces obs(0)
2. **Split** distributes obs(0) to:
   - 3 Barrier nodes → held in FIFO queues
   - Eat_N node → consumed (not passed through)
3. **Eat_N** sends trigger to all 3 Barriers
4. **Barriers** release obs(0) to Networks
5. **Networks** process obs(0) → generate actions
6. **Concat** combines actions → send to Simulator
7. **Simulator** produces obs(1)

### Steady State (t≥1)

For each timestep t:

1. **Simulator** produces obs(t)
2. **Split** distributes obs(t) to:
   - 3 Barrier nodes → held in queues
   - Eat_N node → passed through to Loss
3. **Loss** computes loss from obs(t)
   - Note: This obs(t) is the result of action(t-1)
   - Networks that produced action(t-1) used obs(t-1)
   - So loss correctly captures transition (obs(t-1), action(t-1), obs(t))
4. **SGD Optimizers** receive loss:
   - Compute gradients
   - Update network weights
   - Send step_complete triggers
5. **Barriers** receive triggers → release held obs(t)
6. **Networks** process obs(t) with updated weights
7. **Concat** → **Simulator** → obs(t+1)

## Key Insights

### Temporal Shift
The Eat_N node creates a one-step temporal shift:
- Networks process obs(t-1)
- Loss sees obs(t) (result of action from obs(t-1))
- This naturally aligns the (state, action, next_state) tuple

### Synchronization Points
Two critical synchronization mechanisms:
1. **Bootstrap**: Eat_N trigger starts the pipeline
2. **Steady State**: SGD step_complete maintains synchronization

### Gradient Consistency
Networks don't see new observations until:
- Previous observation's action has been executed
- Resulting observation has been used for loss
- Gradients have been computed and applied

## Configuration Example

### Eat_N Node Setup
```json
{
  "class_type": "Eat_N",
  "inputs": {
    "input": ["split_node", "output_2"],
    "num_to_eat": 1,
    "trigger_mode": "every_eat"
  }
}
```

### Barrier Node Setup (per network)
```json
{
  "class_type": "Barrier",
  "inputs": {
    "input": ["split_node", "output_1"],
    "release": [
      ["eat_n_node", "trigger"],
      ["sgd_node", "step_complete"]
    ],
    "hold_mode": "FIFO"
  }
}
```

## Benefits

1. **Correct Temporal Alignment**: Loss computation sees proper (s, a, s') transitions
2. **No Manual Buffering**: Synchronization handled by reusable nodes
3. **Automatic Bootstrap**: Eat_N handles the special case of first observation
4. **Gradient Synchronization**: Networks always use latest weights
5. **Scalable**: Pattern works for any number of parallel networks

## Common Pitfalls

### Wrong: Direct Connection
```
obs(t) → Networks → action → Simulator → obs(t+1)
   ↓
Loss (sees obs(t), not obs(t+1))
```
Problem: Loss computed from wrong observation

### Wrong: No Synchronization
```
obs(t) → Networks → action
   ↓
SGD (updates while network processes next obs)
```
Problem: Race condition between gradient updates and forward pass

### Wrong: Manual Buffering in Loss
```python
def compute_loss(obs):
    # Don't do this - use Eat_N + Barrier instead
    if not hasattr(self, 'prev_obs'):
        self.prev_obs = obs
        return 0
    loss = calculate(self.prev_obs, obs)
    self.prev_obs = obs
    return loss
```
Problem: Stateful loss computation is error-prone and non-reusable

## Variations

### Single Network
For single network scenarios, use one Barrier:
```
obs → Barrier ← (Eat_N.trigger + SGD.step_complete)
        ↓
     Network
```

### Multiple Losses
For per-network losses, split after Eat_N:
```
obs → Eat_N → Split → [Loss1, Loss2, Loss3]
                          ↓      ↓      ↓
                       [SGD1, SGD2, SGD3]
```

### Delayed Rewards
For N-step returns, use Eat_N with num_to_eat=N:
```
obs → Eat_N(N) → N-step-return → Loss
```

## Implementation Checklist

- [ ] Add Eat_N node after observation split
- [ ] Set num_to_eat=1 for standard RL
- [ ] Connect Eat_N output to loss computation
- [ ] Add Barrier before each network
- [ ] Connect Eat_N.trigger to all Barriers
- [ ] Connect SGD.step_complete to corresponding Barrier
- [ ] Verify loss receives obs(t+1) not obs(t)
- [ ] Test bootstrap with initial observation
- [ ] Verify gradient synchronization

## Related Documentation

- [Eat_N Node](../nodes/utility/eat_n_node.md)
- [Barrier Node](../nodes/utility/barrier_node.md)
- [SGD Optimizer](../nodes/ml/sgd_optimizer.md)
- [Split Node](../nodes/data/split_node.md)
- [Concat Node](../nodes/data/concat_node.md)