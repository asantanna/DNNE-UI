# 4-Node PPO Decomposition

## Priority
Low

## Description
Decompose the current 2-node PPO implementation (PPOAgent + PPOTrainer) into 4 separate nodes for better visibility and understanding of the algorithm components.

## Motivation
- Current 2-node design hides algorithm details
- Educational value in seeing PPO components
- More granular control over algorithm
- Better debugging and experimentation

## Implementation Notes
### Proposed 4-Node Structure
1. **ActorCriticNetwork**: Neural network only
   - No action sampling
   - Outputs raw policy and value
   
2. **ActionSampler**: Sampling and log prob calculation
   - Takes policy output
   - Samples actions
   - Computes log probabilities
   
3. **PPOBuffer**: Trajectory storage
   - Collects experiences
   - Computes advantages
   - Provides minibatches
   
4. **PPOTrainer**: Training logic only
   - Computes losses
   - Updates network
   - No data collection

### Visual Flow
```
Obs → ActorCritic → ActionSampler → Environment
           ↓              ↓
        Value         Action+LogProb → PPOBuffer
                                           ↓
                                      PPOTrainer
```

## Trade-offs
### Pros
- Better algorithm visibility
- Educational value
- Fine-grained control
- Easier debugging

### Cons
- More async overhead
- Complex wiring
- Performance impact
- More nodes to manage

## Dependencies
- Current PPO implementation working well
- Performance optimization completed first

## Estimated Effort
Medium

## Success Metrics
- Maintain current performance
- Clearer algorithm understanding
- Same training results as 2-node version