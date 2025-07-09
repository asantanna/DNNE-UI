# Reinforcement Learning PPO Nodes

## Overview

PPO (Proximal Policy Optimization) nodes implement state-of-the-art reinforcement learning in DNNE. These nodes work together to create complete RL training systems for robotics and control tasks.

---

## PPOAgentNode

### Purpose
Implements an Actor-Critic neural network that outputs both policy actions and value estimates for PPO training.

### Category
`RL/PPO`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| observations | Tensor | Environment state observations |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input_size | int | Required | Observation space dimension |
| output_size | int | Required | Action space dimension |
| hidden_sizes | list[int] | [64, 64] | Hidden layer dimensions |
| activation | string | "elu" | Activation function |
| learning_rate | float | 0.0003 | Learning rate for optimizer |
| action_space | string | "continuous" | "continuous" or "discrete" |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| policy_output | PolicyOutput | Bundle of action, value, log_prob |
| model | nn.Module | The actor-critic network |

### PolicyOutput Structure
```python
class PolicyOutput:
    action: Tensor        # Sampled action to execute
    value: Tensor        # Estimated state value
    log_prob: Tensor     # Log probability of action
    action_params: dict  # Distribution parameters (mean, std)
```

### Architecture Details

#### Shared Backbone
```
Observations → Linear(input_size, hidden[0]) → Activation →
               Linear(hidden[0], hidden[1]) → Activation
                            ↓
                    ┌───────┴────────┐
                    ↓                ↓
               Actor Head       Critic Head
```

#### Actor Head (Policy)
- **Continuous**: Outputs mean and log_std for Gaussian
- **Discrete**: Outputs action logits for Categorical

#### Critic Head (Value)
- Single output estimating state value
- Used for advantage computation

### Usage Example
```
IsaacGymEnv → ORNode → PPOAgent → ActionProcessor → IsaacGymStep
                           ↓
                    PPOTrainer (uses model output)
```

### Export Support
✅ Full support with queue template
- Handles both continuous and discrete actions
- Maintains internal optimizer state
- Thread-safe for async execution

### Configuration Examples

#### Continuous Control (Robotics)
```python
PPOAgentNode(
    input_size=24,      # Joint positions/velocities
    output_size=8,      # Joint torques
    hidden_sizes=[256, 256],
    activation="elu",
    action_space="continuous"
)
```

#### Discrete Control (Games)
```python
PPOAgentNode(
    input_size=84*84,   # Flattened image
    output_size=4,      # Up/Down/Left/Right
    hidden_sizes=[512, 256],
    activation="relu",
    action_space="discrete"
)
```

### Notes
- Automatically initializes weights appropriately
- Supports GPU acceleration
- Action sampling includes exploration noise
- Value estimates used for advantage calculation

---

## PPOTrainerNode

### Purpose
Implements the complete PPO training algorithm including trajectory collection, advantage estimation, and policy updates.

### Category
`RL/PPO`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| states | Tensor | Observations/states |
| policy_outputs | PolicyOutput | Agent's outputs |
| rewards | Tensor | Environment rewards |
| dones | Tensor | Episode termination flags |
| model | nn.Module | Actor-critic network |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| horizon_length | int | 2048 | Steps before update |
| epochs | int | 10 | PPO update epochs |
| minibatch_size | int | 64 | Batch size for updates |
| learning_rate | float | 0.0003 | Optimizer learning rate |
| gamma | float | 0.99 | Discount factor |
| lam | float | 0.95 | GAE lambda |
| clip_param | float | 0.2 | PPO clipping parameter |
| value_coef | float | 0.5 | Value loss coefficient |
| entropy_coef | float | 0.01 | Entropy bonus |
| max_grad_norm | float | 0.5 | Gradient clipping |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| training_complete | Trigger | Signals update completion |
| loss_info | dict | Training statistics |

### Algorithm Flow

1. **Trajectory Collection**
   ```python
   buffer = []
   for step in range(horizon_length):
       buffer.append((state, action, reward, value, log_prob))
   ```

2. **Advantage Computation (GAE)**
   ```python
   advantages = compute_gae(rewards, values, dones, gamma, lam)
   returns = advantages + values
   ```

3. **PPO Update**
   ```python
   for epoch in range(epochs):
       for batch in get_minibatches(buffer):
           # Compute ratio
           ratio = exp(new_log_prob - old_log_prob)
           
           # Clipped objective
           surr1 = ratio * advantages
           surr2 = clip(ratio, 1-ε, 1+ε) * advantages
           policy_loss = -min(surr1, surr2).mean()
           
           # Value loss
           value_loss = MSE(new_values, returns)
           
           # Total loss
           loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
   ```

### Usage Example
```
[States, PolicyOutputs, Rewards, Dones, Model] → PPOTrainer
                                                      ↓
                                              training_complete → StepEnv
```

### Export Support
✅ Full support with comprehensive features
- Async trajectory collection
- Efficient minibatch sampling
- Automatic advantage normalization
- GPU-optimized operations

### Hyperparameter Guidelines

#### Stable Training (Default)
```python
PPOTrainerNode(
    horizon_length=2048,
    epochs=10,
    minibatch_size=64,
    clip_param=0.2
)
```

#### Fast Learning (Aggressive)
```python
PPOTrainerNode(
    horizon_length=512,    # Faster updates
    epochs=15,            # More learning
    clip_param=0.3,       # Larger updates
    learning_rate=0.0005  # Higher LR
)
```

#### Sample Efficient (Conservative)
```python
PPOTrainerNode(
    horizon_length=4096,   # More data
    epochs=5,             # Fewer updates
    clip_param=0.1,       # Smaller changes
    minibatch_size=256    # Larger batches
)
```

### Advanced Features

#### Advantage Normalization
- Automatically normalizes advantages per batch
- Stabilizes training across different reward scales

#### Learning Rate Scheduling
```python
# Linearly decay learning rate
lr_schedule = lambda progress: learning_rate * (1 - progress)
```

#### Early Stopping
- Monitors KL divergence between old and new policy
- Stops epoch early if divergence too large

### Notes
- Buffer automatically manages memory
- Supports recurrent policies (future)
- Includes comprehensive logging
- Compatible with distributed training

---

## PPO Workflow Patterns

### Basic PPO Loop
```
Environment → ORNode → PPOAgent → Action → Step
                ↑                            ↓
                └─── PPOTrainer ←────────────┘
```

### Multi-Environment PPO
```
Env1 → ORNode1 ↘
Env2 → ORNode2 → PPOAgent → Actions → Steps
Env3 → ORNode3 ↗                          ↓
                                    PPOTrainer
```

### Hierarchical PPO
```
HighLevelPPO → SubGoal → LowLevelPPO → Actions → Environment
```

## Best Practices

### Network Architecture
- **Simple tasks**: [64, 64] hidden layers
- **Complex tasks**: [256, 256] or deeper
- **Vision tasks**: Add CNN feature extractor
- **Continuous**: Separate std parameters

### Hyperparameter Tuning
1. Start with defaults
2. Adjust horizon based on episode length
3. Increase epochs if sample efficient needed
4. Tune clip_param based on policy change

### Training Stability
- Use advantage normalization
- Monitor value function accuracy
- Check entropy for exploration
- Watch for policy collapse

### Performance Optimization
- Use larger minibatches for GPU
- Vectorize environments
- Profile trajectory collection
- Consider mixed precision

## Common Issues

### Issue: Policy not improving
**Solution**: 
- Increase learning rate
- Check reward scaling
- Verify environment reset
- Add entropy bonus

### Issue: Training unstable
**Solution**:
- Reduce learning rate
- Decrease clip parameter
- Use gradient clipping
- Check for reward outliers

### Issue: High variance returns
**Solution**:
- Increase horizon length
- Use more environments
- Tune GAE lambda
- Normalize observations

### Issue: Slow training
**Solution**:
- Parallelize environments
- Increase batch size
- Use GPU if available
- Profile bottlenecks

## Integration Examples

### With Isaac Gym
```python
# Typical hyperparameters for robotics
PPOTrainerNode(
    horizon_length=16,    # Short for reactive control
    epochs=4,            # Quick updates
    gamma=0.99,          # Standard discount
    clip_param=0.2       # Moderate clipping
)
```

### With Atari Games
```python
# Game-specific settings
PPOTrainerNode(
    horizon_length=128,   # Episode-based
    epochs=4,
    entropy_coef=0.01,   # Exploration important
    value_coef=1.0       # Strong value learning
)
```

## Monitoring and Debugging

### Key Metrics to Track
- **Policy Loss**: Should decrease but not too fast
- **Value Loss**: Indicates value function quality
- **Entropy**: Measures exploration (shouldn't collapse)
- **KL Divergence**: Policy change magnitude
- **Episode Rewards**: Ultimate performance metric

### Debugging Tools
```python
# Add to training loop
print(f"Policy Loss: {policy_loss:.4f}")
print(f"Value Loss: {value_loss:.4f}")
print(f"Entropy: {entropy:.4f}")
print(f"Mean Reward: {rewards.mean():.4f}")
```

## Future Enhancements

- Recurrent policy support (LSTM/GRU)
- Multi-agent PPO
- Curiosity-driven exploration
- Automated hyperparameter tuning
- Distributed training support