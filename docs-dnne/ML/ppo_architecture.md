# PPO Node Architecture in DNNE

## Overview

DNNE provides two specialized nodes for implementing Proximal Policy Optimization (PPO) - one of the most popular and effective reinforcement learning algorithms. These nodes integrate seamlessly with Isaac Gym and other RL environments.

## PPOAgentNode

### Purpose
The PPOAgentNode implements the Actor-Critic neural network that forms the core of PPO. It outputs both policy actions and value estimates needed for the PPO algorithm.

### Architecture

```python
class PPOAgentNode:
    def __init__(self, input_size, output_size, hidden_sizes, activation, learning_rate):
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            activation(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_sizes[-1], output_size)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_sizes[-1], 1)
```

### Key Features

1. **Dual Output Design**
   - Actor head: Produces action distributions
   - Critic head: Estimates state values
   - Shared feature extraction layers

2. **Action Space Support**
   - Continuous: Outputs mean and log_std for Gaussian distribution
   - Discrete: Outputs action logits for categorical distribution

3. **Configurable Architecture**
   - Variable hidden layer sizes
   - Choice of activation functions (ReLU, ELU, Tanh)
   - Adjustable learning rate

### Outputs

The node produces a `PolicyOutput` object containing:
```python
class PolicyOutput:
    action: torch.Tensor          # Sampled action
    value: torch.Tensor          # Value estimate
    log_prob: torch.Tensor       # Log probability of action
    action_params: dict          # Distribution parameters (mean, std, etc.)
```

### Usage Example

```python
# In visual workflow or code
agent = PPOAgentNode(
    input_size=4,           # Cartpole observation space
    output_size=1,          # Single continuous action
    hidden_sizes=[64, 64],  # Two hidden layers
    activation="elu",       # ELU activation
    learning_rate=0.0003    # Learning rate for optimizer
)
```

## PPOTrainerNode

### Purpose
The PPOTrainerNode implements the complete PPO training algorithm, including trajectory collection, advantage estimation, and policy updates.

### Core Algorithm

1. **Trajectory Collection**
   ```python
   # Collect experiences for horizon steps
   for t in range(horizon_length):
       states.append(state)
       policy_outputs.append(agent(state))
       rewards.append(reward)
       dones.append(done)
   ```

2. **Advantage Computation**
   ```python
   # Generalized Advantage Estimation (GAE)
   advantages = compute_gae_advantages(
       rewards, values, dones, 
       gamma=0.99, lam=0.95
   )
   ```

3. **PPO Update**
   ```python
   # Multiple epochs over collected data
   for epoch in range(epochs):
       for batch in get_minibatches(data, minibatch_size):
           # Compute clipped surrogate loss
           ratio = torch.exp(new_log_probs - old_log_probs)
           clipped_ratio = torch.clamp(ratio, 1-clip, 1+clip)
           policy_loss = -torch.min(
               ratio * advantages,
               clipped_ratio * advantages
           ).mean()
   ```

### Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| horizon_length | 2048 | Steps before update |
| epochs | 10 | Training epochs per update |
| minibatch_size | 64 | Batch size for updates |
| learning_rate | 3e-4 | Optimizer learning rate |
| gamma | 0.99 | Discount factor |
| lam | 0.95 | GAE lambda |
| clip_param | 0.2 | PPO clipping parameter |
| value_coef | 0.5 | Value loss coefficient |
| entropy_coef | 0.01 | Entropy bonus coefficient |
| max_grad_norm | 0.5 | Gradient clipping threshold |

### Training Flow

1. **Buffer Management**
   - Maintains trajectory buffer of size `horizon_length`
   - Automatically triggers training when full

2. **Advantage Normalization**
   - Normalizes advantages per batch for stability
   - Prevents gradient explosion

3. **Loss Computation**
   - Policy loss (clipped surrogate)
   - Value loss (MSE between predictions and returns)
   - Entropy bonus (encourages exploration)

4. **Optimization**
   - Adam optimizer with configurable learning rate
   - Gradient clipping for stability
   - Separate or shared optimizers for actor/critic

### Output Signals

The trainer outputs control signals:
- `training_complete`: Triggered after each PPO update
- `episode_complete`: Triggered when episodes finish
- `checkpoint_ready`: (Future) When model should be saved

## Integration Patterns

### With Isaac Gym Environments

```python
# Typical connection pattern
IsaacGymEnv → ORNode → PPOAgent → ActionProcessor → IsaacGymStep
                ↑                                          ↓
                └──────────── PPOTrainer ←─────────────────┘
```

### With Custom Environments

PPO nodes work with any environment providing:
- Observations (tensor format)
- Rewards (scalar values)
- Done flags (boolean indicators)

### Multi-Environment Training

```python
# Parallel environments (future enhancement)
num_envs = 16
agent = PPOAgentNode(
    input_size=obs_shape,
    output_size=action_shape,
    # ... other params
)
# Agent automatically handles batched observations
```

## Best Practices

### Network Architecture

1. **Hidden Layer Sizing**
   - Start with [64, 64] for simple tasks
   - Increase to [256, 256] for complex observations
   - Deeper networks for image-based inputs

2. **Activation Functions**
   - ELU: Good default choice
   - ReLU: Faster but may have dead neurons
   - Tanh: For bounded representations

### Hyperparameter Guidelines

1. **Learning Rate**
   - Start with 3e-4
   - Decrease if training unstable
   - Increase for faster initial learning

2. **Horizon Length**
   - Shorter (128-512) for simple tasks
   - Longer (2048-4096) for complex environments
   - Must balance memory usage

3. **PPO Epochs**
   - 3-10 epochs typical
   - More epochs risk overfitting
   - Fewer epochs waste data

4. **Clipping Parameter**
   - 0.1-0.3 range works well
   - Smaller values for conservative updates
   - Larger values for aggressive learning

## Advanced Features

### Adaptive Parameters

The nodes support dynamic adjustment:
```python
# Scheduled learning rate
scheduler = lambda epoch: 3e-4 * (0.99 ** epoch)

# Adaptive clipping
clip_schedule = lambda progress: 0.2 * (1 - progress)
```

### Custom Reward Shaping

```python
# Integrate reward processing nodes
RewardShaper → PPOTrainer
```

### Multi-Task Learning

```python
# Share agent across tasks
TaskSelector → PPOAgent → [Task1Step, Task2Step]
```

## Implementation Details

### Memory Efficiency

- Trajectory data stored in circular buffers
- Old data automatically discarded
- GPU memory managed automatically

### Computational Optimization

- Vectorized operations throughout
- Efficient minibatch sampling
- Parallel advantage computation

### Debugging Support

- Comprehensive logging system
- Training metrics tracking
- Gradient norm monitoring

## Common Issues and Solutions

### Issue: Training Instability

**Symptoms**: Loss explodes, rewards decrease
**Solutions**:
- Reduce learning rate
- Decrease horizon length
- Increase PPO epochs
- Check advantage normalization

### Issue: Slow Learning

**Symptoms**: Rewards plateau early
**Solutions**:
- Increase learning rate
- Add entropy bonus
- Larger network capacity
- Longer horizon length

### Issue: Memory Usage

**Symptoms**: OOM errors during training
**Solutions**:
- Reduce horizon length
- Smaller minibatch size
- Fewer parallel environments
- Use gradient checkpointing

## Future Enhancements

1. **Recurrent Policy Support**
   - LSTM/GRU integration for partial observability
   - Hidden state management

2. **Advanced Algorithms**
   - PPO with auxiliary tasks
   - Curiosity-driven exploration
   - Hindsight experience replay

3. **Distributed Training**
   - Multi-GPU support
   - Asynchronous updates
   - Experience sharing

4. **Automated Hyperparameter Tuning**
   - Population-based training
   - Bayesian optimization
   - Evolutionary strategies

The PPO nodes in DNNE provide a robust foundation for reinforcement learning, combining ease of use with production-ready performance. The visual workflow design makes it simple to experiment with different architectures while the export system ensures efficient execution.