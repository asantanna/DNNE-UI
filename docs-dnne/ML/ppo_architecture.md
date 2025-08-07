# PPO (Proximal Policy Optimization) Architecture in DNNE

## Overview

Proximal Policy Optimization (PPO) is a state-of-the-art reinforcement learning algorithm that strikes an excellent balance between sample efficiency, stability, and ease of implementation. DNNE provides a complete PPO implementation through its node-based visual programming system.

## Algorithm Details

### Core Concept

PPO improves upon vanilla policy gradient methods by preventing large policy updates that could destabilize training. It uses a clipped surrogate objective function to ensure the new policy doesn't deviate too far from the old policy.

### Mathematical Foundation

#### Policy Gradient Objective
The standard policy gradient objective is:
```
L^PG(θ) = E_t[log π_θ(a_t|s_t) * A_t]
```

#### PPO Clipped Objective
PPO modifies this to:
```
L^CLIP(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio
- `ε` is the clip range (typically 0.2)
- `A_t` is the advantage estimate

### Key Components

#### 1. Actor Network (Policy)
- **Input**: Observation from environment
- **Output**: Action distribution parameters
- **Architecture**: Typically MLP with 2-3 hidden layers
- **Activation**: ReLU or Tanh

#### 2. Critic Network (Value Function)
- **Input**: Observation from environment
- **Output**: State value estimate V(s)
- **Architecture**: Similar to actor, often shared backbone
- **Purpose**: Baseline for advantage calculation

#### 3. Advantage Estimation (GAE)
Generalized Advantage Estimation balances bias and variance:
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```
Where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`

## DNNE Implementation

### Node Architecture

```
PPOConfig → configuration
     ↓
PPOAgent ← observation_space, action_space
     ↓
  policy, value_fn
```

### PPOAgent Node Features

1. **Automatic Network Construction**
   - Creates actor and critic networks based on space dimensions
   - Supports continuous and discrete action spaces
   - Configurable hidden layer sizes

2. **Experience Buffer**
   - Stores trajectories for batch updates
   - Implements efficient replay mechanisms
   - Handles variable episode lengths

3. **Loss Computation**
   ```python
   # Policy loss with clipping
   ratio = new_prob / old_prob
   clipped = torch.clamp(ratio, 1-clip_range, 1+clip_range)
   policy_loss = -torch.min(ratio * advantages, clipped * advantages)
   
   # Value loss
   value_loss = (returns - values).pow(2)
   
   # Entropy bonus for exploration
   entropy = -torch.sum(probs * log_probs)
   
   # Combined loss
   total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
   ```

### Integration with Isaac Gym

PPO in DNNE is optimized for robotics simulation:

1. **Parallel Environments**
   - Runs thousands of environments simultaneously
   - GPU-accelerated physics and learning
   - Efficient batch processing

2. **Observation Processing**
   - Automatic normalization
   - Running statistics for mean/std
   - Configurable observation stacking

3. **Action Scaling**
   - Maps network outputs to action space bounds
   - Supports continuous and discrete actions
   - Handles multi-dimensional action spaces

## Hyperparameters

### Learning Parameters
- **Learning Rate**: 3e-4 (adaptive scheduling available)
- **Clip Range (ε)**: 0.2 (prevents large policy updates)
- **Value Coefficient**: 0.5 (weight for value loss)
- **Entropy Coefficient**: 0.01 (exploration bonus)

### Training Parameters
- **Batch Size**: 32-64 per environment
- **Mini-batch Size**: 32-256 for SGD updates
- **Epochs per Update**: 4-10 passes over data
- **Horizon (T)**: 128-2048 steps per rollout

### GAE Parameters
- **Gamma (γ)**: 0.99 (discount factor)
- **Lambda (λ)**: 0.95 (GAE parameter)

## Training Process

### 1. Rollout Collection
```python
for step in range(horizon):
    # Get action from policy
    action = policy.sample(observation)
    
    # Step environment
    next_obs, reward, done = env.step(action)
    
    # Store transition
    buffer.add(obs, action, reward, done, value)
```

### 2. Advantage Calculation
```python
# Bootstrap value for incomplete episodes
last_value = critic(last_obs)

# Calculate returns and advantages using GAE
for t in reversed(range(horizon)):
    if done[t]:
        next_value = 0
    else:
        next_value = values[t + 1]
    
    delta = rewards[t] + gamma * next_value - values[t]
    advantages[t] = delta + gamma * lambda * advantages[t + 1]
```

### 3. Policy Update
```python
for epoch in range(update_epochs):
    for batch in buffer.get_batches(mini_batch_size):
        # Compute losses
        policy_loss, value_loss, entropy = compute_losses(batch)
        
        # Backpropagation
        total_loss.backward()
        
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        
        # Update weights
        optimizer.step()
```

## Best Practices

### 1. Environment Design
- **Reward Shaping**: Design informative rewards
- **Observation Space**: Include relevant state information
- **Action Space**: Appropriate discretization or bounds
- **Episode Length**: Balance exploration and efficiency

### 2. Network Architecture
- **Shared Backbone**: Share features between actor and critic
- **Initialization**: Use orthogonal or Xavier initialization
- **Normalization**: Layer norm or batch norm for stability
- **Activation**: ReLU for hidden, Tanh for outputs

### 3. Training Stability
- **Gradient Clipping**: Prevent exploding gradients
- **Learning Rate Schedule**: Decay over time
- **Early Stopping**: Monitor performance plateau
- **Checkpointing**: Save models regularly

### 4. Hyperparameter Tuning
- **Start Conservative**: Use proven defaults
- **Systematic Search**: Change one parameter at a time
- **Monitor Metrics**: Track KL divergence, explained variance
- **Environment-Specific**: Tune per task requirements

## Common Issues and Solutions

### Issue: Exploding KL Divergence
**Solution**: Reduce learning rate or clip range

### Issue: Poor Sample Efficiency
**Solution**: Increase batch size or replay ratio

### Issue: Unstable Training
**Solution**: Check reward scaling, use gradient clipping

### Issue: Slow Convergence
**Solution**: Increase learning rate or entropy coefficient

## Performance Metrics

Monitor these during training:
- **Episode Reward**: Primary performance metric
- **Policy Loss**: Should decrease with fluctuations
- **Value Loss**: Should steadily decrease
- **Entropy**: Should slowly decrease
- **KL Divergence**: Should stay < 0.01-0.02
- **Explained Variance**: Should approach 1.0

## Export and Deployment

DNNE's PPO implementation exports to:
1. **Standalone Python**: Complete training scripts
2. **Isaac Gym Integration**: Direct simulator control
3. **Real Robot Deployment**: Via ROS bridge
4. **Cloud Training**: Distributed PPO on clusters

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original Proximal Policy Optimization
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) - GPU Physics Simulation
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - PPO Tutorial