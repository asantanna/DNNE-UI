# PPOAgent Node

## Overview
The PPOAgent node implements a Proximal Policy Optimization agent for reinforcement learning, providing stable and efficient policy gradient training.

## Properties

- **Category**: `rl`
- **Color Scheme**: RL nodes
- **Implementation**: `custom_nodes/ppo_agent_visnode.py`

## Inputs

### Required Inputs
- **config** (PPO_CONFIG)
  - PPO hyperparameter configuration
  - From PPOConfig node

- **observation_space** (SPACE)
  - Environment observation space specification
  - Defines input dimensions for networks

- **action_space** (SPACE)
  - Environment action space specification
  - Defines output dimensions and action type

### Optional Inputs
- **actor_network** (NETWORK)
  - Custom actor (policy) network architecture
  - If not provided, creates default MLP

- **critic_network** (NETWORK)
  - Custom critic (value) network architecture
  - If not provided, creates default MLP

## Outputs

- **agent** (PPO_AGENT)
  - Trained PPO agent instance
  - Can select actions and update policy

- **policy** (POLICY)
  - Current policy network
  - For action selection and export

- **value_fn** (VALUE_FUNCTION)
  - Value function network
  - For advantage estimation

## Functionality

### Core PPO Algorithm
1. **Experience Collection**: Gather trajectories from environment
2. **Advantage Calculation**: Compute GAE advantages
3. **Policy Update**: Clipped objective to prevent large updates
4. **Value Update**: Minimize value function prediction error
5. **Entropy Bonus**: Encourage exploration

### Key Components
- **Actor Network**: Outputs action distribution parameters
- **Critic Network**: Estimates state values for advantage calculation
- **Experience Buffer**: Stores trajectories for batch updates
- **Optimizer**: Separate optimizers for actor and critic

## PPO Loss Functions

### Policy Loss (Clipped)
```python
ratio = new_prob / old_prob
clipped_ratio = torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
```

### Value Loss
```python
value_loss = (returns - value_predictions).pow(2).mean()
```

### Total Loss
```python
total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
```

## Usage Example

### In Visual Workflow
1. Create PPOConfig with hyperparameters
2. Connect IsaacGymEnvs spaces to PPOAgent
3. PPOAgent handles training loop internally
4. Export generates complete training script

### Training Loop Integration
```
PPOConfig → config
IsaacGymEnvs → observation_space, action_space
         ↓
    PPOAgent
         ↓
  agent, policy
         ↓
Environment interaction loop
```

### Exported Python Code
```python
class PPOAgentNode(QueueNode):
    def __init__(self, config, observation_space, action_space):
        super().__init__()
        self.agent = PPOAgent(
            config=config,
            observation_space=observation_space,
            action_space=action_space
        )
        
    async def train_step(self, observations, rewards, dones):
        # Collect experience
        actions = self.agent.select_actions(observations)
        
        # Store in buffer
        self.agent.store_transition(observations, actions, rewards, dones)
        
        # Update policy when buffer full
        if self.agent.buffer_full():
            losses = self.agent.update()
            await self.output_queue.put({"losses": losses})
        
        return actions
```

## Network Architectures

### Default Actor Network
- Input: Observation dimension
- Hidden: [256, 256] with ReLU
- Output: Action distribution parameters
- Initialization: Orthogonal

### Default Critic Network
- Input: Observation dimension
- Hidden: [256, 256] with ReLU
- Output: Single value estimate
- Initialization: Orthogonal

## Best Practices

1. **Hyperparameter Tuning**: Start with defaults, tune systematically
2. **Normalization**: Normalize observations and rewards
3. **Parallel Environments**: Use 16+ environments for stability
4. **Learning Rate Schedule**: Decay learning rate over training
5. **Early Stopping**: Monitor performance plateau

## Common Issues

- **Exploding KL Divergence**: Reduce learning rate or clip range
- **Poor Sample Efficiency**: Increase replay ratio or buffer size
- **Unstable Training**: Check reward scaling and normalization
- **Mode Collapse**: Increase entropy coefficient

## Performance Metrics

Monitor these during training:
- **Policy Loss**: Should decrease but may fluctuate
- **Value Loss**: Should decrease steadily
- **Entropy**: Should decrease slowly (not too fast)
- **KL Divergence**: Should stay below target (e.g., 0.01)
- **Explained Variance**: Should approach 1.0

## Related Nodes

- [PPOConfig](ppo_config.md) - Hyperparameter configuration
- [IsaacGymEnvs](../robotics/isaac_gym_envs.md) - RL environments
- [Network](../ml/network.md) - Custom network architectures