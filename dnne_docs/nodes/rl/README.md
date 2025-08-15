# RL (Reinforcement Learning) Nodes Documentation

Reinforcement learning nodes for implementing PPO and other RL algorithms in DNNE.

## Available Nodes

### [PPOAgent](ppo_agent.md)
Proximal Policy Optimization agent for training RL policies.

### [PPOConfig](ppo_config.md)
Configuration node for PPO hyperparameters and training settings.

## Overview

The RL nodes in DNNE provide a complete implementation of Proximal Policy Optimization (PPO), one of the most popular and effective reinforcement learning algorithms. These nodes integrate seamlessly with Isaac Gym environments for robotics simulation.

## PPO Architecture

PPO uses two main components:
1. **Actor Network**: Outputs action probabilities (policy)
2. **Critic Network**: Estimates value function for advantage calculation

## Typical RL Workflow

1. **Environment Setup**: Isaac Gym environment provides observations and rewards
2. **PPO Configuration**: Set hyperparameters via PPOConfig node
3. **PPO Agent**: Handles policy updates and action selection
4. **Training Loop**: Collect experience, compute advantages, update policy

## Key Features

- **Clipped Objective**: Prevents large policy updates for stability
- **Value Function**: Critic network for advantage estimation
- **GAE**: Generalized Advantage Estimation for variance reduction
- **Mini-batch Updates**: Efficient use of collected experience

## Integration with Isaac Gym

The RL nodes are designed to work with:
- [IsaacGymSim](../robotics/isaac_gym_sim.md) - Physics simulation
- [IsaacGymEnvs](../robotics/isaac_gym_envs.md) - Pre-built RL environments

## Common Use Cases

### Robotics Control
- Cartpole balancing
- Humanoid walking
- Robotic manipulation
- Quadruped locomotion

### Training Configuration
```
PPOConfig → PPOAgent
    ↓
IsaacGymEnvs → observations, rewards
    ↓
PPOAgent → actions
    ↓
Environment step
```

## Hyperparameters

Key PPO hyperparameters configured via PPOConfig:
- **Learning Rate**: Policy and value function learning rates
- **Clip Range**: PPO clipping parameter (typically 0.2)
- **Entropy Coefficient**: Exploration bonus
- **Value Loss Coefficient**: Weight for value function loss
- **GAE Lambda**: Bias-variance tradeoff for advantages
- **Discount Factor (Gamma)**: Future reward discounting

## Export Behavior

RL nodes export to:
- Standalone Python training scripts
- Integration with rl_games framework
- Custom Isaac Gym training loops
- Queue-based async execution for real-time control

## Best Practices

1. **Start Simple**: Begin with proven hyperparameters from PPOConfig defaults
2. **Environment Normalization**: Normalize observations and rewards
3. **Parallel Environments**: Use multiple environment instances for faster training
4. **Checkpoint Regularly**: Save model checkpoints during training
5. **Monitor Metrics**: Track policy loss, value loss, and entropy

## Implementation Details

- **Base Class**: All RL nodes inherit from `RoboticsNodeBase`
- **Location**: `/home/asantanna/DNNE/DNNE-UI/custom_nodes/*_visnode.py`
- **Templates**: `/home/asantanna/DNNE/DNNE-UI/export_system/templates/nodes/*_queue.py`
- **Framework**: Integrates with rl_games_dnne for advanced features