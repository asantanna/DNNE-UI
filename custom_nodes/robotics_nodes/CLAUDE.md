# Robotics Nodes CLAUDE.md

Robotics and reinforcement learning nodes for DNNE.

## Overview

Robotics nodes provide Isaac Gym integration and RL training components with async queue-based execution.

## Node Categories

- **Environment Nodes**: IsaacGymEnv, IsaacGymStep
- **RL Agent Nodes**: PPOAgent (complete PPO implementation)
- **Control Nodes**: RobotController, Manipulation nodes
- **Sensor Nodes**: IMUNode, CameraNode
- **Utility Nodes**: ORNode (state routing), RewardComputation

## Key Features

- **Standard RL Interface**: observations, actions, rewards, done, info
- **State Caching**: Synchronization for RL training loops
- **OR Node Routing**: Handles initial state vs step state flow
- **Isaac Gym Integration**: GPU-accelerated physics simulation

## Documentation

For detailed information, see:
- **Node Reference**: `docs-dnne/nodes/robotics/`
- **RL Architecture**: `docs-dnne/nodes/rl/ppo.md`
- **Isaac Gym Integration**: `docs-dnne/nodes/robotics/isaac_gym.md`
- **Examples**: `docs-dnne/examples/cartpole_ppo.md`