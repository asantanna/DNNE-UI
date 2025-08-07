# Robotics Nodes Documentation

Robotics simulation nodes for Isaac Gym integration in DNNE.

## Available Nodes

### [IsaacGymSim](isaac_gym_sim.md)
Core Isaac Gym physics simulator node for robotics simulation.

### [IsaacGymEnvs](isaac_gym_envs.md)
Pre-built reinforcement learning environments from Isaac Gym.

## Overview

The robotics nodes provide integration with NVIDIA Isaac Gym, a high-performance physics simulation platform designed for robot learning. These nodes enable:

- **GPU-Accelerated Physics**: Thousands of parallel simulations
- **RL Environment**: Standard Gym-like interface for reinforcement learning
- **Robot Simulation**: Accurate dynamics for various robot types
- **Real-time Performance**: Suitable for sim-to-real transfer

## Isaac Gym Features

### Physics Simulation
- **PhysX Backend**: NVIDIA's physics engine
- **GPU Acceleration**: Massively parallel simulation
- **Contact Dynamics**: Accurate collision and friction
- **Soft Body Simulation**: Deformable objects

### Supported Robots
- **Manipulators**: Franka, UR10, etc.
- **Humanoids**: Walking and balancing tasks
- **Quadrupeds**: Locomotion and navigation
- **Mobile Robots**: Navigation and manipulation

## Typical Workflow

1. **Initialize Simulator**: IsaacGymSim sets up physics
2. **Load Environment**: IsaacGymEnvs provides task
3. **Connect RL Agent**: PPOAgent learns control policy
4. **Training Loop**: Simulate, learn, repeat

## Environment Configuration

Environments are configured via YAML files:
```yaml
env:
  numEnvs: 4096          # Parallel environments
  envSpacing: 1.5        # Space between environments
  episodeLength: 1000    # Max steps per episode
  
sim:
  dt: 0.01              # Simulation timestep
  substeps: 2           # Physics substeps
  gravity: [0, 0, -9.81] # Gravity vector
```

## Available Environments

### Classic Control
- **Cartpole**: Balance pole on cart
- **Pendulum**: Swing-up task
- **Ant**: Quadruped locomotion

### Manipulation
- **FrankaCubeStack**: Stack cubes with Franka arm
- **ShadowHand**: Dexterous manipulation

### Locomotion
- **Humanoid**: Bipedal walking
- **Anymal**: Quadruped terrain navigation

## Integration with ML/RL

The robotics nodes seamlessly integrate with:
- [PPOAgent](../rl/ppo_agent.md) - RL training
- [Network](../ml/network.md) - Neural network policies
- ML nodes for custom architectures

## Export Behavior

Robotics nodes export to:
- Standalone Isaac Gym training scripts
- Integration with IsaacGymEnvs tasks
- Custom robot controllers
- Real-time control systems

## Performance Optimization

1. **GPU Selection**: Use high-memory GPUs for large simulations
2. **Environment Count**: Balance between diversity and memory
3. **Simulation Timestep**: Smaller dt for accuracy, larger for speed
4. **Substeps**: More substeps for stable contact dynamics

## Common Issues

- **CUDA Errors**: Ensure Isaac Gym CUDA version matches PyTorch
- **Import Order**: Always import isaacgym before torch
- **Memory Limits**: Reduce numEnvs if GPU runs out of memory
- **Simulation Instability**: Reduce timestep or increase substeps

## Best Practices

1. **Start Small**: Test with few environments first
2. **Domain Randomization**: Vary physics parameters for robustness
3. **Curriculum Learning**: Gradually increase task difficulty
4. **Observation Design**: Include relevant state information
5. **Reward Shaping**: Design informative reward functions

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 11+
- **Driver**: Latest NVIDIA driver
- **Memory**: 8GB+ GPU memory recommended
- **OS**: Linux (Ubuntu 20.04/22.04)

## Implementation Details

- **Base Class**: All robotics nodes inherit from `RoboticsNodeBase`
- **Location**: `/home/asantanna/DNNE/DNNE-UI/custom_nodes/*_visnode.py`
- **Isaac Gym Path**: `/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym`
- **IsaacGymEnvs Path**: `/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs`