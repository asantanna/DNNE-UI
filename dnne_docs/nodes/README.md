# DNNE Node Reference Documentation

*Last Updated: 2026-01-11*

This directory contains documentation for DNNE nodes organized by category.

## Node Categories

### [ML Nodes](ml/README.md)
Machine learning nodes for neural networks and training.

- **Datasets**: MNISTDataset, CIFAR10Dataset
- **Layers**: LinearLayer, Network
- **Training**: SGDOptimizer, CrossEntropyLoss, GeometricLoss, TrainingSequencer
- **Data Flow**: BatchSampler, GetBatch, EpochTracker

### [RL Nodes](rl/README.md)
Reinforcement learning nodes.

- **Agents**: PPOAgent
- **Configuration**: PPOConfig

### [Robotics Nodes](robotics/README.md)
Isaac Gym integration nodes.

- **Simulation**: IsaacGymSim, IsaacGymEnvs, SimulationTracker

### [Utility Nodes](utility/README.md)
Workflow control and synchronization.

- **Synchronization**: Barrier, Eat_N
- **Data Flow**: Tensor, Concat, Split, DataStreamer
- **Balancing**: Balancer, BalancerConfig
- **Custom**: CustomComputation

## Quick Reference

| Node | Category | Purpose |
|------|----------|---------|
| MNISTDataset | ML | Load MNIST digits |
| CIFAR10Dataset | ML | Load CIFAR-10 images |
| LinearLayer | ML | Fully connected layer |
| Network | ML | Sequential layer container |
| BatchSampler | ML | Sample batches from datasets |
| GetBatch | ML | Retrieve next batch |
| SGDOptimizer | ML | Gradient descent optimizer |
| CrossEntropyLoss | ML | Classification loss |
| GeometricLoss | ML | Geometric/MSE loss |
| TrainingSequencer | ML | Coordinate training flow |
| EpochTracker | ML | Track epochs and metrics |
| PPOAgent | RL | PPO algorithm implementation |
| PPOConfig | RL | PPO hyperparameters |
| IsaacGymSim | Robotics | Physics simulator |
| IsaacGymEnvs | Robotics | Pre-built RL environments |
| SimulationTracker | Robotics | Track simulation metrics |
| Barrier | Utility | Hold data until triggered |
| Eat_N | Utility | Initial trigger generation |
| Tensor | Utility | Tensor creation/manipulation |
| Concat | Utility | Concatenate tensors (dim=1) |
| Split | Utility | Split tensors (dim=1) |
| DataStreamer | Utility | Stream data from files |
| Balancer | Utility | Measure throughput |
| BalancerConfig | Utility | Balancer configuration |
| CustomComputation | Utility | User-defined Python code |

## Implementation

- **Node code**: `custom_nodes/*_visnode.py`
- **Export templates**: `export_system/templates/nodes/*_queue.tpl`
- **Node exporters**: `export_system/node_exporters/`
