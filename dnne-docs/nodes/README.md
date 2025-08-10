# DNNE Node Reference Documentation

This directory contains comprehensive documentation for all DNNE nodes organized by category.

## Node Categories

### [ML Nodes](ml/README.md)
Machine learning nodes for building neural networks, handling datasets, and training models.

- **Datasets**: MNIST, CIFAR-10
- **Layers**: Linear, Conv2D, BatchNorm, Dropout, Flatten, Activation
- **Training**: SGD Optimizer, Cross Entropy Loss, Training Step, Accuracy
- **Utilities**: Batch Sampler, Get Batch, Epoch Tracker, Tensor Visualizer
- **Networks**: Network (composite node)

### [RL Nodes](rl/README.md)
Reinforcement learning nodes for PPO and other RL algorithms.

- **Agents**: PPO Agent
- **Configuration**: PPO Config

### [Robotics Nodes](robotics/README.md)
Robotics simulation nodes for Isaac Gym integration.

- **Simulation**: Isaac Gym Sim, Isaac Gym Envs

### [Utility Nodes](utility/README.md)
General utility nodes for workflow control and configuration.

- **Logic**: OR Node
- **Configuration**: Balancing Config, Balancing Node

## Node Documentation Format

Each node documentation includes:
- **Purpose**: What the node does
- **Category**: Node category (ml, rl, robotics, utility)
- **Inputs**: Required and optional inputs with types
- **Outputs**: What the node produces
- **Parameters**: Configuration parameters
- **Usage Examples**: How to use the node in workflows
- **Export Behavior**: How the node exports to Python code

## Quick Reference

| Node | Category | Primary Function |
|------|----------|-----------------|
| MNISTDataset | ML | Load MNIST digit dataset |
| CIFAR10Dataset | ML | Load CIFAR-10 image dataset |
| LinearLayer | ML | Fully connected neural network layer |
| Conv2DLayer | ML | 2D convolutional layer |
| BatchNorm | ML | Batch normalization layer |
| Dropout | ML | Dropout regularization |
| Flatten | ML | Flatten tensor dimensions |
| Activation | ML | Apply activation functions (ReLU, Sigmoid, etc.) |
| Network | ML | Composite node for sequential layers |
| BatchSampler | ML | Sample batches from datasets |
| GetBatch | ML | Retrieve next batch from sampler |
| SGDOptimizer | ML | Stochastic gradient descent optimizer |
| CrossEntropyLoss | ML | Calculate cross-entropy loss |
| TrainingStep | ML | Execute single training iteration |
| Accuracy | ML | Calculate model accuracy |
| EpochTracker | ML | Track training epochs and metrics |
| TensorVisualizer | ML | Visualize tensor data |
| PPOAgent | RL | Proximal Policy Optimization agent |
| PPOConfig | RL | PPO hyperparameter configuration |
| IsaacGymSim | Robotics | Isaac Gym physics simulator |
| IsaacGymEnvs | Robotics | Isaac Gym RL environments |
| ORNode | Utility | Logical OR operation |
| BalancingConfig | Utility | Balancing task configuration |
| BalancingNode | Utility | Balancing control logic |

## Implementation Location

All node implementations are located in `/home/asantanna/DNNE/DNNE-UI/custom_nodes/` with the naming pattern `*_visnode.py`.

## Export Templates

Export templates for each node are located in `/home/asantanna/DNNE/DNNE-UI/export_system/templates/nodes/` with queue-based async implementations.