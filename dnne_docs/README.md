# DNNE Documentation

*Last Updated: 2026-01-11*

Welcome to the DNNE (Drag and Drop Neural Network Environment) documentation.

## Documentation Structure

### Node Reference
- **[Node Overview](nodes/)** - Complete reference for all DNNE nodes
- **[ML Nodes](nodes/ml/)** - Datasets, layers, training nodes
- **[RL Nodes](nodes/rl/)** - PPO agent and configuration
- **[Robotics Nodes](nodes/robotics/)** - Isaac Gym integration
- **[Utility Nodes](nodes/utility/)** - Synchronization and data flow

### Examples
- **[MNIST Classification](examples/mnist_classification.md)** - Supervised learning
- **[Cartpole PPO](examples/cartpole_ppo.md)** - Reinforcement learning
- **[Isaac Gym Integration](examples/isaac_gym_integration.md)** - Robotics

### Machine Learning
- **[PPO Architecture](ML/ppo_architecture.md)** - Deep dive into PPO algorithm
- **[Training Workflows](ML/training_workflow.md)** - Guide to creating RL workflows

### Architecture
- **[Export System](architecture/export_system.md)** - Visual workflow to Python code
- **[Queue Framework](architecture/queue_framework.md)** - Async runtime
- **[Templates](architecture/templates.md)** - Code generation templates
- **[Adaptive Yielding](architecture/adaptive_yielding.md)** - Cooperative multitasking

### Development
- **[Gotchas](development/gotchas.md)** - Common pitfalls and solutions
- **[Debugging Techniques](development/debugging-techniques.md)** - Non-obvious debugging

### Future Features
- **[Feature Roadmap](future/)** - Planned features and improvements

## Quick Navigation

### For ML Practitioners
1. Start with [MNIST Example](examples/mnist_classification.md) for supervised learning
2. Browse [ML Nodes](nodes/ml/) to understand available components
3. Check [Training Workflows](ML/training_workflow.md) for best practices

### For RL Practitioners
1. Review [Cartpole PPO Example](examples/cartpole_ppo.md) for a complete RL implementation
2. Study [PPO Nodes](nodes/rl/ppo.md) for algorithm details
3. Explore [Robotics Nodes](nodes/robotics/) for environment integration

### For Developers
1. Understand the [Queue Framework](architecture/queue_framework.md) for the async execution model
2. Learn about the [Export System](architecture/export_system.md) for code generation
3. Browse [Future Features](future/) to contribute ideas

## Key Concepts

### Visual Programming for ML
DNNE transforms complex machine learning workflows into intuitive visual graphs. Nodes represent operations (neural networks, environments, trainers) while connections represent data flow.

### Export to Production
Unlike traditional visual programming, DNNE exports workflows to efficient, standalone Python code that runs on:
- Local machines with GPU support
- Cloud providers (Lambda, AWS, etc.)
- Robotics simulators (NVIDIA Isaac Gym)
- Edge devices

### Async Queue Architecture
All exported code uses an async, queue-based architecture similar to ROS (Robot Operating System), enabling:
- Real-time performance
- Non-blocking execution
- Scalable multi-node systems
- Hardware-accelerated simulation

## Getting Started

1. **Create a Workflow**: Use the visual editor to design your ML system
2. **Export to Code**: Click the Export button to generate Python code
3. **Run Anywhere**: Execute the generated code on your target platform

## Contributing

Documentation improvements are welcome! When adding new documentation:
- Place ML-related docs in the `ML/` directory
- Place architecture docs in the `architecture/` directory
- Update this README with navigation links
- Follow the existing documentation style

## Related Resources

- **Main Project**: [DNNE-UI Repository](https://github.com/asantanna/DNNE-UI)
- **Frontend**: [DNNE-UI-Frontend Repository](https://github.com/asantanna/DNNE-UI-Frontend.git)
- **NVIDIA Isaac Gym**: [Official Documentation](https://developer.nvidia.com/isaac-gym)