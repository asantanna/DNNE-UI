# DNNE Documentation

Welcome to the DNNE (Drag and Drop Neural Network Environment) documentation. This documentation covers the machine learning capabilities, architecture, and usage patterns of DNNE.

## Documentation Structure

### Machine Learning (ML)
- **[Cartpole PPO Analysis](ML/cartpole_ppo_analysis.md)** - Detailed analysis of the PPO implementation for Cartpole, including workflow structure, export system, and performance characteristics
- **[PPO Architecture](ML/ppo_architecture.md)** - Comprehensive guide to the PPOAgentNode and PPOTrainerNode, including configuration, best practices, and troubleshooting
- **[Training Workflows](ML/training_workflow.md)** - *(Coming soon)* Guide to creating RL training workflows in DNNE

### Architecture
- **[Export System](architecture/export_system.md)** - *(Coming soon)* How DNNE converts visual workflows to executable Python code
- **[Queue Framework](architecture/queue_framework.md)** - *(Coming soon)* Understanding the async queue-based architecture

## Quick Navigation

### For RL Practitioners
1. Start with [PPO Architecture](ML/ppo_architecture.md) to understand the available nodes
2. Review [Cartpole PPO Analysis](ML/cartpole_ppo_analysis.md) for a complete implementation example
3. Follow [Training Workflows](ML/training_workflow.md) to build your own RL systems

### For Developers
1. Understand the [Queue Framework](architecture/queue_framework.md) for the async execution model
2. Learn about the [Export System](architecture/export_system.md) for code generation
3. Review implementation examples in the ML section

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