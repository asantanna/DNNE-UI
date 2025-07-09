# DNNE Examples

## Overview

This directory contains complete, working examples of DNNE workflows. Each example demonstrates best practices and serves as a template for your own projects.

## Available Examples

### [MNIST Classification](mnist_classification.md)
A complete neural network training pipeline for handwritten digit recognition:
- Dataset loading and batching
- Two-layer neural network with ReLU activation
- Cross-entropy loss and SGD optimization
- Training loop with epoch tracking
- Export to standalone Python code

**Key Concepts**: Supervised learning, feedforward networks, classification

### [Cartpole PPO](cartpole_ppo.md)
Reinforcement learning example using PPO to balance a pole:
- Isaac Gym environment integration
- Actor-Critic network with PPO training
- Continuous control with exploration
- Async queue-based execution
- Real-time robot control

**Key Concepts**: Reinforcement learning, continuous control, PPO algorithm

## Example Structure

Each example includes:

1. **Overview** - What the example demonstrates
2. **Workflow Diagram** - Visual representation of node connections
3. **Node Breakdown** - Detailed explanation of each node
4. **Parameters** - Key settings and their effects
5. **Running the Example** - Step-by-step instructions
6. **Results** - Expected outcomes and performance
7. **Variations** - How to modify for your use case

## Quick Start

### Running an Example

1. **Load the Workflow**
   ```bash
   # In DNNE UI, load the workflow file
   File → Open → examples/MNIST_Test.json
   ```

2. **Review Parameters**
   - Check node settings match your hardware
   - Adjust batch sizes for your GPU memory

3. **Export the Code**
   ```bash
   # Click Export button in UI
   # Choose export location
   ```

4. **Run the Generated Code**
   ```bash
   # Activate environment
   source /home/asantanna/miniconda/bin/activate DNNE_PY38
   
   # Run exported workflow
   cd export_system/exports/YourWorkflow
   python runner.py
   ```

## Creating Your Own Examples

### Starting from Scratch

1. **Identify Your Task**
   - Classification, regression, or control?
   - What type of data?
   - Performance requirements?

2. **Choose Base Example**
   - MNIST for supervised learning
   - Cartpole for reinforcement learning

3. **Modify Architecture**
   - Add/remove layers
   - Change activation functions
   - Adjust hyperparameters

4. **Test and Iterate**
   - Start with small data
   - Monitor training progress
   - Tune based on results

### Common Modifications

#### Different Dataset
Replace MNISTDatasetNode with:
- Custom dataset loader
- Different built-in dataset
- Streaming data source

#### Deeper Networks
Add more layers:
```
Input → Linear → ReLU → Dropout → 
        Linear → ReLU → Dropout →
        Linear → Output
```

#### Different Optimizer
Replace SGDOptimizerNode with:
- Adam (when available)
- RMSprop
- Custom optimizer

## Best Practices

### Workflow Design
- Keep workflows organized with reroute nodes
- Group related nodes visually
- Use consistent naming conventions
- Save versions as you iterate

### Parameter Selection
- Start with proven defaults
- Change one parameter at a time
- Document what works
- Use version control

### Debugging
- Add DebugPrint nodes at key points
- Start with small datasets
- Check tensor shapes match
- Monitor loss and metrics

### Performance
- Profile before optimizing
- Use appropriate batch sizes
- Enable GPU when available
- Consider distributed training

## Example Categories

### Supervised Learning
- **MNIST Classification** - Digit recognition
- Image Classification (coming soon)
- Text Classification (coming soon)
- Regression Tasks (coming soon)

### Reinforcement Learning
- **Cartpole PPO** - Classic control
- Ant Locomotion (coming soon)
- Humanoid Control (coming soon)
- Multi-Agent RL (coming soon)

### Advanced Patterns
- Transfer Learning (coming soon)
- Multi-Task Learning (coming soon)
- Continual Learning (coming soon)
- Meta-Learning (coming soon)

## Troubleshooting

### Common Issues

**Workflow won't load**
- Check JSON syntax
- Verify node types exist
- Look for version compatibility

**Export fails**
- Check all nodes have templates
- Verify connections are valid
- Review error messages

**Training doesn't converge**
- Adjust learning rate
- Check data preprocessing
- Verify loss computation

**Out of memory**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision

## Contributing Examples

To contribute a new example:

1. Create a working workflow
2. Document thoroughly
3. Test export and execution
4. Write example documentation
5. Submit pull request

Guidelines:
- Examples should be self-contained
- Include clear documentation
- Test on multiple platforms
- Follow existing format

## Resources

- [Node Reference](../nodes/) - Detailed node documentation
- [ML Concepts](../ML/) - Theoretical background
- [Architecture](../architecture/) - System design
- [ComfyUI Basics](https://github.com/comfyanonymous/ComfyUI) - Original project

## Next Steps

1. Try the MNIST example to understand basics
2. Explore Cartpole for RL concepts
3. Modify examples for your use case
4. Create and share your own workflows

Remember: The best way to learn is by experimenting!