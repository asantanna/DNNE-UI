# DNNE Node Reference

## Overview

DNNE nodes are the building blocks of visual workflows. Each node represents a specific operation - from loading data to training neural networks to controlling robots. This documentation provides comprehensive reference for all available nodes.

## Node Categories

### [Machine Learning (ML)](ml/)
Neural network components, training algorithms, and data processing:
- **[Data Processing](ml/data.md)** - Dataset loading, batching, splitting
- **[Neural Layers](ml/layers.md)** - Linear, Conv2D, normalization layers
- **[Training Tools](ml/training.md)** - Optimizers, loss functions, metrics
- **[Activation Functions](ml/activation.md)** - ReLU, ELU, Sigmoid, Tanh

### [Reinforcement Learning (RL)](rl/)
Specialized nodes for RL algorithms:
- **[PPO Implementation](rl/ppo.md)** - Actor-Critic networks and PPO training

### [Robotics](robotics/)
Integration with simulators and robot control:
- **[Isaac Gym](robotics/isaac_gym.md)** - NVIDIA Isaac Gym environments
- **[Control Systems](robotics/control.md)** - Decision networks and controllers

### [Utility](utility/)
Helper nodes for workflow construction:
- **[Data Flow](utility/data_flow.md)** - Routing, merging, and organizing connections
- **[Constants & Debug](utility/constants.md)** - Fixed values and debugging tools

## Understanding Node Documentation

Each node documentation includes:

### Node Header
- **Name**: The node's class name as it appears in the UI
- **Category**: Which menu/submenu contains the node
- **Purpose**: Brief description of what the node does

### Inputs
Table of all input connections:
| Input | Type | Description |
|-------|------|-------------|
| data | Tensor | Input data to process |

### Parameters
Configurable settings:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| learning_rate | float | 0.001 | Step size for optimization |

### Outputs
What the node produces:
| Output | Type | Description |
|--------|------|-------------|
| result | Tensor | Processed output |

### Usage Example
How to use the node in a workflow, including:
- Typical connections
- Parameter configurations
- Common patterns

### Export Support
- Whether the node has an export template
- Any limitations in exported code
- Special considerations

## Node Types

### Sensor Nodes
Nodes that generate or load data:
- No required inputs (or optional trigger inputs)
- Produce data for downstream processing
- Examples: Dataset loaders, environment nodes

### Processing Nodes
Transform data from inputs to outputs:
- Require one or more inputs
- Apply transformations or computations
- Examples: Neural layers, activation functions

### Sink Nodes
Consume data without producing outputs:
- Used for side effects (saving, logging)
- May produce status signals
- Examples: Model savers, visualizers

### Control Nodes
Manage workflow execution:
- Route data based on conditions
- Synchronize multiple paths
- Examples: OR nodes, conditional nodes

## Creating Workflows

### Basic Pattern
1. **Source** → **Processing** → **Sink**
   ```
   DataLoader → NeuralNetwork → SaveModel
   ```

2. **Feedback Loops**
   ```
   Environment → Policy → Action → Environment
         ↑                            ↓
         └──────── Trainer ←─────────┘
   ```

### Connection Rules

1. **Type Matching**: Outputs must match expected input types
2. **Single Source**: Each input accepts one connection
3. **Multiple Targets**: Outputs can connect to multiple inputs
4. **No Cycles**: Except through special control nodes (OR, etc.)

## Common Patterns

### Data Pipeline
```
Dataset → BatchSampler → GetBatch → Network → Loss
```

### Training Loop
```
GetBatch → Network → Loss → Optimizer → TrainingStep
```

### RL Episode
```
Environment → ORNode → Policy → Action → Step → ORNode
```

## Best Practices

### Node Configuration
- Start with default parameters
- Adjust based on task requirements
- Monitor outputs during debugging
- Use appropriate data types

### Workflow Organization
- Use Reroute nodes for visual clarity
- Group related nodes together
- Label important connections
- Save workflow versions frequently

### Performance Tips
- Minimize data copying between nodes
- Use appropriate batch sizes
- Enable GPU where available
- Profile bottlenecks in exported code

## Debugging

### Visual Debugging
- Add DebugPrint nodes at key points
- Use smaller datasets for testing
- Check tensor shapes and values
- Verify connections are correct

### Export Debugging
- Examine generated code
- Add logging to templates
- Use Python debugger on exported code
- Check queue depths and flow

## Node Development

To create custom nodes:

1. **Define Node Class**
   ```python
   class MyNode:
       @classmethod
       def INPUT_TYPES(cls):
           return {"required": {"input": ("TENSOR",)}}
   ```

2. **Implement Processing**
   ```python
   def compute(self, input):
       return (processed_output,)
   ```

3. **Create Export Template**
   ```python
   # In templates/nodes/my_node_queue.py
   class MyNode_{node_id}(QueueNode):
       async def process(self):
           # Implementation
   ```

4. **Register Exporter**
   ```python
   # In node_exporters/
   class MyNodeExporter(BaseNodeExporter):
       # Export configuration
   ```

See the [Node Development Guide](../architecture/node_development.md) for details.

## Quick Reference

### Most Used Nodes

| Node | Purpose | Category |
|------|---------|----------|
| LinearLayerNode | Fully connected layer | ML/Layers |
| PPOAgentNode | RL policy network | RL/PPO |
| IsaacGymEnvNode | Robot simulation | Robotics |
| BatchSamplerNode | Create batches | ML/Data |
| SGDOptimizerNode | Training optimizer | ML/Training |
| ORNode | Merge data flows | Utility |

### Node Naming Convention
- **Functionality + "Node"**: LinearLayerNode, Conv2DNode
- **Action-based**: TrainingStepNode, GetBatchNode
- **Domain-specific**: PPOAgentNode, IsaacGymEnvNode

## Next Steps

- Browse specific node categories for detailed documentation
- Check [Examples](../examples/) for complete workflows
- Read [ML Concepts](../ML/) for theoretical background
- Explore [Architecture](../architecture/) for system design