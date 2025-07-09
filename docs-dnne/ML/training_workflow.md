# Creating RL Training Workflows in DNNE

## Overview

This guide walks through creating reinforcement learning training workflows in DNNE, from basic setup to advanced configurations. We'll use the Cartpole PPO workflow as a reference while explaining general principles.

## Basic Workflow Structure

Every RL workflow in DNNE follows this pattern:

```
Environment → Observation Router → Policy → Action Processor → Environment Step
                    ↑                                                    ↓
                    └─────────────── Trainer ←───────────────────────────┘
```

## Step-by-Step Workflow Creation

### Step 1: Environment Setup

Start by adding an environment node:

1. **Add IsaacGymEnvNode**
   - Set `env_name` (e.g., "Cartpole", "Ant", "Humanoid")
   - Configure `num_envs` (start with 1 for testing)
   - Set `headless` to True for training, False for visualization
   - Choose `device` ("cpu" or "cuda")

```json
{
  "class_type": "IsaacGymEnvNode",
  "inputs": {
    "env_name": "Cartpole",
    "num_envs": 1,
    "headless": true,
    "device": "cuda"
  }
}
```

### Step 2: Observation Routing

Add an ORNode to handle the cyclic nature of RL:

1. **Add ORNode**
   - Connect initial observations from environment
   - Connect step observations from simulation step
   - This creates the feedback loop

The ORNode solves the initialization problem - it routes the first observation from environment setup, then switches to observations from the step node.

### Step 3: Policy Network

Add the PPO agent:

1. **Add PPOAgentNode**
   - Set `input_size` to match observation space
   - Set `output_size` to match action space
   - Configure `hidden_sizes` (e.g., [64, 64])
   - Choose `activation` function
   - Set `learning_rate`

```json
{
  "class_type": "PPOAgentNode",
  "inputs": {
    "observations": ["from_or_node"],
    "input_size": 4,
    "output_size": 1,
    "hidden_sizes": [64, 64],
    "activation": "elu",
    "learning_rate": 0.0003
  }
}
```

### Step 4: Action Processing

Add environment-specific action processing:

1. **Add Action Node** (e.g., CartpoleActionNode)
   - Connects policy output to environment-compatible actions
   - Handles action scaling and formatting

### Step 5: Environment Stepping

Add the simulation step:

1. **Add IsaacGymStepNode**
   - Connect `sim` from environment node
   - Connect `actions` from action processor
   - Connect `trigger` from trainer (for synchronization)

### Step 6: Training Algorithm

Complete the loop with PPO trainer:

1. **Add PPOTrainerNode**
   - Connect all required inputs (states, policy outputs, rewards, dones, model)
   - Configure training hyperparameters
   - Outputs trigger signal to control collection

```json
{
  "class_type": "PPOTrainerNode",
  "inputs": {
    "states": ["from_or_node"],
    "policy_outputs": ["from_ppo_agent"],
    "rewards": ["from_step"],
    "dones": ["from_step"],
    "model": ["from_ppo_agent"],
    "horizon_length": 16,
    "epochs": 4,
    "minibatch_size": 32
  }
}
```

## Hyperparameter Configuration

### Environment Settings

- **num_envs**: Number of parallel environments
  - Start with 1 for debugging
  - Increase to 16-64 for faster training
  - Limited by GPU memory

- **headless**: Rendering mode
  - True: No visualization (faster)
  - False: Shows simulation window

### Network Architecture

- **hidden_sizes**: Layer dimensions
  - Simple tasks: [64, 64]
  - Complex tasks: [256, 256]
  - Vision tasks: [512, 256, 128]

- **activation**: Non-linearity
  - "elu": Smooth, good default
  - "relu": Fast, may have dead neurons
  - "tanh": Bounded outputs

### PPO Parameters

- **horizon_length**: Steps before update
  - Short (128-512): Frequent updates, less stable
  - Long (2048-4096): Stable updates, more memory

- **epochs**: Training iterations per update
  - 3-4: Conservative, stable
  - 8-10: Aggressive, may overfit

- **minibatch_size**: Batch size for SGD
  - 32-64: Good for small networks
  - 128-256: Better GPU utilization

- **learning_rate**: Step size
  - 1e-4 to 3e-4: Standard range
  - Reduce if unstable
  - Increase for faster initial learning

## Advanced Patterns

### Multi-Environment Training

For parallel training:

```python
# In environment node
num_envs = 16  # Run 16 environments in parallel

# PPO agent automatically handles batched observations
# Each environment contributes to the same trajectory buffer
```

### Custom Reward Shaping

Add reward processing between step and trainer:

```
Step → RewardShaper → PPOTrainer
```

Create custom reward nodes for:
- Curriculum learning
- Sparse reward handling
- Multi-objective optimization

### Curriculum Learning

Implement progressive difficulty:

```python
# Environment with configurable difficulty
env_node.difficulty = schedule_function(training_progress)
```

### Model Checkpointing

Add checkpoint nodes (future feature):

```
PPOTrainer → CheckpointSaver
           → CheckpointLoader (for resume)
```

## Debugging Workflows

### Visual Debugging

1. Set `headless=False` in environment
2. Add print nodes at key points
3. Use smaller horizon_length for faster iteration

### Common Issues

**Issue**: Observations have wrong shape
- Check environment observation space
- Verify ORNode connections
- Print observation shapes

**Issue**: Actions cause environment errors
- Verify action space dimensions
- Check action processing node
- Ensure proper scaling

**Issue**: Training doesn't converge
- Start with proven hyperparameters
- Check reward scaling
- Verify advantage computation

### Monitoring Training

Add monitoring nodes:

```
PPOTrainer → MetricsLogger → TensorBoard
           → ConsoleLogger
```

Track key metrics:
- Episode rewards
- Policy loss
- Value loss
- Entropy
- Learning rate

## Best Practices

### 1. Start Simple

- Begin with single environment
- Use default hyperparameters
- Add complexity gradually

### 2. Validate Each Component

- Test environment alone first
- Verify policy outputs make sense
- Check reward signals

### 3. Use Reroute Nodes

For cleaner visual layout:
- Add reroute nodes for long connections
- Group related nodes visually
- Label important connections

### 4. Save Incremental Versions

- Save workflow after each major change
- Use descriptive names
- Document hyperparameter choices

## Example Workflows

### Basic Cartpole

Minimal setup for testing:
- 1 environment
- Simple network [64, 64]
- Short horizon (128)
- Quick iterations

### Advanced Ant Locomotion

Complex continuous control:
- 16 parallel environments
- Deeper network [256, 256, 128]
- Longer horizon (2048)
- Careful hyperparameter tuning

### Vision-Based Control

For camera inputs:
- Add CNN feature extractor
- Larger networks needed
- Frame stacking for temporal info
- Lower learning rates

## Exporting and Running

### Export Process

1. Click "Export" button in UI
2. Choose export location
3. Generated code structure:
   ```
   exports/YourWorkflow/
   ├── runner.py           # Entry point
   ├── generated_nodes/    # Node implementations
   └── framework/          # Queue infrastructure
   ```

### Running Exported Code

```bash
# Activate conda environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Navigate to export
cd export_system/exports/YourWorkflow

# Run training
python runner.py
```

### Customizing Exported Code

The exported code is clean Python:
- Modify hyperparameters directly
- Add logging/debugging
- Integrate with other systems

## Next Steps

1. **Experiment with Hyperparameters**
   - Try different network sizes
   - Adjust learning rates
   - Test horizon lengths

2. **Try Different Environments**
   - Progress from Cartpole to Ant
   - Experiment with custom environments

3. **Implement Advanced Features**
   - Multi-task learning
   - Transfer learning
   - Custom reward functions

4. **Optimize Performance**
   - Use multiple environments
   - Profile bottlenecks
   - GPU optimization

## Troubleshooting Resources

- Check node documentation in `custom_nodes/`
- Review exported code for debugging
- Examine queue communication patterns
- Use logging extensively

Remember: RL training requires patience and experimentation. Start simple, validate each component, and build complexity gradually.