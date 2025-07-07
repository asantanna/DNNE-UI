# DNNE Robotics Nodes

This directory contains robotics-specific nodes for DNNE (Drag and Drop Neural Network Environment), including a sophisticated **RL training synchronization system** for Isaac Gym integration.

## Overview

The robotics nodes provide:
- **Isaac Gym Integration**: Physics simulation environments for robotics
- **Standard RL Interface**: Industry-standard observations/rewards/done/info outputs  
- **Async RL Training**: Queue-based synchronization for real-time RL training loops
- **OR Node Routing**: State routing solution for RL training flow

## Core Architecture

### Standard RL Terminology Alignment

We've aligned DNNE with standard reinforcement learning terminology:

| **Standard RL Term** | **DNNE Type** | **Description** | **Shape** |
|---------------------|---------------|-----------------|-----------|
| `observations` | `TENSOR` | What the agent observes (neural network input) | `[num_envs, obs_dim]` |
| `actions` | `TENSOR` | Agent's control commands | `[num_envs, action_dim]` |
| `rewards` | `TENSOR` | Scalar feedback per environment | `[num_envs]` |
| `done` | `TENSOR` | Episode termination flags | `[num_envs]` (bool) |
| `info` | `DICT` | Additional metadata | Python dict |

**Why This Matters**: Previous DNNE used custom types (`ROBOT_STATE`, `SENSOR_DATA`) which didn't align with RL standards. Neural networks expect tensors, not custom objects.

## RL Training Synchronization Problem

### The Challenge

Reinforcement learning requires **strict sequential execution**:

1. Environment provides initial state
2. Neural network processes state → produces actions  
3. Environment steps with actions → produces new state + rewards
4. Loss computed from rewards + network predictions
5. Backpropagation updates network weights
6. **Only then** can network process new state

This is fundamentally different from supervised learning where batches are independent.

### The DNNE Solution: State Caching + OR Node Routing

We implemented a **state caching mechanism** with **OR node routing** that solves the synchronization problem while preserving DNNE's async queue architecture.

## Implementation Details

### 1. IsaacGymEnvNode - Environment Initialization

**Purpose**: Sets up Isaac Gym simulation and provides initial observations

```python
# Outputs
RETURN_TYPES = ("SIM_HANDLE", "TENSOR", "CONTEXT")  
RETURN_NAMES = ("sim_handle", "observations", "context")

# Key Features
- No inputs (runs immediately at startup)
- Outputs initial observations tensor [num_envs, obs_dim]
- Bootstraps the RL training loop
```

**Observation Space**: 21-dimensional vector
- Joint positions (8 DOF)
- Joint velocities (8 DOF)  
- Base orientation quaternion (4 values)
- Upright indicator (1 value)

### 2. IsaacGymStepNode - Dual-Mode Execution

**Purpose**: Steps simulation and caches state for RL synchronization

```python
# Inputs
- sim_handle: SIM_HANDLE (required)
- actions: ACTION (required)  
- trigger: SYNC (optional) - from TrainingStep

# Outputs  
RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "DICT", "TENSOR", "CONTEXT")
RETURN_NAMES = ("observations", "rewards", "done", "info", "next_observations", "context")
```

**Dual Execution Modes**:

#### Mode 1: Normal Execution (actions provided)
1. Apply actions to Isaac Gym simulation
2. Step physics forward  
3. Compute observations, rewards, done flags
4. **Cache observations** for later trigger-based output
5. Output current step results

#### Mode 2: Trigger-Based Output (trigger provided)  
1. Output **cached observations** as `next_observations`
2. Return dummy values for other outputs
3. This happens **only after** TrainingStep completes

**Why State Caching Works**:
- Environment step produces state at time T
- State gets cached internally  
- Training happens on time T data
- **After training completes**, cached state becomes available for time T+1
- Ensures network doesn't see new state until training on old state finishes

### 3. ORNode - State Routing Solution  

**Purpose**: Routes initial state OR ongoing states to neural network

```python
# Inputs (all optional)
- input_a: TENSOR (initial state from IsaacGymEnv)
- input_b: TENSOR (ongoing state from IsaacGymStepNode)  
- input_c: TENSOR (additional routing if needed)

# Output
RETURN_TYPES = ("TENSOR",)
RETURN_NAMES = ("output",)
```

**Routing Logic**:
- Executes when **ANY** input becomes available
- Priority: A → B → C  
- **Input A**: Initial observations (first execution)
- **Input B**: Cached observations from training triggers
- **Input C**: Reserved for future use

**Why OR Node is Elegant**:
- No complex state management or counters needed
- Natural async execution when any input arrives
- Clean separation of initial vs. ongoing state logic
- Reusable pattern for other scenarios

## RL Training Flow

### Complete Training Loop

```
IsaacGymEnv (startup) → initial_observations ↘
                                              OR_Node → Network → actions → IsaacGymStepNode
                                               ↑                               ↓
                     TrainingStep ← Loss ← CustomReward ← rewards          observations (cached)
                          ↓                                                     ↓
                     ready_trigger                                       next_observations
                          ↓                                                     ↓  
                          └─────────────────────────────────────────────────────┘
```

### Execution Sequence

1. **Startup**: IsaacGymEnv runs (no inputs) → provides initial observations
2. **Route Initial**: OR_Node receives initial state → forwards to Network  
3. **Generate Actions**: Network processes state → outputs actions
4. **Environment Step**: IsaacGymStepNode receives actions → steps simulation → caches new observations
5. **Compute Rewards**: Custom reward nodes process step results
6. **Calculate Loss**: Loss nodes combine rewards + network predictions  
7. **Training Step**: TrainingStep does backpropagation → sends trigger
8. **Release Cached State**: IsaacGymStepNode receives trigger → outputs cached observations
9. **Route Ongoing**: OR_Node receives ongoing state → forwards to Network
10. **Loop Continues**: Network processes new state → generates new actions

### Synchronization Guarantees

✅ **No Race Conditions**: Queue-based execution ensures proper ordering  
✅ **State Continuity**: Network always gets correct sequential states  
✅ **Training Isolation**: Backprop completes before new state becomes available  
✅ **Async Performance**: Maintains DNNE's real-time queue architecture  

## Node Architecture

### Base Classes (`base_node.py`)

- **RoboticsNodeBase**: Common functionality for all robotics nodes
- **SensorNodeBase**: Base for sensor nodes (IMU, cameras, etc.)
- **ControllerNodeBase**: Base for control nodes  
- **LearningNodeBase**: Base for ML/RL nodes (Isaac Gym inherits from this)
- **VisualizationNodeBase**: Base for display/visualization nodes

### Custom Types (`robotics_types.py`)

**Key Types**:
- **SIM_HANDLE**: Isaac Gym simulation reference
- **ROBOT_STATE**: Legacy robot state (being phased out for RL)  
- **ACTION**: Control commands and forces
- **SENSOR_DATA**: Sensor readings and metadata
- **SYNC**: Synchronization signals for training coordination

**Type Conversion**: Helper functions convert between DNNE custom types and standard tensors.

### Node Registration (`__init__.py`)

Registers all robotics nodes with ComfyUI:
```python
NODE_CLASS_MAPPINGS = {
    "IsaacGymEnvNode": IsaacGymEnvNode,
    "IsaacGymStepNode": IsaacGymStepNode,  
    "ORNode": ORNode,
    # ... other nodes
}
```

## Export System Integration

### Templates (`../../export_system/templates/nodes/`)

- **isaac_gym_env_queue.py**: Environment setup with observation generation
- **isaac_gym_step_queue.py**: Dual-mode step execution with state caching
- **or_node_queue.py**: OR node routing logic

### Exporters (`../../export_system/node_exporters/robotics_nodes.py`)

- **IsaacGymEnvExporter**: Handles environment configuration parameters
- **IsaacGymStepExporter**: Manages step node template variables  
- **ORNodeExporter**: Simple OR node export configuration

**Generated Code**: Templates produce standalone Python scripts that run RL training loops independently.

## Design Decisions & Rationale

### Why State Caching Instead of Alternatives?

**Alternative 1: Complex State Router**
- ❌ Would need counters, episode tracking, reset logic
- ❌ More complex, harder to debug
- ❌ Tight coupling between nodes

**Alternative 2: Modify TrainingStep to Forward State**
- ❌ Makes TrainingStep robotics-specific  
- ❌ Breaks backward compatibility with supervised learning
- ❌ Violates separation of concerns

**Our Solution: State Caching**
- ✅ Keeps existing nodes unchanged
- ✅ Simple, focused responsibility per node
- ✅ Works with any RL environment, not just Isaac Gym
- ✅ Natural fit with trigger-based architecture

### Why OR Node Instead of Complex Router?

**OR Node Benefits**:
- ✅ **Simple Logic**: Just route first available input
- ✅ **Reusable**: Works for many scenarios beyond RL
- ✅ **Natural Async**: Executes when ANY input arrives
- ✅ **No State**: Stateless operation, no complex management
- ✅ **Visual Clarity**: Easy to understand in visual graph

## Usage Examples

### Basic RL Training Setup

1. **Environment Setup**:
   ```
   IsaacGymEnvNode (env_name="Cartpole", num_envs=64, headless=True)
   ```

2. **State Routing**:  
   ```
   IsaacGymEnv.observations → OR_Node.input_a
   IsaacGymStep.next_observations → OR_Node.input_b
   ```

3. **Neural Network**:
   ```
   OR_Node.output → Network.input
   Network.output → IsaacGymStep.actions  
   ```

4. **Training Loop**:
   ```
   IsaacGymStep.rewards → Loss.rewards
   Loss.loss → TrainingStep.loss
   TrainingStep.ready → IsaacGymStep.trigger
   ```

### Custom Reward Functions

Insert custom reward nodes between environment and loss:
```
IsaacGymStep.observations → BalanceReward.observations
IsaacGymStep.info → BalanceReward.info  
BalanceReward.rewards → Loss.rewards
```

## Development Guidelines

### Adding New RL Environments

1. **Inherit from LearningNodeBase**
2. **Use standard RL terminology** (observations, rewards, done, info)
3. **Implement state caching** if needed for training synchronization
4. **Provide tensor outputs** with proper shapes `[num_envs, dim]`
5. **Add trigger input** if environment participates in training loop

### Testing RL Nodes

1. **Validate tensor shapes** and device placement
2. **Test state caching logic** with trigger inputs
3. **Verify synchronization** with training step triggers  
4. **Check episode boundaries** and reset behavior
5. **Export and run** generated scripts independently

### Common Pitfalls

❌ **Using custom types instead of tensors** - Neural networks need tensors  
❌ **Forgetting state caching** - Creates race conditions in training  
❌ **Complex routing logic** - Use OR node pattern instead  
❌ **Device mismatches** - Ensure tensors are on correct device  
❌ **Ignoring batch dimensions** - All tensors need `[num_envs, ...]` shape  

## Future Extensions

### Planned Enhancements

- **Multi-Agent Support**: Extend to multiple agents per environment
- **Episode Reset Logic**: Automatic environment reset on done conditions  
- **Observation Preprocessing**: Standard observation normalization nodes
- **Reward Shaping**: Library of common reward function nodes
- **Curriculum Learning**: Progressive difficulty adjustment

### Integration Points

- **Existing ML Nodes**: OR node can route between any tensor sources
- **Other Simulators**: Pattern works with any physics simulator
- **Cloud Deployment**: Generated scripts run on Lambda, AWS, etc.
- **Real Robots**: Same pattern works with real robot interfaces

## Troubleshooting

### Common Issues

**"OR Node: No inputs available"**
- Check that at least one input connection exists
- Verify nodes feeding OR node are executing properly

**"Invalid simulation handle"**  
- Ensure IsaacGymEnvNode runs first and outputs valid SIM_HANDLE
- Check Isaac Gym installation and paths

**Tensor shape mismatches**
- Verify observation dimensions match between environment and network
- Check batch dimensions are consistent `[num_envs, ...]`

**Training not progressing**
- Confirm trigger connection from TrainingStep to IsaacGymStep
- Check state caching is working (enable debug logging)

This architecture represents a sophisticated solution to RL training synchronization that maintains DNNE's visual programming paradigm while enabling robust reinforcement learning workflows.