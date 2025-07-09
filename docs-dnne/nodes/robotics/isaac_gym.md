# Robotics Isaac Gym Nodes

## Overview

Isaac Gym nodes provide integration with NVIDIA's physics simulation platform for robotics and reinforcement learning. These nodes enable high-performance robot simulation with GPU acceleration.

---

## IsaacGymEnvNode

### Purpose
Initializes Isaac Gym simulation environments for robot control tasks.

### Category
`Robotics/Isaac Gym` (registered as "Isaac Gym Environment")

### Inputs
None (Sensor node)

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| env_name | string | "Cartpole" | Environment name (Cartpole, Ant, Humanoid, etc.) |
| num_envs | int | 1 | Number of parallel environments |
| headless | boolean | true | Run without visualization |
| device | string | "cuda" | Compute device ("cuda" or "cpu") |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| sim | SimHandle | Simulation handle |
| observations | Tensor | Initial environment observations |

### Usage Example
```
IsaacGymEnvNode → ORNode → PPOAgent → Action → IsaacGymStepNode
```

Supported environments:
- **Cartpole**: Classic balance control
- **Ant**: Quadruped locomotion
- **Humanoid**: Bipedal walking
- **Generic**: Custom environments

### Export Support
✅ Full support with queue template

### Notes
- Requires NVIDIA GPU for best performance
- Automatically handles device placement
- Supports vectorized environments
- Integrates with IsaacGymEnvs library

---

## IsaacGymStepNode

### Purpose
Steps the physics simulation forward and returns new observations and rewards.

### Category
`Robotics/Isaac Gym` (registered as "Isaac Gym Step")

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| sim | SimHandle | Simulation handle from env node |
| actions | Tensor | Actions to apply |
| trigger | Any | Optional trigger for synchronization |

### Parameters
None

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| observations | Tensor | New state observations |
| rewards | Tensor | Step rewards |
| dones | Tensor | Episode termination flags |

### Usage Example
```
IsaacGymEnvNode → IsaacGymStepNode ← Actions
                        ↓
              [Observations, Rewards, Dones] → PPOTrainer
```

Features:
- Caches observations for efficiency
- Handles environment resets automatically
- Supports multi-environment stepping
- GPU-accelerated physics

### Export Support
✅ Full support with observation caching

### Notes
- Synchronizes with training via trigger input
- Efficiently manages state transitions
- Handles episode boundaries
- Thread-safe for async execution

---

## CartpoleActionNode

### Purpose
Converts neural network outputs to Cartpole-specific control actions.

### Category
`Robotics/Control` (registered as "Cartpole Action Converter")

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| network_output | Tensor/PolicyOutput | Raw network output |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_push_effort | float | 10.0 | Maximum force to apply |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| actions | Tensor | Scaled actions for environment |

### Usage Example
```
PPOAgent → CartpoleActionNode → IsaacGymStep
```

Action scaling:
- Takes continuous values from network
- Scales to [-max_push_effort, max_push_effort]
- Handles both single and batched actions

### Export Support
✅ Full support

### Notes
- Extracts action from PolicyOutput if needed
- Ensures proper tensor format
- Maintains batch dimensions

---

## AntActionNode *(Coming Soon)*

### Purpose
Converts network outputs to Ant robot joint torques.

### Category
`Robotics/Control`

### Expected Features
- 8 joint torque outputs
- Torque limiting
- Action smoothing options

---

## HumanoidActionNode *(Coming Soon)*

### Purpose
Converts network outputs to Humanoid robot control signals.

### Category
`Robotics/Control`

### Expected Features
- 21+ joint controls
- Balance assistance
- Gait pattern support

---

## Environment Integration Patterns

### Basic RL Loop
```
IsaacGymEnv → ORNode → Policy → ActionNode → IsaacGymStep
                 ↑                                  ↓
                 └──────────────────────────────────┘
```

### Multi-Environment Training
```
IsaacGymEnv(num_envs=16) → Parallel stepping and training
```

### Custom Environment
```
IsaacGymEnv(env_name="YourEnv") → Requires custom action node
```

## Best Practices

### Performance Optimization
- Use GPU device when available
- Increase num_envs for parallel training
- Enable headless mode for training
- Batch actions for efficiency

### Environment Setup
- Start with 1 environment for debugging
- Scale to 16-64 for training
- Match action node to environment type
- Monitor GPU memory usage

### Action Processing
- Always use appropriate action node
- Check action space dimensions
- Handle continuous vs discrete actions
- Scale actions appropriately

## Common Issues

### Issue: CUDA out of memory
**Solution**: Reduce num_envs or use smaller networks

### Issue: Environment not found
**Solution**: Ensure IsaacGymEnvs is properly installed

### Issue: Actions have wrong shape
**Solution**: Verify action node matches environment

### Issue: Slow simulation
**Solution**: Enable GPU, use headless mode, check CPU bottlenecks

## Integration with RL

### PPO Training
```python
# Typical setup
env = IsaacGymEnvNode(
    env_name="Cartpole",
    num_envs=16,
    headless=True,
    device="cuda"
)
```

### Observation Processing
- Observations are normalized
- Include positions, velocities, etc.
- Shape: (num_envs, obs_dim)

### Reward Shaping
- Environment-specific rewards
- Can add custom reward nodes
- Supports sparse and dense rewards

## Future Enhancements

- More environment types
- Custom environment support
- Domain randomization
- Sim-to-real transfer tools
- Multi-agent environments