# IsaacGymSimNode

## Overview

The IsaacGymSimNode provides a queue-based interface to Isaac Gym environments, enabling proper integration with DNNE's async architecture. This node acts as a pure environment interface that follows DNNE's queue patterns, unlike PPOAgentNode which wraps the entire training process.

## Why This Node Exists

The existing PPOAgentNode is a monolithic wrapper around IsaacGymEnvs' train.py script. While this works for PPO training, it doesn't allow for:
- Custom RL algorithms beyond PPO
- Step-by-step environment interaction
- Integration with other DNNE components
- Debugging and visualization of individual steps

IsaacGymSimNode solves these issues by providing a clean, queue-based interface to Isaac Gym environments that can be composed with any other DNNE nodes.

## Node Details

### Category
`robotics`

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| config | ISAAC_ENV_CONFIG | Environment configuration from Isaac Gym Environment Config node |
| action | TENSOR | Actions to execute in the environment |
| reset | TRIGGER | (Optional) Manual reset trigger |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| observation | TENSOR | Environment observations after each step |
| done | TRIGGER | (Optional) Sends trigger when episode ends |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| reset_when_done | BOOLEAN | True | Automatically reset environment when episode ends |
| render | BOOLEAN | False | Enable rendering for debugging/visualization |

## How It Works

1. **Initialization**: Uses the config from Isaac Gym Environment Config node to create a single environment instance
2. **Action Loop**: Waits for actions on the input queue
3. **Step**: Executes the action in the environment
4. **Output**: Sends observations to the output queue
5. **Episode End**: 
   - Sends done trigger (if connected)
   - Auto-resets if `reset_when_done` is True
   - Otherwise waits for manual reset trigger

## Usage Examples

### Basic Environment Loop
```
Isaac Gym Env Config → IsaacGymSim
                            ↑ action    ↓ observation
                         Policy ←────────┘
```

### With Manual Reset Control
```
Isaac Gym Env Config → IsaacGymSim
                            ↑ action    ↓ observation
                         Policy ←────────┘
                            ↑ reset     ↓ done
                    Reset Controller ←───┘
```

### Custom RL Training
```
Isaac Gym Env Config → IsaacGymSim
                            ↑ action    ↓ observation
                    Exploration Policy ←─┘
                            ↓
                      Replay Buffer → Training Algorithm
```

## Implementation Notes

- Forces `num_envs=1` since DNNE doesn't support vectorization
- Uses `isaacgymenvs.make()` to create environments
- Handles tensor device placement automatically
- Provides clean async interface following DNNE patterns

## Differences from PPOAgentNode

| Feature | PPOAgentNode | IsaacGymSimNode |
|---------|--------------|-----------------|
| Purpose | Complete PPO training | Environment interface only |
| Architecture | Monolithic wrapper | Queue-based components |
| Flexibility | PPO only | Any algorithm |
| Integration | Standalone | Composable with other nodes |
| Control | Automated training | Step-by-step control |

## Export Support

✅ Full support with queue template
- Generates standalone async environment interface
- Compatible with all Isaac Gym environments
- Handles GPU/CPU device placement

## Future Enhancements

- Support for multiple environments (when DNNE adds vectorization)
- Additional outputs (rewards, info dict)
- Environment-specific parameter overrides
- Visual debugging integration