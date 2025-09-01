# SimulationTracker Node

## Overview

The SimulationTracker node is a critical component for monitoring and controlling reinforcement learning (RL) and robotics training progress. It tracks episodes, rewards, losses, and performance metrics while providing control signals to manage the training loop lifecycle.

## Purpose

SimulationTracker serves as the central monitoring and control hub for RL/robotics simulations by:
- Tracking training progress across episodes and timesteps
- Computing running statistics and performance metrics
- Detecting training completion conditions (max episodes, success threshold, convergence)
- Providing control signals to downstream nodes
- Reporting telemetry data for monitoring and visualization

## Inputs

### Required Inputs

| Input | Type | Description |
|-------|------|-------------|
| **observation** | `*SIM_OBSERVATION_TENSOR` | Observation tensor from the simulator containing state information. Each observation increments the timestep counter. |
| **done** | `*TRIGGER` | Episode completion signal from the simulator. Any value on this input triggers episode end processing. |

### Optional Inputs

| Input | Type | Description |
|-------|------|-------------|
| **loss** | `*LOSS_SCALAR` | Training loss from policy/value networks for tracking learning progress |
| **reward** | `*REWARD_SCALAR` | Current step reward from the environment. Accumulated to compute episode rewards |
| **custom_metrics** | `*METRICS_PYDICT` | Task-specific metrics dictionary (e.g., `{"success": True, "distance": 0.5}`) |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **max_episodes** | INT | 1000 | Maximum number of episodes before stopping |
| **success_threshold** | FLOAT | 0.95 | Success rate threshold for early stopping |
| **telemetry_mode** | COMBO | "time" | Telemetry reporting mode: "time", "steps", or "episodes" |
| **telemetry_interval** | STRING | "10s" | Reporting interval (e.g., "10s", "5m", "100" for steps) |
| **telemetry_stats** | BOOLEAN | True | Include statistical aggregations in telemetry |

## Input Processing Behavior

### When `observation` is received:
- Increments timestep counter
- Updates current episode length
- Marks the start of training if first observation

### When `loss` is received:
- Appends to loss history
- Buffers for telemetry aggregation (if enabled)
- Does NOT report immediately (aggregated reporting only)

### When `reward` is received:
- Accumulates into current episode reward total
- Buffers for telemetry aggregation (if enabled)

### When `done` is received (episode completion):
- Records episode statistics (reward, length)
- Checks for performance improvements
- Evaluates success criteria (from custom_metrics or reward > 0)
- Resets episode accumulators for next episode
- Checks training completion conditions

### When `custom_metrics` is received:
- Extracts "success" field if present for success tracking
- Can be used to pass task-specific completion criteria

## Output

### control_metrics (CONTROL_METRICS_PYDICT)

The node outputs a comprehensive metrics dictionary on every timestep:

```python
{
    # Core control signals
    "episode": 42,              # Current episode number
    "timestep": 4200,           # Total timesteps across all episodes
    "done": False,              # True when training should stop
    
    # Episode-level metrics
    "episode_done": False,      # True when current episode completes
    "episode_reward": 125.3,    # Total reward for current/last episode
    "avg_reward": 98.7,         # Average reward over window (100 episodes)
    
    # Performance tracking
    "success_rate": 0.82,       # Success rate over window
    "improvement_rate": 0.05,   # Relative improvement rate
    "best_reward": 145.2,       # Best episode reward achieved
    
    # Additional statistics
    "avg_episode_length": 100,  # Average episode length
    "episodes_since_improvement": 5,  # Episodes since best_reward updated
    
    # Loss tracking (if available)
    "latest_loss": 0.023,       # Most recent loss value
    "avg_loss": 0.031          # Average loss over window
}
```

## Training Completion Conditions

Training stops (sets `done=True`) when ANY of these conditions are met:

1. **Max Episodes**: Episode count reaches `max_episodes`
2. **Success Threshold**: Success rate ≥ `success_threshold` (after window_size episodes)
3. **Convergence**: No improvement in `convergence_window` episodes (default: 500 or max_episodes/10)

When training completes, the node raises a `CauseExitException` to cleanly stop the graph runner.

## Telemetry Reporting

When telemetry is enabled (`telemetry_enabled=True` in node config), the node reports aggregated statistics at configurable intervals.

### Reporting Modes

#### Time-based (default)
- Reports every N seconds (e.g., "10s", "5m", "2m30s")
- Useful for real-time monitoring

#### Step-based
- Reports every N timesteps (e.g., "100", "500")
- Ensures consistent data density

#### Episode-based
- Reports every N episodes (e.g., "5", "10")
- Aligned with episode boundaries

### Telemetry Metrics

When `telemetry_stats=True`, reports include:
- **Loss statistics**: mean, min, max, std, percentiles (p25, p50, p75)
- **Reward statistics**: mean, min, max, std
- **Episode statistics**: episode rewards/lengths with aggregations
- **Success rate**: Rolling window success percentage

When `telemetry_stats=False`, only latest values are reported.

### Telemetry Data Fields

```python
# Per reporting interval
"loss_mean", "loss_min", "loss_max", "loss_std"
"loss_p25", "loss_p50", "loss_p75"
"reward_mean", "reward_min", "reward_max"
"episode_reward_mean", "episode_length_mean"
"success_rate"
"report_timestep", "report_episode"
```

## Usage Example

### Basic RL Training Loop

```
[Isaac Gym Sim] --observation--> [SimulationTracker] --control_metrics--> [Control Logic]
      |                               ^        ^
      |--done-------------------------|        |
      |--reward---------------------------------|
      
[PPO Agent] --loss--> [SimulationTracker]
```

### With Custom Success Metrics

```python
# In your environment or custom node:
custom_metrics = {
    "success": distance_to_target < 0.1,
    "distance": distance_to_target,
    "energy_used": total_energy
}
# Send to SimulationTracker's custom_metrics input
```

## Integration Notes

1. **Episode Boundaries**: The `done` input accepts ANY value as a trigger - it doesn't need to be a boolean. This allows flexible integration with various simulator outputs.

2. **Timestep Counting**: Only observations increment the timestep counter, ensuring accurate step counting even if other inputs arrive at different rates.

3. **Window Size**: Statistics are computed over a rolling window of 100 episodes by default for stable metrics.

4. **Early Stopping**: The node implements multiple early stopping criteria to prevent unnecessary training and detect convergence.

5. **Telemetry Overhead**: Telemetry reporting is designed to have minimal impact on training performance through buffering and aggregation.

## Configuration Tips

- **For Quick Experiments**: Use default settings with `max_episodes=100`
- **For Production Training**: Increase `max_episodes` and adjust `success_threshold` based on task difficulty
- **For Debugging**: Set `telemetry_mode="steps"` with `telemetry_interval="10"` for frequent updates
- **For Long Runs**: Use `telemetry_mode="time"` with `telemetry_interval="30s"` to reduce data volume

## See Also

- [EpochTracker](../ml/epoch_tracker.md) - Similar tracking for supervised learning
- [Isaac Gym Sim](./isaac_gym_sim.md) - Primary simulator integration
- [PPO Agent](../rl/ppo_agent.md) - Common RL algorithm used with SimulationTracker