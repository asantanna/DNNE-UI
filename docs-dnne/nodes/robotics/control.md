# Robotics Control Nodes

## Overview

Control nodes provide decision-making and control capabilities for robotics applications. These nodes bridge perception and action in robotic systems.

---

## DecisionNetworkNode *(Coming Soon)*

### Purpose
Implements decision-making networks for robot control systems.

### Category
`Robotics/Control`

### Expected Inputs
| Input | Type | Description |
|-------|------|-------------|
| sensor_data | Tensor | Processed sensor inputs |
| state | Tensor | Current robot state |

### Expected Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| architecture | string | "mlp" | Network architecture type |
| hidden_sizes | list | [128, 64] | Hidden layer dimensions |
| decision_type | string | "continuous" | Output type |

### Expected Outputs
| Output | Type | Description |
|--------|------|-------------|
| decision | Tensor | Control decision |
| confidence | Tensor | Decision confidence |

### Planned Features
- Multiple decision architectures
- Hierarchical decision making
- Uncertainty estimation
- Multi-modal input fusion

---

## CartpoleRewardNode

### Purpose
Calculates custom rewards for Cartpole balancing task.

### Category
`Robotics/Control` (registered as "Cartpole Reward Calculator")

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| observations | Tensor | Environment observations |
| actions | Tensor | Applied actions |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| alive_bonus | float | 1.0 | Reward per timestep |
| angle_penalty | float | 0.1 | Penalty for angle deviation |
| position_penalty | float | 0.01 | Penalty for cart position |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| rewards | Tensor | Computed rewards |

### Usage Example
```
IsaacGymStep → CartpoleRewardNode → PPOTrainer
```

### Export Support
✅ Full support

### Notes
- Allows custom reward shaping
- Can replace default environment rewards
- Useful for curriculum learning

---

## IMUSensorNode

### Purpose
Simulates Inertial Measurement Unit data for robotics applications.

### Category
`Robotics/Sensors` (registered as "IMU Sensor")

### Inputs
| Input | Type | Description |
|-------|------|-------------|
| robot_state | Tensor | Current robot state |

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| noise_level | float | 0.01 | Sensor noise magnitude |
| sample_rate | int | 100 | Sampling frequency (Hz) |

### Outputs
| Output | Type | Description |
|--------|------|-------------|
| imu_data | Tensor | Accelerometer and gyroscope readings |

### Usage Example
```
RobotState → IMUSensorNode → DecisionNetwork
```

### Export Support
✅ Full support

### Notes
- Simulates realistic sensor noise
- Provides 6-DOF measurements
- Can be used for state estimation

---

## Control Patterns

### Reactive Control
```
Sensors → DecisionNetwork → Actions → Robot
```

### Hierarchical Control
```
HighLevel → Goals → LowLevel → Actions
```

### Sensor Fusion
```
[Camera, IMU, Lidar] → Fusion → DecisionNetwork
```

## Best Practices

### Control Design
- Start with simple reactive controllers
- Add complexity gradually
- Test in simulation first
- Validate with real hardware

### Sensor Integration
- Handle sensor noise appropriately
- Fuse multiple sensor modalities
- Account for sensor delays
- Implement fallback strategies

### Decision Making
- Use appropriate network architectures
- Consider computational constraints
- Implement safety checks
- Monitor decision confidence

## Common Issues

### Issue: Unstable control
**Solution**: Reduce control gains, add damping, check sensor data

### Issue: Delayed responses
**Solution**: Minimize processing time, use predictive control

### Issue: Sensor noise affecting control
**Solution**: Implement filtering, use robust control methods

## Integration Examples

### With PPO
```
IMU → DecisionNetwork → PPOAgent → Actions
```

### With Classical Control
```
Sensors → PIDController → Actions
        → DecisionNetwork (supervisor)
```

### Multi-Modal Control
```
Vision → FeatureExtractor ↘
IMU → Processing         → DecisionNetwork → Actions
Touch → ContactDetection ↗
```

## Future Enhancements

- Model Predictive Control nodes
- Adaptive control algorithms
- Learning-based sensor fusion
- Safety-critical control systems
- Real-time performance guarantees