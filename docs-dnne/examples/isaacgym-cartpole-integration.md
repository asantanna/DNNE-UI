# IsaacGym Cartpole Integration Reference

## Overview

This document provides a comprehensive analysis of the IsaacGymEnvs Cartpole implementation, serving as a reference for proper DNNE integration with Isaac Gym environments.

## Problem Analysis

### Current DNNE Issues
The DNNE Cartpole integration shows static, identical cartpoles because:
1. **No randomized initialization** - all environments start with identical state
2. **No action application** - forces aren't applied to the simulation
3. **Hardcoded observations** - returning zeros instead of reading physics state
4. **No episode management** - no resets or termination conditions

### Expected Behavior
Proper cartpole environments should:
- Start with randomized cart positions and pole angles
- Respond to policy actions with realistic physics movement
- Automatically reset when termination conditions are met
- Provide accurate state observations from the physics simulation

## IsaacGymEnvs Reference Implementation

### 1. Environment Structure

**File**: `/home/asantanna/IsaacGymEnvs/isaacgymenvs/tasks/cartpole.py`

#### DOF Configuration
- **DOF 0**: Cart position (prismatic joint, y-axis, range -4 to +4m)
- **DOF 1**: Pole angle (continuous joint, x-axis, unlimited rotation)

```python
# From cartpole.py lines 109-114
dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT  # Cart controlled by force
dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE    # Pole free-swinging
dof_props['stiffness'][:] = 0.0
dof_props['damping'][:] = 0.0
```

#### Action/Observation Spaces
- **Action Space**: 1D force applied to cart (-400N to +400N)
- **Observation Space**: 4D [cart_pos, cart_vel, pole_angle, pole_angular_vel]

### 2. Randomized Initialization Pattern

**Critical Feature**: `reset_idx()` method with proper randomization

```python
# From cartpole.py lines 144-149
def reset_idx(self, env_ids):
    positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
    velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
    
    self.dof_pos[env_ids, :] = positions[:]
    self.dof_vel[env_ids, :] = velocities[:]
```

**Key Insights**:
- Uses `torch.rand() - 0.5` for symmetric random ranges around zero
- Cart position: -0.1 to +0.1 meters
- Pole angle: -0.1 to +0.1 radians
- Cart velocity: -0.25 to +0.25 m/s
- Pole angular velocity: -0.25 to +0.25 rad/s

### 3. Force-Based Action Application

**Method**: `pre_physics_step(actions)`

```python
# From cartpole.py lines 159-163
def pre_physics_step(self, actions):
    actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
    actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
    forces = gymtorch.unwrap_tensor(actions_tensor)
    self.gym.set_dof_actuation_force_tensor(self.sim, forces)
```

**Key Patterns**:
- Actions applied as **forces** (not positions/velocities)
- Only **DOF 0 (cart)** receives force input
- Uses **strided indexing** `[::self.num_dof]` to apply forces to cart DOF only
- **max_push_effort = 400.0N** from config provides force scaling

### 4. Observation Extraction

**Method**: `compute_observations(env_ids=None)`

```python
# From cartpole.py lines 131-142
def compute_observations(self, env_ids=None):
    if env_ids is None:
        env_ids = np.arange(self.num_envs)

    self.gym.refresh_dof_state_tensor(self.sim)  # CRITICAL: Must refresh first!

    self.obs_buf[env_ids, 0] = self.dof_pos[env_ids, 0].squeeze()  # Cart position
    self.obs_buf[env_ids, 1] = self.dof_vel[env_ids, 0].squeeze()  # Cart velocity  
    self.obs_buf[env_ids, 2] = self.dof_pos[env_ids, 1].squeeze()  # Pole angle
    self.obs_buf[env_ids, 3] = self.dof_vel[env_ids, 1].squeeze()  # Pole angular velocity

    return self.obs_buf
```

**Critical**: Must call `refresh_dof_state_tensor()` before reading state!

### 5. Episode Management

#### Reward Function
```python
# From cartpole.py lines 185-190 (JIT compiled)
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos, reset_dist, ...):
    # Reward for staying upright with minimal movement
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
    
    # Penalty for failure conditions
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
```

#### Termination Conditions
- Cart position too far: `|cart_pos| > 3.0m` (reset_dist)
- Pole fallen: `|pole_angle| > π/2 radians`
- Episode timeout: `progress_buf >= 500 steps` (max_episode_length)

#### Automatic Reset Trigger
```python
# From cartpole.py lines 165-173
def post_physics_step(self):
    self.progress_buf += 1
    
    env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(env_ids) > 0:
        self.reset_idx(env_ids)  # Trigger randomized reset
```

### 6. Asset Configuration

**URDF Structure** (`/home/asantanna/IsaacGymEnvs/assets/urdf/cartpole.urdf`):
- **slider** (fixed base): Rail for cart movement
- **cart** (mass=1kg): Movable cart on rail  
- **pole** (mass=1kg, length=1m): Pole attached to cart

**Joint Configuration**:
- `slider_to_cart`: Prismatic joint (y-axis, limits -4 to +4m)
- `cart_to_pole`: Continuous joint (x-axis, unlimited rotation)

### 7. Simulation Parameters

**From Cartpole.yaml**:
```yaml
sim:
  dt: 0.0166          # 1/60 s (60 FPS)
  substeps: 2
  up_axis: "z"
  gravity: [0.0, 0.0, -9.81]

env:
  resetDist: 3.0      # Cart position limit
  maxEffort: 400.0    # Maximum force magnitude
```

## DNNE Integration Requirements

### 1. Proper Class Architecture
- Base `IsaacGymEnvironment` class with abstract methods
- `CartpoleEnvironment` subclass implementing Cartpole-specific logic
- Clean separation of concerns

### 2. Required Methods
- `reset_environments(env_ids)` - Randomized initialization
- `apply_actions(actions)` - Force-based action application  
- `get_observations()` - Physics state extraction with refresh
- `compute_rewards()` - Environment-specific reward calculation
- `check_termination()` - Failure condition detection

### 3. State Management
- DOF state tensors (`dof_pos`, `dof_vel`)
- Observation buffers with proper device placement
- Reset and progress tracking per environment

### 4. Integration Points
- Initialize environments with randomization on startup
- Apply actions before each physics step
- Extract observations after each physics step
- Handle automatic resets when termination conditions are met

## Implementation Priority

1. **Base Environment Class** - Abstract interface
2. **Cartpole Environment Class** - Concrete implementation
3. **Template Integration** - Update Isaac Gym node templates
4. **Testing** - Verify randomized initialization and physics response

This reference provides the foundation for implementing proper Isaac Gym Cartpole integration in DNNE that matches the behavior and performance of the original IsaacGymEnvs implementation.