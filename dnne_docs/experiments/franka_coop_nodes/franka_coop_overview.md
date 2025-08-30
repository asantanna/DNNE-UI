# Franka Cooperative Control Experiment Overview

## Experiment Hypothesis
**Can multiple independent neural networks learn to cooperatively control a robot's joints using only a shared global loss signal, without explicit inter-network communication?**

## Background

### FrankaDNNE Environment
FrankaDNNE is a hierarchical Isaac Gym environment cloned from the Franka cube stacking task, designed specifically for DNNE experimentation.

#### Environment Hierarchy
```
FrankaDNNE (Task)
└── Subtask Selection (What to do)
    ├── random_target: Reach/touch a randomly placed target
    │   ├── joint_tor: Joint torque control
    │   └── osc: Operational Space Control
    └── pick_place: Pick and place objects
        ├── joint_tor: Joint torque control
        └── osc: Operational Space Control
```

#### Schema System Purpose
The schemas in FrankaDNNE.yaml serve as **UI documentation**, displaying human-readable descriptions of tensor elements in the DNNE interface. This helps users:
- Understand what each observation index represents
- Configure Split/Concat nodes correctly
- Map between semantic names and tensor indices

### Current Implementation Status
- **YAML Configuration**: Updated to match actual implementation ✅
- **Python Task**: Exists and working (random_target subtask)
- **Workflow**: Franka_Coop_Nodes with 31 nodes, 45 connections
- **Loss Function**: Implemented distance-based loss ✅
- **Split Nodes**: Updated to use correct field names ✅

## Workflow Architecture

### Core Design: Distributed Control
```
Isaac Gym Simulator
        ↓ (observations)
    Split Node
    ↙   ↓   ↘
Net1  Net2  Net3  (Independent controllers)
  ↓     ↓     ↓   (torque outputs)
    Concat Node
        ↓ (combined actions)
Isaac Gym Simulator
        ↓
Custom Loss Computation
    ↙   ↓   ↘
Train1 Train2 Train3  (Shared global loss)
```

### Key Components

#### 1. Observation Distribution
- **Split Node 56**: Extracts `target_pos, eef_pos` (shared global state)
- **Split Node 45**: Extracts individual joint angles from 9 total joints
- **Concat Nodes (47, 55, 57)**: Combine shared state with individual joint angle

**What each network sees**:
- **Network 33 (Joint 0)**: `[target_pos(3), eef_pos(3), joint_theta[0](1)]` = 7 dims
- **Network 54 (Joint 1)**: `[target_pos(3), eef_pos(3), joint_theta[1](1)]` = 7 dims  
- **Network 62 (Joint 2)**: `[target_pos(3), eef_pos(3), joint_theta[2](1)]` = 7 dims

Each network receives the same global state (where to reach) but only its own joint's current angle, forcing specialization.

#### 2. Independent Networks
- **Network 33**: Controls Joint 0 (base rotation)
- **Network 54**: Controls Joint 1 (shoulder)
- **Network 62**: Controls Joint 2 (elbow)
- Each has 3 layers: 128→128→1 neurons
- ReLU activation, dropout=0

#### 3. Action Aggregation
- **Concat Node 42**: Combines individual torque outputs
- Mode: "as available" with "hold previous"
- **Tensor Node 69**: Provides 5 zeros for uncontrolled joints (3-6)
- **Final Action**: `[net33_output, net54_output, net62_output, 0, 0, 0, 0]` = 7 torques

Note: Currently controlling joints 0, 1, 2 while joints 3-6 receive zero torque (free to move passively).

#### 4. Shared Loss Signal
- **CustomComputation Node 67**: Computes distance to target
- Broadcasts same loss to all three training steps
- No gradient sharing between networks

## Implementation Challenges

### 1. ✅ RESOLVED: Schema-Implementation Alignment
**Solution**: Updated schema to match actual implementation (implementation is ground truth)

**Actual Observations** (joint_tor mode, 20 dims):
```python
target_pos: [0, 2]       # Target position (x,y,z)
eef_pos: [3, 5]          # End-effector position (x,y,z)
eef_quat: [6, 9]         # End-effector quaternion (x,y,z,w)
joint_theta: [10, 18]    # 9 joint angles in radians (7 arm + 2 gripper)
episode_time: [19, 19]   # Episode elapsed time in seconds
```

### 2. ✅ RESOLVED: Loss Function
Implemented proper distance-based loss:
1. Extracts eef_pos (indices 3-5)
2. Extracts target_pos (indices 0-2)
3. Computes L2 distance using `torch.norm`
4. Returns scalar loss (mean for batching)

### 3. Joint Selection and Locking Strategy ✅ RESEARCHED

#### Franka Joint Structure
The Franka Panda robot has 7 revolute joints (+ 2 gripper fingers):

| Joint Index | URDF Name | Description | Typical Role |
|------------|-----------|-------------|--------------|
| 0 | panda_joint1 | Base rotation | Rotates around vertical axis |
| 1 | panda_joint2 | Shoulder pitch | First major arm movement |
| 2 | panda_joint3 | Shoulder roll | Secondary shoulder movement |
| 3 | panda_joint4 | Elbow pitch | Elbow flexion/extension |
| 4 | panda_joint5 | Forearm roll | Forearm rotation |
| 5 | panda_joint6 | Wrist pitch | Wrist up/down |
| 6 | panda_joint7 | Wrist roll | Wrist rotation |
| 7-8 | gripper | Finger joints | Gripper open/close |

#### Recommended 3-Joint Configuration
For simplified control with good workspace coverage:
- **Joint 0 (Base)**: Provides rotation around vertical axis for orientation control
- **Joint 1 (Shoulder)**: Provides vertical reach and primary arm movement
- **Joint 3 (Elbow)**: Provides extension/retraction for reaching targets

Note: Skipping joint 2 (shoulder roll) and using joint 3 (elbow) gives better workspace coverage than consecutive joints.

#### Freezing Non-Controlled Joints

**Option 1: Position Control Mode (Recommended)**
Set joints [2, 4, 5, 6] to position control mode with high stiffness:
```python
# For frozen joints
franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
franka_dof_props['stiffness'][i] = 5000.0  # High stiffness to lock in place
franka_dof_props['damping'][i] = 100.0      # High damping for stability
```

**Option 2: Zero Torque with Damping**
Keep joints in effort mode but apply zero torque and rely on damping:
```python
# For frozen joints in effort mode
franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
franka_dof_props['stiffness'][i] = 0.0
franka_dof_props['damping'][i] = 50.0  # Damping prevents drift
# Then always apply zero torque to these joints
```

#### Safe Torque Limits for Training
For safe initial training with untrained networks:
- Joint 0 (Base): ±1.0 Nm
- Joint 1 (Shoulder): ±1.0 Nm  
- Joint 3 (Elbow): ±1.0 Nm

These are much smaller than the robot's actual limits but prevent instability during early training.

## Experimental Phases

### Phase 1: Two-Joint Control (Simplified)
- **Active Joints**: 0 (base), 3 (elbow)
- **Locked Joints**: 1 (shoulder), 2 (shoulder roll) - position hold
- **Frozen Joints**: 4-6 (wrist/forearm) - position hold
- **Expected Behavior**: Planar reaching in cylindrical coordinates
- **Success Criteria**: Consistent reaching within 5cm of target

### Phase 2: Three-Joint Control (Recommended)
- **Active Joints**: 0 (base), 1 (shoulder), 3 (elbow)
- **Locked Joint**: 2 (shoulder roll) - position hold
- **Frozen Joints**: 4-6 (wrist/forearm) - position hold
- **Expected Behavior**: Full 3D reaching capability
- **Coordination Test**: Do joints learn complementary strategies?
  - Base for azimuth positioning
  - Shoulder for elevation
  - Elbow for reach extension

### Phase 3: Analysis
- **Emergent Behaviors to Look For**:
  - Temporal coordination (who moves first?)
  - Role specialization (coarse vs fine control)
  - Error correction patterns
- **Metrics**:
  - Convergence speed vs single-network baseline
  - Final accuracy (mean distance to target)
  - Trajectory smoothness
  - Energy efficiency (total torque squared)

## Success Metrics

1. **Primary**: Distance to target < 5cm after 1000 episodes
2. **Coordination**: 
   - Joint velocities show correlation patterns
   - Reduced oscillation/fighting between controllers
3. **Efficiency**: 
   - Learning speed vs single 7-DOF network
   - Total torque usage (energy efficiency)
4. **Robustness**: 
   - Performance with added noise
   - Adaptation to new target ranges
5. **Emergent Specialization**:
   - Measure mutual information between joint actions
   - Identify leader-follower dynamics

## Technical Notes

### Tensor Indices (joint_tor mode)
Actual observations from implementation:
- `[0-2]`: target_pos (x,y,z)
- `[3-5]`: eef_pos (x,y,z)
- `[6-9]`: eef_quat (quaternion)
- `[10-18]`: joint_theta (9 joint angles in radians - 7 arm + 2 gripper)
- `[19]`: episode_time (seconds)

### Queue-Based Execution
All nodes use async queue-based patterns for real-time performance, similar to ROS (Robot Operating System).

### Balancer Nodes
Three balancer nodes (44, 50, 64) ensure synchronized execution across parallel training paths.

## CSV Data Generation for Training

### Requirements for Exploration Data
For training data collection with 3-joint control, we need slow, controlled torque changes to allow the robot to respond and reach steady states.

### Two CSV Generation Modes

#### Mode 1: Smooth Sinusoidal Exploration
Generate smooth, continuous torque commands using sinusoidal patterns:
```python
# Example pattern for each joint
torque_joint0 = 0.5 * sin(2π * 0.1 * t + phase0)  # Slow 10-second period
torque_joint1 = 0.5 * sin(2π * 0.15 * t + phase1) # Different frequency
torque_joint3 = 0.5 * sin(2π * 0.08 * t + phase3) # Another frequency
```
- Use different frequencies for each joint (0.05-0.2 Hz)
- Random phase offsets to create variety
- Amplitude range: [-1.0, 1.0] Nm
- Benefits: Smooth trajectories, continuous exploration, no sudden jerks

#### Mode 2: Step-and-Hold Exploration
Generate random torque values and hold them steady for extended periods:
```python
# Pseudocode
every 1-2 seconds:
    torque_joint0 = random.uniform(-1.0, 1.0)
    torque_joint1 = random.uniform(-1.0, 1.0)
    torque_joint3 = random.uniform(-1.0, 1.0)
    hold these values for next 1-2 seconds
```
- Hold duration: 1-2 seconds (randomized)
- Allows robot to reach steady state
- Creates long sweeping motions
- Benefits: Clear cause-effect relationships, steady-state data points

### Data Fields to Collect
For both modes, collect at 100 Hz:
- **Applied torques** (3 values for joints 0, 1, 3)
- **All joint positions** (7 values in radians)
- **All joint velocities** (7 values in rad/s)
- **End-effector position** (3 values: x, y, z)
- **End-effector quaternion** (4 values: x, y, z, w)
- **Target position** (3 values: x, y, z) - can be constant or varying
- **Timestamp** (seconds from start)

### File Naming Convention
- `franka_sinusoidal_exploration_001.csv` - For smooth sinusoidal data
- `franka_step_hold_exploration_001.csv` - For step-and-hold data

## Next Steps

1. **Implement Joint Configuration**: Modify `franka_dnne.py` to use joints [0, 1, 3] with others locked
2. **Create CSV Generators**: Build both sinusoidal and step-hold data generation scripts
3. **Test Export**: Run `programmatic_export.py Shadow_Train` with generated CSV
4. **Run Training**: Monitor convergence and coordination patterns
5. **Iterate on Architecture**:
   - Adjust network sizes
   - Tune learning rates
   - Experiment with different loss functions

## Related Files

- **Workflow**: `/home/asantanna/DNNE/DNNE-UI/user/default/workflows/Franka_Coop_Nodes.json`
- **Environment Config**: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg/task/FrankaDNNE.yaml`
- **Task Implementation**: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/tasks/franka_dnne.py`
- **Loss Function**: `/home/asantanna/DNNE/DNNE-UI/user/default/custom_compute_funcs/franka_coop_nodes_loss.py`
- **Task Tracking**: `/home/asantanna/DNNE/DNNE-UI/dnne_docs/for_claude/tasks/franka_coop/TASKS.md`
- **Workflow Analyzer**: `/home/asantanna/DNNE/DNNE-UI/claude_scripts/analyze_workflow.py`