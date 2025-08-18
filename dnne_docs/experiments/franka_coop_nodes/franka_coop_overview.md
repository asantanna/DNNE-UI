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
- **Split Node 45**: Extracts individual joint angles `joint_theta[0], [1], [2]`
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

**Actual Observations** (joint_tor mode, 18 dims):
```python
target_pos: [0, 2]       # Target position (x,y,z)
eef_pos: [3, 5]          # End-effector position (x,y,z)
eef_quat: [6, 9]         # End-effector quaternion (x,y,z,w)
joint_theta: [10, 16]    # 7 joint angles in radians
episode_time: [17, 17]   # Episode elapsed time in seconds
```

### 2. ✅ RESOLVED: Loss Function
Implemented proper distance-based loss:
1. Extracts eef_pos (indices 3-5)
2. Extracts target_pos (indices 0-2)
3. Computes L2 distance using `torch.norm`
4. Returns scalar loss (mean for batching)

### 3. Joint Locking Strategy ⚠️ TODO
For 2-3 joint control while Franka has 7 DOF:

**Current Implementation**: Zero torque (joints 3-6 free to move)

**Better Options to Implement**:
1. **Position Hold**: PD controller to maintain initial position
   - Add PD control nodes for joints 3-6
   - Use initial joint positions as setpoints
2. **High Damping**: Apply `-k_d * joint_velocity` torques
   - Would need joint velocities in observations
3. **Gravity Compensation**: Counter gravity effects on unused joints
   - Requires dynamics model

## Experimental Phases

### Phase 1: Two-Joint Control (Simplified)
- **Active Joints**: 0 (base), 2 (elbow)
- **Locked Joint**: 1 (shoulder) - needs position hold
- **Free Joints**: 3-6 (wrist/gripper) - currently zero torque
- **Expected Behavior**: Planar reaching in cylindrical coordinates
- **Success Criteria**: Consistent reaching within 5cm of target

### Phase 2: Three-Joint Control (Current)
- **Active Joints**: 0 (base), 1 (shoulder), 2 (elbow)
- **Free Joints**: 3-6 (zero torque)
- **Expected Behavior**: Full 3D reaching capability
- **Coordination Test**: Do joints learn complementary strategies?
  - Base for azimuth positioning
  - Shoulder/elbow for elevation and distance

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
- `[10-16]`: joint_theta (7 joint angles in radians)
- `[17]`: episode_time (seconds)

### Queue-Based Execution
All nodes use async queue-based patterns for real-time performance, similar to ROS (Robot Operating System).

### Balancer Nodes
Three balancer nodes (44, 50, 64) ensure synchronized execution across parallel training paths.

## Next Steps

1. **Test Export**: Run `programmatic_export.py Franka_Coop_Nodes`
2. **Implement Joint Locking**: Add PD controllers for unused joints
3. **Run Training**: Monitor convergence and coordination patterns
4. **Iterate on Architecture**:
   - Adjust network sizes (currently 128→128→1)
   - Tune learning rates (currently 0.01)
   - Experiment with different activation functions

## Related Files

- **Workflow**: `/home/asantanna/DNNE/DNNE-UI/user/default/workflows/Franka_Coop_Nodes.json`
- **Environment Config**: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg/task/FrankaDNNE.yaml`
- **Task Implementation**: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/tasks/franka_dnne.py`
- **Loss Function**: `/home/asantanna/DNNE/DNNE-UI/user/default/custom_compute_funcs/franka_coop_nodes_loss.py`
- **Task Tracking**: `/home/asantanna/DNNE/DNNE-UI/dnne_docs/for_claude/tasks/franka_coop/TASKS.md`
- **Workflow Analyzer**: `/home/asantanna/DNNE/DNNE-UI/claude_scripts/analyze_workflow.py`