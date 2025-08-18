# Franka Cooperative Control - History

## 2025-08-18: FrankaDNNE Environment Fixed ✅

### Completed
- **Fixed FrankaDNNE Task Initialization**:
  - Renamed `franka_dnne.py` to `franka_dnne_task.py` to avoid circular imports
  - Added `target_radius = 0.05` for sphere creation
  - Removed all cube-related code from base class (DNNE-only)
  - Added initialization checks for all tensor operations
  - Fixed action dimensions (8 → 7) for OSC control in YAML

- **Franka_Minimal_Test Workflow Working**:
  - Robot arm appears and responds to data streamer
  - Visual rendering confirmed working
  - 60Hz control loop stable
  - All fixes permanent in IsaacGymEnvs codebase

## 2025-01-18: Schema Alignment & Loss Implementation ✅

### Completed
- **Schema-Implementation Alignment**: Updated FrankaDNNE.yaml to match actual franka_dnne.py observations
  - Changed from incorrect schema to actual: target_pos, eef_pos, eef_quat, joint_theta, episode_time
  - Fixed Split node configurations to use correct field names
  
- **Loss Function Implementation**: Replaced placeholder with distance-based loss
  - Computes L2 norm between end-effector and target positions
  - Located in `custom_compute_funcs/franka_coop_nodes_loss.py`

- **Workflow Analysis Tool**: Created `claude_scripts/analyze_workflow.py`
  - Comprehensive analysis of 31-node, 45-connection workflow
  - Extracts Split/Concat patterns and widget values
  - Generates organized output in /tmp directory

- **Documentation**: Created comprehensive experiment overview
  - `experiments/franka_coop_nodes/franka_coop_overview.md`
  - Reorganized TASKS.md for clarity (214→44 lines)

### Key Insights
- Implementation is ground truth - schemas must match code, not vice versa
- Workflow uses 3 independent networks controlling joints 0,1,2
- Cooperative control emerges through shared global loss signal only
- Joints 3-6 currently free-floating (zero torque)