# Node System - Historical Accomplishments

*This file contains the historical record of completed work moved from TASKS.md*

## Session: 2025-08-19 - IsaacGymEnvs Widget Save/Load Fix

### Widget Value Persistence ✅
- Fixed issue where dynamic widget values reset to defaults on workflow load
- Modified onLoad callback to use `node_data` from `event_params` for actual loaded values
- Removed default value fallbacks - always use loaded values from workflow
- Schema display now correctly shows loaded configuration, not defaults
- Tested successfully with Franka_Coop_Nodes and Cartpole_PPO workflows

### Hierarchical Schema System Complete ✅
- All phases of implementation completed
- Dynamic widgets properly save/restore with correct labels and values
- Schema display updates correctly based on selections
- Split node works with all schema variants
- Full backward compatibility maintained

## Session: 2025-08-17 - Schema Format Enhancement

### Schema Format Support ✅
- Added support for single-number schema format in YAML (e.g., `gripper_width: 6`)
- Updated Split node exporter to handle both formats:
  - Array format `[start, end]` for ranges
  - Single number format for single elements
- Enhanced IsaacGymEnvs display to show single elements as `[x]` for aesthetics
- Maintains backward compatibility with existing YAML files

## Session: 2025-08-15 - LinearLayer/Network Refactoring

### Virtual Node Architecture ✅
- Made LinearLayer nodes properly virtual (no standalone export)
- Network node orchestrates layer code generation
- Clean delegation pattern with `get_layer_pytorch_code()`
- Removed network_consumed_nodes hack

### Node Export System ✅
- All nodes have proper exporters
- Template-based code generation
- Queue-based async architecture
- Proper import management

## Previous Sessions

### Core ML Nodes ✅
- LinearLayer with activation/dropout
- Conv2D with padding/stride
- Network consolidation node
- Loss functions (CrossEntropy, MSE)
- Optimizers (SGD, Adam planned)

### Training Infrastructure ✅
- TrainingStep with trigger system
- EpochTracker with progress display
- BatchSampler with shuffle support
- GetBatch with trigger coordination

### RL Integration ✅
- PPO Agent node
- PPO Config node
- BalancerConfig (virtual)
- Isaac Gym environment support

### Robotics Nodes ✅
- IsaacGymSim core simulator
- IsaacGymEnvs with YAML config
- Camera position control
- Telemetry collection

### Utility Nodes ✅
- DataStreamer for CSV trajectories
- CustomComputation for user functions
- GeometricLoss for robotics
- TensorVisualization
- OR logic node

### Type System ✅
- Consistent color coding
- Proper type validation
- Schema resolution through connections
- Wildcard matching support