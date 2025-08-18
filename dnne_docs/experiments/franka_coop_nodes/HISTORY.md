# Franka Cooperative Control Nodes - Implementation History

## 2025-01-18: Circular Dependency Resolution

### Problem
The Franka_Coop_Nodes workflow was hanging after "Starting Concat node 42 in 'wait for all' mode" due to a circular dependency:
- IsaacGymSim needs actions to produce observations
- Actions come from networks that need observations
- null_action was supposed to bootstrap this loop but wasn't being used

### Root Causes Identified
1. **Missing Bootstrap**: IsaacGymSim was waiting for action input before initializing, preventing the first observation
2. **Uninitialized null_action**: Widget in UI showed null_action as empty despite being in YAML schema
3. **Broken Fail-Fast**: Template had contradictory validation logic that should have caught missing null_action
4. **No Widget Updates**: When task/schema changed, null_action widget didn't update
5. **Asyncio Blocking**: GraphRunner's `await asyncio.sleep(0)` after task creation blocked other tasks
6. **Import Errors**: Missing __init__.py in franka_dnne directory

### Solutions Implemented

#### 1. Fixed IsaacGymSim Template
- Removed action from required inputs
- Added custom run() method to bootstrap with null_action
- Fixed broken fail-fast validation logic
- Template: `export_system/templates/nodes/isaac_gym_sim_queue.tpl`

#### 2. Added Widget Update Callbacks
- IsaacGymEnvs node now sends JavaScript through WebSocket to update connected nodes
- Created DRY utility function `extract_null_action_from_schema()`
- Updates propagate when task or schema changes
- File: `custom_nodes/isaac_gym_envs_visnode.py`

#### 3. Fixed Asyncio Blocking
- Removed `await asyncio.sleep(0)` after task creation in GraphRunner
- All tasks now created before yielding control
- File: `export_system/exports/Franka_Coop_Nodes/framework/graph_runner.py`

#### 4. Changed Concat Nodes Configuration
- Switched from "wait for all" to "as available" mode
- Added "hold previous" padding to handle missing inputs
- Hardcoded connected inputs to work around missing set_connections() calls
- Nodes: 42, 47, 55, 57

#### 5. Fixed FrankaDNNE Import
- Created proper __init__.py in franka_dnne directory
- Imports: `from ..franka_dnne import Franka_DNNE_RandomTarget as FrankaDNNE`
- File: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/tasks/franka_dnne/__init__.py`

### Testing Results
- Workflow now starts successfully without hanging
- IsaacGymSim bootstraps with null_action properly
- All nodes create and start as expected
- FrankaDNNE task imports correctly

### Lessons Learned
1. "Prioritize RIGHT over QUICK every single time" - User feedback
2. Don't use hacks (like making IsaacGymSim a SensorNode) - override methods properly
3. Follow DRY principles - create shared utilities for common operations
4. Fail-fast violations hide bugs - fix them immediately
5. Asyncio task creation order matters - create all tasks before yielding