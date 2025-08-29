# Franka Cooperative Control Nodes - Implementation History

## 2025-08-29: Critical Deadlock Fix - System Runs Forever!

### Problem
The Franka_Coop_Nodes workflow was experiencing deadlocks after 12-13 iterations where:
- IsaacGymSim (node 25) would stop producing outputs
- SimulationTracker nodes (72, 132) would fail with TypeError when receiving partial inputs
- System would hang with all nodes idle but messages queued

### Root Causes
1. **IsaacGym Custom Queue Handling**: Complex custom run() method with internal queueing caused "double-getter deadlock"
2. **SimulationTracker Partial Inputs**: Required positional args (observation, loss) but MultiWaiter would call compute() with just optional inputs
3. **Misleading Documentation**: base_nodes.py comment incorrectly stated when to override run()
4. **Workflow Bug**: Eat_N trigger wasn't connected to barriers

### Solutions Implemented

#### 1. Simplified IsaacGymSim to Use Standard MultiWaiter
- Removed custom queue handling entirely
- Changed to: `self.setup_inputs(required=[], optional=["action", "reset"])`
- Fixed syntax error with nested quotes in f-strings
- Template: `export_system/templates/nodes/isaac_gym_sim_queue.tpl`

#### 2. Fixed SimulationTracker to Handle Partial Inputs
- Made observation and loss optional with None defaults
- Added guards to only process inputs that are provided
- Prevents TypeError when optional inputs arrive alone
- Template: `export_system/templates/nodes/simulation_tracker_queue.py`

#### 3. Fixed Misleading Comment in base_nodes.py
- Clarified that inputs is always a dict from MultiWaiter
- Override run() only for exotic cases like manual queue reads
- File: `export_system/templates/framework/base_nodes.py`

#### 4. Connected Eat_N Trigger to Barriers (User fix)
- Ensures proper data flow through the pipeline
- Workflow: `user/default/workflows/Franka_Coop_Nodes.json`

### Testing Results
- System runs indefinitely without deadlock!
- Deadlock analyzer confirms no issues
- All nodes continue processing data correctly

### Commits
- `32603cbb`: Fix critical deadlock in DNNE queue framework

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