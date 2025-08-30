# DNNE Documentation Index

## System Status Overview

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| MultiWaiter | ✅ Fixed | - | Race condition resolved with required_and_received tracking |
| DataStreamer | ✅ Fixed | - | External sync mode working with wait_for_optionals |
| Shadow_Train | ✅ Working | - | Proper lockstep simulation/training synchronization |
| Deadlock Tool | ✅ Enhanced | - | Added DataStreamer simulator |
| Export System | ✅ Enhanced | - | Connection validation at export time |
| Queue Framework | ✅ Stable | - | Async queue architecture working correctly |

## Today's Achievements (Aug 30, 2025)
- Fixed MultiWaiter race condition causing listener errors
- Resolved DataStreamer busy loop consuming 1.2M+ events
- Added wait_for_optionals parameter for proper optional input handling
- Shadow_Train workflow running correctly with proper synchronization

## Active Priorities

### CRITICAL - Remove HACKS
1. Fix UI dimension configuration (concat/split using wrong dims)
2. Fix device management (tensors on wrong device)
3. Enable gradient tracking for Isaac Gym observations
4. Fix tensor shape consistency in Network nodes

### High Priority
1. Fix --timeout to be more reliable (almost never works)
2. Test complete Franka workflow with actual training

### Medium Priority  
1. Add YAML caching for get_task_schema_info
2. Update PPOAgent/PPOConfig on IsaacGymEnvs changes

### Low Priority
1. Add more FrankaDNNE subtasks (reach_pose, trajectory_follow)
2. Export profiling and metrics

## Documentation Structure

### Core Documentation
- [`CLAUDE.md`](../CLAUDE.md) - AI assistant instructions
- [`dev-status.md`](dev-status.md) - Development status

### Architecture
- [`architecture/`](architecture/) - System design
  - [`ui_callbacks.md`](architecture/ui_callbacks.md) - WebSocket-based UI widget callback system
  - [`websocket_not_rest.md`](architecture/websocket_not_rest.md) - WebSocket communication principles
- [`nodes/`](nodes/) - Node guides
- [`tasks/`](tasks/) - Task tracking

## Quick Access

### Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Test all exports
python claude_scripts/test_all_exports.py

# Run unit tests
./dnne_test quick

# Start server (Windows)
dnne.bat
```