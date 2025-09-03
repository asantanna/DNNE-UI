# DNNE Documentation Index

## System Status Overview

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Export System | ✅ Simplified | - | Removed unnecessary gradient isolation mechanism |
| Shadow_Train | ✅ Working | - | Learning correctly (1.23 → 0.71 loss in 40 steps) |
| MultiWaiter | ✅ Fixed | - | Race condition resolved with required_and_received tracking |
| DataStreamer | ✅ Fixed | - | External sync mode working with wait_for_optionals |
| Queue Framework | ✅ Stable | - | Async queue architecture working correctly |

## Today's Achievements (Sep 2, 2025)
- Removed gradient isolation complexity - PyTorch handles optimizer isolation naturally
- Verified Shadow_Train learning with simplified architecture
- Cleaned up network_queue.tpl, sgd_optimizer_queue.tpl, and globals.py

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