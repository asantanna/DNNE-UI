# DNNE Documentation Index

## System Status Overview

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Export System | ✅ Complete | - | Widget encapsulation implemented |
| LinearLayer/Network | ✅ Refactored | - | Virtual nodes architecture |
| Isaac Gym Integration | ✅ Fixed | - | Circular dependency resolved |
| Test Suite | ✅ Passing | - | 164 tests pass, 0 skipped |
| Franka Coop Workflow | ✅ Running | - | Bootstrap with null_action working |
| Balancer Node | ✅ Fixed | - | Naming consistency resolved |

## Today's Achievements (Jan 18, 2025)
- Fixed Franka_Coop_Nodes circular dependency blocking workflow execution
- Implemented IsaacGymSim bootstrap with null_action via custom run() method  
- Added JavaScript widget update callbacks for task/schema changes
- Fixed asyncio blocking and FrankaDNNE import issues

## Active Priorities

### High Priority
1. Fix --timeout to be more reliable (almost never works)
2. Test complete Franka workflow with training loop

### Medium Priority  
1. Add YAML caching for get_task_schema_info
2. Update PPOAgent/PPOConfig on IsaacGymEnvs changes
3. Fix Concat node set_connections() calls

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
  - [`websocket-not-rest.md`](architecture/websocket-not-rest.md) - WebSocket communication principles
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