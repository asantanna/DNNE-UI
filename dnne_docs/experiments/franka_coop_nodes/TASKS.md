# Franka Cooperative Control - Active Tasks

**Status**: ✅ DEADLOCK FIXED! Workflow runs indefinitely without hanging!

## Pending TODOs

### High Priority
- [x] Fix --timeout to be more reliable (almost never works currently)
- [x] Test complete workflow execution to verify control loop works

### Medium Priority  
- [ ] Add caching for YAML files in get_task_schema_info
- [ ] Update PPOAgent and PPOConfig when IsaacGymEnvs changes (for other workflows)
- [ ] Fix set_connections() not being called for Concat nodes
- [x] Implement proper connection tracking instead of hardcoding

### Low Priority
- [ ] Add more subtasks beyond random_target (reach_pose, trajectory_follow)
- [ ] Improve logging to be less verbose during normal operation
- [ ] Add unit tests for null_action extraction and widget updates

## Quick Reference

**Workflow**: `user/default/workflows/Franka_Coop_Nodes.json`  
**Export Location**: `export_system/exports/Franka_Coop_Nodes/`  
**Task Config**: `/home/asantanna/DNNE/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg/task/FrankaDNNE.yaml`

**Run Command**:
```bash
cd export_system/exports/Franka_Coop_Nodes
python runner.py
```