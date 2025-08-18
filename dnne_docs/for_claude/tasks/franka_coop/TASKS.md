# Franka Cooperative Control Workflow Tasks

## Current Status
✅ **FrankaDNNE environment fully operational** - All initialization issues fixed, visual rendering working
✅ **Franka_Minimal_Test workflow working** - Robot arm responding correctly to data streamer
✅ **All fixes permanent** - Changes in IsaacGymEnvs codebase persist across exports

## 📚 Documentation
See: [`dnne_docs/experiments/franka_coop_nodes/franka_coop_overview.md`](../../experiments/franka_coop_nodes/franka_coop_overview.md)

---

## Active Tasks

### 1. Export and Testing Franka_Coop_Nodes
**Status**: 🟡 Ready to test  
**Priority**: HIGH  
**Next Step**: Run `python claude_scripts/programmatic_export.py Franka_Coop_Nodes` with working FrankaDNNE

### 2. Joint Locking Implementation  
**Status**: 🔴 Not Started  
**Priority**: MEDIUM  
**TODO**: Implement PD control for joints 3-6 (currently free-floating with zero torque)

### 3. Training Experiments
**Status**: 🔴 Not Started  
**Priority**: HIGH  
**TODO**: Run training and monitor coordination emergence between the 3 controllers

---

## Quick Reference

**Workflow Structure**:
- 31 nodes, 45 connections
- 3 parallel networks (128→128→1)
- Split 56: `target_pos, eef_pos`
- Split 45: `joint_theta[0], [1], [2]`
- Concat 42: mode="as available"

**Key Files**:
- Workflow: `user/default/workflows/Franka_Coop_Nodes.json`
- Loss: `user/default/custom_compute_funcs/franka_coop_nodes_loss.py`
- Analyzer: `claude_scripts/analyze_workflow.py`