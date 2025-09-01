# Franka Cooperative Control Workflow Tasks

## Current Status
✅ **Workflow repaired and exports successfully** - Removed phantom connections (2025-08-31)
✅ **Export system enforces integrity** - Fail-fast on broken connections
🎉 **READY FOR EXPERIMENTS** - Workflow validated and clean

## 📚 Documentation
See: [`dnne_docs/experiments/franka_coop_nodes/franka_coop_overview.md`](../../experiments/franka_coop_nodes/franka_coop_overview.md)

## Active Tasks

### Next Steps for Experiments
1. ⬜ Generate CSV training data (sinusoidal and step-hold patterns)
2. ⬜ Test Shadow Environment with repaired workflow
3. ⬜ Run cooperative control experiments with 3 networks

## Quick Reference

**Workflow Structure**:
- 81 nodes, 84 connections (after repair)
- 3 parallel networks (128→128→1)
- Split 56: `target_pos, eef_pos`
- Split 45: `joint_theta[0], [1], [2]`
- Concat 42: mode="wait for all"

**Key Files**:
- Workflow: `user/default/workflows/Franka_Coop_Nodes.json`
- Loss: `user/default/custom_compute_funcs/franka_coop_nodes_loss.py`
- Export: `export_system/exports/Franka_Coop_Nodes/`