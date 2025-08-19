# DNNE Tasks

## Current Status
Franka_Coop_Nodes workflow exports and runs but requires multiple HACKS to patch dimension mismatches, device issues, and gradient tracking problems. Core architectural issues need addressing.

## Active TODOs

### Critical - Fix Underlying Issues (Remove HACKS)
1. **Dimension Configuration**
   - [ ] Fix UI to correctly set concat/split dimensions (currently hardcoded to dim=1)
   - [ ] Ensure nodes output consistent tensor shapes
   
2. **Device Management**
   - [ ] Fix nodes to output tensors on correct device from the start
   - [ ] Remove device-fixing hacks from Concat nodes
   
3. **Gradient Tracking**
   - [ ] Enable gradient tracking for Isaac Gym observations when training
   - [ ] Fix Network nodes to properly handle gradient flow

### High Priority
1. **Export System Enhancement**
   - [ ] Add progress tracking to exported runners
   - [ ] Implement distributed training support
   - [ ] Test GeometricLoss node export with all metrics

### Medium Priority
1. **Node Development**
   - [ ] Complete RL node implementations (PPO, A2C)
   - [ ] Add transformer architecture nodes

### Low Priority
1. **UI Improvements**
   - [ ] Add node search functionality
   - See frontend/TASKS.md for UI-specific tasks

## Quick Reference

### Testing Commands
```bash
# Test export system
python claude_scripts/programmatic_export.py

# Run exported workflow
cd export_system/exports/{workflow_name}
python runner.py
```

### Key Files
- `export_system/graph_exporter.py` - Core export logic with dependency system
- `custom_nodes/*_visnode.py` - Node implementations
- `export_system/templates/framework/math_utils.py` - Shared math utilities