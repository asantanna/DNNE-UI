# DNNE Tasks

## Current Status
MultiWaiter race condition fixed - proper synchronization for required/optional inputs.
DataStreamer external sync mode working correctly with wait_for_optionals parameter.

## Active TODOs

### Critical - Frontend Integration
1. **Patch Verification System**
   - [ ] Integrate dnne_patches verification into DNNE backend startup
   - [ ] Add --ignore-patch-errors flag to bypass patch checks
   - [ ] Call verify_all_patches() from main.py or server.py startup
   - [ ] Display clear error messages when patches don't match
   - [ ] Document patch reapplication process in error messages

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
- `../DNNE-UI-Frontend/dnne_patches/` - Frontend patch system for npm packages
- `../DNNE-UI-Frontend/dnne_patches/dnne_patches.py` - Patch verification script