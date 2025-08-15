# DNNE Tasks

## Current Status
Type system refactored (LOSS_TENSOR → LOSS_SCALAR). New nodes: Concat, GeometricLoss with math_utils dependency system.

## Active TODOs

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
   - [ ] Add copy, paste and paste with links to Edit menu
   - [ ] Add node search functionality

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