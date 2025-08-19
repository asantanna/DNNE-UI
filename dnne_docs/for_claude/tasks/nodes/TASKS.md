# Node System Tasks

*For historical accomplishments, see HISTORY.md*

## Current Status
**Operational** - All core nodes working, IsaacGymEnvs widget save/load fixed
- Hierarchical schema system fully implemented
- Dynamic widgets properly save/restore values from workflows

## 📋 TODO

### Medium Priority

1. **Widget Display Improvements**
   - [ ] Fix widget hiding without gaps (need to reposition Y coordinates)

2. **Add 'group' widget to Balancer nodes**
   - Add string widget for group identification
   - Nodes in same group can enforce percent-based execution rates
   - Enable coordination between multiple balancing points
   - Consider other group-based metrics

### Low Priority

1. **Fix balancing_node → network_node connection color**
   - balancing_node.output → network_node.to_output has wrong color link
   - Should match proper type color scheme

2. **Rename GeometricLoss output**
   - Current output name may be generic
   - Should be called "loss" for clarity

## 💡 Quick Reference

### Node Categories
- `ml` - Machine learning nodes
- `rl` - Reinforcement learning nodes  
- `robotics` - Robotics and simulation nodes
- `utility` - Utility and control flow nodes

### Adding New Nodes
1. Create `custom_nodes/{name}_visnode.py`
2. Add exporter in `export_system/node_exporters/`
3. Create template in `export_system/templates/nodes/`
4. Register in appropriate category
5. Test with example workflow