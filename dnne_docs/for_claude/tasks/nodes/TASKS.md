# Node System Tasks

*For historical accomplishments, see HISTORY.md*

## Current Status
**Operational** - All core nodes working
- ML nodes (LinearLayer, Network, Loss, Optimizers)
- RL nodes (PPO Agent, Config)
- Robotics nodes (Isaac Gym integration)
- Utility nodes (EpochTracker, DataStreamer)

## 📋 TODO

### Medium Priority

1. **Add 'group' widget to Balancer nodes**
   - Add string widget for group identification
   - Nodes in same group can enforce percent-based execution rates
   - Enable coordination between multiple balancing points
   - Consider other group-based metrics

2. **Create Split node**
   - Opposite of Concat node
   - Up to 4 outputs
   - Widget specifies slices for each output
   - Example: input[0:10] → output1, input[10:20] → output2
   - Category: utility

### Low Priority

1. **Fix balancing_node → network_node connection color**
   - balancing_node.output → network_node.to_output has wrong color link
   - Should match proper type color scheme
   - Check type system for correct color assignment

2. **Rename GeometricLoss output**
   - Current output name may be generic
   - Should be called "loss" for clarity
   - Update both UI and export templates

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

### Type System
- Colors defined in `nodes.py`
- Virtual nodes don't generate standalone code
- Context passed implicitly through global state

## Future Enhancements
1. Dynamic node generation from templates
2. Custom node marketplace/sharing
3. Node versioning for backward compatibility
4. Visual node editor in UI
5. Node performance profiling