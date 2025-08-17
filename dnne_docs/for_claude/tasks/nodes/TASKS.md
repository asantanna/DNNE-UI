# Node System Tasks

*For historical accomplishments, see HISTORY.md*

## Current Status
**Operational** - All core nodes working
- ML nodes (LinearLayer, Network, Loss, Optimizers)
- RL nodes (PPO Agent, Config)
- Robotics nodes (Isaac Gym integration)
- Utility nodes (EpochTracker, DataStreamer)

## 📋 TODO

### High Priority

1. **Hierarchical Schema System Implementation**
   
   **Phase 1: YAML Structure** 
   - [x] Update FrankaDNNE.yaml with schema_levels: ["subtask", "controlType"]
   - [x] Add subtask_options and controlType_options arrays
   - [x] Restructure using nested_schemas hierarchy
   - [x] Document schema format in architecture/yaml_schema.md
   
   **Phase 2: Dynamic Widget System**
   - [x] Implement dynamic widget creation in IsaacGymEnvs visnode
   - [x] Add callbacks for dynamic widgets (subtask, controlType)
   - [x] Add schema_display widget (multiline text, always last)
   - [x] Implement widget ordering: task → [dynamic] → fixed → schema_display
   
   **Phase 2.1: Widget Display Issues** (IN PROGRESS)
   - [x] Fix initial widget update on workflow load (dynamic_1/2/3 names showing)
   - [ ] Fix widget hiding without gaps (need to reposition Y coordinates)
   - [x] Update dynamic widget labels to show actual names (subtask, controlType)
   - [x] Ensure schema display updates when selections change
   
   **Phase 3: Index Management**
   - [x] Index management not necessary since we only show/hide widgets
      
   **Phase 4: Schema Resolution**
   - [x] Navigate nested_schemas based on all widget selections
   - [x] Update schema display on any widget change (backend ready)
   - [x] Support both array [x,x] and single number x schema formats
   - [x] Display single elements as [x] in UI for aesthetics
   - [x] Propagate complete schema through graph
   - [x] Ensure Split node receives and handles both schema formats
   
   **Testing Requirements:**
   - [x] Test Cartpole with direct schema (no levels)
   - [x] Test FrankaDNNE with 2 levels (subtask, controlType)
   - [x] Create and test 1-level example
   - [x] Verify Split node works with all schema variants
   - [x] Test slice notation with new schemas
   
   **Acceptance Criteria:**
   - Dynamic widgets appear/disappear based on task selection - OK
   - Schema display updates immediately on any change - OK
   - Split node can use semantic names from any schema type - OK
   - All existing workflows continue to work - OK

### Medium Priority

1. **Add 'group' widget to Balancer nodes**
   - Add string widget for group identification
   - Nodes in same group can enforce percent-based execution rates
   - Enable coordination between multiple balancing points
   - Consider other group-based metrics

2. ~~**Create Split node**~~ ✅ COMPLETED
   - ~~Opposite of Concat node~~
   - ~~Up to 4 outputs~~
   - ~~Widget specifies slices for each output~~
   - ~~Example: input[0:10] → output1, input[10:20] → output2~~
   - ~~Category: utility~~
   - **Enhancement added**: "by name" mode with schema support and slice notation

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