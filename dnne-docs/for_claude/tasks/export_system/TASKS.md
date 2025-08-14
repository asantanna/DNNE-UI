# Export System Tasks

*For historical accomplishments, see HISTORY.md*

## Current Status
**Working** - Test suite fully passing (163 tests)
- Converts visual workflows to executable Python code
- Queue-based async architecture
- All nodes use FUNCTION = None (visual-only)
- Custom Computation node with file export support

## 📋 Active TODOs

### Critical - In Progress
1. **Fix UI export widget_values issue**
   - UI export reconstructs from prompt format which lacks widget_values
   - Exporters that read config from connected nodes fail (IsaacGymSim, PPOAgent)
   - Solution: Update exporters to check both widgets_values and inputs fields
   - Workaround: Use programmatic export or recreate affected nodes

### Medium Priority
1. **Optimize export for large workflows**
   - Currently exports all nodes even if not connected
   - Should prune disconnected subgraphs
   - Add validation for required connections

2. **Add export validation**
   - Verify all required node inputs are connected
   - Check for circular dependencies
   - Validate data type compatibility between connections

### Low Priority
1. **Export profiling and metrics**
   - Track export time and size metrics
   - Identify bottlenecks in export process
   - Add progress reporting for large exports

2. **Custom node template support**
   - Allow users to provide custom templates
   - Template validation and testing framework
   - Documentation for template creation

## 💡 Quick Reference

### Directory Structure
```
exports/{workflow_name}_wf_{hash}/
├── runner.py              # Main entry point
├── metadata.json          # Workflow metadata
├── framework/             # Queue framework
├── nodes/                 # Generated node code
├── custom_compute_funcs/  # Custom functions
└── telemetry/            # Telemetry output
```

### Template System
- Templates in `export_system/templates/nodes/`
- Queue-based templates for async execution
- String formatting for parameter injection