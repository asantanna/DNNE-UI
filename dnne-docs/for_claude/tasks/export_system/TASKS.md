# Export System Tasks

*Last Updated: 2025-08-13*

## Quick Stats
- **Status**: Working - Test Suite Fixed
- **Priority**: Medium
- **Completion**: ~95%
- **Dependencies**: Node implementations, Runner framework

## Current Status

The export system successfully converts visual workflows to executable Python code. It handles ML, RL, and robotics nodes with queue-based async architecture. Visual node architecture cleaned up - all nodes now properly use FUNCTION = None with no dead execution code. Test suite fully passing (163 tests).

## ✅ Completed

### Core Export Functionality
- [x] Graph traversal and dependency resolution
- [x] Node template generation with queue-based patterns
- [x] Import management and deduplication
- [x] Runner.py generation with proper error handling
- [x] Metadata.json creation with workflow info
- [x] Content-based workflow ID generation (SHA256)

### Node Support
- [x] ML nodes (LinearLayer, Conv2D, Dropout, etc.)
- [x] Dataset nodes (MNIST, CIFAR-10)
- [x] Training nodes (SGDOptimizer, CrossEntropyLoss)
- [x] RL nodes (PPO Agent, PPO Config)
- [x] Robotics nodes (Isaac Gym integration)
- [x] Utility nodes (EpochTracker, TensorVisualization)

### Advanced Features
- [x] Telemetry collection system
- [x] Remote deployment support
- [x] Runner arguments configuration
- [x] Workflow packaging for distribution
- [x] Queue-based async execution framework
- [x] DataStreamer node for CSV trajectory streaming
- [x] Isaac Gym camera position configuration

### Test Suite & Architecture (2025-08-13)
- [x] Fixed visual node architecture - FUNCTION = None on all nodes
- [x] Removed all dead execution methods from visual nodes
- [x] Deleted 7 incomplete node implementations
- [x] Updated all tests to check UI interface instead of execution
- [x] Fixed workflow metadata for slot correction
- [x] Template naming consistency (removed "simple" suffix)
- [x] Runner args sync tests handle intentional UI omissions
- [x] All 163 tests passing (0 failures)

## 📋 TODO

### Critical - In Progress

1. **Fix UI export widget_values issue**
   - Status: IN PROGRESS
   - UI export reconstructs from prompt format which lacks widget_values
   - Exporters that read config from connected nodes fail (IsaacGymSim, PPOAgent)
   - Solution: Update exporters to check both widgets_values and inputs fields
   - Workaround: Use programmatic export or recreate affected nodes

### High Priority

1. **Include data files during export**
   - Status: PENDING
   - Copy dataset files to export directory during export
   - Avoid re-downloading datasets on every deployment
   - Check for existing data files locally first
   - Update node code to look for local data before downloading
   - Pattern already exists in telemetry test suite

### Medium Priority

1. **Optimize export for large workflows**
   - Status: PENDING
   - Currently exports all nodes even if not connected
   - Should prune disconnected subgraphs
   - Add validation for required connections

2. **Add export validation**
   - Status: PENDING
   - Verify all required node inputs are connected
   - Check for circular dependencies
   - Validate data type compatibility between connections

### Low Priority

1. **Export profiling and metrics**
   - Status: PENDING
   - Track export time and size metrics
   - Identify bottlenecks in export process
   - Add progress reporting for large exports

2. **Custom node template support**
   - Status: PENDING
   - Allow users to provide custom templates
   - Template validation and testing framework
   - Documentation for template creation

## 💡 Notes

### Export Process Flow
1. User clicks Export button in UI
2. Server receives workflow JSON
3. GraphExporter processes nodes and connections
4. Templates are loaded and populated
5. Runner.py and supporting files generated
6. Package saved to `export_system/exports/{workflow_name}/`
7. Optional deployment to remote client

### Directory Structure
```
exports/
└── {workflow_name}_wf_{hash}/
    ├── runner.py           # Main entry point
    ├── metadata.json       # Workflow metadata
    ├── queue_framework.py  # Async queue system
    ├── nodes/             # Generated node code
    │   ├── node_1.py
    │   ├── node_2.py
    │   └── ...
    ├── data/              # (TODO) Dataset files
    │   ├── MNIST/
    │   └── CIFAR10/
    └── telemetry/         # Telemetry output
        └── telem_{timestamp}/
```

### Template System
- Templates in `export_system/templates/nodes/`
- Queue-based templates for async execution
- String formatting for parameter injection
- Import statements managed separately

## Future Enhancements

1. **Export optimization**
   - Compile Python code to bytecode
   - Bundle dependencies for offline execution
   - Generate Docker containers

2. **Multi-target export**
   - Export to different frameworks (TensorFlow, JAX)
   - Generate C++ code for embedded systems
   - ONNX export for inference

3. **Export versioning**
   - Track export history
   - Diff between versions
   - Rollback capability

4. **Cloud deployment**
   - Direct deployment to AWS/GCP/Azure
   - Kubernetes manifest generation
   - Serverless function export

---
*Focus: Add data file inclusion to avoid re-downloading datasets*