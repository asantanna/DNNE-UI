# DNNE Task History

## Completed Features (Jan 2025)

### Session: Virtual Connection System Implementation
**Date**: Jan 28, 2025

#### Completed:
1. **Virtual Connection System**
   - Implemented "virtual" connections - UI-only connections resolved at runtime
   - Created OUTPUT_DICT system for cleaner output declarations with virtual flag
   - Added OutputDictMixin to auto-generate RETURN_TYPES/RETURN_NAMES from OUTPUT_DICT
   - Modified server.py to extract virtual flags from OUTPUT_DICT and INPUT_TYPES
   - Converted all 24 visnodes to use OUTPUT_DICT system

2. **Graph Exporter Virtual Support**
   - Added _is_virtual_output() and _is_virtual_input() methods
   - Virtual connections properly skipped when generating runner.py
   - No queues created for virtual connections

3. **Runtime Resolution Pattern**
   - SGD optimizer template uses g.graph_runner.get_node() to resolve connections
   - Connection resolved in run() method after all nodes created
   - Added get_node() method to GraphRunner template
   - Store graph_runner globally for node access

4. **Testing**
   - MNIST_Test workflow successfully exports and runs with virtual connections
   - SGD properly finds Network node for model parameters
   - No unnecessary queues created for model connection

### Session: Connection Validation & Test Cleanup
**Date**: Jan 20, 2025

#### Completed:
1. **Export System Connection Validation**
   - Added automatic validation of required input connections at export time
   - Implemented validate_required_connections() in ExportableNode base class
   - Added get_required_input_names() for optional input support
   - Created prepare_template_vars_with_validation() wrapper
   - Prevents runtime failures with clear error messages

2. **Test Suite Enhancement**
   - Added 8 comprehensive unit tests for connection validation
   - Fixed test cleanup to remove temporary export directories
   - All 173 tests passing with proper cleanup

3. **Repository Maintenance**
   - Fixed CIFAR10_Test.json workflow (added missing SGD loss connection)
   - Added tracking for example_reshape.py and franka_coop_nodes_loss.py
   - Ensured custom_compute_funcs directory is properly tracked

## Completed Features (Jan 2025)

### Session: Fail-Fast Enforcement & Code Review
**Date**: Jan 17, 2025

#### Completed:
1. **Systematic Fail-Fast Enforcement**
   - Removed 71+ `.get(key, default)` patterns that hide errors
   - Fixed hasattr/getattr anti-patterns with direct access
   - Enforced proper error propagation throughout codebase
   - Focus areas: export_system/node_exporters/, custom_nodes/

2. **Config Loader Improvements**
   - Made PPO config optional (workflows can be incomplete)
   - Removed nullAction from config loader (requires schema selection)
   - Made enableCameraSensors properly optional
   - Skip configs with Hydra `defaults:` inheritance

3. **Isaac Gym Config Handling**
   - Identified configs using Hydra inheritance
   - Skip unsupported configs with informative logging
   - Fixed _resolve_value to properly extract defaults

4. **Code Reviewer Enhancement**
   - Merged ui_auditor.md into dnne-code-reviewer.md
   - Documented review process with /tmp review plans
   - Added audit commands for finding violations
   - Simplified to: no COMMENT = approved for fixing

5. **Testing**
   - All 164 unit tests passing
   - Fixed test failures from overly aggressive validation
   - Proper handling of optional vs required fields

### Session: Type System Refactoring & New Nodes
**Date**: Jan 14, 2025

#### Completed:
1. **Type System Refactoring**
   - Changed LOSS_TENSOR to LOSS_SCALAR throughout codebase
   - Updated CrossEntropyLoss, TrainingStep, EpochTracker nodes
   - Updated frontend color configurations in dark.json and dnneColors.ts
   - Added SCALAR type with training color

2. **Concat Utility Node**
   - 4 optional tensor inputs (input_a through input_d)
   - Two modes: "wait for all" and "as available"
   - Pad modes: "pad with zeros" and "hold previous"
   - State management for hold previous mode
   - Full export system support

3. **GeometricLoss Node**
   - 5 error metrics: Max Abs Error, Euclidean Dist, Manhattan Dist, KL Div, Norm KL Div
   - Normalized KL Divergence (0-1 scale) using log(n) normalization
   - Created math_utils.py with reusable metric functions
   - Implemented proper dependency system for framework files

4. **Export System Enhancement**
   - Modified _copy_dependency to support any relative path from templates/
   - Framework dependencies now properly placed in framework directory
   - Dependency system tested with math_utils.py

5. **UI Fixes**
   - Disabled auto-rewiring on node deletion
   - Fixed input connector outlines (removed HollowCircle shape)
   - Fixed server URL display (shows localhost instead of 0.0.0.0)
   - Removed unused Queue and Model Library sidebar tabs

### Previous Sessions
See git history for earlier completed work.