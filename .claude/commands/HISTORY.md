# DNNE Development History

*This file contains the historical record of development sessions moved from dev-status.md*

## 2025-01-15

### @dnne_node Decorator System ✅
- Implemented `@dnne_node(*, is_virtual)` decorator with required keyword-only parameter
- Automatic node registration - no more manual editing of registration files
- Virtual node status enforcement at node definition
- Auto-discovery of exporters based on strict naming conventions
- Template validation - virtual nodes must NOT have templates, non-virtual must have them
- Removed is_virtual() methods from exporters - decorator handles this
- Updated all 22 nodes with decorator (4 virtual, 18 non-virtual)
- Renamed mismatched exporter files to conform to naming convention
- All 164 tests passing (100% success)

## 2025-08-14

### Custom Computation Node ✅
- User-defined tensor operations via external Python files
- File export with automatic copying to export package
- Filter/sink capability (returning None = no output)
- Example functions: identity, filter, sink

### Isaac Gym Improvements ✅
- FrankaDNNE environment now visible
- Config loader handles environments without PPO configs
- Widget reordering (subtask/dt at top)
- Added FrankaDNNE to IsaacGymEnvs repository

### Code Organization ✅
- Moved utilities to custom_nodes/utils/
- Renamed base.py → visnode_base.py
- Created standard custom_compute_funcs directory

## 2025-08-13

### DataStreamer File Copy Mechanism ✅
- Implemented file copy mechanism for exporting data files with workflows
- Added get_export_files() method to ExportableNode base class
- Updated DataStreamer node to use new src_path/dest_dir widgets
- GraphExporter now collects and processes file copy requests with collision detection
- Supports both individual files and entire directories
- Fail-fast behavior on file collisions between nodes
- Fixed path separator issues (using forward slashes for cross-platform compatibility)
- Generated test data for Isaac Gym environments (299 samples @ 60Hz for 5 seconds)
- **Result**: Franka_Coop_Nodes workflow successfully exports with data files

### Visual Node Architecture & Test Suite Fixes ✅
- Fixed visual node architecture - all nodes properly use FUNCTION = None
- Removed all dead execution methods (load_dataset, create_dataloader, etc.)
- Deleted 7 incomplete node implementations and their exporters
- Deleted BalancerConfig exporter/template (it's a virtual node)
- Updated all tests to check UI interface instead of execution behavior
- Fixed test data formats to use widgets_values arrays
- Added workflow_name to fixtures for slot correction
- Template naming consistency (removed "simple" suffix)
- Fixed runner args sync tests to handle intentional UI omissions
- **Result**: All 163 tests passing (reduced from 28 failures to 0)

### PPOAgent Export System Fix ✅
- Fixed widget update mechanism for task-specific configurations
- Resolved backend/frontend naming mismatch (isaac_gym_env_node → isaac_gym_env)
- Re-added missing balancing_config input to PPOAgent node
- Implemented YAML-based network configuration loading for PPOAgent
- PPOAgent now loads network architecture from task-specific YAML files
- Fixed Cartpole_PPO workflow export - correct [32, 32] network layers
- Maintained hybrid approach: UI widgets for training control, YAML for architecture

## 2025-08-11 Session 2

### Telemetry Test Suite Optimization ✅
- Refactored telemetry_overhead_test.py to deploy workflow only once
- Added start_existing_workflow() to deployment_helper for workflow reuse
- Added wait_for_workflow_completion() to wait for workflow exit
- Simplified deployment_helper API - separated starting from waiting
- Eliminated redundant exports and data copies (saves ~170MB per iteration)
- Tests now handle their own timing for better control
- **Discovered bug**: DNNE server crashes on startup after MCP restart

## 2025-08-11 Session 1

### Telemetry Test Suite Completion ✅
- Created telemetry_runner_aggregation.py for testing aggregation, batching, and grouping
- Fixed critical bug: `dnne_test telemetry` only ran basic test due to `set -e` in scripts
- Removed `set -e` from dnne_test and commands.sh to allow all tests to run
- Standardized node IDs across all tests (nodes 10-14) for consistency
- Simplified test_telemetry.py validation logic
- All 5 telemetry tests now run successfully: basic, long, ratelimit, aggregation, overhead

## 2025-01-10 Session 3

### Telemetry Testing & Overhead Measurement ✅
- Fixed telemetry overhead test with CIFAR-10 dataset (0.6% overhead - excellent!)
- Added copy_dir parameter to deployment_helper for dataset caching
- Fixed deployment confirmation flow - wait for client confirmation before deploy_success
- Fixed monitor_workflow_execution timeout bug (was exiting after 1 second)
- Integrated ALL telemetry tests into `dnne_test telemetry` command
- Telemetry storage VERIFIED WORKING - files created successfully!

## 2025-01-10 Session 2

### MCP UI Automation & Telemetry Testing ✅
- Enhanced MCP with checkbox and input field interaction capabilities
- Fixed click_droplist_item to work with PrimeVue tieredmenu components
- Tested telemetry pipeline up to DNNE reception
- Fixed critical telemetry storage bugs (missing log_dir, wrong start_time type)
- Verified telemetry flows: runner → agent client → agent server → DNNE

## 2025-01-10 Session 1

### Runner Arguments Dialog Implementation ✅
- Implemented dynamic, JSON-driven UI for configuring runner.py arguments
- Two-column layout (900px width) with flexible field positioning
- Override mode allows manual command line editing
- All field types working: checkbox, text, number, select, select_or_text
- Professional styling with dark background (#252525) for readonly command input
- No frontend rebuild needed for layout changes - reads runner_args.json fresh
- SplitButton with "Export with Arguments..." option
- Real-time command line preview generation

## 2025-01-09

### Telemetry Pipeline Implementation ✅
- Implemented complete telemetry system from exported nodes to DNNE
- Added rate-limited violation reporting (10 msgs/sec) with optional `extra_args` grouping
- Created agent-side ViolationAggregator (first 5 details, then summaries every 10s)
- Efficient file storage in `telemetry/telem_{timestamp}/` directories
- Fire-and-forget UDP from nodes, smart batching at agent level
- Comprehensive documentation in `dnne_docs/architecture/telemetry.md`
- Test scripts: `test_telemetry_simple.py` for verification

## 2025-08-08 Session 4

### Logging Infrastructure Improvements ✅
- Created centralized `dnne_logs` directory for all DNNE components
- Configured all loggers (DNNE server, agent server/client, MCP) to use centralized directory
- Fixed critical race condition causing status bar not to update when workflows terminated
- Changed all log files to overwrite mode (mode='w') for fresh logs each run
- Agent client log reader now completes before cancellation to send "terminated" status

## 2025-08-08 Session 3

### STOP Button & Workflow Termination ✅
- Organized DNNE code into dnne_hooks directory for separation from ComfyUI
- Implemented STOP button functionality through WebSocket chain
- Made interrupt_processing async for proper stop signal handling  
- Fixed race condition in agent client when workflow terminates during stop
- Added robust error handling with proper status reporting
- Workflow termination messages injected into log stream

## 2025-08-08 Session 2

### Export System Fix ✅
- Fixed critical issue where export failed after server restart
- Frontend now sends workflow path with every export request
- Server extracts workflow name from path (no fallbacks)
- Fail-fast principle: clear errors instead of timestamp workarounds

### Log Window Improvements ✅
- Fixed UTF-8 encoding for emoji support in logs
- Historical log retrieval for completed workflows
- UI requests logs even when no active workflows
- Proper log file naming (dnne_agent_server.log)

### MCP Utility Functions ✅
- Added util_restart_dnne() with optional agent restart
- Added util_is_DNNE_running() health check
- Renamed get_agent_status to get_viewer_client_log
- Support for command-line arguments in restart (--verbose DEBUG)

## Previous Commits

### 2025-08-13
- `51974e60` - Fix runner args sync tests to handle intentional UI design decisions
- `41466e51` - Fix DNNE visual node architecture and tests

### 2025-08-11
- `4dc4d99b` - Optimize telemetry test suite to reuse workflows
- `e70fb355` - Fix telemetry test suite and add aggregation test

### 2025-08-08
- `4baf02fa` - Change all log files to overwrite mode instead of append
- `d8b8fcaf` - Fix agent client log reader cancellation race condition  
- `4a587423` - Fix status bar not updating for terminated workflows
- `0c644c71` - Add MCP server to centralized logging
- `e7b9c0b2` - Centralize logging in dnne_logs directory
- `b8546638` - Fix termination message not appearing in workflow logs
- `150ed4f2` - Fix STOP button workflow termination and error handling
- `e65deac5` - Organize DNNE code into dnne_hooks directory

### Earlier Commits
- `a7163565` - Fix logging issues and MCP export_workflow reporting
- `74952554` - Implement content-based workflow IDs and remote logging infrastructure
- `e90d3989` - Add run_after_export functionality for remote clients
- `2ee6dc85` - Remove link visibility functions from MCP DNNE-UI