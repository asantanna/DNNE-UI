
# DNNE Development Status

**📋 TASK TRACKING**: See `docs-dnne/for_claude/tasks/INDEX.md` for current task status and priorities across all components.

## Latest Achievements (2025-01-10)

### Runner Arguments Dialog Implementation ✅
- Implemented dynamic, JSON-driven UI for configuring runner.py arguments
- Two-column layout (900px width) with flexible field positioning
- Removed groups system in favor of direct column/order per field
- Override mode allows manual command line editing
- All field types working: checkbox, text, number, select, select_or_text
- Professional styling with dark background (#252525) for readonly command input
- No frontend rebuild needed for layout changes - reads runner_args.json fresh
- SplitButton with "Export with Arguments..." option
- Real-time command line preview generation

## Previous Session (2025-01-09)

### Telemetry Pipeline Implementation ✅
- Implemented complete telemetry system from exported nodes to DNNE
- Added rate-limited violation reporting (10 msgs/sec) with optional `extra_args` grouping
- Created agent-side ViolationAggregator (first 5 details, then summaries every 10s)
- Efficient file storage in `telemetry/telem_{timestamp}/` directories
- Fire-and-forget UDP from nodes, smart batching at agent level
- Comprehensive documentation in `docs-dnne/architecture/telemetry.md`
- Test scripts: `test_telemetry_simple.py` for verification

## Previous Session (2025-08-08 Session 4)

### Logging Infrastructure Improvements ✅
- Created centralized `dnne_logs` directory for all DNNE components
- Configured all loggers (DNNE server, agent server/client, MCP) to use centralized directory
- Fixed critical race condition causing status bar not to update when workflows terminated
- Changed all log files to overwrite mode (mode='w') for fresh logs each run
- Agent client log reader now completes before cancellation to send "terminated" status

## Previous Session (2025-08-08 Session 3)

### STOP Button & Workflow Termination ✅
- Organized DNNE code into dnne_hooks directory for separation from ComfyUI
- Implemented STOP button functionality through WebSocket chain
- Made interrupt_processing async for proper stop signal handling  
- Fixed race condition in agent client when workflow terminates during stop
- Added robust error handling with proper status reporting
- Workflow termination messages injected into log stream

## Previous Session (2025-08-08 Session 2)

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

## Previous Achievements (2025-08-06)

### Content-Based IDs & Remote Logging ✅
- Workflow IDs now use SHA256 content hash (wf_{hash[:12]}) for deterministic identification
- Complete remote logging infrastructure captures all workflow output
- Logs saved to `remote_clients/{client}/{workflow}_wf_{id}/run_logs/`
- Metadata.json tracks deployment information
- Clean deployment ensures no leftover files

### Run After Export Feature ✅
- Checkbox properly disabled for Local exports
- State preserved when switching between Local/remote clients
- Workflows auto-start on remote clients when enabled
- MCP functions control and test the feature
- End-to-end tested with MNIST achieving 99.58% accuracy

## Current Work: Pending Tasks
- Add WARNING header for historical logs in viewer
- Test click_button with all locations/controls  
- Replace all evaluate() calls in tool files with js_* functions
- BUG: Export should fail with error when no client connected (currently silent)

## Claude Code Capabilities
- **Server Control**: Can restart DNNE server via `/remote_command` endpoint
- **Browser Automation**: Can view and interact with UI via MCP
- **Status Monitoring**: Can check server status, uptime, and node count
- **Access from WSL2**: Server accessible at `http://172.22.160.1:8188`


## Quick Reference

### Essential Commands
```bash
# Activate environment (exported code)
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Activate environment (Claude Code, MCP)
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Start DNNE UI (Windows)
./dnne.bat

# Start Agent Client (WSL2)
python dnne-agent/dnne_agent_client.py

# Export workflow
python claude_scripts/programmatic_export.py MNIST_Test

# Run exported workflow
cd export_system/exports/MNIST_Test
python runner.py --epochs 10

# Run with telemetry enabled
python runner.py --enable-telemetry 10,11 --timeout 30s

# Test telemetry
python test_telemetry_simple.py

# Build the frontend
./build_frontend.sh
```

### Claude Code Server Control
```bash
# Restart server
curl -X POST http://172.22.160.1:8188/remote_command \
  -H "Content-Type: application/json" \
  -d '{"command": "restart", "args": {"delay": 3}}'

# Check server status
curl -X POST http://172.22.160.1:8188/remote_command \
  -H "Content-Type: application/json" \
  -d '{"command": "get_status"}'

# Test all commands
python claude_scripts/test_remote_command.py
```

### UI Interaction (via MCP)
See `mcp-dnne-ui/README.md`

### Key Ports
See `dnne_config.json`

### Key Documentation
- **Task Index**: `docs-dnne/for_claude/tasks/INDEX.md` - Quick overview of all component tasks
- **Agent**: `docs-dnne/architecture/dnne-agent.md` - Agent architecture
- **Telemetry**: `docs-dnne/architecture/telemetry.md` - Telemetry system architecture
- **Runner**: `docs-dnne/development/runner.md` - Command line switches for runner.py
- **CLAUDE.md**: Project overview and development guidance

### Recent Commits (2025-08-08)
- `4baf02fa` - Change all log files to overwrite mode instead of append
- `d8b8fcaf` - Fix agent client log reader cancellation race condition  
- `4a587423` - Fix status bar not updating for terminated workflows
- `0c644c71` - Add MCP server to centralized logging
- `e7b9c0b2` - Centralize logging in dnne_logs directory
- `b8546638` - Fix termination message not appearing in workflow logs
- `150ed4f2` - Fix STOP button workflow termination and error handling
- `e65deac5` - Organize DNNE code into dnne_hooks directory

### Previous Commits
- `a7163565` - Fix logging issues and MCP export_workflow reporting
- `74952554` - Implement content-based workflow IDs and remote logging infrastructure
- `e90d3989` - Add run_after_export functionality for remote clients
- `2ee6dc85` - Remove link visibility functions from MCP DNNE-UI