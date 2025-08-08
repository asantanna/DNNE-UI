
# DNNE Development Status

**📋 TASK TRACKING**: See `docs-dnne/for_claude/tasks/INDEX.md` for current task status and priorities across all components.

## Latest Achievements (2025-08-08 Session 2)

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
- Implement 5 remaining MCP log management functions
- Add util_set_DNNE_log_level() and util_set_agent_server_log_level()
- Test get_viewer_client_log() retrieves content from UI
- Fix export to fail with error when no client connected (currently silent)

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
- **Runner**: `docs-dnne/development/runner.md` - Command line switches for runner.py
- **CLAUDE.md**: Project overview and development guidance

### Recent Commits (2025-08-08)
- `7aaf47b2` - Fix export after server restart by sending workflow path from frontend
- `9d9f2a8` (Frontend) - Send workflow path with export requests
- `1fe20ca0` - Fix health endpoint import error
- `a359c19f` - Fix MCP tool name updates and permission script
- `5fc5ac70` - Major MCP and server improvements for logging and health monitoring

### Previous Commits
- `a7163565` - Fix logging issues and MCP export_workflow reporting
- `74952554` - Implement content-based workflow IDs and remote logging infrastructure
- `e90d3989` - Add run_after_export functionality for remote clients
- `2ee6dc85` - Remove link visibility functions from MCP DNNE-UI