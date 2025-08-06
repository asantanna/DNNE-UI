# DNNE Development Status

## Latest Achievements (2025-08-06)

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

## Current Work: UI Polish & Missing Features
- Log viewer modal needs frontend implementation
- Status bar should always show "Active Workflows: 0" when none
- Telemetry pipeline not yet processed
- Investigate programmatic agent server restart issue

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
- **Agent**: `docs-dnne/architecture/dnne-agent.md` - Agent architecture
- **Runner**: `docs-dnne/development/runner.md` - Command line switches for runner.py
- **CLAUDE.md**: Project overview and development guidance
- **Tasks**: `docs-dnne/for_claude/TASKS.md` - Complete task tracking

### Recent Commits
- `a7163565` - Fix logging issues and MCP export_workflow reporting
- `74952554` - Implement content-based workflow IDs and remote logging infrastructure
- `e90d3989` - Add run_after_export functionality for remote clients
- `2ee6dc85` - Remove link visibility functions from MCP DNNE-UI