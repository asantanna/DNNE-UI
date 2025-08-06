# DNNE Development Status

## Current Work: MCP Server Improvements (2025-08-04)
Working on DNNE UI automation via MCP (Model Context Protocol) server using a Python/Playwright solution.

### Active Work
See `mcp-dnne-ui/DEVELOPMENT.md` for technical details and `mcp-dnne-ui/TASKS.md` for current issues.

### TODO List
See `mcp-dnne-ui/TASKS.md`

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