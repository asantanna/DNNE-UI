# DNNE Development Status

## Current Work
See `docs-dnne/for_claude/TASKS.md` for the complete task list and project roadmap.

## Claude Code Capabilities
- **Server Control**: Can restart DNNE server via `/remote_command` endpoint
- **Browser Automation**: Can view and interact with UI via MCP Puppeteer
- **Status Monitoring**: Can check server status, uptime, and node count
- **Access from WSL2**: Server accessible at `http://172.22.160.1:8188`

## Recent Accomplishments (2025-08-04)
- ✅ Implemented remote command endpoint for server control
- ✅ Can now restart server programmatically (no manual intervention needed)
- ✅ Simplified WSL2 access with --listen 0.0.0.0 flag
- ✅ Consolidated agent documentation into single dnne-agent.md file
- ✅ Removed unnecessary Chrome proxy complexity

## Recent Accomplishments (2025-08-02)
- ✅ Refactored dnne-agent system for production readiness
- ✅ Implemented asyncio-based UDP telemetry (replaced busy-wait polling)
- ✅ Added test port architecture (8768) for isolated testing
- ✅ All agent tests passing: connectivity, deployment, execution, telemetry

## Recent Accomplishments (2025-02-02)
- ✅ Centralized all paths through dnne_config.json
- ✅ Made exported packages self-sufficient (copy framework files)
- ✅ All tests passing: 171 unit tests + 3 integration tests

## Quick Reference

### Essential Commands
```bash
# Activate environment
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

### UI Interaction (via MCP Puppeteer)
```python
# Navigate to DNNE UI
await mcp__puppeteer__puppeteer_navigate(url="http://172.22.160.1:8188")

# Take screenshot
await mcp__puppeteer__puppeteer_screenshot(name="dnne-ui", encoded=False)

# Click elements, fill forms, etc.
await mcp__puppeteer__puppeteer_click(selector="#button-id")
```

### Key Ports
- **8188**: DNNE UI
- **8766-8769**: Agent system
- **9999**: Telemetry UDP

### Key Documentation
- **Tasks**: `docs-dnne/for_claude/TASKS.md` - Current work items
- **Agent**: `docs-dnne/architecture/dnne-agent.md` - Agent architecture
- **Runner**: `docs-dnne/development/runner.md` - Command line switches for runner.py
- **CLAUDE.md**: Project overview and development guidance