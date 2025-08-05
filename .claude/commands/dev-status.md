# DNNE Development Status

## Current Work: MCP Server Improvements (2025-08-04)
Working on DNNE UI automation via MCP (Model Context Protocol) server that replaces Puppeteer with a more reliable Python/Playwright solution.

### Today's Accomplishments
- ✅ Fixed critical UI selectors (status bar, dialog close buttons)
- ✅ Implemented fail-fast error handling (no more silent failures)
- ✅ Created comprehensive test suite - 100% coverage, 79.5% passing
- ✅ Migrated to MCP_PY310 conda environment
- ✅ Fixed dialog dismiss functionality with bulletproof UI restoration

### Active Work
See `mcp-dnne-ui/DEVELOPMENT.md` for technical details and `mcp-dnne-ui/TASKS.md` for current issues.

### TODO List
- [ ] Fix missing workflow tool implementations (load_workflow, export_workflow)
- [ ] Fix client dropdown and log button selectors
- [ ] Fix IsaacGymEnvs node widget mismatch on export (BUG: "node 12 has 0 widget values, expected at least 15" when exporting Cartpole_PPO)
- [ ] Remove non-functional items from Workflow menu: Browse Templates, Export, Export (API)

## Claude Code Capabilities
- **Server Control**: Can restart DNNE server via `/remote_command` endpoint
- **Browser Automation**: Can view and interact with UI via MCP Puppeteer
- **Status Monitoring**: Can check server status, uptime, and node count
- **Access from WSL2**: Server accessible at `http://172.22.160.1:8188`

### Puppeteer Configuration for DNNE UI
**Use this exact configuration:**

```javascript
await mcp__puppeteer__puppeteer_navigate({
  url: "http://172.22.160.1:8188",
  launchOptions: {"headless": false, "defaultViewport": null, "args": ["--start-maximized"]}
});
```

**Important**: 
- Must use --start-maximized to see the status bar
- Status bar shows: "Agent: ⚪ Connected | Clients: 0" on left, Export/Run controls on right
- Tested on 1920x1080 displays
- **Known Issue**: Taking screenshots causes the window to start clipping the status bar - reason unknown

## Recent Accomplishments (2025-08-04)
- ✅ Created comprehensive Puppeteer debugging documentation at `docs-dnne/development/using_puppeteer_for_debug.md`
- ✅ Successfully tested all UI elements with Puppeteer (sidebar tabs, canvas controls, export button, menus)
- ✅ Documented working selectors for workflows tree navigation using aria-label attributes
- ✅ Fixed default workflow to load blank instead of ComfyUI image generation workflow
- ✅ Documented JavaScript evaluation capabilities for accessing canvas state and UI information

## Previous Accomplishments (2025-08-04)
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