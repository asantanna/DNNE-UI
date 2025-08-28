# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-08-28)

### Virtual Connection System ✅
- **UI-Only Connections** - Resolved at runtime, no queues created
  - Implemented OUTPUT_DICT system with virtual flags
  - Graph exporter skips virtual connections properly
  - SGD optimizer uses g.graph_runner.get_node() for runtime resolution
  - Network.model → SGD.model connection now efficient
  - All 24 visnodes converted to OUTPUT_DICT system

### Previous Achievements (2025-08-27)
- Multi-optimizer support with retain_graph override system
- Export system bug fixes and debug cleanup
- Workflow analysis & repair tools with property-based validation

## Quick Reference

### Essential Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Restart DNNE Server (runs on Windows)
use MCP function: "util_restart_dnne"

# Start Agent Client (WSL2)
python dnne_agent/dnne_agent_client.py

# Export workflow
python claude_scripts/programmatic_export.py MNIST_Test

# Run exported workflow
cd export_system/exports/MNIST_Test
python runner.py --epochs 10

# Test telemetry
./dnne_test telemetry

# Build frontend
./build_frontend.sh
```

### Key Ports
- DNNE UI: 8188
- Agent Server: 8767
- Agent Health: 8769
- Telemetry: 8770

### Key Documentation
- **Task Index**: `dnne_docs/for_claude/tasks/INDEX.md`
- **Architecture**: `dnne_docs/architecture/`
- **CLAUDE.md**: Project overview

## Claude Code Capabilities
- **Server Control**: Restart via `/remote_command` endpoint
- **Browser Automation**: UI interaction via MCP
- **WSL2 Access**: Server at `http://172.22.160.1:8188`

## Recent Commits
- Remove debug print statements from export system
- Add retain_graph override support for multi-optimizer workflows
- Fix SimulationTracker template output method bug
- Fix Network node output schema reporting bug
- Update analyze_workflow tool with property-based validation and repair
- Add label rats nest visualization feature
- Clean up debug output and document label improvements

---
*Focus on active tasks in INDEX.md*