# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-08-27)

### Multi-Optimizer Support & Export Improvements ✅
- **Retain Graph Override System** - Support for cooperative learning
  - Added `--override all:retain_graph=True` for multi-optimizer workflows
  - Special "all" subsystem in runners applies settings to all nodes
  - SGDOptimizer template checks runtime configuration
  - Enables Franka_Coop_Nodes workflow without hardcoding

### Export System Refinements ✅
- **Critical Bug Fixes**
  - Fixed SimulationTracker using wrong output method (template fix)
  - Fixed Network nodes reporting incorrect output dimensions
  - All template fixes ensure future exports are correct
- **Debug Output Cleanup**
  - Removed all debug prints from exporters and templates
  - Clean programmatic export output

### Previous Achievements (2025-08-24)
- Workflow analysis & repair tools with property-based validation
- Label rats nest visualization in frontend
- Dictionary-free label system implementation

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