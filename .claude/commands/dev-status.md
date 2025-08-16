# DNNE Development Status

*For historical development sessions, see HISTORY.md*

## READ THESE IMPORTANT GUIDELINES !!
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

## Latest Achievements (2025-01-16)

### Dynamic Widget System (IN PROGRESS)
- ✅ Backend endpoint enhanced to handle widget updates and schema display
- ✅ Frontend callback system for task and dynamic widget changes
- ✅ Widget update mechanism for showing/hiding dynamic dropdowns
- 🔧 Issues found with LiteGraph widget rendering:
  - Initial widget labels show as "dynamic_1/2/3" on load
  - Hidden widgets leave gaps (Y positioning not updated)
  - Need to manually recompute widget positions

### @dnne_node Decorator System ✅  
- Automatic node registration via decorator
- Virtual node status enforcement  
- Auto-discovery of exporters based on naming
- All 164 tests passing (100% success)

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
- Implement @dnne_node decorator system
- Auto-discovery for node registration and exporters  
- Update all nodes with decorator
- Fix naming convention mismatches
- 100% test success

---
*Focus on active tasks in INDEX.md*