# DNNE Development Status

$ARGUMENTS

*For historical development sessions, see HISTORY.md*

## READ THESE IMPORTANT GUIDELINES !!
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne-docs/for_claude/tasks/INDEX.md`

## Latest Achievements (2025-08-14)

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

## Quick Reference

### Essential Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Restart DNNE Server (runs on Windows)
use MCP function: "util_restart_dnne"

# Start Agent Client (WSL2)
python dnne-agent/dnne_agent_client.py

# Export workflow
python claude_scripts/programmatic_export.py MNIST_Test

# Run exported workflow
cd export_system/exports/MNIST_Test
python runner.py --epochs 10

# Test telemetry
./dnne-test telemetry

# Build frontend
./build_frontend.sh
```

### Key Ports
- DNNE UI: 8188
- Agent Server: 8767
- Agent Health: 8769
- Telemetry: 8770

### Key Documentation
- **Task Index**: `dnne-docs/for_claude/tasks/INDEX.md`
- **Architecture**: `dnne-docs/architecture/`
- **CLAUDE.md**: Project overview

## Claude Code Capabilities
- **Server Control**: Restart via `/remote_command` endpoint
- **Browser Automation**: UI interaction via MCP
- **WSL2 Access**: Server at `http://172.22.160.1:8188`

## Recent Commits
- `63627331` - Add example filter and sink functions
- `3d83c7b8` - Add Custom Computation node file export
- `ba4fd4c` - Add setup.py for rl_games_dnne
- `d6f20be` - Add FrankaDNNE environment

---
*Focus on active tasks in INDEX.md*