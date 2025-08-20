# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design-rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-08-20)

### FrankaDNNE Environment Reset Control ✅
- Disabled auto-reset in `post_physics_step` for manual control
- Added proper `reset()` method for trigger-based resets
- Implemented target-based episode completion (10cm threshold)
- Episodes now end on timeout OR reaching target
- Debug output added when target is reached

### IsaacGymSim Node Trigger Support ✅
- Verified "done" trigger properly emitted when episode ends
- Verified "reset" input trigger properly handled
- `reset_when_done=True` auto-resets while still sending done signal
- `reset_when_done=False` waits for manual reset trigger
- Full workflow control over reset timing achieved

### Previous Week (2025-08-17)
- Tensor Constant Node with 9 initialization modes
- Export System Schema Resolution fixes
- Schema Format Enhancement for YAML

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
- Disable FrankaDNNE auto-reset for manual control
- Add target-based done trigger to FrankaDNNE
- Add manual reset method to FrankaDNNE
- Verify IsaacGymSim done/reset trigger handling
- Implement Tensor constant node with 9 initialization modes
- Support simplified schema format (single number for single elements)
- Implement @dnne_node decorator system

---
*Focus on active tasks in INDEX.md*