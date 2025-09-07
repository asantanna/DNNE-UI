# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-09-07)

### TrainingSequencer Complete ✅
- **Fixed deadlock** - Resolved Franka_Coop_V2 circular dependency
  - Added step_complete signals to SGDOptimizer.step_only()
  - Made step_only() async and properly awaited
- **Export system fixes** - TrainingSequencerExporter working
  - Fixed missing import of export_utils
  - Changed SUBSYSTEM_ML to SUBSYSTEM_TRAINING
- **Template improvements** - Pass loss tensors, not metadata dicts

### Previous Achievements (2025-09-03)
- Telemetry System Complete - Unified client, 50% data reduction
- Gradient Isolation Removal - Simplified export templates
- Export System Hardening - Fail-fast validation
- Franka_Coop Workflow Repair - Fixed phantom connections

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
- Fix TrainingSequencer deadlock in Franka_Coop_V2 (2025-09-07)
- Add step_complete signals to SGDOptimizer.step_only()
- Fix TrainingSequencerExporter imports and subsystem
- Complete telemetry system refactoring (2025-09-03)
- Remove gradient isolation mechanism from export templates (2025-09-02)

---
*Focus on active tasks in INDEX.md*