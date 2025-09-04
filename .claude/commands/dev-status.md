# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-09-03)

### Telemetry System Complete ✅
- **Unified system** - Merged TelemetryClient and MetricsLogger
  - Single fire-and-forget UDP telemetry client
  - Configurable intervals with runtime overrides
  - 50% reduction in telemetry data volume achieved
- **Biologically plausible** - Removed all reward tracking
  - Loss-only tracking for biological algorithms
  - Split done inputs: step_done/episode_done for clarity
- **Improved metrics** - Better naming and essential-only defaults
  - `elapsed_seconds`, `total_timesteps`, `loss_mean`
  - Fixed --enable-telemetry flag to set telemetry_level
- **Testing** - 28 unit tests created (all passing)

### Previous Achievements (2025-09-02)
- Gradient Isolation Removal - Simplified export by removing unnecessary mechanism
- Export System Hardening - Fail-fast validation prevents invalid exports
- Franka_Coop_Nodes Workflow Repair - Fixed phantom connections and Barrier nodes

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
- Complete telemetry system refactoring (2025-09-03)
- Fix --enable-telemetry flag to set telemetry_level
- Improve telemetry metric naming and remove redundant metrics
- Remove gradient isolation mechanism from export templates (2025-09-02)
- Add retain_graph override support for multi-optimizer workflows

---
*Focus on active tasks in INDEX.md*