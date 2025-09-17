# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-09-17)

### Debug Sphere Visual Rendering ✅
- **Replaced physics-based sphere with visual-only wireframe**
  - Uses WireframeSphereGeometry for pure visual rendering
  - No physics interactions - eliminated collision issues
  - Increased density to 24x24 for better visibility
  - Always visible when position set (independent of debug_viz mode)

### Episode Tracking Fix ✅
- **Fixed done signal propagation in SimulationTracker**
  - Save episode state before auto-reset to prevent signal loss
  - Separated loss_mean from episode_loss_mean metrics
  - Eliminated jumps in loss graphs from mixed averaging methods
  - Episodes now correctly counted in telemetry

### Previous Achievements (2025-09-09)
- CustomComputation Debug Visualization - extra_args for dynamic debug data
- Debug augmenter script - Attaches data to action.extra_args
- FrankaDNNE integration - Initial debug sphere implementation

### Earlier Achievements (2025-09-08)
- SGDOptimizer Gradient Accumulation - batch_size widget added
- Fixed sync checker - execution_count increments every step
- All 12 workflows export successfully

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
- Fix episode done signal propagation in IsaacGymSim and SimulationTracker (2025-09-17)
- Replace physics-based debug sphere with visual-only wireframe rendering (2025-09-17)
- Fix end-of-run checkpoint trigger type (2025-09-17)
- Add inference mode checks to SGDOptimizer TrainingSequencer methods (2025-09-17)
- Enable checkpoint loading via --load-checkpoint-dir (2025-09-16)

---
*Focus on active tasks in INDEX.md*