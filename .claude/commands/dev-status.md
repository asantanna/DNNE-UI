# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2026-01-11)

### Documentation Cleanup ✅
- **Reorganized dnne_docs/ structure**
  - Archived completed yield_tests/ experiments
  - Moved features/ implementation notes to development/
  - Fixed broken links and outdated paths
  - Added cross-references between architecture docs
  - Updated node documentation with accurate 25-node list

### Previous Achievements (2025-09-17)
- Debug Sphere Visual Rendering - wireframe geometry, no physics
- Episode Tracking Fix - done signal propagation in SimulationTracker
- CustomComputation Debug Visualization - extra_args for debug data

### Earlier Achievements (2025-09-08)
- SGDOptimizer Gradient Accumulation - batch_size widget added
- Fixed sync checker - execution_count increments every step

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
- docs: Clean up and reorganize dnne_docs/ documentation (2026-01-11)
- Fix episode done signal propagation in IsaacGymSim and SimulationTracker (2025-09-17)
- Replace physics-based debug sphere with visual-only wireframe rendering (2025-09-17)
- Fix end-of-run checkpoint trigger type (2025-09-17)
- Add inference mode checks to SGDOptimizer TrainingSequencer methods (2025-09-17)

---
*Focus on active tasks in INDEX.md*