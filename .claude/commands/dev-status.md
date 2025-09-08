# DNNE Development Status

## !!IMPORTANT!! READ THESE IMPORTANT GUIDELINES AND DOCUMENTS !!
- **Why Things Are The Way They Are**: `dnne_docs/architecture/design_rationale.md`
- **Gotchas That Will Burn You**: `dnne_docs/development/gotchas.md`
- **Non-Obvious Debugging Techniques** - `dnne_docs/development/debugging-techniques.md`
- **Coding Guidelines**: `.claude/commands/rules-for-DNNE.md`
- **Task Tracking**: `dnne_docs/for_claude/tasks/INDEX.md`

$ARGUMENTS$

*For historical development sessions, see HISTORY.md*

## Latest Achievements (2025-09-08)

### SGDOptimizer Gradient Accumulation ✅
- **Added batch_size widget** - Enables gradient accumulation over N steps
  - Accumulates gradients without stepping optimizer
  - Automatic averaging via loss scaling (loss/batch_size)
  - Independent batch sizes per optimizer supported
- **Fixed sync checker** - execution_count increments every step
- **All workflows export** - 12/12 workflows tested successfully

### Previous Achievements (2025-09-07)
- TrainingSequencer Complete - Fixed Franka_Coop_V2 deadlock
- Export system fixes - TrainingSequencerExporter working
- Template improvements - Pass loss tensors, not metadata dicts

### Earlier Achievements (2025-09-03)
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
- Add gradient accumulation to SGDOptimizer (2025-09-08)
- Fix sync checker execution count for batch accumulation
- Fix TrainingSequencer deadlock in Franka_Coop_V2 (2025-09-07)
- Add step_complete signals to SGDOptimizer.step_only()
- Fix TrainingSequencerExporter imports and subsystem

---
*Focus on active tasks in INDEX.md*