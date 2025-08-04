# DNNE Development Status

## Current Work
See `docs-dnne/for_claude/TASKS.md` for the complete task list and project roadmap.

## Recent Accomplishments (2025-08-04)
- ✅ Simplified WSL2 access with --listen 0.0.0.0 flag
- ✅ Consolidated agent documentation into single dnne-agent.md file
- ✅ Removed unnecessary Chrome proxy complexity

## Recent Accomplishments (2025-08-02)
- ✅ Refactored dnne-agent system for production readiness
- ✅ Implemented asyncio-based UDP telemetry (replaced busy-wait polling)
- ✅ Added test port architecture (8768) for isolated testing
- ✅ All agent tests passing: connectivity, deployment, execution, telemetry

## Recent Accomplishments (2025-02-02)
- ✅ Centralized all paths through dnne_config.json
- ✅ Made exported packages self-sufficient (copy framework files)
- ✅ All tests passing: 171 unit tests + 3 integration tests

## Quick Reference

### Essential Commands
```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Start DNNE UI (Windows)
./dnne.bat

# Start Agent Client (WSL2)
python dnne-agent/dnne_agent_client.py

# Export workflow
python claude_scripts/programmatic_export.py MNIST_Test

# Run exported workflow
cd export_system/exports/MNIST_Test
python runner.py --epochs 10
```

### Key Ports
- **8188**: DNNE UI
- **8766-8769**: Agent system
- **9999**: Telemetry UDP

### Key Documentation
- **Tasks**: `docs-dnne/for_claude/TASKS.md` - Current work items
- **Agent**: `docs-dnne/architecture/dnne-agent.md` - Agent architecture
- **Runner**: `docs-dnne/development/runner.md` - Command line switches for runner.py
- **CLAUDE.md**: Project overview and development guidance