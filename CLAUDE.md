# CLAUDE.md

This file provides essential information for Claude Code when working with the DNNE repository.

## Project Overview

**DNNE** (Distributed Neural Network Editor) is a visual programming environment for building neural networks and robotics control systems. It exports visual node graphs to standalone Python scripts that run on GPU clusters and robotics simulators.

## Repository Locations

- **Backend (this repo)**: `/mnt/e/ALS-Projects/DNNE/DNNE-UI`
- **Frontend**: `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend`
- **Linux Support**: `/home/asantanna/DNNE-LINUX-SUPPORT`

## Essential Commands

```bash
# Activate conda environment (REQUIRED)
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Start server (Windows only)
python main.py

# Export workflow
python claude_scripts/programmatic_export.py [workflow_name]

# Run exported code
cd export_system/exports/{workflow_name}
python runner.py
```

## Documentation

For detailed documentation, see:
- **Architecture**: `docs-dnne/architecture/`
- **Node Reference**: `docs-dnne/nodes/`
- **Examples**: `docs-dnne/examples/`
- **Export System**: `docs-dnne/architecture/export_system.md`
- **Development**: `docs-dnne/development/`

## Key Files
- `server.py` - Web server with export functionality
- `export_system/` - Code generation system
- `custom_nodes/` - Node implementations
- `docs-dnne/` - All documentation

## Important Notes
- Always activate conda environment before running
- Import isaacgym before torch to avoid conflicts
- Use `dnne-test` script for running tests