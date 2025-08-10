# Export System CLAUDE.md

The export system converts visual node graphs into standalone Python scripts.

## Overview

The export system is DNNE's core innovation - it generates production-ready Python code from visual workflows that can run independently on GPU clusters, robotics simulators, or edge devices.

## Directory Structure

```
export_system/
├── graph_exporter.py      # Core export logic
├── node_exporters/        # Node-specific exporters
├── templates/             # Code generation templates
│   ├── base/             # Queue framework
│   └── nodes/            # Node templates
└── exports/              # Generated scripts
```

## Key Components

- **Graph Exporter**: Parses workflows and orchestrates export
- **Node Exporters**: Extract parameters from UI nodes
- **Templates**: Python code templates with variable substitution
- **Queue Framework**: Async execution architecture

## Documentation

For detailed information, see:
- **Export System Architecture**: `dnne-docs/architecture/export_system.md`
- **Template System**: `dnne-docs/architecture/templates.md`
- **Queue Framework**: `dnne-docs/architecture/queue_framework.md`