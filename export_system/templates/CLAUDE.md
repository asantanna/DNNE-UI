# Templates CLAUDE.md

Code generation templates for converting visual nodes to Python code.

## Overview

Templates use Python string formatting to generate node classes from UI parameters.

## Directory Structure

```
templates/
├── base/              # Framework templates (QueueNode, GraphRunner)
└── nodes/             # Node-specific templates (*_queue.py)
```

## Template Files

- **Queue templates**: `{node_type}_queue.py` (async queue-based)
- **Base classes**: `queue_framework.py`, `graph_runner.py`

## Naming Conventions

- Template variables: `{UPPER_CASE}`
- Generated classes: `{CLASS_NAME}_{NODE_ID}`
- Queue methods: `setup_inputs()`, `setup_outputs()`, `compute()`

## Documentation

For detailed information, see:
- **Template System**: `dnne-docs/architecture/templates.md`
- **Variable Substitution**: `dnne-docs/architecture/templates.md#variable-substitution-rules`
- **Creating Templates**: `dnne-docs/architecture/templates.md#template-development-guidelines`