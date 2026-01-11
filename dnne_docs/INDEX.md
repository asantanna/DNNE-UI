# DNNE Documentation Index

*Last Updated: 2026-01-11*

## Documentation Structure

### Core Documentation
- [`README.md`](README.md) - User-facing overview and navigation
- [`CLAUDE.md`](../CLAUDE.md) - Project overview and AI assistant context

### Architecture (`architecture/`)
System design and technical details:
- `export_system.md` - Workflow to Python code conversion
- `queue_framework.md` - Async queue-based execution
- `templates.md` - Code generation templates
- `adaptive_yielding.md` - Cooperative multitasking
- `system_balancing.md` - Load balancing and scheduling
- `telemetry.md` - Metrics and monitoring
- `design_rationale.md` - Why things are the way they are
- `ui_callbacks.md` - WebSocket-based UI widget callback system
- `websocket_not_rest.md` - WebSocket communication principles

### Development (`development/`)
Practical guides for developers:
- `gotchas.md` - Common pitfalls and solutions
- `debugging-techniques.md` - Non-obvious debugging approaches
- `code-quality-checklist.md` - Quality standards
- `deadlock_analysis.md` - Deadlock detection and prevention
- `sync_check.md` - Synchronization checking

### Nodes (`nodes/`)
Node reference documentation:
- `ml/` - ML nodes (datasets, layers, training)
- `rl/` - RL nodes (PPO agent/config)
- `robotics/` - Isaac Gym integration
- `utility/` - Data flow and control nodes

### Examples (`examples/`)
Complete working examples:
- `mnist_classification.md` - Supervised learning
- `cartpole_ppo.md` - RL with PPO
- `isaac_gym_integration.md` - Robotics integration

### Theory (`theory/`)
Research and advanced concepts:
- `shadow_environment.md` - Differentiable control through non-differentiable simulators

### Experiments (`experiments/`)
Research and experimental work:
- `franka_coop_nodes/` - Franka robot collaboration research
- `performance/` - Performance analysis
- `archive/` - Completed experiments (yield_tests)

### Future (`future/`)
Planned features and improvements:
- Organized by category (ML, robotics, system, UI)
- Priority and effort estimates

### For Claude (`for_claude/`)
Claude Code session context:
- `tasks/INDEX.md` - Current task status overview
- Per-component task tracking and history

## Quick Links

| Need | Location |
|------|----------|
| Current tasks | `for_claude/tasks/INDEX.md` |
| Common gotchas | `development/gotchas.md` |
| Design decisions | `architecture/design_rationale.md` |
| Export details | `architecture/export_system.md` |

## Commands Reference

```bash
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Export workflow
python claude_scripts/programmatic_export.py WORKFLOW_NAME

# Run tests
./dnne_test quick
```
