# DNNE Documentation Index

## System Status Overview

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Export System | ✅ Functional | - | Dependency system working |
| Type System | ✅ Complete | - | LOSS_SCALAR refactoring done |
| ML Nodes | ✅ Enhanced | - | GeometricLoss with 5 metrics |
| Utility Nodes | ✅ Enhanced | - | Concat node with modes |
| RL Nodes | 🚧 In Progress | Medium | PPO partially implemented |
| UI Improvements | 📋 Planned | Low | Copy/paste menu items |

## Today's Achievements (Jan 14, 2025)
- Refactored LOSS_TENSOR → LOSS_SCALAR system-wide
- Created Concat node with wait/async modes
- Implemented GeometricLoss with Norm KL Div
- Fixed dependency system for framework files

## Active Priorities

### High Priority
1. Test GeometricLoss export with all metrics
2. Implement distributed training support

### Medium Priority
1. Complete RL node implementations
2. Add transformer architecture nodes

## Documentation Structure

### Core Documentation
- [`CLAUDE.md`](../CLAUDE.md) - AI assistant instructions
- [`dev-status.md`](dev-status.md) - Development status

### Architecture
- [`architecture/`](architecture/) - System design
- [`nodes/`](nodes/) - Node guides
- [`tasks/`](tasks/) - Task tracking

## Quick Access

### Commands
```bash
./dnne.bat              # Start server
./build_frontend.sh     # Build UI
python claude_scripts/programmatic_export.py  # Test export
```