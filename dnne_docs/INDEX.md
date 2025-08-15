# DNNE Documentation Index

## System Status Overview

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Export System | ✅ Complete | - | All workflows export cleanly |
| LinearLayer/Network | ✅ Refactored | - | Virtual nodes architecture |
| Isaac Gym Integration | ✅ Fixed | - | YAML-based configuration |
| Test Suite | ✅ Passing | - | 164 tests pass |
| RL Nodes | ✅ Working | - | PPO with BalancingConfig |
| Other Isaac Envs | 📋 Planned | Low | Need dnne: sections in YAMLs |

## Today's Achievements (Aug 15, 2025)
- Refactored LinearLayer as virtual nodes within Networks
- Created export utilities with context management
- Fixed Isaac Gym integration with YAML configuration
- All 7 workflows export with zero warnings
- All unit tests passing

## Active Priorities

### High Priority
None - System is fully operational

### Low Priority
1. Add dnne: sections to other Isaac Gym environment YAMLs
2. Export profiling and metrics
3. Custom node template support

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
# Activate environment
source /home/asantanna/miniconda/bin/activate DNNE_PY38

# Test all exports
python claude_scripts/test_all_exports.py

# Run unit tests
./dnne_test quick

# Start server (Windows)
dnne.bat
```