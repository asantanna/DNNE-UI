# DNNE Documentation Index

## System Status Overview

| Component | Status | Priority | Notes |
|-----------|--------|----------|-------|
| Export System | ✅ Complete | - | Widget encapsulation implemented |
| LinearLayer/Network | ✅ Refactored | - | Virtual nodes architecture |
| Isaac Gym Integration | ✅ Fixed | - | YAML-based configuration |
| Test Suite | ✅ Passing | - | 164 tests pass, 0 skipped |
| RL Nodes | ✅ Working | - | PPO with BalancerConfig |
| Balancer Node | ✅ Fixed | - | Naming consistency resolved |

## Today's Achievements (Jan 17, 2025)
- Enforced fail-fast principles across entire codebase
- Removed 71+ silent defaults that were hiding errors
- Fixed Isaac Gym config loader to skip Hydra inheritance
- Enhanced code reviewer with streamlined review process

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
  - [`ui_callbacks.md`](architecture/ui_callbacks.md) - WebSocket-based UI widget callback system
  - [`websocket-not-rest.md`](architecture/websocket-not-rest.md) - WebSocket communication principles
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