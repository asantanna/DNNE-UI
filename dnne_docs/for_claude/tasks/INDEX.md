# DNNE Task Index

*Last Updated: 2025-09-01*  
*For historical achievements, see HISTORY.md*

## Task Status Overview

| Component | Status | Progress | Priority | Last Updated |
|-----------|--------|----------|----------|--------------|
| **Export System** | 🟢 Complete | Fail-fast validation, no partial exports | - | 2025-08-31 |
| **Franka Coop Control** | 🟢 Complete | Workflow repaired and exports cleanly | - | 2025-08-31 |
| **Label Connections** | 🟢 Complete | Rats nest + workflow repair tool | - | 2025-08-24 |
| **Type System** | 🟢 Complete | 100% - Color system fixed | - | 2025-08-15 |
| **MCP Integration** | 🟢 Complete | 42 tools implemented | Low | 2025-08-12 |
| **DNNE Agent** | 🟢 Complete | Fully functional | - | 2025-08-11 |
| **Log Viewer** | 🟡 Working | ~95% - Stream end issue | Medium | 2025-08-15 |
| **Core Infrastructure** | 🟢 Complete | 100% - All features done | - | 2025-08-15 |
| **Runner Args Dialog** | 🟢 Complete | 100% - All features done | - | 2025-08-12 |
| **Node System** | 🟢 Complete | Eat_N & Barrier synchronization nodes | - | 2025-08-21 |
| **DNNE Combo Widget** | 🟢 Complete | 100% - Generic WebSocket callbacks | - | 2025-08-16 |
| **UI Proxy** | 🟡 Design | Architecture documented, ready for implementation | Medium | 2025-08-18 |
| **Telemetry System** | 🟡 In Progress | Phase 3/4 - Templates done, TelemetryClient pending | High | 2025-09-03 |

## Today's Achievements (2025-09-03)

✅ **Telemetry System Implementation**: Phases 1-3 complete
- **Phase 1**: Deleted MetricsLogger system entirely
- **Phase 2**: Updated all UI nodes and exporters
  - Added report_interval to BalancerNode
  - Added telemetry_level to all nodes
  - Removed window-based telemetry from EpochTracker
  - Updated SimulationTracker to simplified interval format
- **Phase 3**: Updated node templates
  - Implemented telemetry levels (off/essential/extended/debug)
  - Made all intervals configurable
  - Simplified metrics to essential only
- **Remaining**: TelemetryClient enhancement and testing

✅ **Telemetry System Planning**: Complete refactoring plan created
- Analyzed current system and identified redundancies
- Created policy document defining essential vs optional metrics
- Designed phase-based implementation plan
- SimulationTracker refocused on loss (simulator-specific, not RL)
- All intervals will be configurable via --override flags

## Previous Achievements (2025-09-01)

✅ **SimulationTracker Telemetry**: Reduced volume with configurable reporting
- Added time/step/episode-based reporting modes with intervals
- Implemented statistical aggregation (min/max/mean/std/percentiles)
- Created shared time_utils.py for duration parsing ("10s", "2m30s")
- Added comprehensive documentation at dnne_docs/nodes/robotics/

✅ **Split Node Enhancement**: Index range support for cleaner workflows
- Added support for inclusive ranges like "[3:5], [10:18]"
- Maintains backward compatibility with integer format
- Created 23 unit tests for comprehensive coverage

✅ **Documentation Cleanup**: Centralized all TASKS.md/HISTORY.md files
- Removed 4 redundant files outside dnne_docs/for_claude/tasks
- All task tracking now in proper location

## Active Priority Items

### Medium Priority
1. **UI Proxy**: Implement server-initiated JavaScript execution
   - Enable asynchronous UI state updates from backend
   - Primary use case: Split node updating output labels dynamically
   - Architecture documented, ready for implementation
2. **Log Viewer**: Fix streaming logs missing final DNNE stop line
3. **Node System**: Add 'group' widget to Balancer nodes
4. **Widget Display**: Fix widget hiding without gaps (Y positioning)

### Low Priority
1. **Log Viewer**: Fix dropdown clickable when Local selected
2. **Log Viewer**: Fix run logs briefly appearing in telemetry view
3. **MCP**: Add util_set_DNNE_log_level and util_set_agent_server_log_level functions
4. **Node System**: Fix balancing_node → network_node connection color
5. **Node System**: Rename GeometricLoss output to "loss"

## Component Details

### Telemetry System (`telemetry/TASKS.md`)
- Complete refactoring to eliminate redundancy
- Single TelemetryClient system (removing MetricsLogger)
- Configurable intervals via --override
- 50% reduction in data volume target
- Phase-based implementation planned

### Type System (`nodes/type_system.md`)
- Refined types implemented (e.g., BATCH_IMAGE_TENSOR, NETWORK_MODEL_OBJ)
- Wildcard validation system working (*TENSOR matches any _TENSOR suffix)
- Dynamic color palette substitution system implemented
- CONFIG types now correctly show red instead of green
- All node definitions updated with PYDICT suffixes for config types

### Node System (`nodes/TASKS.md`)
- All core nodes operational
- New features planned: Split node, group widgets
- Minor fixes needed for colors and naming

### Export System (`export_system/TASKS.md`)
- All tests passing (164 tests)
- LinearLayer/Network architecture refactored
- Configuration-based paths implemented

### Log Viewer (`log_viewer/TASKS.md`)
- Telemetry viewing implemented
- Minor dropdown behavior issues remain

### MCP Integration (`MCP/TASKS.md`)
- All 42 tools implemented and tested
- Few low-priority utility functions to add

### DNNE Combo Widget (`dnne_combo_widget/TASKS.md`)
- Generic callback-based widget system
- WebSocket protocol for widget events
- Replaces hardcoded IsaacGymEnvs hack

### UI Proxy (`ui_proxy/TASKS.md`)
- Server-initiated JavaScript execution system
- Enables dynamic UI updates from backend logic
- Architecture phase complete, implementation pending

### DNNE Agent (`dnne_agent/TASKS.md`)
- Phase 12 complete - fully functional
- Remote deployment and telemetry working

### Runner Args Dialog (`runner_args_dialog/TASKS.md`)
- Complete - JSON-driven UI working perfectly

### Core Infrastructure (`core_infrastructure/TASKS.md`)
- All features operational
- Minor Windows URL display issue

## Quick Links
- [Task Template](TEMPLATE.md) - For creating new task files
- [Main README](../README.md) - Documentation for Claude sessions
- [CLAUDE.md](../../CLAUDE.md) - Project overview

## How to Use
1. Check status table for quick overview
2. See individual TASKS.md files for active work
3. Check HISTORY.md files for completed work
4. Update this index when priorities change

---
*This index helps Claude Code quickly understand current DNNE component status*