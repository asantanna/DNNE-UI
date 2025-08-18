# DNNE Task Index

*Last Updated: 2025-01-18*  
*For historical achievements, see HISTORY.md*

## Task Status Overview

| Component | Status | Progress | Priority | Last Updated |
|-----------|--------|----------|----------|--------------|
| **Type System** | 🟢 Complete | 100% - Color system fixed | - | 2025-08-15 |
| **MCP Integration** | 🟢 Complete | 42 tools implemented | Low | 2025-08-12 |
| **DNNE Agent** | 🟢 Complete | Fully functional | - | 2025-08-11 |
| **Log Viewer** | 🟡 Working | ~95% - Stream end issue | Medium | 2025-08-15 |
| **Core Infrastructure** | 🟢 Complete | 100% - All features done | - | 2025-08-15 |
| **Export System** | 🟢 Complete | 100% - Schema resolution fixed | - | 2025-08-17 |
| **Runner Args Dialog** | 🟢 Complete | 100% - All features done | - | 2025-08-12 |
| **Node System** | 🟡 Working | Tensor node added, widget issues remain | High | 2025-08-17 |
| **DNNE Combo Widget** | 🟢 Complete | 100% - Generic WebSocket callbacks | - | 2025-08-16 |
| **Franka Coop Control** | 🟡 Working | Schema aligned, loss implemented | High | 2025-01-18 |

## Active Priority Items

### High Priority
1. **Franka Cooperative Control**: Export and test the workflow
   - Export using programmatic_export.py
   - Test training and monitor coordination emergence
   - Implement PD control for joints 3-6
   
2. **Node System**: Fix dynamic widget display issues
   - Initial widget labels showing as "dynamic_1/2/3"
   - Widget hiding leaves gaps (Y positioning)
   - Widget labels not updating to actual names

### Medium Priority
1. **Log Viewer**: Fix streaming logs missing final DNNE stop line
2. **Node System**: Add 'group' widget to Balancer nodes

### Low Priority
1. **Log Viewer**: Fix dropdown clickable when Local selected
2. **Log Viewer**: Fix run logs briefly appearing in telemetry view
3. **MCP**: Add util_set_DNNE_log_level and util_set_agent_server_log_level functions
4. **Node System**: Fix balancing_node → network_node connection color
5. **Node System**: Rename GeometricLoss output to "loss"

## Component Details

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