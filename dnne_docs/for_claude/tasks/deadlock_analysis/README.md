# Deadlock Analysis Tool Development

## Overview
Development of an event-driven dataflow simulator to automatically identify root causes of deadlocks in DNNE workflows, particularly for complex multi-node synchronization patterns like Franka_Coop_Nodes.

## Problem Statement
- Current tool only logs queue operations but doesn't capture causality
- Franka_Coop_Nodes deadlocks after ~4.6 seconds with complex circular dependencies
- Manual analysis is error-prone and doesn't scale to larger workflows
- Need to understand why workflows that initially run successfully eventually deadlock

## Solution Approach
Event-driven simulation that models actual dataflow and node state transitions to identify deadlock patterns.

### Implementation Architecture
- **Parallel Structure**: Each node template (`/export_system/templates/nodes/{node}_queue.tpl`) has a corresponding simulator (`/deadlock_tool/node_simulators/{node}_queue_sim.py`)
- **Consistent Naming**: Simulators mirror template names for predictability
- **Modular Design**: Each node simulator is self-contained and testable
- **Behavioral Parity**: Simulators accurately model runtime behavior of templates

## Development Phases

### Phase 1: Core Infrastructure ✅ [COMPLETE]
- Basic deadlock data collection
- Event logging framework
- Initial analysis tool

### Phase 2: Graph Model [IN PROGRESS]
- Parse graph structure and connections
- Model node types and behaviors
- Define state transitions

### Phase 3: Event Replay System [TODO]
- Chronological event processing
- Pending data tracking
- Node state management

### Phase 4: Deadlock Detection [TODO]
- Progress monitoring
- Wait-for graph construction
- Circular dependency detection

### Phase 5: Root Cause Analysis [TODO]
- Minimal cycle identification
- Deadlock chain explanation
- Fix recommendations

### Phase 6: Testing & Validation [TODO]
- Local development testing in `/deadlock_tool/test_scripts/`
- Test individual node simulators
- Test on Franka_Coop_Nodes data
- Test on simpler deadlock patterns
- Migration to formal test suite after development

## Key Files
- `/home/asantanna/DNNE/DNNE-UI/deadlock_tool/` - Main tool implementation
- `/tmp/dnne_deadlock_data/` - Runtime data collection
- `dnne_docs/experiments/franka_coop_nodes/` - Test case documentation

## Success Criteria
- Correctly identifies Franka_Coop_Nodes deadlock after 4.6s
- Provides clear, actionable root cause explanation
- Handles complex synchronization patterns (barriers, Eat_N, multi-input nodes)
- Generates minimal deadlock cycles, not just all blocked nodes