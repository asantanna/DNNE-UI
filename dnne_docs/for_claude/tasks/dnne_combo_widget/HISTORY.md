# DNNE Combo Widget - History

## 2025-08-16: Feature Completed ✅

### Phase 1: Initial Implementation
- Identified need to replace hardcoded IsaacGymEnvs hack
- Designed generic WebSocket-based callback system
- Created `useDnneComboWidget.ts` with WebSocket callback support
- Registered DNNE_COMBO widget type in widgets.ts
- Added widget_callback handler to server.py WebSocket processing
- Updated IsaacGymEnvsNode to use new widget with callbacks

### Phase 2: Schema Display Fix
- **Problem**: Dynamic widgets weren't updating schema_display
- **Challenge**: Backend needed complete widget state but only received changed value
- **Solution**: Frontend sends all widget values as `node_data` in event_params
- **Implementation**:
  - Frontend collects all widget values from node.widgets array
  - Sends complete state in `node_data` field (completely generic)
  - Backend uses node_data to calculate correct schema
  - JavaScript updates schema_display widget with new text
- **Result**: All widgets (task, subtask, controlType) now update schema correctly

### Key Innovation
The `node_data` approach is completely generic - the widget doesn't need to know anything about specific node types. It simply iterates `node.widgets` and sends all values, giving the backend complete context for any calculations.