# Log Window Tasks

*Last Updated: 2025-08-07*

## Quick Stats
- **Status**: Testing Phase
- **Priority**: High
- **Completion**: ~80%
- **Dependencies**: DNNE Agent integration ✅

## Current Status

The log window functionality is implemented and in testing phase. Core features are working but several UI elements need testing and debugging, particularly the status bar workflow count updates when workflows are exported to remote agents.

## ✅ Completed

### Phase 1: Backend Implementation ✅
- [x] Implement log storage system (`remote_clients/{client}/{workflow}/run_logs/`)
- [x] Create log retrieval API endpoints
- [x] Add log streaming for active workflows
- [x] Implement log filtering by client/workflow
- [x] Handle log file lifecycle (open on start, close on stop)

### Phase 2: Frontend UI ✅
- [x] Create LogsTerminal.vue component
- [x] Add agent dropdown for selecting which agent's logs to view
- [x] Add log type dropdown for filtering log types
- [x] Implement auto-scroll functionality for following new log entries
- [x] Add visual indicator for agent running status
- [x] Create AgentStatusBar.vue with workflow counts per client

### Phase 3: Integration ✅
- [x] Connect to DNNE Agent log streams
- [x] Handle remote client log collection
- [x] Implement workflow_log message handler in server.py
- [x] Create metadata.json for each deployment
- [x] Add message handling for workflow_started/workflow_stopped

## 🚧 In Progress - Testing Phase

### Critical Bug: Status Bar Updates
- [ ] **Debug status bar workflow count updates** - Messages arrive but counts don't update
- [ ] Trace message flow from agent → server → UI → agentStore → status bar
- [ ] Verify workflow_started/workflow_stopped message handling
- [ ] Check clientWorkflows Map updates in agentStore
- [ ] Ensure AgentStatusBar computed properties are reactive

### UI Component Testing
- [ ] **Test auto-scroll functionality** - Verify it follows new log entries properly
- [ ] **Test agent dropdown** - Ensure proper agent selection and log filtering
- [ ] **Test log type dropdown** - Verify filtering works for different log types
- [ ] **Test visual indicators** - Confirm "agent running" indicator updates correctly

### Integration Testing
- [ ] Test with multiple concurrent workflows
- [ ] Verify remote log collection from multiple agents
- [ ] Test log streaming for long-running workflows
- [ ] Verify log file size handling

## 📋 TODO

### Bug Fixes (After Testing)
- [ ] Fix status bar workflow count update issue
- [ ] Address any auto-scroll issues found in testing
- [ ] Fix dropdown filtering issues if any
- [ ] Resolve visual indicator update problems

### Performance & Polish
- [ ] Add performance optimizations for large log files
- [ ] Implement log rotation/cleanup policies
- [ ] Add error highlighting in logs
- [ ] Create log level filtering (debug, info, warning, error)
- [ ] Add log export functionality

### Documentation
- [ ] Create user documentation for log window
- [ ] Document log retention policies
- [ ] Add troubleshooting guide

## 🐛 Known Issues

### Status Bar Not Updating
- **Problem**: When workflows are exported to remote agents and executed, notification messages arrive at the UI but the status bar doesn't update workflow counts
- **Impact**: Users can't see active workflow counts in status bar
- **Workaround**: Check logs directly for workflow status
- **Investigation**: Need to trace message flow through the system

### Testing Needed
- Auto-scroll behavior not fully verified
- Agent dropdown selection needs testing with multiple agents
- Log type filtering not fully tested
- Visual "agent running" indicator needs state change testing

## 💡 Notes

### Implementation Details
- Logs stored in `remote_clients/{client}/{workflow}_wf_{id}/run_logs/`
- Content-based workflow IDs using SHA256 hash (wf_{hash[:12]})
- LogsTerminal.vue handles display
- AgentStatusBar.vue shows per-client workflow counts
- agentStore manages workflow state with clientWorkflows Map

### Message Flow for Debugging
1. Agent sends workflow_started/workflow_stopped to agent server
2. Agent server forwards to DNNE server via WebSocket
3. DNNE server broadcasts to UI WebSocket clients
4. UI receives in WebSocket handler
5. agentStore.handleAgentMessage processes the message
6. clientWorkflows Map should update
7. AgentStatusBar computed properties should reactively update

### Testing Commands
```bash
# Start agent client with a workflow
python dnne-agent/dnne_agent_client.py

# Export and run workflow on remote
# Use UI to export to remote agent with "Run after export" checked

# Monitor WebSocket messages in browser console
# Check Network tab for WebSocket frames
```

## Future Enhancements

1. Advanced log analysis tools
2. Log pattern detection for common errors
3. Performance metrics extraction from logs
4. Log comparison between runs
5. Integration with monitoring systems
6. Automated error detection and alerting
7. Log search functionality with regex support
8. Implement "All" option in log viewer for multiple agents

---
*Focus: Debug status bar updates and complete UI testing*