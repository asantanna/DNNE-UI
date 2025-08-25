# UI Proxy Tasks

## Current Status
UI Proxy is in initial design phase.

### Important Documents/Files
- main docs: `dnne_docs/for_claude/tasks/ui_proxy/README.md`
- Architecture specification: `dnne_docs/architecture/ui_proxy.md`
- Related: UI Callbacks: `dnne_docs/architecture/ui_callbacks.md`
- WebSocket architecture: `dnne_docs/architecture/websocket_not_rest.md`
- Server WebSocket handler: `server.py` (WebSocket message handling)
- Frontend WebSocket client: `DNNE-UI-Frontend/src/scripts/api.js`

## Active TODOs

### High Priority
1. **Protocol Design**
   - [ ] Define `ui_proxy_execute` message structure
   - [ ] Define `ui_proxy_response` message structure
   - [ ] Design request_id tracking mechanism for async responses
   - [ ] Specify node context access patterns

2. **Server Implementation**
   - [ ] Add `ui_proxy_execute` handler in server.py WebSocket handler
   - [ ] Implement request tracking for matching responses
   - [ ] Add helper methods for nodes to trigger UI updates
   - [ ] Create base class method for UI proxy execution
   
### Medium Priority
1. **Frontend Implementation**
   - [ ] Add `ui_proxy_execute` message handler in api.js
   - [ ] Implement JavaScript execution sandbox/context
   - [ ] Add response message generation
   - [ ] Ensure node context is accessible to executed code

2. **Node Integration**
   - [ ] Update Split node to use UI proxy for label updates
   - [ ] Add UI proxy support to Network node for layer display
   - [ ] Document UI proxy usage patterns for node developers

### Low Priority
1. **Security & Validation**
   - [ ] Implement JavaScript execution limits (timeout, memory)
   - [ ] Add validation for node_id existence
   - [ ] Consider sandboxing options for JavaScript execution
   - [ ] Add rate limiting for UI proxy requests

2. **Advanced Features**
   - [ ] Batch UI updates (multiple nodes in one message)
   - [ ] Broadcast UI updates to all connected clients
   - [ ] Add UI proxy debugging/logging capabilities
   - [ ] Create UI proxy testing utilities

### Future Considerations
1. **Widget Library Extensions**
   - [ ] Dynamic widget creation via UI proxy
   - [ ] Widget removal/hiding via UI proxy
   - [ ] Widget property animation support
   
2. **State Synchronization**
   - [ ] UI state persistence across reconnects
   - [ ] Multi-client UI state sync
   - [ ] Conflict resolution for concurrent UI updates