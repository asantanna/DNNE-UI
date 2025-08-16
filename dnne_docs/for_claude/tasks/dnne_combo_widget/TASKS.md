# DNNE Combo Widget - Task Tracking

*For historical accomplishments, see HISTORY.md*

## Current Status
**In Development** - Creating generic callback-based combo widget (~0%)
- Replacing hardcoded IsaacGymEnvs hack with clean generic system
- WebSocket-based callbacks for any node to use

## 📋 TODO

### High Priority - Core Implementation

1. **Create useDnneComboWidget.ts**
   - [ ] Implement base combo widget with callback override
   - [ ] Add WebSocket message sending for events
   - [ ] Handle callback responses with code execution
   - [ ] Support onLoad, onChange events initially

2. **Register widget in system**
   - [ ] Add DNNE_COMBO to widgets.ts registry
   - [ ] Ensure proper widget type selection logic

3. **Backend WebSocket handler**
   - [ ] Add widget_callback message handler in server.py
   - [ ] Route callbacks to appropriate node handlers
   - [ ] Send widget_callback_response messages

4. **Update IsaacGymEnvsNode**
   - [ ] Change INPUT_TYPES to use widgetType: "DNNE_COMBO"
   - [ ] Add widget_id and listen_to parameters
   - [ ] Implement callback handler for task widget

### Medium Priority - Cleanup

5. **Remove old hack code**
   - [ ] Delete useDNNEComboWidget.ts (the hack)
   - [ ] Remove /dnne/env_config endpoint from routes.py
   - [ ] Clean up hardcoded IsaacGymEnvs logic

6. **Test with other nodes**
   - [ ] Verify generic nature with non-IsaacGymEnvs nodes
   - [ ] Add examples to other node types

### Low Priority - Future Enhancements

7. **Additional event support**
   - [ ] onFocus/onBlur events
   - [ ] onHover events
   - [ ] Custom event types

8. **Documentation**
   - [ ] Create usage guide for node developers
   - [ ] Document WebSocket protocol
   - [ ] Add examples

## 💡 Quick Reference

### Widget Specification (Backend)
```python
"my_param": (["option1", "option2"], {
    "widgetType": "DNNE_COMBO",
    "widget_id": "MyNode.my_param",
    "listen_to": ["onChange", "onLoad"]
})
```

### WebSocket Protocol
Frontend → Backend:
```json
{
  "type": "widget_callback",
  "widget_id": "MyNode.my_param",
  "event": "onChange",
  "event_params": { "value": "option2" }
}
```

Backend → Frontend:
```json
{
  "type": "widget_callback_response",
  "widget_id": "MyNode.my_param",
  "code_payload": "// JavaScript to execute",
  "chain": true
}
```

### Key Design Principles
1. **Generic** - Any node can use without frontend changes
2. **WebSocket-based** - Follows DNNE architecture (no REST)
3. **Event-driven** - Nodes specify which events they care about
4. **Code injection** - Backend can execute JS in response
5. **Clean separation** - No node-specific logic in widget

## Implementation Notes

### Why This Approach?
- Current `useDNNEComboWidget` hardcodes IsaacGymEnvs logic
- REST endpoint `/dnne/env_config` violates DNNE WebSocket principle
- Need generic solution for all nodes to use callbacks

### Technical Details
- Widget extends standard ComboWidget
- Overrides event handlers for listened events
- Uses `api.socket.send()` for communication
- Executes returned JavaScript via `eval()` or safer method

## Future Enhancements
1. Support for other widget types (String, Int, Float)
2. Batch callback responses for efficiency
3. Widget state synchronization across clients
4. Callback filtering/throttling options
5. Security sandbox for code execution