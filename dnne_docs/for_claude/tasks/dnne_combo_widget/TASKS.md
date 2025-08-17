# DNNE Combo Widget - Task Tracking

*For historical accomplishments, see HISTORY.md*

## Current Status
**✅ COMPLETE** - Generic callback-based combo widget fully implemented (100%)
- Clean generic system with node_data context passing
- WebSocket-based callbacks work perfectly  
- All dynamic widgets update schema properly

## 📋 TODO

### Future Enhancements (Optional)
- [ ] Test with additional node types beyond IsaacGymEnvs
- [ ] Add onFocus/onBlur event support if needed
- [ ] Create developer guide with examples

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