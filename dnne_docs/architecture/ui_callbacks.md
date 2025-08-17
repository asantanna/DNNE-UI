# UI Widget Callbacks Architecture

*Last Updated: 2025-08-16*

## Overview

DNNE implements a generic WebSocket-based callback system that allows backend nodes to respond to UI widget events. This replaces node-specific hardcoding in the frontend with a clean, extensible protocol.

## Design Principles

1. **WebSocket-Only Communication**: All dynamic UI callbacks use WebSocket messages, never REST endpoints
2. **Generic Implementation**: Frontend widgets have no node-specific logic
3. **Event-Driven**: Nodes explicitly declare which events they want to handle
4. **Code Injection**: Backend can execute JavaScript in the frontend as response
5. **Bidirectional**: Supports both UI→Backend events and Backend→UI updates

## Protocol Specification

### Widget Declaration (Backend)

Nodes declare callback-enabled widgets in their `INPUT_TYPES`:

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "my_param": (["option1", "option2", "option3"], {
                "widgetType": "DNNE_COMBO",           # Use callback-enabled widget
                "widget_id": "MyNode.my_param",       # Unique identifier
                "listen_to": ["onChange", "onLoad"],  # Events to handle
                # Note: No default needed - onLoad callback will initialize
            })
        }
    }
```

### Message Flow

#### 1. UI → Backend: Widget Event Notification

When a listened event occurs, the widget sends:

```json
{
  "type": "widget_callback",
  "widget_id": "MyNode.my_param",
  "event": "onChange",
  "event_params": {
    "value": "option2",
    "oldValue": "option1",
    "node_id": 42,
    "node_data": {  // All widget values from the node
      "my_param": "option2",
      "other_widget": "value1",
      "another_widget": 123
    }
  }
}
```

**Key Feature**: The `node_data` field contains ALL widget values from the node, giving the backend complete context for calculating updates.

#### 2. Backend Processing

The backend's WebSocket handler routes the message:

```python
# In server.py WebSocket handler
if msg_type == "widget_callback":
    widget_id = message["widget_id"]
    node_type, widget_name = widget_id.split(".")
    
    # Find handler for this node type
    handler = widget_callback_handlers.get(node_type)
    if handler:
        response = await handler(message)
        await ws.send_json(response)
```

#### 3. Backend → UI: Callback Response

The backend responds with instructions:

```json
{
  "type": "widget_callback_response",
  "widget_id": "MyNode.my_param",
  "code_payload": "// JavaScript code to execute\nwidget.value = 'option2';\nupdateOtherWidget();",
  "chain": true  // Whether to call base widget implementation
}
```

#### 4. UI Execution

The widget receives the response and:
1. Executes the `code_payload` JavaScript (if provided)
2. Calls the base implementation (if `chain` is true)
3. Updates the UI accordingly

## Supported Events

### Standard Events
- **onLoad**: Widget initialization (replaces default values)
- **onChange**: Value changed by user
- **onFocus**: Widget gains focus
- **onBlur**: Widget loses focus

### Custom Events
Widgets can define custom events specific to their functionality.

## Implementation Details

### Frontend Widget Structure

```typescript
// useDnneComboWidget.ts
const createDnneComboWidget = (node, inputSpec) => {
  const widget = node.addWidget('combo', inputSpec.name, null, 
    async (value) => {
      // Check if this event should trigger callback
      if (inputSpec.listen_to?.includes('onChange')) {
        // Collect all widget values for context
        const nodeData = {}
        if (node.widgets) {
          for (const widget of node.widgets) {
            nodeData[widget.name] = widget.value
          }
        }
        
        // Send callback message with full context
        api.socket.send(JSON.stringify({
          type: 'widget_callback',
          widget_id: inputSpec.widget_id,
          event: 'onChange',
          event_params: { 
            value, 
            oldValue: widget.lastValue,
            node_id: node.id,
            node_data: nodeData  // Complete node state (all widgets)
          }
        }))
        
        // Wait for and process response
        // ... handle response ...
      }
    },
    { values: inputSpec.options }
  )
  
  // Trigger onLoad event
  if (inputSpec.listen_to?.includes('onLoad')) {
    // Send initialization callback
    // ...
  }
  
  return widget
}
```

### Backend Handler Example

```python
class IsaacGymEnvsNode:
    @classmethod
    def handle_widget_callback(cls, message):
        """Handle widget callbacks for this node type"""
        widget_name = message["widget_id"].split(".")[1]
        event = message["event"]
        event_params = message["event_params"]
        
        if widget_name == "task" and event == "onChange":
            new_task = event_params["value"]
            
            # Generate JavaScript to update related widgets
            js_code = f"""
            // Update dependent widgets
            const node = app.graph.getNodeById({event_params['node_id']});
            updateDynamicWidgets(node, {json.dumps(get_task_config(new_task))});
            """
            
            return {
                "type": "widget_callback_response",
                "widget_id": message["widget_id"],
                "code_payload": js_code,
                "chain": True
            }
        
        elif widget_name.startswith("dynamic_"):
            # Access complete node state from node_data
            node_data = event_params.get("node_data", {})
            task = node_data.get("task")
            
            # Build complete selections from all widget values
            selections = {}
            for i in range(3):
                value = node_data.get(f"dynamic_{i+1}")
                if value and value != "none":
                    selections[f"level_{i}"] = value
            
            # Calculate updates based on complete state
            schema_text = cls.format_schema(task, selections)
            
            # Return JavaScript to update display
            return {
                "type": "widget_callback_response",
                "widget_id": message["widget_id"],
                "code_payload": f"updateSchema({json.dumps(schema_text)})",
                "chain": True
            }
```

## Security Considerations

### Code Execution
- **Sandboxing**: JavaScript execution should be sandboxed or use safe evaluation
- **Validation**: All code payloads should be validated before execution
- **Limits**: Impose limits on code payload size and execution time

### Message Validation
- Widget IDs must match registered widgets
- Events must be in the widget's `listen_to` list
- Node IDs must correspond to existing nodes

## Migration from Hardcoded System

### Old System (Hardcoded)
- `useDNNEComboWidget` contained IsaacGymEnvs-specific logic
- REST endpoint `/dnne/env_config/{task}` for configuration
- Frontend knew about specific node types

### New System (Generic)
- `useDnneComboWidget` is completely generic
- WebSocket messages for all callbacks
- Backend nodes handle their own logic

## Benefits

1. **Extensibility**: Any node can use callbacks without frontend changes
2. **Maintainability**: Node logic stays in node files
3. **Consistency**: Single communication pattern for all UI updates
4. **Performance**: WebSocket reuses connection, no HTTP overhead
5. **Flexibility**: Backend has full control over UI behavior

## Future Enhancements

1. **Batch Updates**: Single response updating multiple widgets
2. **Event Filtering**: Rate limiting and debouncing options
3. **State Sync**: Widget state synchronization across clients
4. **Type Safety**: TypeScript definitions for callback messages
5. **Widget Library**: Extend to other widget types (String, Int, Float)

## Related Documentation

- `/dnne_docs/for_claude/tasks/dnne_combo_widget/` - Implementation tracking
- `/dnne_docs/architecture/websocket-not-rest.md` - WebSocket architecture principles