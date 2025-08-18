# UI Proxy Architecture

*Last Updated: 2025-08-18*

## Overview

The UI Proxy is a mechanism for the DNNE server to execute arbitrary JavaScript on the UI side and receive responses. It enables asynchronous modification of UI state based on backend logic, allowing nodes to dynamically update their visual representation without user interaction.

## Primary Use Case: Asynchronous UI State Modification

The key innovation is allowing backend nodes to update their UI representation based on configuration changes or processing results. For example:

- **Split Node**: When configured with `"theta_pos[0:2], theta_pos[2:4]"`, the node can update its output port labels from generic "output_a, output_b" to meaningful "theta_pos[0:2], theta_pos[2:4]"
- **Network Node**: Dynamically display the layers it contains as they're added
- **Training Nodes**: Update progress indicators during training
- **Data Nodes**: Show data shape/statistics after loading

## Design Principles

1. **Server-Initiated**: Unlike UI callbacks (which are UI-initiated), UI proxy is server-initiated
2. **Asynchronous**: Updates can happen at any time, not just in response to user actions
3. **WebSocket-Based**: Follows DNNE's WebSocket-only communication pattern
4. **Node-Scoped**: JavaScript executes in the context of a specific node
5. **Response-Capable**: Can return data back to the server after execution

## Relationship to UI Callbacks

UI Proxy and UI Callbacks form a symmetric bidirectional communication system:

| Feature | UI Callbacks | UI Proxy |
|---------|-------------|----------|
| **Initiator** | UI (user action) | Server (backend logic) |
| **Trigger** | Widget events (onChange, onLoad) | Backend processing/configuration |
| **Direction** | UI → Server → UI | Server → UI → Server |
| **Purpose** | Handle user interactions | Update UI state asynchronously |
| **JavaScript** | Response contains JS to execute | Request contains JS to execute |

Together, they provide complete bidirectional control over the UI from the backend.

## Protocol Specification

### Server → UI: Execute JavaScript

```json
{
  "type": "ui_proxy_execute",
  "node_id": 42,
  "request_id": "req_123",
  "javascript": "// JavaScript code to execute\nconst node = this;\nnode.widgets[0].label = 'theta_pos[0:2]';\nreturn { success: true, oldLabel: node.widgets[0].label };",
  "timeout": 5000  // Optional, milliseconds
}
```

**Fields:**
- `type`: Always "ui_proxy_execute"
- `node_id`: The node in whose context to execute
- `request_id`: Unique ID for matching responses
- `javascript`: JavaScript code to execute
- `timeout`: Optional execution timeout

### UI Execution Context

The JavaScript executes with the following context:
```javascript
{
  this: node,           // The target node object
  node: node,          // Alias for this
  app: app,            // The ComfyUI app instance
  graph: app.graph,    // The workflow graph
  widgets: node.widgets, // Node's widgets array
  
  // Helper functions
  updateWidget: (name, props) => { /* ... */ },
  findWidget: (name) => { /* ... */ },
  
  // Response object (writable)
  response: {}
}
```

### UI → Server: Execution Response

```json
{
  "type": "ui_proxy_response",
  "request_id": "req_123",
  "node_id": 42,
  "success": true,
  "result": {
    "success": true,
    "oldLabel": "output_a"
  },
  "error": null
}
```

**Fields:**
- `type`: Always "ui_proxy_response"
- `request_id`: Matching request ID
- `node_id`: The node that was modified
- `success`: Whether execution succeeded
- `result`: Return value from JavaScript execution
- `error`: Error message if execution failed

## Implementation Examples

### Server-Side: Split Node Updates Output Labels

```python
class SplitNode:
    def update_output_labels(self, split_config):
        """Update UI output labels based on split configuration"""
        
        # Parse split configuration to get output names
        outputs = self.parse_split_config(split_config)
        
        # Generate JavaScript to update labels
        js_code = """
        const node = this;
        const outputs = %s;
        
        // Update output slot labels
        if (node.outputs) {
            outputs.forEach((label, index) => {
                if (node.outputs[index]) {
                    node.outputs[index].label = label;
                    node.outputs[index].name = label;
                }
            });
        }
        
        // Update any output widgets
        node.widgets.forEach(widget => {
            if (widget.name.startsWith('output_')) {
                const index = parseInt(widget.name.split('_')[1]) - 1;
                if (outputs[index]) {
                    widget.label = outputs[index];
                }
            }
        });
        
        // Trigger graph update
        app.graph.setDirtyCanvas(true);
        
        return { updated: outputs.length, labels: outputs };
        """ % json.dumps(outputs)
        
        # Send UI proxy request
        request_id = f"split_{self.node_id}_{time.time()}"
        await self.send_ui_proxy(
            node_id=self.node_id,
            javascript=js_code,
            request_id=request_id
        )
```

### Frontend-Side: Message Handler

```javascript
// In api.js or WebSocket handler
socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'ui_proxy_execute') {
        handleUIProxyExecute(message);
    }
});

async function handleUIProxyExecute(message) {
    const { node_id, request_id, javascript, timeout = 5000 } = message;
    
    try {
        // Find the target node
        const node = app.graph.getNodeById(node_id);
        if (!node) {
            throw new Error(`Node ${node_id} not found`);
        }
        
        // Create execution context
        const context = {
            node,
            app,
            graph: app.graph,
            widgets: node.widgets || [],
            
            // Helper functions
            updateWidget(name, props) {
                const widget = node.widgets?.find(w => w.name === name);
                if (widget) {
                    Object.assign(widget, props);
                }
                return widget;
            },
            
            findWidget(name) {
                return node.widgets?.find(w => w.name === name);
            }
        };
        
        // Execute with timeout
        const result = await executeWithTimeout(
            javascript,
            context,
            timeout
        );
        
        // Send response
        socket.send(JSON.stringify({
            type: 'ui_proxy_response',
            request_id,
            node_id,
            success: true,
            result,
            error: null
        }));
        
    } catch (error) {
        // Send error response
        socket.send(JSON.stringify({
            type: 'ui_proxy_response',
            request_id,
            node_id,
            success: false,
            result: null,
            error: error.message
        }));
    }
}

async function executeWithTimeout(code, context, timeout) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            reject(new Error('Execution timeout'));
        }, timeout);
        
        try {
            // Create function with context
            const func = new Function(
                ...Object.keys(context),
                code
            );
            
            // Execute with context values
            const result = func.call(
                context.node,  // 'this' binding
                ...Object.values(context)
            );
            
            clearTimeout(timer);
            resolve(result);
        } catch (error) {
            clearTimeout(timer);
            reject(error);
        }
    });
}
```

## Use Cases

### 1. Dynamic Label Updates (Split Node)
```javascript
// Update output labels based on split configuration
node.outputs.forEach((output, i) => {
    output.label = splitNames[i];
});
app.graph.setDirtyCanvas(true);
```

### 2. Progress Indicators (Training Nodes)
```javascript
// Update training progress
const widget = findWidget('progress');
widget.value = `Epoch ${epoch}/${totalEpochs}`;
widget.color = epoch === totalEpochs ? '#00ff00' : '#ffff00';
```

### 3. Dynamic Widget Creation
```javascript
// Add widgets based on configuration
const numOutputs = config.outputs;
for (let i = node.widgets.length; i < numOutputs; i++) {
    node.addWidget('text', `output_${i}`, '');
}
```

### 4. State Visualization (Data Nodes)
```javascript
// Show data statistics
updateWidget('shape', { value: `Shape: ${shape}` });
updateWidget('dtype', { value: `Type: ${dtype}` });
updateWidget('stats', { value: `Mean: ${mean.toFixed(2)}` });
```

## Security Considerations

### Execution Sandboxing
- JavaScript runs in the browser's JavaScript engine (inherently sandboxed)
- No access to filesystem, network, or system resources
- Limited to DOM and ComfyUI app context

### Validation
- Verify node_id exists before execution
- Validate request_id format
- Check message size limits
- Implement rate limiting per client

### Timeout Protection
- Default 5-second timeout for execution
- Prevents infinite loops or hanging code
- Server can specify custom timeout per request

### Code Injection Prevention
- JavaScript is generated server-side by trusted code
- No user-provided code execution
- Parameters are JSON-encoded to prevent injection

## Performance Considerations

### Batching Updates
Multiple UI updates can be batched in a single JavaScript execution:
```javascript
// Update multiple widgets at once
['widget1', 'widget2', 'widget3'].forEach(name => {
    updateWidget(name, { value: newValues[name] });
});
```

### Throttling
For high-frequency updates (e.g., training progress):
- Server should throttle updates (e.g., max 1 per second)
- Use request deduplication for identical updates

### Graph Redraws
Minimize canvas redraws:
```javascript
// Batch all changes, then single redraw
makeAllChanges();
app.graph.setDirtyCanvas(true);  // Only once
```

## Error Handling

### Server-Side
```python
try:
    response = await self.send_ui_proxy(node_id, js_code)
    if not response['success']:
        logger.error(f"UI proxy failed: {response['error']}")
        # Fallback behavior
except TimeoutError:
    logger.error("UI proxy request timed out")
    # Handle timeout
```

### Client-Side
```javascript
try {
    const result = executeCode(javascript);
    sendResponse({ success: true, result });
} catch (error) {
    console.error('UI proxy execution failed:', error);
    sendResponse({ success: false, error: error.message });
}
```

## Testing Approach

### Unit Tests
- Mock WebSocket messages
- Test JavaScript execution in isolated context
- Verify error handling

### Integration Tests
- Test full round-trip: server → UI → server
- Verify node state changes
- Test timeout handling

### Example Test
```python
async def test_split_node_label_update():
    # Create split node
    node = SplitNode(node_id=42)
    
    # Configure split
    node.configure("theta_pos[0:2], theta_pos[2:4]")
    
    # Verify UI proxy message sent
    assert mock_websocket.sent_message['type'] == 'ui_proxy_execute'
    assert 'theta_pos[0:2]' in mock_websocket.sent_message['javascript']
    
    # Simulate response
    await mock_websocket.receive({
        'type': 'ui_proxy_response',
        'success': True,
        'result': {'updated': 2, 'labels': ['theta_pos[0:2]', 'theta_pos[2:4]']}
    })
    
    # Verify node state
    assert node.ui_labels == ['theta_pos[0:2]', 'theta_pos[2:4]']
```

## Future Enhancements

1. **Broadcast Mode**: Send UI updates to all connected clients
2. **Persistent State**: Store UI modifications for reconnecting clients
3. **Undo/Redo**: Track UI state changes for undo functionality
4. **Animation Support**: Smooth transitions for UI updates
5. **Widget Templates**: Predefined UI modification patterns
6. **Debug Mode**: Log all UI proxy executions for debugging

## Related Documentation

- `/dnne_docs/architecture/ui_callbacks.md` - UI-initiated callback system
- `/dnne_docs/architecture/websocket-not-rest.md` - WebSocket architecture
- `/dnne_docs/for_claude/tasks/ui_proxy/` - Implementation tracking