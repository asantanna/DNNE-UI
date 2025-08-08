# IMPORTANT: WebSocket Architecture (NOT REST APIs)

## ⚠️ CRITICAL DESIGN PRINCIPLE ⚠️

**DNNE uses WebSocket for ALL client-server communication. DO NOT create REST API endpoints for features that should use WebSocket.**

## Why This Matters

Claude Code repeatedly suggests creating REST endpoints like:
- `/api/logs/{client_id}/{workflow_id}`
- `/api/logs/{client_id}`
- `/api/logs`

**THIS IS WRONG.** DNNE's architecture uses WebSocket messages for all dynamic data exchange.

## Correct Pattern

Instead of REST endpoints, use WebSocket messages:

### ❌ WRONG Approach (REST):
```python
@routes.get("/api/logs/{workflow_id}")
async def get_logs(request):
    # DO NOT DO THIS
    pass
```

### ✅ CORRECT Approach (WebSocket):
```python
# In the WebSocket handler:
if msg_type == 'request_logs':
    workflow_id = data.get('workflow_id')
    await self.send_workflow_history(ws, workflow_id)
```

## Existing WebSocket Messages

The system already handles these message types:
- `request_logs` - Request historical logs for a workflow
- `workflow_log` - Real-time log streaming
- `workflow_status` - Workflow state changes
- `client_status_update` - Client connection status

## How to Add New Features

1. **Define a new message type** in the WebSocket handler
2. **Send response via WebSocket** using `ws.send_json()` or `self.send_sync()`
3. **Frontend listens for the message** in the WebSocket client

## Examples in Codebase

- **Log retrieval**: See `send_workflow_history()` in server.py
- **Real-time logs**: See `_write_workflow_log()` broadcasting via `self.send_sync()`
- **Client status**: See `client_status_update` messages

## Remember

- **ALL UI updates go through WebSocket**
- **ALL data requests use WebSocket messages**
- **REST is ONLY for static resources and initial page load**
- **The frontend maintains a persistent WebSocket connection**

---
*This document exists to prevent Claude Code from repeatedly suggesting REST API endpoints when WebSocket should be used.*