# DNNE Agent Integration Architecture

This document describes the integration between DNNE UI, DNNE Server, and the DNNE Agent Server for remote workflow deployment and execution.

## Performance

- **Telemetry batching**: 100ms intervals, ~1000 metrics/sec throughput
- **WebSocket reconnection**: Exponential backoff (1-16s), 5 retry attempts
- **Agent server overhead**: <50MB RAM, minimal CPU usage when idle

## Quick Setup

1. **Windows**: Run `dnne.bat` to start DNNE UI with agent server
2. **WSL2/Linux**: Run `python dnne-agent/dnne_agent_client.py` to connect
3. **Access from WSL2**: Use `http://172.22.160.1:8188` (adjust IP as needed)
4. **Ports**: 8188 (UI), 8766-8769 (agent), 9999 (telemetry UDP)

## Architecture Overview

### System Components

```
┌─────────────────┐
│   DNNE UI       │ (Vue.js Frontend)
│  (Browser)      │
└────────┬────────┘
         │ WebSocket
         │ (existing)
┌────────▼────────┐
│  DNNE Server    │ (main.py/server.py)
│  (Windows)      │ Port 8188
└────────┬────────┘
         │ WebSocket Client
         │ Port 8767 (UI connections)
┌────────▼────────────┐
│  DNNE Agent Server  │ (dnne_agent_server.py)
│    (Windows)        │ Ports: 8766 (clients), 8767 (UI)
└──────────┬──────────┘
           │ WebSocket
           │ Port 8766 (client connections)
┌──────────▼──────────┐
│  DNNE Agent Client  │ (dnne_agent_client.py)
│   (Linux/WSL)       │ Multiple instances possible
└─────────────────────┘
```
### Connection Establishment
- On startup, DNNE starts Agent Server if it isn't already started
- DNNE starts a websocket port for the browser UI (default: 8188)
- DNNE connects to the Agent Server's UI port (default: 8767)
- Agent Client connects to Agent Server's Client port (default: 8766)
- Exported code sends telemetry locally to Agent Client's UDP port (default: 9999)
- Some test programs impersonate the UI by connecting to Server Agent's Test port (default: 8768)

### Port Assignments
- **8188**: DNNE Server HTTP/WebSocket (UI connections)
- **8766**: Agent Server WebSocket (agent client connections)
- **8767**: Agent Server WebSocket (UI connections from DNNE)
- **8768**: Agent Server WebSocket (test connections - optional)
- **9999**: UDP telemetry port on each agent client

### Connection Types
1. **UI ↔ DNNE Server**: Existing WebSocket for UI operations
2. **DNNE Server → Agent Server**: New WebSocket client connection to UI port
3. **Agent Client → Agent Server**: Client WebSocket for workflow execution
4. **Exported Workflow → Agent Client**: UDP telemetry to local port 9999

## Message Protocol Reference

### 1. Server State Message
**Direction**: Agent Server → DNNE Server (on connection)
**Trigger**: When DNNE connects to agent server UI port
```json
{
  "type": "server_state",
  "clients": {
    "client_abc123": {
      "client_id": "client_abc123",
      "hostname": "wsl-machine",
      "platform": "Linux",
      "connected_at": "2025-08-02T10:30:00Z",
      "connected": true
    }
  },
  "workflows": {
    "wf_12345678": {
      "workflow_id": "wf_12345678",
      "client_id": "client_abc123",
      "status": "running",
      "deployed_at": "2025-08-02T10:35:00Z"
    }
  },
  "server_uptime": 3600.5
}
```
**DNNE Action**: Cache client list for UI requests

### 2. Client Connected
**Direction**: Agent Server → DNNE Server (broadcast)
**Trigger**: When new agent client connects
```json
{
  "type": "client_connected",
  "client_id": "client_def456",
  "info": {
    "hostname": "ubuntu-box",
    "platform": "Linux",
    "connected_at": "2025-08-02T10:40:00Z"
  }
}
```
**DNNE Action**: Update cached client list, notify UI

### 3. Client Disconnected
**Direction**: Agent Server → DNNE Server (broadcast)
**Trigger**: When agent client disconnects
```json
{
  "type": "client_disconnected",
  "client_id": "client_abc123",
  "reason": "Connection lost"
}
```
**DNNE Action**: Update cached client list, notify UI

### 4. Deploy Workflow
**Direction**: DNNE Server → Agent Server
**Trigger**: User clicks Export with remote client selected
```json
{
  "type": "deploy_workflow",
  "client_id": "client_abc123",
  "files": {
    "runner.py": "#!/usr/bin/env python3\\n# Generated workflow code...",
    "framework/queue_framework.py": "# Queue framework code...",
    "framework/dnne_config.py": "# Config reader...",
    "framework/dnne_config.json": "{\"dnne\": {...}}",
    "nodes/__init__.py": "# Node initialization..."
  }
}
```
**Response**: Workflow Deployed confirmation

### 5. Workflow Deployed
**Direction**: Agent Server → DNNE Server
**Trigger**: Response to deploy_workflow
```json
{
  "type": "workflow_deployed",
  "workflow_id": "wf_87654321",
  "client_id": "client_abc123"
}
```
**DNNE Action**: Store workflow ID, optionally send start command

### 6. Start Workflow
**Direction**: DNNE Server → Agent Server
**Trigger**: If "run after export" is checked
```json
{
  "type": "start_workflow",
  "workflow_id": "wf_87654321",
  "args": ["--epochs", "10", "--debug", "balancing"]
}
```
**Agent Action**: Forward to client, start runner.py

### 7. Workflow Status
**Direction**: Agent Server → DNNE Server (broadcast)
**Trigger**: Workflow state changes
```json
{
  "type": "workflow_status",
  "workflow_id": "wf_87654321",
  "status": "running",  // or "completed", "failed", "stopped"
  "details": {
    "pid": 12345,
    "start_time": 1234567890.5
  }
}
```
**DNNE Action**: Update UI with workflow status

### 8. Workflow Log
**Direction**: Agent Server → DNNE Server
**Trigger**: Workflow outputs log messages
```json
{
  "type": "workflow_log",
  "workflow_id": "wf_87654321",
  "log": {
    "timestamp": 1234567890.5,
    "level": "info",
    "message": "Training epoch 5/10 completed"
  }
}
```
**DNNE Action**: Forward to UI for display

### 9. Telemetry Update
**Direction**: Agent Server → DNNE Server (batched)
**Trigger**: Every 100ms if metrics available
```json
{
  "type": "telemetry_update",
  "batch": {
    "node_42": [
      {
        "timestamp": 1234567890.5,
        "metric": "throughput",
        "value": 150.5,
        "unit": "items/sec"
      }
    ]
  }
}
```
**DNNE Action**: Forward to UI for visualization

### 10. Error Message
**Direction**: Agent Server → DNNE Server
**Trigger**: Error conditions
```json
{
  "type": "error",
  "message": "Client client_xyz789 not connected"
}
```
**DNNE Action**: Display error to user

## Interaction Scenarios

### Scenario 1: DNNE Startup
1. **main.py starts**
2. **Check agent server**: Try connecting to localhost:8767
3. **If connection fails**:
   - Start `dnne_agent_server.py` as subprocess
   - Wait up to 5 seconds for it to start
   - Retry connection
4. **Connect to agent server** on port 8767
5. **Receive `server_state`** message
6. **Cache client list** for UI requests
7. **Start DNNE server** on port 8188

**Note**: When DNNE exits, the agent server is left running. This allows:
- Quick reconnection when DNNE restarts
- Multiple DNNE instances to share the same agent server
- Persistent client connections across DNNE restarts

### Scenario 2: UI Requests Client List
1. **UI loads** and connects to DNNE WebSocket
2. **UI requests** client list via new API: `GET /api/agent/clients`
3. **DNNE returns** cached client list:
```json
{
  "clients": [
    {"id": "local", "type": "local", "display": "Local"},
    {"id": "client_abc123", "type": "remote", "display": "wsl-machine", "hostname": "wsl-machine", "platform": "Linux"}
  ]
}
```
**Note**: "local" is a special reserved export_target that means export to DNNE server's filesystem
4. **UI populates** export dropdown

### Scenario 3: Export Workflow (Unified)
1. **User selects** target from dropdown (e.g., "Local" or "wsl-machine")
2. **User optionally checks** "Run after export" checkbox (future feature)
3. **User clicks** Export button
4. **UI sends** to DNNE (same format for both local and remote):
```json
{
  "prompt": {...workflow data...},
  "export_target": "local",  // or "client_abc123" for remote
  "run_after_export": false  // true for remote with run option
}
```
5. **DNNE exports** to local filesystem
6. **If target is "local"**:
   - DNNE returns success with local path
   - Process ends here
7. **If target is a client**:
   - DNNE reads all exported files into memory
   - DNNE sends `deploy_workflow` to agent server
8. **Agent server** forwards files to client
9. **Client saves** files and responds with success
10. **Agent server** sends `workflow_deployed` to DNNE
11. **DNNE sends** `start_workflow` (if run_after_export is true)
12. **Client starts** runner.py process
13. **Agent server** broadcasts `workflow_status: running`
14. **DNNE forwards** status to UI

### Scenario 4: Client Connects Mid-Session
1. **New client** connects to agent server
2. **Agent server** broadcasts `client_connected` to all UI connections
3. **DNNE receives** broadcast and updates cached client list
4. **DNNE notifies UI** via WebSocket:
```json
{
  "type": "agent_update",
  "action": "client_connected",
  "client": {
    "id": "client_ghi789",
    "hostname": "ubuntu-box",
    "platform": "Linux"
  }
}
```
5. **UI updates dropdown** dynamically to include new client

### Scenario 5: Client Disconnects
1. **Client** loses connection to agent server
2. **Agent server** detects disconnection
3. **Agent server** broadcasts `client_disconnected`
4. **DNNE** updates cached list, removes client
5. **DNNE notifies UI** of the disconnection
6. **UI removes client** from dropdown
7. **If user had that client selected**, UI defaults to "Local"

**Note**: Disconnected clients are removed from the list immediately. The system does not track disconnected clients.

### Scenario 6: Export Failure
1. **Export attempt** fails (e.g., client disconnected mid-transfer)
2. **Agent server** sends error to DNNE
3. **DNNE** logs error and returns to UI:
```json
{
  "success": false,
  "error": "Failed to deploy to client_abc123: Client disconnected",
  "fallback": "Workflow exported locally to: export_system/exports/MyWorkflow"
}
```
4. **UI shows** error message with local export path as fallback

## Implementation Details

### DNNE Server Modifications

#### 1. Add Agent Client to PromptServer
```python
class PromptServer:
    def __init__(self, loop):
        # ... existing code ...
        self.agent_connection = None
        self.agent_clients = {}
        self.agent_reconnect_task = None
        
    async def connect_to_agent_server(self):
        """Establish connection to agent server UI port"""
        # Implementation details in code
        
    async def handle_agent_message(self, message):
        """Process messages from agent server"""
        # Update client cache, forward to UI, etc.
```

#### 2. Client List Endpoint
```python
@routes.get("/api/agent/clients")
async def get_agent_clients(request):
    return web.json_response({
        "clients": [
            {"id": "local", "type": "local", "display": "Local"},
            *[{"id": cid, "type": "remote", "display": info.get("hostname"), **info} 
              for cid, info in self.agent_clients.items()]
        ]
    })
```

#### 3. Modified Export Endpoint
```python
@routes.post("/prompt")
async def post_prompt(request):
    json_data = await request.json()
    export_target = json_data.get("export_target", "local")
    run_after_export = json_data.get("run_after_export", False)
    
    # ... existing export code ...
    
    if export_target != "local":
        # Package and send to agent server
        await self.deploy_to_agent(export_target, exported_files, run_after_export)
```

### Agent Server Connection Management

#### Startup Sequence
1. Try connecting before starting subprocess
2. Use exponential backoff for reconnection
3. Maximum 5 reconnection attempts
4. Fall back to local-only mode if agent unavailable

#### Message Handling
- Use asyncio.Queue for incoming messages
- Process messages in order
- Update internal state before notifying UI
- Handle partial message reception

#### Error Recovery
- Automatic reconnection on connection loss
- Preserve client list during brief disconnections
- Clear error indication in UI when recovered

### Export Package Format

When sending to agent server, package structure:
```python
{
  "type": "deploy_workflow",
  "client_id": "client_abc123",
  "files": {
    "runner.py": "...",
    "framework/queue_framework.py": "...",
    "framework/dnne_config.py": "...",
    "framework/dnne_config.json": "...",
    "nodes/__init__.py": "...",
    "nodes/ml_nodes.py": "...",
    # ... all other exported files
  },
  "metadata": {
    "workflow_name": "MNIST_Classifier",
    "exported_at": "2025-08-02T11:00:00Z",
    "run_after_export": true
  }
}
```

## API Changes Summary

### New Endpoints
- `GET /api/agent/clients` - Get available export targets

### Modified Endpoints
- `POST /prompt` - Added `export_target` and `run_after_export` fields

### New WebSocket Messages (DNNE → UI)
- `agent_update` - Client list changes
- `workflow_status` - Remote workflow status
- `workflow_log` - Remote workflow output

## User Interface Design

### Toolbar Layout
```
[Export] [📍 Local ▼] [⏹️] [📊] [📋]
   |          |         |     |     |
   |          |         |     |     └─ Show logs (opens log viewer)
   |          |         |     └─ Dashboard (opens metrics/telemetry panel)
   |          |         └─ Stop button (enabled only for remote workflows)
   |          └─ Target selector dropdown (📍 for local, 🖥️ for remote clients)
   └─ Export action button
```

### Status Bar
```
Agent: 🟢 Connected | Clients: 2 (wsl-machine, ubuntu-box) | Active Workflows: 1
```

### Connection Status Indicators
- **🟢 Green**: Agent server connected + at least 1 client connected
- **🟡 Yellow**: Agent server connected but no clients
- **🔴 Red**: Agent server disconnected (temporary connection issue)
- **⚫ Gray**: Agent server disabled/not started

### Target Dropdown Behavior
- First item: "📍 Local" (always available)
- Remote clients: "🖥️ {hostname}" (e.g., "🖥️ wsl-machine")
- Updates dynamically as clients connect/disconnect
- Defaults to "Local" when selected client disconnects

### Log Viewer Features
- Filter by client or show all logs
- Color-coded by log level (info, warning, error)
- Auto-scroll with pause option
- Search functionality

### UI Requirements
- Update export dropdown on `agent_update` messages
- "Run after export" functionality embedded in export request
- Show remote workflow status/logs when available
- Handle dynamic client list updates
- Clear indication when only local export is available
- Stop button disabled for local exports