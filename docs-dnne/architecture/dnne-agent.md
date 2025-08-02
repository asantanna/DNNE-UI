# DNNE Server/Client Architecture

## Overview

The DNNE Server/Client system enables remote workflow execution and real-time telemetry. It consists of:
- **dnne_server**: Persistent Windows service managing workflows and telemetry
- **dnne_client**: Linux/WSL client executing workflows and forwarding metrics
- **Dual-channel communication**: UDP for high-frequency telemetry, WebSocket for control

## Architecture

```
Windows:
├── dnne_server.py (Persistent service)
│   ├── Client WebSocket (port 8766)
│   ├── UI WebSocket (port 8767)
│   ├── Workflow management
│   ├── Telemetry aggregation
│   └── Log collection
│
├── DNNE-UI (main.py)
│   ├── Auto-starts dnne_server if needed
│   └── Connects to dnne_server:8767
│
WSL/Linux:
└── dnne_client.py
    ├── Connects to dnne_server:8766
    ├── UDP listener (localhost:9999)
    ├── WebSocket server (localhost:9998)
    └── Process management
```

## Communication Flow

### Telemetry (High-frequency, Fire-and-forget)
```
[Node] --UDP--> [dnne_client:9999] --WebSocket--> [dnne_server] --WebSocket--> [UI]
```

### Control (Reliable, Bidirectional)
```
[UI] <--WebSocket--> [dnne_server] <--WebSocket--> [dnne_client] <--WebSocket--> [Node]
```

## Core Components

### dnne_server.py

Persistent Windows service that:
- Manages client connections
- Handles workflow deployment
- Aggregates telemetry data
- Collects and stores logs
- Survives UI restarts

```python
class DNNEServer:
    def __init__(self):
        self.clients = {}         # Connected dnne_clients
        self.workflows = {}       # Active workflows
        self.ui_connections = set()  # Connected UIs
        self.telemetry_buffer = defaultdict(deque)
        
    async def start(self):
        # Client connections (WSL/Linux)
        await websockets.serve(self.handle_client, "0.0.0.0", 8766)
        
        # UI connections
        await websockets.serve(self.handle_ui, "0.0.0.0", 8767)
```

### dnne_client.py

Linux/WSL client that:
- Executes workflows locally
- Forwards telemetry from nodes
- Manages process lifecycle
- Provides control channel for nodes

```python
class DNNEClient:
    def __init__(self):
        self.server_url = "ws://localhost:8766"
        self.workflows = {}
        self.telemetry_buffer = []
        
    async def run(self):
        # Connect to server
        async with websockets.connect(self.server_url) as ws:
            # Start telemetry listener
            asyncio.create_task(self.telemetry_listener())
            
            # Start node control server
            asyncio.create_task(self.node_control_server())
            
            # Handle server messages
            await self.handle_messages(ws)
```

### Telemetry System

#### framework/telemetry.py
```python
class TelemetryClient:
    """Fire-and-forget UDP telemetry for nodes"""
    
    def __init__(self, host="localhost", port=9999):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self.target = (host, port)
        
    def report_metric(self, node_id: str, metric: str, value: float):
        """Send metric via UDP (non-blocking)"""
        try:
            packet = f"{node_id}|{metric}|{value}|{time.time()}"
            self.socket.sendto(packet.encode(), self.target)
        except:
            pass  # Fire-and-forget
```

## Implementation Phases

### Phase 1: Basic Infrastructure (Current)
- Single client support (WSL)
- Basic deploy/run/stop operations
- No authentication
- Manual client start

### Phase 2: Telemetry Integration
- UDP telemetry client in framework
- Telemetry forwarding in dnne_client
- Real-time UI visualization
- BalancingNode integration

### Phase 3: UI Polish
- Export button states
- Connection status display
- Workflow monitoring
- Log viewing

### Phase 4: Production Features (Future)
- Multiple client support
- Authentication (API keys)
- Cloud deployment
- Error recovery
- Persistent storage

## Developer Workflow

### Initial Setup
1. **Start DNNE-UI** (Windows): `python main.py`
   - Automatically starts dnne_server if not running
   - Shows "No Client Connected" initially

2. **Start Client** (WSL): `python dnne_client.py`
   - Connects to dnne_server
   - UI shows "Client Connected ✅"

3. **Use Normally**
   - Click "Export and Run" to deploy workflows
   - See real-time metrics on nodes
   - Click "Stop" to terminate

### Persistent Service
- dnne_server continues running when UI closes
- Workflows keep executing
- Logs are preserved
- Next UI start reconnects to existing server

## Protocol Messages

### Registration
```json
{
    "type": "register",
    "client_id": "wsl-1",
    "capabilities": {
        "hostname": "WSL",
        "gpu": true,
        "gpu_count": 1
    }
}
```

### Workflow Deployment
```json
{
    "type": "deploy",
    "workflow_id": "wf_123",
    "files": {
        "runner.py": "...",
        "nodes/__init__.py": "..."
    }
}
```

### Telemetry Batch
```json
{
    "type": "telemetry",
    "metrics": [
        {"node_id": "balancing_42", "metric": "throughput", "value": 87.3, "timestamp": 1234567890.123}
    ]
}
```

## Cloud Deployment (Future)

### Simple Installation
```bash
# 1. Setup conda environment
conda create -n DNNE_PY38 python=3.8
conda activate DNNE_PY38

# 2. Install dnne_client
git clone https://github.com/dnne/dnne-client
cd dnne-client
pip install -r requirements.txt

# 3. Configure
echo '{"server": "ws://YOUR_IP:8766"}' > dnne_config.json

# 4. Run
python dnne_client.py
```

### Benefits
- Easy GPU provider deployment
- No SSH required after setup
- Remote monitoring and control
- Automatic dependency management

## Security Considerations

### Current (Development)
- No authentication
- Local network only
- Trust-based model

### Future (Production)
- API key authentication
- TLS encryption
- IP whitelisting
- Resource isolation

## Performance Characteristics

### Telemetry Overhead
- UDP packets: ~100 bytes each
- 100 nodes @ 100Hz = 1MB/sec locally
- Batching reduces network traffic 100x
- Zero blocking on nodes

### Resource Usage
- dnne_server: ~50MB RAM, minimal CPU
- dnne_client: ~30MB RAM + workflow memory
- Network: ~10KB/sec with batching