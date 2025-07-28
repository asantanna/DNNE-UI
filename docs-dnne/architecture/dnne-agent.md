# DNNE Agent Architecture

## Overview

The DNNE Agent is a server process that runs on Linux machines (local, cloud, or cluster) to manage DNNE workflow execution. It provides a REST/WebSocket API for the UI to deploy, execute, monitor, and control workflows remotely.

## Motivation

Current workflow:
1. UI exports to local filesystem
2. User manually copies to Linux machine
3. User SSH's in to run workflows
4. No easy way to monitor or control

With DNNE Agent:
1. UI exports directly to agent
2. Agent manages execution lifecycle
3. Real-time monitoring and control
4. Cloud and cluster deployment ready

## Architecture

```
[DNNE UI] <--HTTPS/WSS--> [DNNE Agent] <--Local--> [Workflow Processes]
                                |
                                ├── File Management
                                ├── Process Management
                                ├── Resource Monitoring
                                ├── Log Aggregation
                                └── Telemetry Relay
```

## Core Features

### 1. Workflow Deployment

```python
# REST API endpoint
POST /api/workflows/deploy
{
    "workflow_name": "MNIST_Training",
    "files": {
        "runner.py": "...",
        "nodes/__init__.py": "...",
        "nodes/mnist_dataset_node_1.py": "...",
        # ... all exported files
    },
    "metadata": {
        "dnne_version": "1.0.0",
        "export_time": "2024-01-27T10:00:00Z",
        "node_count": 12
    }
}

Response:
{
    "workflow_id": "wf_abc123",
    "deployment_path": "/home/dnne/workflows/wf_abc123",
    "status": "deployed"
}
```

### 2. Workflow Execution

```python
# Start workflow
POST /api/workflows/{workflow_id}/start
{
    "args": [
        "--timeout", "300",
        "--save-checkpoint",
        "--telemetry"
    ],
    "env": {
        "CUDA_VISIBLE_DEVICES": "0"
    }
}

Response:
{
    "process_id": "proc_xyz789",
    "pid": 12345,
    "status": "running",
    "telemetry_port": 9999
}
```

### 3. Process Management

```python
# List running processes
GET /api/processes

# Get process details
GET /api/processes/{process_id}

# Stop process
POST /api/processes/{process_id}/stop
{
    "signal": "SIGTERM",  # or "SIGKILL"
    "timeout": 30  # seconds to wait before SIGKILL
}

# Get process logs
GET /api/processes/{process_id}/logs?lines=100&follow=true
```

### 4. Resource Monitoring

```python
# Get system resources
GET /api/system/resources
{
    "cpu": {
        "cores": 32,
        "usage_percent": 45.2,
        "per_core": [...]
    },
    "memory": {
        "total_gb": 128,
        "used_gb": 48.5,
        "available_gb": 79.5
    },
    "gpu": [
        {
            "id": 0,
            "name": "NVIDIA A100",
            "memory_total_gb": 40,
            "memory_used_gb": 12.3,
            "utilization_percent": 78
        }
    ],
    "disk": {
        "workspace_total_gb": 1000,
        "workspace_used_gb": 234
    }
}
```

## Agent Implementation

### Core Server

```python
# dnne_agent.py
from fastapi import FastAPI, WebSocket, UploadFile
from fastapi.responses import StreamingResponse
import asyncio
import subprocess
import psutil
import uuid
from pathlib import Path
from typing import Dict, List

app = FastAPI()

class WorkflowManager:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.workflows: Dict[str, WorkflowInfo] = {}
        self.processes: Dict[str, ProcessInfo] = {}
    
    async def deploy_workflow(self, name: str, files: Dict[str, str]) -> str:
        """Deploy workflow files to filesystem"""
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        deploy_path = self.base_path / workflow_id
        deploy_path.mkdir(parents=True)
        
        # Write all files
        for file_path, content in files.items():
            full_path = deploy_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # Make runner.py executable
        (deploy_path / "runner.py").chmod(0o755)
        
        # Store workflow info
        self.workflows[workflow_id] = WorkflowInfo(
            id=workflow_id,
            name=name,
            path=deploy_path,
            deployed_at=datetime.now()
        )
        
        return workflow_id
    
    async def start_workflow(self, workflow_id: str, args: List[str], 
                           env: Dict[str, str] = None) -> str:
        """Start workflow execution"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        process_id = f"proc_{uuid.uuid4().hex[:8]}"
        
        # Prepare command
        cmd = ["python", "runner.py"] + args
        
        # Start process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=workflow.path,
            env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Store process info
        self.processes[process_id] = ProcessInfo(
            id=process_id,
            workflow_id=workflow_id,
            process=process,
            pid=process.pid,
            started_at=datetime.now(),
            args=args
        )
        
        # Start log collection
        asyncio.create_task(self._collect_logs(process_id))
        
        return process_id
    
    async def stop_process(self, process_id: str, signal: str = "SIGTERM",
                          timeout: int = 30) -> bool:
        """Stop a running process"""
        proc_info = self.processes.get(process_id)
        if not proc_info:
            return False
        
        # Send signal
        if signal == "SIGTERM":
            proc_info.process.terminate()
        else:
            proc_info.process.kill()
        
        # Wait for termination
        try:
            await asyncio.wait_for(proc_info.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Force kill if not terminated
            proc_info.process.kill()
            await proc_info.process.wait()
        
        proc_info.stopped_at = datetime.now()
        return True

# Global manager
manager = WorkflowManager(Path("/home/dnne/workflows"))

@app.post("/api/workflows/deploy")
async def deploy_workflow(request: WorkflowDeployRequest):
    workflow_id = await manager.deploy_workflow(
        request.workflow_name,
        request.files
    )
    return {
        "workflow_id": workflow_id,
        "deployment_path": str(manager.workflows[workflow_id].path),
        "status": "deployed"
    }

@app.post("/api/workflows/{workflow_id}/start")
async def start_workflow(workflow_id: str, request: WorkflowStartRequest):
    process_id = await manager.start_workflow(
        workflow_id,
        request.args,
        request.env
    )
    return {
        "process_id": process_id,
        "pid": manager.processes[process_id].pid,
        "status": "running"
    }

@app.websocket("/ws/processes/{process_id}/logs")
async def stream_logs(websocket: WebSocket, process_id: str):
    """Stream logs via WebSocket"""
    await websocket.accept()
    
    proc_info = manager.processes.get(process_id)
    if not proc_info:
        await websocket.close(code=1008, reason="Process not found")
        return
    
    # Stream logs from circular buffer
    while True:
        if proc_info.new_logs:
            logs = proc_info.get_new_logs()
            await websocket.send_json({
                "type": "logs",
                "lines": logs
            })
        
        if proc_info.process.returncode is not None:
            # Process ended
            await websocket.send_json({
                "type": "exit",
                "code": proc_info.process.returncode
            })
            break
        
        await asyncio.sleep(0.1)
```

### Telemetry Relay

The agent can also relay telemetry from local workflows to remote UI:

```python
class TelemetryRelay:
    """Relay UDP telemetry to WebSocket clients"""
    
    def __init__(self, agent_port: int = 9999):
        self.agent_port = agent_port
        self.process_ports: Dict[str, int] = {}  # process_id -> telemetry_port
        self.ws_clients: Dict[str, Set[WebSocket]] = defaultdict(set)
    
    async def assign_telemetry_port(self, process_id: str) -> int:
        """Assign unique telemetry port for process"""
        port = 10000 + len(self.process_ports)
        self.process_ports[process_id] = port
        
        # Start relay for this port
        asyncio.create_task(self._relay_telemetry(process_id, port))
        
        return port
    
    async def _relay_telemetry(self, process_id: str, port: int):
        """Relay telemetry from UDP port to WebSocket clients"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('localhost', port))
        sock.setblocking(False)
        
        while process_id in self.process_ports:
            try:
                data, addr = await asyncio.get_event_loop().sock_recvfrom(sock, 1024)
                
                # Relay to all WebSocket clients watching this process
                for ws in self.ws_clients[process_id]:
                    try:
                        await ws.send_json({
                            "type": "telemetry",
                            "data": data.decode()
                        })
                    except:
                        pass
                        
            except Exception:
                await asyncio.sleep(0.001)
```

## Security Considerations

### Authentication & Authorization

```python
# API key authentication
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or not verify_key(api_key):
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    return await call_next(request)

# Or JWT tokens for more complex auth
```

### Process Isolation

```python
# Run workflows in containers or with restricted permissions
async def start_workflow_sandboxed(self, workflow_id: str, args: List[str]):
    cmd = [
        "docker", "run",
        "--rm",
        "--gpus", "all",
        "-v", f"{workflow.path}:/workspace",
        "-w", "/workspace",
        "dnne-runtime:latest",
        "python", "runner.py"
    ] + args
```

### Network Security

- HTTPS/WSS only for remote agents
- Firewall rules to restrict access
- VPN for cloud deployments

## Cloud Integration

### Multi-Agent Architecture

```
                    ┌─────────────┐
                    │   DNNE UI   │
                    └──────┬──────┘
                           │
                    ┌──────┴──────┐
                    │ Agent Proxy  │  (Load balancer)
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────┴────┐       ┌────┴────┐       ┌────┴────┐
   │ Agent 1 │       │ Agent 2 │       │ Agent 3 │
   │ (Local) │       │  (AWS)  │       │  (GCP)  │
   └─────────┘       └─────────┘       └─────────┘
```

### Agent Discovery

```python
# Agent registry service
class AgentRegistry:
    def register_agent(self, agent_info: AgentInfo):
        """Register new agent with capabilities"""
        
    def find_agents(self, requirements: Requirements) -> List[AgentInfo]:
        """Find agents matching requirements (GPU type, memory, etc)"""
        
    def get_agent_status(self, agent_id: str) -> AgentStatus:
        """Get current agent status and load"""
```

## UI Integration

### Export Dialog Enhancement

```javascript
// Export to agent instead of local file
async function exportToAgent() {
    const agent = await selectAgent();  // Show agent selection dialog
    
    const exportData = await prepareExport();
    
    const response = await fetch(`${agent.url}/api/workflows/deploy`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': agent.apiKey
        },
        body: JSON.stringify({
            workflow_name: currentWorkflow.name,
            files: exportData.files,
            metadata: exportData.metadata
        })
    });
    
    const result = await response.json();
    
    // Start workflow immediately if desired
    if (autoStart) {
        await startWorkflow(agent, result.workflow_id);
    }
}
```

### Process Management UI

- List of deployed workflows per agent
- Running processes with status
- Real-time logs viewer
- Resource usage graphs
- Start/stop/restart controls

## Benefits

1. **Remote Deployment**: No manual file copying
2. **Process Lifecycle**: Full control over execution
3. **Multi-Machine**: Distribute workflows across machines
4. **Cloud Ready**: Deploy to AWS, GCP, Azure instances
5. **Monitoring**: Centralized logs and metrics
6. **Scalability**: Add more agents as needed

## Deployment

### Quick Start Script

A single script to set up a DNNE Agent on any Linux machine:

```bash
#!/bin/bash
# install-dnne-agent.sh

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip git

# Install NVIDIA drivers and CUDA if needed
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA..."
    # Install CUDA toolkit
fi

# Create dnne user and directories
sudo useradd -m -s /bin/bash dnne
sudo mkdir -p /opt/dnne-agent
sudo chown dnne:dnne /opt/dnne-agent

# Install DNNE Agent
cd /opt/dnne-agent
sudo -u dnne git clone https://github.com/dnne/dnne-agent.git .
sudo -u dnne pip3 install -r requirements.txt

# Configure agent
sudo -u dnne python3 configure.py \
    --name "$(hostname)" \
    --port 8080 \
    --workspace /home/dnne/workflows

# Install as systemd service
sudo cp dnne-agent.service /etc/systemd/system/
sudo systemctl enable dnne-agent
sudo systemctl start dnne-agent

# Show connection info
echo "✅ DNNE Agent installed successfully!"
echo "📍 Agent URL: https://$(hostname -I | awk '{print $1}'):8080"
echo "🔑 API Key: $(cat /opt/dnne-agent/api_key.txt)"
```

### Cloud Provider Templates

#### AWS EC2 User Data Script
```bash
#!/bin/bash
# Runs automatically when EC2 instance starts

# Install DNNE Agent
curl -sSL https://dnne.ai/install-agent.sh | bash

# Configure for AWS
/opt/dnne-agent/configure.py \
    --cloud aws \
    --instance-id $(ec2-metadata --instance-id) \
    --gpu-type $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

# Register with DNNE Cloud Registry (optional)
/opt/dnne-agent/register.py \
    --registry https://registry.dnne.ai \
    --tags "aws,gpu,${AWS_REGION}"
```

#### Docker Deployment
```dockerfile
# Dockerfile.agent
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Install DNNE Agent
RUN pip3 install dnne-agent

# Configure
ENV DNNE_AGENT_PORT=8080
ENV DNNE_AGENT_WORKSPACE=/workspace

EXPOSE 8080
VOLUME /workspace

CMD ["dnne-agent", "serve"]
```

```bash
# Run with Docker
docker run -d \
    --gpus all \
    -p 8080:8080 \
    -v /home/ubuntu/dnne-workflows:/workspace \
    dnne/agent:latest
```

### Terraform Module for Multi-Cloud

```hcl
# terraform/dnne-agent/main.tf
module "dnne_agent" {
  source = "dnne/agent/cloud"
  
  providers = {
    aws   = var.deploy_to_aws
    gcp   = var.deploy_to_gcp
    azure = var.deploy_to_azure
  }
  
  instance_type = "gpu.large"  # Maps to provider-specific GPU instance
  regions       = ["us-west-2", "europe-west1", "eastus"]
  
  agent_config = {
    version = "latest"
    port    = 8080
    auth    = "api_key"
  }
  
  scaling = {
    min_instances = 1
    max_instances = 10
    scale_on      = "queue_depth"
  }
}

output "agent_endpoints" {
  value = module.dnne_agent.endpoints
}
```

### UI Connection

Once an agent is running, connecting from DNNE UI is simple:

```javascript
// In UI settings
async function addAgent() {
    const agent = {
        name: "AWS GPU Server",
        url: "https://ec2-54-123-456-789.compute-1.amazonaws.com:8080",
        apiKey: "dnne_agent_key_abc123..."
    };
    
    // Test connection
    const info = await testAgentConnection(agent);
    console.log(`Connected to ${info.name} with ${info.gpu_count} GPUs`);
    
    // Save to config
    saveAgentConfig(agent);
}
```

### Zero-Config Discovery

For local network agents:

```python
# Agent broadcasts presence on local network
class AgentBroadcaster:
    def __init__(self):
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    async def broadcast_presence(self):
        while True:
            message = json.dumps({
                "service": "dnne-agent",
                "version": "1.0.0",
                "hostname": socket.gethostname(),
                "port": 8080,
                "capabilities": {
                    "gpu": torch.cuda.is_available(),
                    "gpu_count": torch.cuda.device_count(),
                    "memory_gb": psutil.virtual_memory().total / 1e9
                }
            })
            
            self.broadcast_socket.sendto(
                message.encode(),
                ('<broadcast>', 9988)
            )
            
            await asyncio.sleep(5)  # Broadcast every 5 seconds
```

```javascript
// UI auto-discovers local agents
async function discoverLocalAgents() {
    const agents = await listenForBroadcasts(5000);  // Listen for 5 seconds
    
    console.log(`Found ${agents.length} local DNNE agents`);
    agents.forEach(agent => {
        console.log(`  - ${agent.hostname}: ${agent.capabilities.gpu_count} GPUs`);
    });
}
```

## Implementation Phases

### Phase 1: Local Agent
- Basic REST API
- Local process management
- Simple authentication
- One-line installer script

### Phase 2: Remote Deployment
- HTTPS/WSS support
- Cloud provider templates
- Docker images
- UI integration

### Phase 3: Cloud Features
- Multi-agent support
- Agent discovery
- Load balancing
- Auto-scaling

### Phase 4: Advanced Features
- Container isolation
- Resource limits
- Scheduling
- Kubernetes operator