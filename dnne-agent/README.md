# DNNE Agent

The DNNE Agent system enables remote workflow execution and real-time telemetry for DNNE.

## Components

- **dnne_server.py**: Persistent Windows service that manages workflows and telemetry
- **dnne_client.py**: Linux/WSL client that executes workflows and forwards metrics

## Quick Start (Development with WSL)

### 1. Install Dependencies

```bash
# In the dnne-agent directory
pip install -r requirements.txt
```

### 2. Start the Server (Windows)

The server is automatically started by the DNNE UI, but you can also run it manually:

```bash
# Windows terminal
cd dnne-agent
python dnne_server.py
```

The server will start on:
- Port 8766: Client connections
- Port 8767: UI connections

### 3. Start the Client (WSL)

```bash
# WSL terminal
cd dnne-agent
python dnne_client.py
```

The client will:
- Connect to the server on localhost:8766
- Start UDP telemetry listener on localhost:9999
- Wait for workflow deployments

## Architecture

```
Windows:
├── DNNE-UI (main.py)
│   └── Auto-starts dnne_server.py
│
├── dnne_server.py
│   ├── Client WebSocket (8766)
│   ├── UI WebSocket (8767)
│   └── Telemetry aggregation
│
WSL:
└── dnne_client.py
    ├── Connects to server
    ├── UDP listener (9999)
    └── Workflow execution
```

## Workflow Execution

1. UI exports workflow and sends to dnne_server
2. Server forwards to connected client
3. Client saves files and starts runner.py
4. Nodes send telemetry via UDP to client
5. Client batches and forwards to server
6. Server broadcasts to UI for visualization

## Configuration

### Environment Variables

```bash
# For dnne_client
export DNNE_SERVER_URL=ws://localhost:8766  # Default

# For telemetry (in exported workflows)
export DNNE_TELEMETRY_HOST=localhost  # Default
export DNNE_TELEMETRY_PORT=9999       # Default
export DNNE_TELEMETRY_DISABLED=0      # Set to 1 to disable
```

### Conda Environment

The client automatically activates the DNNE_PY38 conda environment if available:

```bash
# Expected location
~/miniconda3/envs/DNNE_PY38
```

## Development Mode

For development with WSL on the same machine:

1. No authentication required
2. Uses localhost connections
3. Single client support
4. Manual client start

## Logs

- **Server**: `dnne_server.log` (in dnne-agent directory)
- **Client**: Console output
- **Workflows**: Streamed to server and UI

## Troubleshooting

### Client Can't Connect
- Check server is running: `netstat -an | grep 8766`
- Check Windows firewall allows connections
- Try `telnet localhost 8766` from WSL

### Telemetry Not Working
- Check UDP port: `netstat -an | grep 9999`
- Verify nodes have telemetry.py in framework
- Check DNNE_TELEMETRY_DISABLED environment variable

### Workflow Won't Start
- Check conda environment: `conda info --envs`
- Verify runner.py exists in workspace
- Check process permissions

## Testing

The dnne-agent includes comprehensive test scripts for validating functionality.

### Test Scripts

#### test_dnne_server.py (Windows - Passive)
Connects to dnne_server as a UI client and displays activity:
```bash
# Basic usage - auto-starts dnne_server if needed
python test_dnne_server.py

# Options
python test_dnne_server.py --verbose      # Show detailed output
python test_dnne_server.py --quiet        # Minimal output
python test_dnne_server.py --timeout 300  # Exit after 5 minutes
```

#### test_dnne_client.py (Linux - Active Controller)
Controls and executes various test scenarios:

**Basic Operations:**
```bash
# Test connectivity
python test_dnne_client.py --test-connectivity

# Ensure agent is running
python test_dnne_client.py --ensure-agent

# Stop agent
python test_dnne_client.py --stop-agent
```

**Telemetry Tests:**
```bash
# Send single metric
python test_dnne_client.py --send-metric throughput:100
python test_dnne_client.py --send-metric latency:25.5

# Send burst of packets
python test_dnne_client.py --send-burst 1000

# Send continuous metrics
python test_dnne_client.py --send-continuous --duration 30
```

**Workflow Tests:**
```bash
# Run predefined test
python test_dnne_client.py --run hello_world
python test_dnne_client.py --run long_running
python test_dnne_client.py --run crash_test
python test_dnne_client.py --run telemetry_test

# Run test and wait for completion
python test_dnne_client.py --run hello_world --wait-complete

# Stop running workflow
python test_dnne_client.py --stop-workflow
```

### Predefined Tests

Located in `dnne-agent/tests/`:
- **hello_world.py**: Simple test that prints and exits
- **long_running.py**: Runs until stopped (for testing stop functionality)
- **crash_test.py**: Intentionally crashes (for error handling)
- **telemetry_test.py**: Sends various telemetry metrics

### Running Full Test Suite

1. **Start test server on Windows:**
   ```bash
   python test_dnne_server.py --verbose
   ```

2. **Run tests from Linux/WSL:**
   ```bash
   # Basic connectivity test
   python test_dnne_client.py --test-connectivity
   
   # Full test sequence
   python test_dnne_client.py --ensure-agent --test-connectivity
   python test_dnne_client.py --send-burst 100
   python test_dnne_client.py --run hello_world --wait-complete
   python test_dnne_client.py --run telemetry_test --wait-complete
   ```

### Exit Codes

test_dnne_client.py returns specific exit codes for test automation:
- 0: Success
- 1: Connection failed
- 2: Agent error
- 3: Deployment failed
- 4: Workflow error
- 5: Timeout

### Integration with dnne-test

Future integration:
```bash
# Run all agent tests
./dnne-test dnne-agent

# Run specific test category
./dnne-test dnne-agent connectivity
./dnne-test dnne-agent telemetry
./dnne-test dnne-agent workflow
```

## Future Features

- Multiple client support
- Authentication (API keys)
- Remote deployment (beyond localhost)
- Persistent workflow storage
- Automatic client discovery
- WebSocket control channel for nodes