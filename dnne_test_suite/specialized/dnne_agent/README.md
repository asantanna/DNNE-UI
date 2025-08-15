# DNNE Agent Tests

This directory contains tests for the DNNE Agent system (dnne_agent_server.py and dnne_agent_client.py).

## Test Files

### Main Test Scripts
- `test_dnne_agent_client.py` - Tests for the Linux/WSL client that executes workflows
- `test_dnne_agent_server.py` - Tests for the Windows server that manages clients and workflows

### Helper Scripts (in helpers/)
- `hello_world.py` - Simple test workflow that prints and exits
- `crash_test.py` - Test workflow that crashes after a delay
- `long_running.py` - Test workflow that runs for a specified duration
- `telemetry_test.py` - Test workflow that sends UDP telemetry packets

## Running the Tests

### Prerequisites
1. **Windows Side**: Start the dnne_agent_server
2. **Linux Side**: Ensure DNNE_PY38 conda environment is activated
3. Update paths if needed in the test scripts

### Basic Usage

#### On Windows (Server Side)
First, start the DNNE agent server:
```bash
# In Windows terminal
cd dnne_agent
python dnne_agent_server.py
```

#### On Linux/WSL (Client Side)
With the server running on Windows, you can run tests:

**Automated Test** (recommended):
```bash
# From project root
./dnne_test agent
```

**Manual Client Test**:
```bash
cd dnne_test_suite/specialized/dnne_agent
python test_dnne_agent_client.py
```

**Server Monitor** (optional, runs on Linux to observe server activity):
```bash
cd dnne_test_suite/specialized/dnne_agent
python test_dnne_agent_server.py
```

### Important Notes
- The dnne_agent_server MUST be running on Windows before starting any tests
- All test commands shown above run on Linux/WSL
- The automated test (`./dnne_test agent`) will check if the server is reachable
- **WSL Networking**: The tests automatically detect the Windows host IP using `ip route`. If auto-detection fails or you need to specify a different IP:
  1. Find your Windows host IP manually: `ip route | grep default | awk '{print $3}'`
  2. Use `--server-host YOUR_IP` when running tests manually
  3. The default fallback IP is `172.22.160.1`

### Test Options

Both test scripts support various command-line options:
- `--server-host`: Server hostname (default: localhost)
- `--client-port`: Client WebSocket port (default: 8766)
- `--ui-port`: UI WebSocket port (default: 8767)
- Additional test-specific options (use `--help` to see all options)

## Test Scenarios

The test scripts simulate various scenarios:
- Client registration and capabilities reporting
- Workflow deployment and execution
- Telemetry forwarding
- Process lifecycle management
- Error handling and recovery
- Multi-workflow support

## Configuration

Tests use the exported_config.json for client configuration. Ensure the config files are properly set up in the dnne_agent directory.