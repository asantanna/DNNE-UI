# DNNE Telemetry Architecture

This document describes the telemetry system for collecting performance metrics and violations from exported DNNE workflows.

## Overview

The DNNE telemetry system provides lightweight, non-blocking performance monitoring for exported workflows. It uses a fire-and-forget UDP protocol to ensure telemetry never impacts node execution performance.

### Key Design Principles

1. **Fire-and-forget**: Nodes send telemetry via UDP with no acknowledgment required
2. **Minimal overhead**: Rate limiting at node level, aggregation at agent level
3. **Efficient storage**: Append-only files with simple formats
4. **Optional**: Telemetry is disabled by default, enabled via `--enable-telemetry` flag

## Architecture

### Data Flow

```
[Node] --UDP--> [Agent Client] --WebSocket--> [Agent Server] --WebSocket--> [DNNE Server] --File--> [Telemetry Files]
```

1. **Nodes** send telemetry packets via UDP to localhost:9999
2. **Agent Client** aggregates violations and forwards batches every 100ms
3. **Agent Server** relays telemetry_update messages to DNNE
4. **DNNE Server** writes telemetry to timestamped directories
5. **UI** can poll files for visualization (not real-time)

### Components

#### TelemetryClient (Node-side)
- Located in: `export_system/templates/framework/telemetry.py`
- Sends UDP packets to agent client
- Rate limits violations to 10 msgs/sec
- Supports custom grouping via `extra_args`

#### ViolationAggregator (Agent-side)
- Located in: `dnne-agent/dnne_agent_client.py`
- Groups violations by `node:type` or `node:type:extra_args`
- Forwards first 5 details, then summaries every 10 seconds
- Reduces network traffic through batching

#### Telemetry Storage (Server-side)
- Located in: `server.py::_write_telemetry_batch()`
- Creates timestamped telemetry directories
- Writes efficient file formats
- Closes files cleanly on workflow termination

## Message Types

### 1. Simple Metrics (Pipe-delimited)
Fast metrics use pipe-delimited format for efficiency:
```
metric_type|node_id|value|timestamp
```

Example:
```
throughput|10|150.5|1234567890.5
latency|10|2.3|1234567890.6
```

### 2. Violation Detail (JSON)
First 5 violations of each type are sent as details:
```json
{
  "type": "violation_detail",
  "node_id": "10",
  "violation_type": "frequency_below_minimum",
  "expected": 30.0,
  "actual": 25.5,
  "timestamp": 1234567890.5,
  "extra_args": "input_queue"  // Optional
}
```

### 3. Violation Summary (JSON)
After 5 details, violations are summarized every 10 seconds:
```json
{
  "type": "violation_summary",
  "interval_seconds": 10.0,
  "violations": [
    {
      "node_id": "10",
      "violation_type": "frequency_below_minimum",
      "count": 237,
      "expected": 30.0,
      "actual_range": [24.1, 26.8],
      "last_actual": 25.2,
      "extra_args": "input_queue"  // Optional
    }
  ]
}
```

### 4. Queue Depth (JSON)
Queue metrics for monitoring backpressure:
```json
{
  "type": "queue",
  "node_id": "10",
  "queue": "input_data",
  "depth": 5,
  "timestamp": 1234567890.5
}
```

## File Storage

### Directory Structure
```
remote_clients/
  {client_hostname}/
    {workflow_name}_wf_{id}/
      run_logs/
        2025-01-09_10-30-00.log     # Standard output
        metadata.json                # Deployment info
      telemetry/
        telem_2025-01-09_10-30-00/  # Timestamped run
          node_10.dat                # Metrics
          node_10_violations.log     # Violations
          node_11.dat
```

### File Formats

#### Metrics File (`node_{id}.dat`)
Simple pipe-delimited format for fast appends:
```
1234567890.5|throughput|150.5
1234567890.6|latency|2.3
1234567890.7|queue_input_queue|5
1234567890.8|queue_output_queue|3
1234567890.9|frequency|28.5
```

#### Violations File (`node_{id}_violations.log`)
Human-readable format with details and summaries:
```
2025-01-09T10:30:00.123 frequency_below_minimum exp=30.0 act=25.5
2025-01-09T10:30:00.234 frequency_below_minimum exp=30.0 act=24.8
2025-01-09T10:30:00.345 frequency_below_minimum[input_queue] exp=30.0 act=26.1
2025-01-09T10:30:10.000 SUMMARY frequency_below_minimum count=237 exp=30.0 range=[24.1,26.8] last=25.2
2025-01-09T10:30:10.001 SUMMARY frequency_below_minimum[input_queue] count=45 exp=30.0 range=[24.3,26.1] last=25.8
```

## API Reference

### TelemetryClient Methods

```python
# Create client (usually done by framework)
client = TelemetryClient(
    enabled=True,           # Enable telemetry
    host="localhost",       # UDP destination
    port=9999,             # UDP port
    violation_rate_limit=10 # Max violations/sec
)

# Report metrics
client.report_throughput(node_id, items_per_second)
client.report_latency(node_id, latency_ms)
client.report_queue_depth(node_id, queue_name, depth)
client.report_custom(node_id, metric_name, value)

# Report violations with optional grouping
client.report_violation(
    node_id="10",
    violation_type="frequency_below_minimum",
    expected=30.0,
    actual=25.5,
    extra_args="input_queue"  # Optional context
)
```

### Violation Grouping

Violations are grouped by:
- **Basic**: `node:type` (e.g., `"10:frequency_below_minimum"`)
- **With context**: `node:type:extra_args` (e.g., `"10:memory_exceeded:gpu_0"`)

This allows fine-grained tracking when needed while keeping the default simple.

### Rate Limiting

The `SimpleRateLimiter` class limits messages per second:
```python
limiter = SimpleRateLimiter(max_msgs_per_sec=10)
if limiter.should_send():
    # Send message
```

Rate limiting only applies to violations to prevent flooding during error conditions.

## Testing

### Prerequisites

1. **Start DNNE Server** (Windows):
   ```bash
   python main.py
   ```

2. **Start Agent Client** (WSL/Linux):
   ```bash
   cd dnne-agent
   python dnne_agent_client.py
   ```

### Test Scripts

#### 1. Simple Telemetry Test
Tests the telemetry client directly:
```bash
python test_telemetry_simple.py
```

This script:
- Sends various telemetry types
- Tests rate limiting
- Demonstrates `extra_args` usage

#### 2. Workflow Test
Run an exported workflow with telemetry:
```bash
cd /tmp/dnne_work_areas/{workflow_id}
python runner.py --enable-telemetry 10,11 --timeout 30s
```

Where `10,11` are node IDs to enable telemetry for.

### Verification

1. **Check Agent Client Logs**:
   ```bash
   tail -f dnne_logs/dnne_agent_client.log
   ```
   Look for: "Forwarding telemetry batch"

2. **Check DNNE Server Logs**:
   ```bash
   tail -f dnne_logs/DNNE.log
   ```
   Look for: "📊 Created telemetry directory"

3. **Examine Telemetry Files**:
   ```bash
   ls -la remote_clients/*/*/telemetry/telem_*/
   cat remote_clients/*/*/telemetry/telem_*/node_*_violations.log
   ```

## Examples

### Example 1: Basic Performance Monitoring
Enable telemetry for a balancing node:
```bash
python runner.py --enable-telemetry 10 --timeout 60s
```

Output in `node_10.dat`:
```
1234567890.5|frequency|28.5
1234567890.6|latency|2.3
1234567891.5|frequency|29.1
1234567891.6|latency|2.1
```

### Example 2: Violation Detection
With min_hz=30.0 configured:
```
2025-01-09T10:30:00.123 frequency_below_minimum exp=30.0 act=28.5
2025-01-09T10:30:00.234 frequency_below_minimum exp=30.0 act=27.9
2025-01-09T10:30:10.000 SUMMARY frequency_below_minimum count=89 exp=30.0 range=[27.5,29.8] last=28.2
```

### Example 3: Multi-Device Monitoring
Using `extra_args` for GPU-specific violations:
```python
telemetry.report_violation(node_id, "memory_exceeded", 8192, 9500, "gpu_0")
telemetry.report_violation(node_id, "memory_exceeded", 8192, 8800, "gpu_1")
```

Output groups violations separately:
```
2025-01-09T10:30:00.123 memory_exceeded[gpu_0] exp=8192 act=9500
2025-01-09T10:30:00.234 memory_exceeded[gpu_1] exp=8192 act=8800
2025-01-09T10:30:10.000 SUMMARY memory_exceeded[gpu_0] count=12 exp=8192 range=[9200,9500] last=9350
2025-01-09T10:30:10.001 SUMMARY memory_exceeded[gpu_1] count=5 exp=8192 range=[8700,8900] last=8750
```

## Configuration

### Enable Telemetry
Telemetry is disabled by default. Enable via command line:
```bash
# Enable for specific nodes
python runner.py --enable-telemetry 10,11

# Enable for all nodes
python runner.py --enable-telemetry all
```

### Configuration Options
In `dnne_config.json`:
```json
{
  "agent_client": {
    "telemetry_port": 9999,
    "telemetry_buffer_size": 1000,
    "telemetry_batch_interval": 0.1,
    "violation_summary_interval": 10.0
  }
}
```

## Performance Considerations

### Node Impact
- **UDP send**: ~0.01ms per message
- **Rate limiting check**: ~0.001ms
- **No blocking**: Fire-and-forget pattern

### Network Usage
- **Metrics**: ~50 bytes per message
- **Violations**: ~100-200 bytes per message
- **Batching**: 100ms intervals, up to 100 messages per batch

### Storage Usage
- **Metrics**: ~50 bytes per line
- **Violations**: ~100 bytes per line
- **Growth rate**: ~1MB per hour at 100 msgs/sec

## Limitations

1. **Remote only**: Telemetry requires agent client (not available for local exports)
2. **No real-time UI**: Files only, UI must poll
3. **Single workflow**: Agent client currently supports one active workflow
4. **Rate limiting**: Violations limited to 10/sec per node

## Future Enhancements

1. **UI Integration**: Real-time telemetry dashboard
2. **Alerting**: Threshold-based notifications
3. **Analytics**: Aggregated statistics and trends
4. **Compression**: Binary format for high-frequency metrics
5. **Multi-workflow**: Support concurrent workflow telemetry

## Troubleshooting

### No Telemetry Files Created
- Verify agent client is running
- Check `--enable-telemetry` flag includes correct node IDs
- Ensure workflow is marked as "running"
- Check DNNE logs for "📊 Created telemetry directory"

### Missing Violations
- Check rate limiting (max 10/sec)
- Verify violation thresholds are configured
- Look for summaries after first 5 details

### High CPU/Memory Usage
- Reduce telemetry frequency in node
- Increase batch interval in agent
- Disable telemetry for high-frequency nodes

## Related Documentation

- [DNNE Agent Architecture](dnne-agent.md) - Agent system overview
- [Queue Framework](queue_framework.md) - Node execution model
- [Export System](export_system.md) - Workflow export process