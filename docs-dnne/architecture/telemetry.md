# Telemetry Architecture

## Overview

DNNE telemetry provides real-time performance metrics from exported workflows back to the UI. This enables visualization of throughput, latency, queue depths, and other metrics without affecting the performance of the running system.

## Design Goals

1. **Minimal Overhead**: Telemetry should not significantly impact the measured system
2. **Fire-and-Forget**: Exported code should never block on telemetry
3. **Graceful Degradation**: System continues running even if UI disconnects
4. **High Frequency**: Support 100+ Hz reporting from multiple nodes
5. **Simple Integration**: Easy to add telemetry to any node

## Architecture

### Three-Layer Design

```
[Exported Nodes] --UDP--> [Telemetry Aggregator] --WebSocket--> [UI]
```

1. **Exported Nodes**: Send lightweight UDP packets
2. **Telemetry Aggregator**: Receives UDP, buffers, aggregates
3. **UI**: Receives batched updates via WebSocket

### Why This Design?

- **UDP for nodes**: Non-blocking, minimal overhead
- **Aggregator**: Handles buffering, batching, protocol conversion
- **WebSocket to UI**: Reuses existing infrastructure, handles backpressure

## Telemetry Client (Exported Code)

### Minimal Implementation

```python
# framework/telemetry.py
import socket
import time
from typing import Optional

class TelemetryClient:
    """Lightweight telemetry client for exported nodes"""
    
    def __init__(self, enabled: bool = True, host: str = "localhost", port: int = 9999):
        self.enabled = enabled
        if enabled:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.setblocking(False)  # Never block
                self.target = (host, port)
            except:
                self.enabled = False  # Disable if can't create socket
    
    def report_throughput(self, node_id: str, items_per_second: float):
        """Report node throughput in items/second"""
        if not self.enabled:
            return
        self._send(f"THR|{node_id}|{items_per_second:.2f}")
    
    def report_latency(self, node_id: str, latency_ms: float):
        """Report node processing latency in milliseconds"""
        if not self.enabled:
            return
        self._send(f"LAT|{node_id}|{latency_ms:.3f}")
    
    def report_queue_depth(self, node_id: str, queue_name: str, depth: int):
        """Report queue depth for a node input/output"""
        if not self.enabled:
            return
        self._send(f"QUE|{node_id}|{queue_name}|{depth}")
    
    def report_custom(self, node_id: str, metric_name: str, value: float):
        """Report custom metric"""
        if not self.enabled:
            return
        self._send(f"CUS|{node_id}|{metric_name}|{value:.3f}")
    
    def _send(self, message: str):
        """Send UDP packet, ignore all errors"""
        if not self.enabled:
            return
        try:
            packet = f"{message}|{time.time():.3f}".encode()
            self.socket.sendto(packet, self.target)
        except:
            pass  # Truly fire-and-forget

# Global instance
telemetry = TelemetryClient()
```

### Integration in Nodes

```python
# In QueueNode base class
class QueueNode:
    async def run(self):
        while self.running:
            start_time = time.perf_counter()
            
            # Regular node execution
            inputs = await self._gather_inputs()
            outputs = await self.compute(**inputs)
            
            # Report metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            telemetry.report_latency(self.node_id, latency_ms)
            
            # Report queue depths periodically
            if self.iteration % 10 == 0:
                for name, queue in self.input_queues.items():
                    telemetry.report_queue_depth(self.node_id, f"input_{name}", queue.qsize())
```

### Command Line Integration

```python
# In runner.py
parser.add_argument('--telemetry', action='store_true',
                    help='Enable telemetry reporting to UI')
parser.add_argument('--telemetry-host', type=str, default='localhost',
                    help='Telemetry aggregator host')
parser.add_argument('--telemetry-port', type=int, default=9999,
                    help='Telemetry aggregator port')

# Initialize telemetry
from framework.telemetry import telemetry
telemetry.enabled = args.telemetry
if args.telemetry:
    telemetry.target = (args.telemetry_host, args.telemetry_port)
```

## Telemetry Aggregator

### Server-Side Component

```python
# telemetry_aggregator.py
import asyncio
import socket
import time
import json
from collections import defaultdict, deque
from typing import Dict, List, Deque

class TelemetryAggregator:
    """Receives UDP telemetry and aggregates for WebSocket delivery"""
    
    def __init__(self, udp_port: int = 9999, buffer_time: float = 0.1):
        self.udp_port = udp_port
        self.buffer_time = buffer_time  # Batch window in seconds
        
        # Metrics storage
        self.metrics_buffer: Dict[str, Deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_flush = time.time()
        
        # WebSocket connections
        self.ws_clients = set()
    
    async def start(self):
        """Start UDP receiver and WebSocket broadcaster"""
        # Create UDP socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setblocking(False)
        self.udp_socket.bind(('localhost', self.udp_port))
        
        # Start tasks
        await asyncio.gather(
            self.udp_receiver(),
            self.websocket_broadcaster()
        )
    
    async def udp_receiver(self):
        """Receive and parse UDP packets"""
        while True:
            try:
                data, addr = await asyncio.get_event_loop().sock_recvfrom(
                    self.udp_socket, 1024
                )
                self.parse_packet(data.decode())
            except Exception as e:
                await asyncio.sleep(0.001)  # Brief pause on error
    
    def parse_packet(self, packet: str):
        """Parse telemetry packet and store metric"""
        try:
            parts = packet.split('|')
            metric_type = parts[0]
            
            if metric_type == 'THR':  # Throughput
                _, node_id, value, timestamp = parts
                self.store_metric(node_id, 'throughput', float(value), float(timestamp))
                
            elif metric_type == 'LAT':  # Latency
                _, node_id, value, timestamp = parts
                self.store_metric(node_id, 'latency', float(value), float(timestamp))
                
            elif metric_type == 'QUE':  # Queue depth
                _, node_id, queue_name, depth, timestamp = parts
                self.store_metric(node_id, f'queue_{queue_name}', int(depth), float(timestamp))
                
            elif metric_type == 'CUS':  # Custom
                _, node_id, metric_name, value, timestamp = parts
                self.store_metric(node_id, metric_name, float(value), float(timestamp))
                
        except Exception as e:
            pass  # Ignore malformed packets
    
    def store_metric(self, node_id: str, metric_name: str, value: float, timestamp: float):
        """Store metric in buffer"""
        key = f"{node_id}:{metric_name}"
        self.metrics_buffer[key].append({
            'value': value,
            'timestamp': timestamp
        })
    
    async def websocket_broadcaster(self):
        """Periodically send aggregated metrics to WebSocket clients"""
        while True:
            await asyncio.sleep(self.buffer_time)
            
            if self.ws_clients and self.metrics_buffer:
                # Prepare batch
                batch = self.prepare_batch()
                
                # Broadcast to all clients
                message = json.dumps({
                    'type': 'telemetry_batch',
                    'data': batch,
                    'timestamp': time.time()
                })
                
                # Send to all connected clients
                disconnected = set()
                for ws in self.ws_clients:
                    try:
                        await ws.send(message)
                    except:
                        disconnected.add(ws)
                
                # Remove disconnected clients
                self.ws_clients -= disconnected
    
    def prepare_batch(self) -> Dict:
        """Prepare metrics batch for transmission"""
        batch = {}
        
        for key, metrics in self.metrics_buffer.items():
            if not metrics:
                continue
                
            node_id, metric_name = key.split(':', 1)
            
            if node_id not in batch:
                batch[node_id] = {}
            
            # Calculate summary statistics
            values = [m['value'] for m in metrics]
            batch[node_id][metric_name] = {
                'latest': values[-1],
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values),
                'history': list(metrics)[-20:]  # Last 20 points for graphing
            }
        
        # Clear old metrics
        for key in self.metrics_buffer:
            self.metrics_buffer[key].clear()
        
        return batch
    
    def add_websocket_client(self, ws):
        """Register new WebSocket client"""
        self.ws_clients.add(ws)
    
    def remove_websocket_client(self, ws):
        """Unregister WebSocket client"""
        self.ws_clients.discard(ws)
```

## UI Integration

### WebSocket Handler

```javascript
// In UI telemetry handler
class TelemetryHandler {
    constructor() {
        this.metrics = new Map();  // node_id -> metrics
        this.charts = new Map();   // node_id -> chart instance
    }
    
    handleTelemetryBatch(data) {
        // Update metrics for each node
        for (const [nodeId, nodeMetrics] of Object.entries(data)) {
            this.updateNodeMetrics(nodeId, nodeMetrics);
            this.updateNodeDisplay(nodeId, nodeMetrics);
        }
    }
    
    updateNodeDisplay(nodeId, metrics) {
        // Update node's visual display
        const node = this.getNodeById(nodeId);
        if (!node) return;
        
        // Show throughput on node
        if (metrics.throughput) {
            node.setThroughputDisplay(`${metrics.throughput.latest.toFixed(1)} Hz`);
            
            // Color code based on targets
            if (node.hasBalancingConfig()) {
                const config = node.getBalancingConfig();
                if (metrics.throughput.latest < config.min_hz) {
                    node.setStatusColor('red');  // Below minimum
                } else if (metrics.throughput.latest > config.max_hz) {
                    node.setStatusColor('yellow');  // Above maximum
                } else {
                    node.setStatusColor('green');  // Within range
                }
            }
        }
        
        // Update mini-chart if visible
        if (metrics.throughput && metrics.throughput.history) {
            this.updateMiniChart(nodeId, metrics.throughput.history);
        }
    }
}
```

### Visual Elements

1. **On-Node Display**
   - Current throughput (e.g., "87.3 Hz")
   - Status indicator (green/yellow/red)
   - Mini sparkline chart

2. **Detailed View** (on node selection)
   - Historical charts
   - Latency histogram
   - Queue depth over time
   - Statistical summary

3. **Global Dashboard**
   - System-wide metrics
   - Balancing violations
   - Resource utilization

## Packet Format

### UDP Packet Structure

```
TYPE|node_id|...values...|timestamp

Examples:
THR|node_42|156.7|1706371234.567      # Throughput
LAT|node_42|12.345|1706371234.568     # Latency
QUE|node_42|input_data|5|1706371234.569  # Queue depth
CUS|node_42|accuracy|0.923|1706371234.570  # Custom metric
```

- **TYPE**: 3-letter metric type code
- **Pipe-delimited**: Simple to parse
- **Timestamp**: Unix time with milliseconds
- **Small packets**: Typically under 100 bytes

## Performance Considerations

### Overhead Analysis

```python
# Worst case: 100 nodes reporting at 100 Hz
packets_per_second = 100 * 100  # 10,000 packets/sec
bytes_per_packet = 100  # Conservative estimate
bandwidth = 10_000 * 100  # 1 MB/sec

# This is negligible on modern networks
```

### Optimization Strategies

1. **Sampling**: Don't report every iteration
   ```python
   if self.iteration % 10 == 0:  # Report every 10th iteration
       telemetry.report_throughput(self.node_id, self.current_rate)
   ```

2. **Rate Limiting**: Built into aggregator
   ```python
   # Aggregator only sends to UI every 100ms regardless of input rate
   ```

3. **Local Aggregation**: Nodes could aggregate before sending
   ```python
   # Report average of last N iterations instead of each one
   ```

## Configuration

### Environment Variables

```bash
# Disable telemetry completely
DNNE_TELEMETRY_DISABLED=1

# Change aggregator location
DNNE_TELEMETRY_HOST=192.168.1.100
DNNE_TELEMETRY_PORT=9999

# Debug mode - log all packets
DNNE_TELEMETRY_DEBUG=1
```

### Per-Node Control

```python
# Nodes can have telemetry configuration
class NetworkNode:
    def __init__(self):
        self.telemetry_enabled = True
        self.telemetry_sample_rate = 10  # Report every 10th iteration
        self.telemetry_metrics = ['throughput', 'latency']  # Which metrics to report
```

## Security Considerations

1. **Local Only by Default**: Aggregator only binds to localhost
2. **No Authentication**: Assumed to run on trusted network
3. **Rate Limiting**: Aggregator can drop packets if overwhelmed
4. **Sanitization**: All data sanitized before display in UI

## Future Extensions

1. **Persistent Storage**: Save metrics to time-series database
2. **Remote Monitoring**: Secure tunnel for remote UI access
3. **Alerting**: Thresholds and notifications
4. **Trace Context**: Correlate metrics across nodes
5. **Binary Protocol**: More efficient than text for high-volume
6. **Compression**: Batch compression for remote telemetry

## Implementation Plan

### Phase 1: Basic Telemetry
- TelemetryClient in framework
- Simple UDP aggregator
- Basic throughput display in UI

### Phase 2: Full Metrics
- Latency, queue depth, custom metrics
- Historical charts in UI
- Color coding for targets

### Phase 3: Advanced Features
- Sampling strategies
- Performance optimizations
- Global dashboard view

### Phase 4: Production Features
- Persistent storage option
- Remote monitoring
- Alerting system