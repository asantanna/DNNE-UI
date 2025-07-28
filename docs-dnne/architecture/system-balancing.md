# System Balancing in DNNE

## Overview

System balancing in DNNE is about ensuring different parts of a workflow execute at their optimal rates, not about giving each node equal execution time. Different nodes have fundamentally different performance requirements based on their role in the system.

## Core Philosophy

Traditional load balancing aims for equal resource distribution. However, in ML/robotics workflows:
- A camera sensor might need exactly 30Hz for video processing
- A control loop might require 100Hz for stability
- A planning algorithm might only need 1Hz updates
- Training nodes want maximum throughput without starving monitoring

The goal is to meet each node's specific requirements while efficiently using system resources.

## Balancing Node Types

DNNE provides two types of balancing nodes:

### 1. Balancing Node
A regular node that acts as a passthrough while measuring and enforcing performance targets:
- Has input and output ports
- Inserted into data flow paths
- Minimal overhead (just timestamps and forwards data)
- Actively participates in execution

### 2. Balancing Config (Virtual)
A configuration-only node for monolithic nodes like PPO_Agent:
- Only has output port for visual wiring
- Connects to special nodes but produces no runtime node
- Configuration is read by the target node
- Target node manually reports metrics

### Design
- **Input**: Accepts any data type (passthrough)
- **Output**: Forwards input unchanged
- **Purpose**: Measures and enforces performance targets
- **Placement**: Insert at strategic points in the workflow

### Configuration Options
```python
class MonitorNode:
    # Frequency-based targets (robotics/real-time)
    min_hz: Optional[float] = None          # Minimum frequency
    max_hz: Optional[float] = None          # Maximum frequency  
    target_hz: Optional[float] = None       # Desired frequency
    
    # Throughput-based targets (batch processing)
    target_percentage: Optional[float] = None  # % of total system throughput
    
    # Priority settings
    priority: int = 0                       # Higher = more important
    guaranteed: bool = False                # Must meet targets vs best-effort
    
    # Latency requirements
    max_latency_ms: Optional[float] = None  # Maximum processing time
```

## Balance Strategies

### 1. Frequency-Based (Real-time/Robotics)

For nodes that must maintain specific execution rates:
- **Use Case**: Sensor sampling, control loops, rendering
- **Target**: Maintain X Hz regardless of system load
- **Example**: Camera must capture at 30 FPS for video

```yaml
Camera_Monitor:
  target_hz: 30
  min_hz: 25      # Warn if drops below
  max_hz: 35      # Throttle if exceeds
  guaranteed: true
```

### 2. Throughput-Based (Batch Processing)

For nodes that should get a fair share of processing:
- **Use Case**: Training, data processing, analytics
- **Target**: X% of total system throughput
- **Example**: Training gets 80%, logging gets 20%

```yaml
Training_Monitor:
  target_percentage: 80
  priority: 1
  
Logging_Monitor:
  target_percentage: 20
  priority: 0
```

### 3. Priority-Based (Safety Critical)

For nodes with different importance levels:
- **Use Case**: Emergency stops, safety monitors, optimization
- **Target**: Guarantee execution for critical nodes
- **Example**: Safety monitor always runs, optimizer is best-effort

```yaml
Safety_Monitor:
  min_hz: 10
  guaranteed: true
  priority: 100
  
Optimizer_Monitor:
  target_hz: 1
  guaranteed: false
  priority: 0
```

## Metrics Tracked

Monitor nodes would track:
1. **Throughput**: Items processed per second
2. **Frequency**: Actual execution rate (Hz)
3. **Latency**: Time from input received to output sent
4. **Queue Depth**: Upstream congestion indicators
5. **Starvation Time**: Duration since last execution
6. **Jitter**: Variance in execution timing

## Integration with Adaptive Yielding

Monitor nodes feed metrics to the global adaptive yielding system:

```python
# In Global._compute_adaptive_delay()
def _compute_adaptive_delay(cls) -> float:
    # Get monitor node metrics
    for monitor in cls._monitor_nodes.values():
        if monitor.below_minimum():
            # Node is starved, reduce its yield time
            node_delays[monitor.node_id] *= 0.9
        elif monitor.above_maximum():
            # Node is over-provisioned, increase yield
            node_delays[monitor.node_id] *= 1.1
            
    # Compute global delay based on worst-case starvation
    return compute_balanced_delay(node_delays)
```

## Example Workflows

### 1. Yield_Test with Balancing
```
MNIST Subgraph:
TrainingStep.ready → [Balancing Node] → GetBatch.trigger
                     (max_hz: 100)

PPO Subgraph:
[Balancing Config] → PPO_Agent
(target_percentage: 70)
```

The Balancing Node in MNIST enforces a maximum training rate, while the Balancing Config tells PPO_Agent it should aim for 70% of system resources.

### 2. Robotics Control System
```
Sensors(100Hz) → [Balancing] → Processing(50Hz) → [Balancing] → Control(100Hz)
                 (guaranteed)                       (guaranteed)
                      ↓
                  Logger(10Hz)
```

Balancing nodes ensure control loop maintains timing while logger is best-effort.

### 3. ML Training Pipeline
```
Dataset → Augmentation → Training → [Balancing] → Validation
                             ↓       (target_percentage: 90)
                         Checkpointing(every 100 batches)
```

Training gets 90% throughput, validation 8%, checkpointing 2%.

## Open Questions & Considerations

### 1. Conflicting Requirements
What happens when monitor nodes have incompatible targets?
- Total percentages > 100%
- Frequency requirements exceed system capacity
- **Possible Solution**: Priority-based degradation

### 2. Cascade Effects
Slowing one node affects downstream nodes:
- If camera slows to 15Hz, can fusion still run at 30Hz?
- **Possible Solution**: Decouple with queues, interpolate missing data

### 3. Burst Handling
Some workflows have bursty patterns:
- Process batch quickly, then idle
- **Possible Solution**: Time-windowed metrics (average over X seconds)

### 4. Resource Saturation
System can't meet targets if CPU/GPU is saturated:
- Need to detect and report infeasible configurations
- **Possible Solution**: Admission control, warn user about conflicts

### 5. Dynamic Adaptation
Requirements might change during execution:
- Training needs more resources during backward pass
- **Possible Solution**: Phase-aware monitoring

## Implementation Details

### Throttling Mechanism: yield_slice()

Instead of using uninterruptible `asyncio.sleep()`, balancing nodes use a custom yielding mechanism:

```python
async def yield_slice(seconds: float):
    """Yield control for specified duration, but interruptible by balancing system"""
    start = time.perf_counter()
    chunk_size = 0.001  # 1ms chunks for interruptibility
    remaining = seconds
    
    while remaining > 0:
        # Check if balancing system wants to interrupt
        if Global.should_interrupt_yield(node_id):
            break
            
        # Yield for small chunk
        await asyncio.sleep(min(chunk_size, remaining))
        remaining = seconds - (time.perf_counter() - start)
```

This allows the balancing system to:
- Interrupt yields if high-priority nodes need to run
- Extend yields if the system is overloaded
- Adapt to changing system conditions

### Violation Logging

To avoid log spam while still tracking violations:

```python
class ViolationLogger:
    def __init__(self, dump_interval=10.0, max_details=5):
        self.violations = []
        self.dump_interval = dump_interval
        self.max_details = max_details
    
    def log_violation(self, node_id, violation_type, details):
        self.violations.append({
            'timestamp': time.time(),
            'node_id': node_id,
            'type': violation_type,
            'details': details
        })
        
        # Dump periodically
        if time.time() - self.last_dump > self.dump_interval:
            self.dump_violations()
    
    def dump_violations(self):
        count = len(self.violations)
        logger.warning(f"⚠️ {count} balancing violations in last {self.dump_interval}s")
        
        # Show first few violations
        for v in self.violations[:self.max_details]:
            logger.warning(f"  - Node {v['node_id']}: {v['type']} violation")
        
        if count > self.max_details:
            logger.warning(f"  ... and {count - self.max_details} more")
```

### Execution Logging

Each run creates a fresh `execution.log` file:

```python
# In runner.py
log_file = Path("execution.log")
logging.basicConfig(
    filename=log_file,
    filemode='w',  # Overwrite each run
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Log key events
logger.info(f"Starting {workflow_name} execution")
logger.info(f"Arguments: {args}")
logger.warning(f"Node {node_id} below minimum rate: {actual_hz} < {min_hz}")
logger.info(f"Balance report: PPO={ppo_pct}%, MNIST={mnist_pct}%")
```

### PPO_Agent Integration

For monolithic nodes like PPO_Agent:

```python
class PPOAgentNode:
    def __init__(self):
        # Check for connected Balancing Config
        if self.has_balancing_config():
            self.perf_targets = self.get_balancing_config()
            Global.register_virtual_monitor(self.node_id, self.perf_targets)
    
    async def training_loop(self):
        # Manually report metrics
        for episode in range(max_episodes):
            Global.report_iteration_complete(self.node_id)
            
            # Check if we should yield more/less based on targets
            if Global.should_adjust_yielding(self.node_id):
                self.yield_frequency = Global.get_recommended_yield_frequency(self.node_id)
```

## Implementation Considerations

### Balancing Node Placement
- After high-frequency sources (sensors)
- Before resource-intensive operations (training)
- At workflow convergence points (fusion nodes)
- Between independent subgraphs (e.g., MNIST and PPO)

### Performance Overhead
- Balancing nodes should be near-zero overhead
- Just timestamp and forward data
- Metrics reported asynchronously
- Use yield_slice() instead of sleep() for throttling

### User Interface
- Visual distinction between Balancing Node and Balancing Config
- Real-time rate display on nodes (e.g., "87.3 Hz")
- Green/red indicators for meeting/missing targets
- Configuration through node properties panel

## Implementation Priority

1. **Phase 1: Measurement Only**
   - Implement basic Balancing Node that measures but doesn't enforce
   - Add execution.log to runner.py
   - Test with Yield_Test to understand system behavior

2. **Phase 2: Basic Enforcement**
   - Implement yield_slice() mechanism
   - Add throttling to Balancing Node for max_hz
   - Implement violation logging with batching

3. **Phase 3: Virtual Nodes**
   - Add Balancing Config node type
   - Integrate with PPO_Agent
   - Test percentage-based balancing

4. **Phase 4: Advanced Features**
   - Priority-based scheduling
   - Guaranteed vs best-effort modes
   - Multi-phase awareness

## Future Extensions

1. **Auto-Tuning**: System learns optimal balance over time
2. **Multi-Resource**: Consider GPU, memory, network separately  
3. **Distributed**: Balance across multiple machines
4. **Predictive**: Anticipate load changes and pre-adjust
5. **SLA Enforcement**: Formal service level agreements
6. **Idle Task**: Special node that runs when nothing else needs resources

## Conclusion

System balancing in DNNE should move beyond equal time-slicing to requirement-based scheduling. The distinction between Balancing Nodes (active participants) and Balancing Config (virtual configuration) allows both fine-grained control and support for monolithic nodes. The yield_slice() mechanism provides interruptible yielding, while batched violation logging keeps the system observable without overwhelming logs. This approach better serves the diverse needs of ML and robotics workflows where different components have fundamentally different performance characteristics and requirements.