# System Balancer in DNNE

## Overview

System balancing in DNNE is about ensuring different parts of a workflow execute at their optimal rates, not about giving each node equal execution time. Different nodes have fundamentally different performance requirements based on their role in the system.

## Core Philosophy

Traditional load balancing aims for equal resource distribution. However, in ML/robotics workflows:
- A camera sensor might need exactly 30Hz for video processing
- A control loop might require 100Hz for stability
- A planning algorithm might only need 1Hz updates
- Training nodes want maximum throughput without starving monitoring

The goal is to meet each node's specific requirements while efficiently using system resources.

## Balancer Node Types

DNNE provides two types of balancing nodes:

### 1. Balancer Node
A regular node that acts as a passthrough while measuring and enforcing performance targets:
- Has input and output ports
- Inserted into data flow paths
- Minimal overhead (just timestamps and forwards data)
- Actively participates in execution

### 2. Balancer Config (Virtual)
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

The balancing system integrates with DNNE's adaptive yielding through subgraph metrics:

```python
# Current implementation in Global._compute_adaptive_delay()
@classmethod
def _compute_adaptive_delay(cls) -> float:
    """
    Compute appropriate yield delay based on system metrics.
    
    Returns:
        Delay in seconds (0 for minimal yield)
    """
    # For now, always use minimal yield
    # TODO: Implement adaptive delay based on subgraph requirements
    return 0.0
```

Future implementations will consult balancing node requirements to adjust yield delays.

## Example Workflows

### 1. Yield_Test with Balancer
```
MNIST Subgraph:
TrainingStep.ready → [Balancer Node] → GetBatch.trigger
                     (max_hz: 100)

PPO Subgraph:
[Balancer Config] → PPO_Agent
(target_percentage: 70)
```

The Balancer Node in MNIST enforces a maximum training rate, while the Balancer Config tells PPO_Agent it should aim for 70% of system resources.

### 2. Robotics Control System
```
Sensors(100Hz) → [Balancer] → Processing(50Hz) → [Balancer] → Control(100Hz)
                 (guaranteed)                       (guaranteed)
                      ↓
                  Logger(10Hz)
```

Balancer nodes ensure control loop maintains timing while logger is best-effort.

### 3. ML Training Pipeline
```
Dataset → Augmentation → Training → [Balancer] → Validation
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

## Current Implementation Details

### PPO_Agent Integration

For monolithic nodes like PPO_Agent:

```python
class PPOAgentNode:
    def __init__(self):
        # Register with balancing system
        Global.register_sync_node(
            self.node_id, 
            subgraph="ppo",
            item_unit="env_steps",
            requirements=self.balancing_config if self.has_balancing_config else None
        )
    
    async def training_loop(self):
        # Use unified yield API
        for step in range(horizon_length):
            # ... environment step ...
            Global.sync_adaptive_yield(subgraph="ppo", is_item_ref=True)
            
        # Training yields for responsiveness
        for mini_epoch in range(mini_epochs):
            # ... training ...
            Global.sync_adaptive_yield(subgraph="ppo", is_item_ref=False)
```

## Implementation Considerations

### Balancer Node Placement
- After high-frequency sources (sensors)
- Before resource-intensive operations (training)
- At workflow convergence points (fusion nodes)
- Between independent subgraphs (e.g., MNIST and PPO)

### Performance Overhead
- Balancer nodes have near-zero overhead
- Just timestamp and forward data
- Metrics tracked through unified yield API
- Minimal yielding (0.0 delay) for maximum performance

### User Interface
- Visual distinction between Balancer Node and Balancer Config
- Real-time rate display on nodes (e.g., "87.3 Hz")
- Green/red indicators for meeting/missing targets
- Configuration through node properties panel

## Current Implementation Status

### What's Implemented ✅

1. **Measurement-Only Balancer**
   - Balancer Node measures throughput and frequency
   - No enforcement of targets (measurement only)
   - Metrics collection through unified yield API

2. **Subgraph-Based Metrics**
   - Tracks metrics per subgraph (PPO, MNIST, etc.)
   - Distinguishes sync vs async nodes
   - Custom item units (env_steps, batches, etc.)

3. **Virtual Balancer Config**
   - Balancer Config node for monolithic nodes like PPO_Agent
   - PPO_Agent reads config and registers with balancing system

4. **Unified Yield API**
   - Consistent interface for sync and async nodes
   - `is_item_ref` parameter for accurate throughput tracking
   - Minimal yield delays (0.0) for performance

### What's NOT Implemented ❌

1. **Enforcement**: No throttling or rate limiting
2. **Adaptive Delays**: Always uses minimal yield (0.0)
3. **Priority Scheduling**: No priority-based decisions
4. **Violation Handling**: No automated responses to violations
5. **Multi-Phase Awareness**: No different behavior for different execution phases

## Future Extensions

1. **Adaptive Yielding**: Adjust delays based on requirements
2. **Rate Enforcement**: Actually throttle nodes that exceed max_hz
3. **Priority Scheduling**: Higher priority nodes get more resources
4. **Auto-Tuning**: System learns optimal balance over time
5. **Multi-Resource**: Consider GPU, memory, network separately
6. **Distributed**: Balance across multiple machines

### How Subgraph Metrics Work

Yield functions accept subgraph identification:

```python
# Sync nodes (like PPO)
Global.sync_adaptive_yield(subgraph="ppo", is_item_ref=True)  # For work items
Global.sync_adaptive_yield(subgraph="ppo", is_item_ref=False)  # For responsiveness

# Async nodes (via Balancer nodes) 
await Global.async_adaptive_yield(subgraph="mnist", is_item_ref=True)  # For work items
await Global.async_adaptive_yield(subgraph="mnist", is_item_ref=False)  # For responsiveness
```

### Metrics Collection Strategy

Track different metrics based on node type:

**For Synchronous Nodes (PPO)**:
- **CPU Time**: Exact execution time between yields
- **Throughput**: Items/second (training iterations)
- Both metrics are accurate and meaningful

**For Asynchronous Nodes (MNIST)**:
- **Throughput**: Items/second through Balancer nodes
- **CPU Time**: Not measurable due to async switching
- Focus on throughput as primary metric

**System-Wide Metrics**:
- **Total Execution Time**
- **Total Sync CPU Time**: Sum of all sync node times
- **Total Async Time**: execution_time - sum(sync_times)
  - Represents all async activity combined
  - Helps identify if async nodes are starved

### Subgraph Registration

Nodes register with the balancer during initialization:

```python
# In Balancer node init
Global.register_balancing_node(
    node_id=self.node_id,
    subgraph="mnist",  # Identified from configuration
    item_unit=self.item_name,  # From item_name widget (e.g., "batches")
    requirements=self.config  # From Balancer Config node
)

# In PPO_Agent init (reads Balancer Config)
Global.register_sync_node(
    node_id=self.node_id,
    subgraph="ppo",
    item_unit="env_steps",  # Hardcoded for PPO
    requirements=balancing_config
)
```

### Balance Reports

```
============================================================
🔄 EXECUTION BALANCE REPORT
============================================================
Total execution time: 60.00s
Sync nodes CPU time:  51.60s (86.0%)  
Async nodes time:      8.40s (14.0%)

Subgraph Performance:
  ppo     :   15.8 env_steps/sec (77.0% CPU)
  mnist   :   18.0 batches/sec (async - CPU % N/A)
  
Note: Async time includes all async activity (MNIST, 
      system overhead, idle time)
============================================================
```

### Data Structures

```python
@dataclass
class SubgraphMetrics:
    """Metrics for a computational subgraph"""
    subgraph_name: str
    node_type: str  # "sync" or "async"
    item_unit: str = "items"  # e.g., "env_steps", "batches", "frames"
    
    # Common metrics
    items_processed: int = 0
    last_item_time: float = 0.0
    
    # Sync-only metrics
    cpu_time: float = 0.0  # Total CPU seconds used
    
    @property
    def throughput(self) -> float:
        """Items per second"""
        if self.last_item_time == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.items_processed / elapsed if elapsed > 0 else 0.0
    
    @property
    def cpu_percentage(self) -> Optional[float]:
        """CPU percentage (sync nodes only)"""
        if self.node_type != "sync":
            return None
        elapsed = time.time() - self.start_time
        return (self.cpu_time / elapsed * 100) if elapsed > 0 else 0.0
```


## Unified Yield API Design

### Overview

To provide a consistent interface for both synchronous and asynchronous nodes, we implement a unified yield API where both `sync_adaptive_yield()` and `async_adaptive_yield()` have identical signatures and behavior.

### API Signatures

```python
@classmethod
def sync_adaptive_yield(cls, *, subgraph: str, is_item_ref: bool = False) -> None:
    """
    Synchronous adaptive yield for thread-based nodes.
    
    Args:
        subgraph: Name of the subgraph (e.g., "ppo")
        is_item_ref: True if this yield represents one work item completion
    """

@classmethod
async def async_adaptive_yield(cls, *, subgraph: str, is_item_ref: bool = False) -> None:
    """
    Asynchronous adaptive yield for async nodes.
    
    Args:
        subgraph: Name of the subgraph (e.g., "mnist")
        is_item_ref: True if this yield represents one work item completion
    """
```

### Key Concepts

#### 1. Item Reference (`is_item_ref`)

Complex algorithms need to yield at multiple points for responsiveness, but only specific yields should count as "items" for throughput metrics:

```python
# PPO example - yields in multiple places
for n in range(horizon_length):
    # ... environment step ...
    Global.sync_adaptive_yield(subgraph="ppo", is_item_ref=True)  # Counts as 1 env_step

for batch in dataset:
    # ... training ...
    Global.sync_adaptive_yield(subgraph="ppo", is_item_ref=False)  # Just for responsiveness

# MNIST example
async def compute(self, input):
    # ... process batch ...
    await Global.async_adaptive_yield(subgraph="mnist", is_item_ref=True)  # Counts as 1 batch
    return output
```

#### 2. Meaningful Units (`item_unit`)

Different subgraphs measure different work units:

```python
# Registration includes unit specification
Global.register_sync_node(
    node_id="ppo_66",
    subgraph="ppo",
    item_unit="env_steps"
)

Global.register_balancing_node(
    node_id="balancing_64",
    subgraph="mnist",
    item_unit="batches",  # From user-configured item_name widget
    requirements=config
)
```

This produces readable metrics:
```
Subgraph Performance:
  ppo   :   15.8 env_steps/sec (77.0% CPU)
  mnist :   18.0 batches/sec (async - CPU % N/A)
```

### Implementation Benefits

1. **Unified Interface**: Both sync and async nodes use the same pattern
2. **Self-Documenting**: Keyword-only parameters make intent clear
3. **Accurate Metrics**: Only real work units are counted
4. **Flexible Yielding**: Can yield frequently without inflating metrics
5. **Clean Design**: Yielding and metrics tracking in one place

### Future Adaptive Delay

Currently `_compute_adaptive_delay()` returns `0.0` (minimal yield). Future implementations will:
1. Consult registered balancing requirements
2. Calculate delays to meet frequency/throughput targets
3. Handle conflicts based on priority and guaranteed flags
4. Provide true adaptive balancing based on requirements

## Conclusion

System balancing in DNNE provides requirement-based measurement and monitoring for diverse workflows. The current implementation offers:

- **Measurement without enforcement**: Balancer nodes track metrics without interfering with execution
- **Subgraph-level metrics**: Clear visibility into which parts of the workflow are running
- **Unified yield API**: Consistent interface for both sync and async nodes
- **Custom item units**: Meaningful metrics like "env_steps/sec" instead of generic "items/sec"

This foundation enables future enhancements like adaptive yielding, rate enforcement, and priority scheduling while already providing valuable insights into workflow execution patterns.