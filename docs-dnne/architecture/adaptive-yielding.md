# Adaptive Yielding System

## Overview

The adaptive yielding system is DNNE's solution for achieving concurrent execution of independent computational subgraphs within a single Python process. It enables multiple neural network training workflows (e.g., supervised learning and reinforcement learning) to run simultaneously without blocking each other, maximizing hardware utilization.

## Problem Statement

When running complex workflows with multiple independent subgraphs:
- **Traditional approach**: Sequential execution wastes resources
- **Multi-process approach**: High overhead, complex data sharing
- **DNNE's approach**: Cooperative multitasking via adaptive yielding

### Example: Yield_Test Workflow
The Yield_Test workflow demonstrates two causally independent subgraphs:
1. **MNIST Subgraph**: Supervised learning classification network
2. **PPO Subgraph**: Reinforcement learning with IsaacGym environment

Without yielding, one subgraph would block the other. With adaptive yielding, both execute concurrently with ~50/50 time distribution.

## Architecture

### Core Components

#### 1. Global State Manager (`export_system/templates/framework/globals.py`)
Centralized management of:
- Execution metrics and timing
- Yield statistics and counters
- Concurrency tracking (PPO vs non-PPO execution time)
- Node-specific metrics (starvation time, queue pressure)

#### 2. Thread-Safe Yielding (`export_system/templates/framework/globals_threadsafe.py`)
Enables synchronous code (like PPO training) to yield from within threads:
```python
class ThreadSafeYielder:
    """Singleton that manages yield requests from threads"""
    _instance = None
    _loop = None
    _yield_queue = None
    
    async def process_yield_requests(self):
        """Main loop processes yield requests from threads"""
        while True:
            delay = await self._yield_queue.get()
            await asyncio.sleep(delay)
            self._yield_queue.task_done()
```

#### 3. Async Queue Framework (`export_system/templates/base/queue_framework.py`)
All DNNE nodes use async queues for communication:
- Natural yield points during `queue.get()` and `queue.put()`
- Non-blocking execution model
- Automatic concurrency for async nodes

### Yielding Mechanisms

#### Async Yielding (MNIST and most nodes)
```python
async def compute(self, input_data):
    # Natural yield point when waiting for input
    data = await self.input_queues["input"].get()
    
    # Process data
    result = self.process(data)
    
    # Natural yield point when sending output
    await self.output_queues["output"].put(result)
```

#### Sync Yielding (PPO and thread-based nodes)
```python
def sync_adaptive_yield(cls):
    """Synchronous yield for thread context"""
    if cls._yield_disabled > 0 or cls.no_yield:
        return
    
    # Use thread-safe yielding mechanism
    thread_safe_sync_adaptive_yield(delay=cls._compute_adaptive_delay())
```

## Implementation Details

### PPO Agent Execution Model

The PPO agent runs synchronous IsaacGym training in a thread pool:

```python
async def _run_training_async(self, train_config):
    """Run PPO training in thread pool with yielding"""
    
    def run_training_with_yielding():
        # Enable adaptive yielding for this thread
        os.environ['DNNE_ADAPTIVE_YIELD'] = '1'
        
        # Import rl_games (which imports our yielding Global)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Run synchronous training
        result = runpy.run_path("train.py", run_name="__main__")
        
        # Training loop will call Global.sync_adaptive_yield()
        return result
    
    # Execute in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_training_with_yielding)
```

### Yield Timing Calculation

The system dynamically adjusts yield delays based on execution metrics:

```python
def _compute_adaptive_delay(cls) -> float:
    """Compute yield delay based on node starvation"""
    if not cls._node_metrics:
        return 0.0  # Minimal yield
    
    # Find maximum starvation time
    max_starvation = max(
        metrics.starvation_time for metrics in cls._node_metrics.values()
    )
    
    # Scale delay: 0ms (low starvation) to 10ms (high starvation)
    if max_starvation < 0.01:  # Less than 10ms
        return 0.0
    elif max_starvation < 0.1:  # 10-100ms
        return 0.001 * (max_starvation / 0.1)
    else:  # Over 100ms
        return 0.01  # Max 10ms yield
```

### Concurrency Metrics

The system tracks execution balance between subgraphs:

```python
class ConcurrencyStats:
    def __init__(self):
        self.ppo_time = 0.0      # Time in PPO subgraph
        self.non_ppo_time = 0.0  # Time in other subgraphs
        self.start_time = time.perf_counter()
        self.current_context = None
        self.context_start = None
```

## Integration with rl_games

The rl_games library was modified to support adaptive yielding:

```python
# In rl_games_dnne/common/a2c_common.py
if DNNE_ADAPTIVE_YIELD:
    from framework.globals import Global
    
    # In training loop
    for n in range(self.horizon_length):
        # ... environment step ...
        
        # Yield to allow other subgraphs to execute
        if DNNE_ADAPTIVE_YIELD:
            Global.sync_adaptive_yield()
```

## Debugging and Monitoring

### Concurrency Report
Shows execution balance between subgraphs:
```
============================================================
🔄 CONCURRENT EXECUTION BALANCE REPORT
============================================================
Total execution time: 19.04s
PPO subgraph time:    8.83s (46.4%)
MNIST subgraph time:  10.21s (53.6%)
Total yields:         0

✅ Both subgraphs are receiving execution time!
   Execution is well-balanced between subgraphs.
============================================================
```

### Debug Prints
All debug prints are tagged with `#DBG_TAG#`:
```python
print(f"[FREEZE_DEBUG 13] About to call Global.sync_adaptive_yield()", flush=True) #DBG_TAG#
```

Toggle script for easy management:
```bash
python claude_scripts/toggle_DBG_TAG.py /path/to/file.py
```

## Known Issues and Limitations

### 1. Total Yields Counter
Currently shows 0 despite yielding occurring. Investigation needed to determine:
- Whether yields below a threshold aren't counted
- If the thread-safe path bypasses the counter
- If the metric calculation needs adjustment

### 2. Output Buffering
Console output may buffer until program completion. Flush calls are included but may not always work in all environments.

### 3. Fixed Queue Sizes
Current implementation uses `maxsize=2` for all queues. This may need tuning for workflows with bursty data patterns.

## Design Decisions

### Why Not Use Python's Native Async/Await Everywhere?
- **IsaacGym requires synchronous execution**: The physics simulation runs in a tight synchronous loop
- **Legacy code compatibility**: Many ML libraries aren't async-native
- **Thread pool solution**: Allows sync code to cooperate with async framework

### Why Track PPO vs Non-PPO?
- **Different execution patterns**: PPO runs long synchronous loops, MNIST is naturally async
- **Balance monitoring**: Ensures neither subgraph dominates execution time
- **Future optimization**: Can adjust yielding strategy based on workload type

### Why Adaptive Delays?
- **Performance optimization**: Minimal yields when not needed
- **Starvation prevention**: Increased yields when nodes are waiting
- **System responsiveness**: Balances throughput with latency

## Future Enhancements

### 1. Per-Node Yield Strategies
Allow nodes to specify their yielding preferences:
```python
class NetworkNode(QueueNode):
    yield_strategy = "aggressive"  # vs "conservative", "adaptive"
```

### 2. Multi-Level Yielding
Implement yielding at multiple granularities:
- **Micro-yields**: Within tight computation loops
- **Macro-yields**: Between major computation phases
- **Scheduled yields**: Time-based yielding for long operations

### 3. Yield Prediction
Use ML to predict optimal yield points:
- Learn from execution patterns
- Predict starvation before it occurs
- Dynamically adjust yield frequency

## Best Practices

### 1. When to Yield
- **After I/O operations**: File reads, network requests
- **Between computation batches**: Every N iterations
- **At natural boundaries**: Between training epochs

### 2. When NOT to Yield
- **Critical sections**: During model updates
- **Tiny operations**: Overhead exceeds benefit
- **Time-critical paths**: Real-time control loops

### 3. Testing Concurrency
- **Use Yield_Test workflow**: Contains example concurrent subgraphs
- **Monitor execution balance**: Check concurrency reports
- **Verify both subgraphs progress**: Watch for starvation

## Code Examples

### Adding Yielding to a Custom Node

```python
class CustomProcessingNode(QueueNode):
    async def compute(self, input_data):
        # For async nodes, yielding is automatic via queues
        
        # For long synchronous operations:
        for i in range(1000):
            self.process_item(input_data[i])
            
            # Yield every 100 items
            if i % 100 == 0 and hasattr(Global, 'sync_adaptive_yield'):
                Global.sync_adaptive_yield()
        
        return {"output": processed_data}
```

### Disabling Yields for Critical Sections

```python
with Global.no_yield():
    # Critical section - no yields allowed
    self.model.update_weights()
    self.save_checkpoint()
```

## Conclusion

The adaptive yielding system enables DNNE to achieve true concurrent execution of independent neural network workflows within a single process. By combining async queue-based communication with thread-safe synchronous yielding, the system maximizes hardware utilization while maintaining code compatibility with existing ML frameworks.

The key insight is that different types of computational workloads (async-native vs synchronous) can cooperate effectively through a unified yielding mechanism, with adaptive timing that responds to actual execution patterns rather than fixed schedules.