# Adaptive Yielding System

*See also: [System Balancing](system_balancing.md) for metrics tracking and balancer nodes*

## Overview

DNNE's queue-based architecture supports multiple concurrent workflows, but compute-intensive nodes can starve other workflows of execution time. The Adaptive Yielding System provides a cooperative multitasking solution that ensures fair execution across all active workflows while maintaining high performance.

## The Concurrency Challenge

### Problem Statement

DNNE workflows often contain tight computational loops:
- PPO training loops that run for thousands of steps
- Large batch processing in neural networks  
- Physics simulations with many environments
- Data preprocessing over large datasets

Without yielding, these loops monopolize execution:
```python
# This blocks all other workflows!
for epoch in range(1000):
    for step in range(10000):
        actions = ppo.get_actions(observations)
        observations, rewards = env.step(actions)
    ppo.update()
```

### Sequential Nature of Workflows

Most DNNE workflows are sequential due to data dependencies:
```
Node A → Node B → Node C → Node D
   ↑                          ↓
   └──────── trigger ─────────┘
```

Only one node executes at a time in a given workflow, but multiple independent workflows should be able to interleave execution.

## The Solution: Dual Adaptive Yield Methods

### The Sync/Async Challenge

DNNE operates in two contexts:
1. **Async contexts**: Queue-based nodes with `async def compute()`
2. **Sync contexts**: PPO training loops, Isaac Gym physics steps, NumPy/PyTorch operations

To support both, we provide two yield methods:

### async_adaptive_yield() - For Async Contexts

Clean and straightforward for async functions:

```python
async def compute(self, inputs):
    for item in large_dataset:
        result = process(item)
        await g.async_adaptive_yield()  # Clean async yield
    return result
```

### sync_adaptive_yield() - For Sync Contexts

For synchronous code that needs to yield:

```python
def ppo_training_loop(self):
    for episode in range(1000):
        for step in range(100):
            actions = self.policy(observations)
            g.sync_adaptive_yield()  # Sync yield (uses event loop voodoo)
            observations, rewards = self.env.step(actions)
```

**⚠️ Warning**: `sync_adaptive_yield()` uses private asyncio APIs (`loop._run_once()`) to achieve yielding from synchronous code. This is hacky but necessary for integrating with synchronous libraries.

### Adaptive Algorithm

Both methods use the same adaptive delay calculation:

```python
class Global:
    @classmethod
    async def async_adaptive_yield(cls):
        """Async version - use in async functions"""
        if cls._yield_disabled > 0:
            return
            
        delay = cls._compute_adaptive_delay()
        await asyncio.sleep(delay)
        cls._yield_stats.update(delay)
    
    @classmethod
    def sync_adaptive_yield(cls):
        """Sync version - use in synchronous functions"""
        if cls._yield_disabled > 0:
            return
        
        # Must have event loop running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "sync_adaptive_yield() called but no event loop is running! "
                "All DNNE workflows must run within an async context."
            )
        
        delay = cls._compute_adaptive_delay()
        
        if delay == 0:
            # Quick yield - just run one iteration
            loop._run_once()
        else:
            # Timed delay using event loop internals
            done = False
            def set_done(): 
                nonlocal done
                done = True
            
            loop.call_later(delay, set_done)
            while not done:
                loop._run_once()
        
        cls._yield_stats.update(delay)
```

## Implementation Details

### Metrics Tracking

The system tracks multiple metrics to determine appropriate yield behavior:

#### 1. Queue Pressure
```python
# High pressure indicates workflows are waiting
total_queued = sum(q.qsize() for q in all_queues)
queue_pressure = total_queued / total_queue_capacity
```

#### 2. Node Starvation
```python
# Track how long each node waits for execution
node_starvation = {
    node_id: time_since_last_execution
    for node_id in active_nodes
}
max_starvation = max(node_starvation.values())
```

#### 3. Execution Fairness
```python
# Ensure balanced execution across workflows
execution_counts = defaultdict(int)  # node_id -> count in window
fairness_variance = statistics.variance(execution_counts.values())
```

### Adaptive Delay Calculation

```python
def _compute_adaptive_delay():
    metrics = Global._gather_metrics()
    
    # Critical starvation - aggressive yielding
    if metrics.max_starvation > CRITICAL_THRESHOLD:
        return 0.01  # 10ms
    
    # Moderate starvation - balanced yielding
    elif metrics.max_starvation > WARNING_THRESHOLD:
        return 0.001  # 1ms
    
    # High queue pressure - minor yielding
    elif metrics.queue_pressure > 0.8:
        return 0.0001  # 0.1ms
    
    # Normal operation - minimal yielding
    else:
        return 0  # Just check event loop
```

## The no_yield() Context Manager

### Purpose

For critical sections that should not be interrupted by cooperative yields:

```python
with Global.no_yield():
    # Critical GPU kernel execution
    cuda_kernel.launch()
    cuda.synchronize()
    # Don't yield during atomic GPU operations
```

### Implementation

Uses a counter to handle nested contexts correctly:

```python
class Global:
    _yield_disabled = 0  # Counter for nested contexts
    
    @staticmethod
    @contextmanager
    def no_yield():
        """
        Disable Global.AdaptiveYield() within this context.
        
        NOTE: This does NOT affect:
        - Direct asyncio.sleep() calls
        - Async I/O operations
        - Queue blocking operations
        - Third-party library yields
        """
        Global._yield_disabled += 1
        try:
            yield
        finally:
            Global._yield_disabled -= 1
```

### Nested Context Handling

The counter ensures correct behavior with nested contexts:

```python
async def outer_function():
    with Global.no_yield():  # Counter: 0 → 1
        await inner_function()
        # Still no yields here!
    # Counter: 1 → 0, yields re-enabled

async def inner_function():
    with Global.no_yield():  # Counter: 1 → 2
        # Critical work
    # Counter: 2 → 1, still disabled for outer context
```

## Integration Patterns

### Async Contexts - Use async_adaptive_yield()

For async functions like queue node compute methods:

```python
async def compute(self, batch):
    results = []
    for i, item in enumerate(batch):
        result = self.model(item)
        results.append(result)
        
        # Yield periodically in async context
        if i % 10 == 0:
            await g.async_adaptive_yield()
    
    return results
```

### Sync Contexts - Use sync_adaptive_yield()

For synchronous code like PPO training loops:

```python
class PPOTrainer:
    def collect_rollouts(self):
        for step in range(self.rollout_steps):
            # Get actions from policy (sync)
            actions = self.policy(self.observations)
            g.sync_adaptive_yield()
            
            # Step environments (sync)
            self.observations, rewards, dones = self.env.step(actions)
            g.sync_adaptive_yield()
            
            # Store in buffer
            self.buffer.add(self.observations, actions, rewards)
            
            # Handle episode resets
            if any(dones):
                self.handle_resets(dones)
                g.sync_adaptive_yield()
    
    def update_policy(self):
        for epoch in range(self.ppo_epochs):
            for batch in self.buffer.get_batches():
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                g.sync_adaptive_yield()
```

### Mixed Async/Sync Example

```python
# Async node that calls sync PPO code
class PPONode(QueueNode):
    async def compute(self, observations):
        # Async context
        await g.async_adaptive_yield()
        
        # Call sync PPO implementation
        self.ppo.train_step(observations)  # Uses sync_adaptive_yield internally
        
        # Back to async
        await g.async_adaptive_yield()
        return self.ppo.get_actions()
```

### Template Integration

Node templates should include yields automatically:

```python
# In template generation
class {CLASS_NAME}_{NODE_ID}(QueueNode):
    async def compute(self, **inputs):
        results = []
        
        for i, item in enumerate(inputs['data']):
            # Process item
            result = self.process_item(item)
            results.append(result)
            
            # Auto-generated yield
            if i % {YIELD_FREQUENCY} == 0:
                await Global.AdaptiveYield()
        
        return results
```

## Performance Considerations

### Yield Frequency Guidelines

1. **Inside loops**: Yield every N iterations, not every iteration
2. **After expensive operations**: Always yield after operations > 10ms
3. **Between phases**: Yield between major algorithm phases
4. **Avoid in tight numerical loops**: Don't yield in BLAS/LAPACK operations

### Performance Comparison with --no-yield

To measure the overhead of adaptive yielding, use the `--no-yield` flag:

```bash
# Normal execution with adaptive yielding
python runner.py

# Full speed execution without yielding
python runner.py --no-yield

# Compare training times
time python runner.py --epochs 10
time python runner.py --epochs 10 --no-yield
```

When `--no-yield` is enabled:
- Both `async_adaptive_yield()` and `sync_adaptive_yield()` return immediately
- No cooperative multitasking occurs
- Useful for benchmarking the yielding overhead
- Should show faster execution but at the cost of blocking other workflows

### Overhead Analysis

**async_adaptive_yield()** overhead:
- `await asyncio.sleep(0)`: ~1μs when nothing else ready
- Clean and predictable

**sync_adaptive_yield()** overhead:
- `loop._run_once()`: ~2-5μs depending on queue state
- Slightly higher due to event loop manipulation
- Worth it for enabling sync code integration

For a loop with 1M iterations:
- Yielding every iteration: ~1-5s overhead (bad)
- Yielding every 1000 iterations: ~1-5ms overhead (good)

### Performance Monitoring

The Global class tracks yield statistics:

```python
class Global:
    yield_count = 0
    yield_time = 0.0
    yield_overhead_ns = 0
    
    @staticmethod
    def get_yield_stats():
        return {
            'total_yields': Global.yield_count,
            'total_yield_time': Global.yield_time,
            'avg_yield_time_us': (Global.yield_time / Global.yield_count) * 1e6,
            'overhead_percentage': (Global.yield_time / total_runtime) * 100
        }
```

## Best Practices

### DO:
- ✅ Use `async_adaptive_yield()` in async functions
- ✅ Use `sync_adaptive_yield()` in synchronous functions
- ✅ Yield in loops over environments/batches
- ✅ Yield after file I/O or network operations
- ✅ Yield between training epochs
- ✅ Use no_yield() for atomic GPU operations
- ✅ Monitor yield overhead in performance-critical code

### DON'T:
- ❌ Mix up async/sync yield methods
- ❌ Yield inside matrix multiplication kernels
- ❌ Yield in time-critical control loops
- ❌ Yield more than once per millisecond
- ❌ Forget to yield in long-running operations
- ❌ Assume no_yield() affects all async operations

### Choosing the Right Method

```python
# If your function signature is:
async def something():
    await g.async_adaptive_yield()  # Use async version

# If your function signature is:
def something():
    g.sync_adaptive_yield()  # Use sync version
```

## Future Enhancements

### Planned Features

1. **Automatic Yield Injection**: Compiler-style pass to inject yields
2. **Priority-Based Scheduling**: Higher priority workflows get more time
3. **Deadline Support**: Ensure time-critical nodes meet deadlines
4. **Profiler Integration**: Identify nodes that need yield optimization
5. **Dynamic Frequency**: Adjust yield frequency based on workload

### Integration with Virtual Nodes

The adaptive yielding system is crucial for the planned PPO virtual node implementation:

```python
# Visual representation: Two nodes
IsaacGymEnvs → PPO_Agent

# Exported reality: Single training loop with yields
class PPOTrainer:
    async def train(self):
        for epoch in range(epochs):
            # Collect experience
            await self.collect_rollouts()
            await Global.AdaptiveYield()
            
            # Update policy
            await self.update_policy()
            await Global.AdaptiveYield()
```

This ensures the monolithic PPO implementation plays nicely with other concurrent DNNE workflows.

## Conclusion

The Adaptive Yielding System enables DNNE to support multiple concurrent workflows while maintaining high performance. By requiring cooperative yielding in all nodes and providing adaptive behavior based on system metrics, we achieve fair scheduling without the complexity of preemptive multitasking.