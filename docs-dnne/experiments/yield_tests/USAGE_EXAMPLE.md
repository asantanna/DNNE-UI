# Using Thread-Safe Yielding in Your DNNE Nodes

## Quick Start Example

If you have a node that runs synchronous code (like PPO, physics simulations, or legacy algorithms) and need it to yield control for other nodes to execute, follow this pattern:

### 1. Basic Node Structure

```python
from framework import QueueNode
from framework.globals import Global
from framework.globals_threadsafe import ThreadSafeYielder, thread_safe_sync_adaptive_yield
import asyncio

class YourSyncNode(QueueNode):
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["input_data"])
        self.setup_outputs(["results"])
        
    async def compute(self, input_data):
        """Run synchronous code with yielding"""
        
        # Set up thread-safe yielder
        loop = asyncio.get_running_loop()
        yielder = ThreadSafeYielder.get_instance()
        await yielder.start(loop)
        
        # Patch Global.sync_adaptive_yield
        original_yield = Global.sync_adaptive_yield
        Global.sync_adaptive_yield = classmethod(
            lambda cls: thread_safe_sync_adaptive_yield()
        )
        
        try:
            # Run your sync code in executor
            result = await loop.run_in_executor(
                None, 
                self._run_sync_algorithm,
                input_data
            )
            
            return {"results": result}
            
        finally:
            # Restore original yield
            Global.sync_adaptive_yield = original_yield
    
    def _run_sync_algorithm(self, data):
        """Your synchronous algorithm that needs to yield"""
        
        results = []
        for i in range(1000):
            # Do some work
            result = self._process_step(data, i)
            results.append(result)
            
            # Yield periodically
            if i % 10 == 0:
                # This will now work from thread!
                Global.sync_adaptive_yield()
        
        return results
```

### 2. For Existing Code Using sync_adaptive_yield

If you have existing code (like rl_games_dnne) that already calls `Global.sync_adaptive_yield()`:

```python
async def compute(self, config):
    """Wrapper for existing sync code"""
    
    # Setup (same as above)
    loop = asyncio.get_running_loop()
    yielder = ThreadSafeYielder.get_instance()
    await yielder.start(loop)
    
    # Patch yielding
    original = Global.sync_adaptive_yield
    Global.sync_adaptive_yield = classmethod(
        lambda cls: thread_safe_sync_adaptive_yield()
    )
    
    try:
        # Your existing code just works!
        result = await loop.run_in_executor(
            None,
            run_existing_training,  # Already has yield calls
            config
        )
        return {"metrics": result}
    finally:
        Global.sync_adaptive_yield = original
```

### 3. Custom Yield Patterns

For more control over yielding:

```python
def _run_with_custom_yields(self, data):
    """Example with different yield strategies"""
    
    # Yield after time elapsed
    last_yield = time.time()
    
    for batch in data:
        result = process_batch(batch)
        
        # Time-based yielding
        if time.time() - last_yield > 0.1:  # 100ms
            Global.sync_adaptive_yield()
            last_yield = time.time()
        
        # Work-based yielding
        if batch.is_heavy_computation:
            Global.sync_adaptive_yield()
        
        # Queue pressure yielding
        if self.output_queue_size() > 50:
            # Yield longer if backed up
            thread_safe_sync_adaptive_yield(delay=1.0)
```

## Common Patterns

### Pattern 1: Wrapper Node for Legacy Code

```python
class LegacyAlgorithmNode(QueueNode):
    """Wraps synchronous legacy algorithm"""
    
    async def compute(self, **inputs):
        # One-time setup
        if not hasattr(self, '_yielder_setup'):
            loop = asyncio.get_running_loop()
            yielder = ThreadSafeYielder.get_instance()
            await yielder.start(loop)
            self._yielder_setup = True
            
            # Permanent patch
            Global.sync_adaptive_yield = classmethod(
                lambda cls: thread_safe_sync_adaptive_yield()
            )
        
        # Run legacy code
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            legacy_algorithm_main,
            inputs
        )
```

### Pattern 2: Chunked Processing

```python
class ChunkedProcessorNode(QueueNode):
    """Process data in chunks with yields between"""
    
    async def compute(self, large_dataset):
        loop = asyncio.get_running_loop()
        
        # Process in chunks
        chunk_size = 100
        results = []
        
        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i:i + chunk_size]
            
            # Process chunk in thread
            chunk_result = await loop.run_in_executor(
                None,
                self._process_chunk,
                chunk
            )
            results.extend(chunk_result)
            
            # Natural yield between chunks
            await asyncio.sleep(0.001)
        
        return {"results": results}
```

### Pattern 3: Monitoring Yields

```python
class MonitoredYieldNode(QueueNode):
    """Track yield effectiveness"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.yield_count = 0
        self.total_yield_time = 0
    
    async def compute(self, data):
        # ... setup code ...
        
        # Add yield monitoring
        original_yield = thread_safe_sync_adaptive_yield
        
        def monitored_yield(delay=0.5):
            start = time.time()
            original_yield(delay)
            self.total_yield_time += time.time() - start
            self.yield_count += 1
        
        # Patch with monitoring version
        Global.sync_adaptive_yield = classmethod(
            lambda cls: monitored_yield()
        )
        
        try:
            result = await loop.run_in_executor(None, self._run, data)
            
            # Report yield stats
            self.logger.info(
                f"Yielded {self.yield_count} times, "
                f"total yield time: {self.total_yield_time:.2f}s"
            )
            
            return {"results": result}
        finally:
            # Restore
            Global.sync_adaptive_yield = original
```

## Best Practices

1. **Always restore patches** - Use try/finally to ensure cleanup
2. **Reuse yielder instance** - It's a singleton, only needs setup once
3. **Choose appropriate yield frequency** - Balance between responsiveness and performance
4. **Monitor performance** - Track yield overhead in production
5. **Test both modes** - Verify behavior with and without yielding

## Troubleshooting

### "No event loop in thread"
- Make sure ThreadSafeYielder is started before running executor
- Verify you're using thread_safe_sync_adaptive_yield

### Yields not happening
- Check if sync_adaptive_yield is properly patched
- Verify executor is actually running in a thread
- Add debug prints to confirm yield calls

### Performance degradation
- Reduce yield frequency
- Use larger chunks between yields
- Profile to find yield hotspots

This solution enables any synchronous code to cooperate with DNNE's async architecture!