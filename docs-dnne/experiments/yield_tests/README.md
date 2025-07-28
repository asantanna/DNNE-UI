# DNNE Yield Test Experiments

This directory contains the research and solution for enabling concurrent execution of independent subgraphs in DNNE workflows, specifically addressing the challenge of allowing synchronous code (like PPO training) to yield control to the async event loop.

## Problem Statement

In DNNE's Yield_Test workflow, we have two independent subgraphs:
1. **MNIST subgraph**: ML training pipeline (async queue-based)
2. **PPO subgraph**: RL training with Isaac Gym (synchronous Python code)

The goal was to allow PPO's synchronous training code to periodically yield control so MNIST nodes could execute concurrently.

## The Challenge

The initial approach using `sync_adaptive_yield()` that calls `loop._run_once()` failed with:
```
RuntimeError: Cannot enter into task while another task is being executed
```

This happens because calling `_run_once()` from within an async task violates asyncio's execution model.

## Research Process

### Test Programs (`test_programs/`)

1. **`test_yield_simple.py`** - Simplified test with 3 async tasks to reproduce the issue
2. **`test_yield_runpy.py`** - Simulates PPO's use of runpy to execute sync code
3. **`test_thread_yield.py`** - Tests thread-based approach (failed due to no event loop in threads)
4. **`test_executor_yield.py`** - Tests run_in_executor approach
5. **`test_async_ppo.py`** - Tests converting PPO to full async (works but requires major refactoring)

### Documentation (`documentation/`)

1. **`dev-status.md`** - Detailed investigation log and findings
2. **`yield-solutions.md`** - Analysis of different approaches and final solution

## Final Solution (`solution/`)

### Thread-Safe Yielding Mechanism

1. **`globals_threadsafe.py`** - Thread-safe yielding infrastructure
   - `ThreadSafeYielder`: Singleton managing yield requests from threads
   - Uses thread-safe queues for communication
   - Processes yields in main event loop via `await asyncio.sleep()`

2. **`ppoagentnode_58_threadsafe.py`** - Modified PPO node
   - Runs training in `run_in_executor` (thread pool)
   - Patches `Global.sync_adaptive_yield` to use thread-safe version
   - Maintains compatibility with existing code

## How It Works

```python
# In thread (PPO training):
sync_adaptive_yield()  # Detects thread context, sends yield request

# In main event loop:
ThreadSafeYielder processes request, performs actual yield
Other async tasks (MNIST) get execution time

# Thread resumes after yield completes
```

## Results

✅ **Both subgraphs execute concurrently**
- PPO yields control every 500ms
- MNIST nodes receive execution time during yields  
- No asyncio violations
- Full compatibility with existing rl_games_dnne code

## Usage

To apply this solution to other workflows:

1. Copy `globals_threadsafe.py` to your framework
2. Modify your synchronous node to:
   - Run in `run_in_executor`
   - Set up ThreadSafeYielder
   - Patch sync_adaptive_yield

Example:
```python
# In your async compute method:
loop = asyncio.get_running_loop()
yielder = ThreadSafeYielder.get_instance()
await yielder.start(loop)

# Patch yielding
Global.sync_adaptive_yield = classmethod(lambda cls: thread_safe_sync_adaptive_yield())

# Run sync code in executor
result = await loop.run_in_executor(None, your_sync_function)
```

## Key Insights

1. **Asyncio's execution model is strict** - Cannot call `_run_once()` from within a task
2. **Thread isolation works** - Running sync code in threads avoids task context issues
3. **Communication is key** - Thread-safe queues enable yielding across contexts
4. **Compatibility matters** - Solution works without modifying existing sync code

## Testing

Run the test programs to understand the problem and solution:

```bash
# See the original problem:
python test_programs/test_yield_runpy.py -test-yield

# See the working async approach:
python test_programs/test_async_ppo.py -test-yield

# Test the thread-safe solution in your workflow
```

## Future Improvements

1. **Performance tuning** - Optimize yield frequency based on workload
2. **Dynamic yielding** - Adjust yield duration based on queue pressure
3. **Monitoring** - Add metrics for yield effectiveness
4. **Generalization** - Create reusable pattern for other sync/async integration

---

*This research was conducted to enable DNNE's vision of concurrent execution of independent subgraphs, maintaining the benefits of async queue-based architecture while supporting legacy synchronous code.*