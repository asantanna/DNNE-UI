# Thread-Safe Yielding Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Event Loop (asyncio)                 │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ MNIST Nodes │  │ Other Async │  │ ThreadSafeYielder│    │
│  │  (Async)    │  │    Tasks    │  │  (Yield Processor)│    │
│  └─────────────┘  └─────────────┘  └────────┬─────────┘    │
│                                              │               │
└──────────────────────────────────────────────┼───────────────┘
                                               │
                              Thread-safe      │
                              Queue Comm       │
                                               │
┌──────────────────────────────────────────────┼───────────────┐
│                    Thread Pool Executor      │               │
│                                              ▼               │
│  ┌────────────────────────────────────────────────────┐     │
│  │              PPO Training (Sync Code)              │     │
│  │                                                    │     │
│  │  while training:                                   │     │
│  │      # Do work...                                  │     │
│  │      if need_yield:                                │     │
│  │          sync_adaptive_yield()  ──────────────────┘     │
│  │          # ^ Sends request to main loop                 │
│  │          # Blocks until yield completes                 │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Main Event Loop
- Runs all async tasks (MNIST nodes, etc.)
- Hosts the ThreadSafeYielder's yield processor
- Processes yield requests from threads

### 2. ThreadSafeYielder
- **Singleton pattern** - One instance manages all thread yields
- **Yield processor task** - Runs in main loop, checks for yield requests
- **Thread-safe queues** - Communication between threads and main loop
- **Response mechanism** - Signals threads when yield completes

### 3. Thread Pool Executor
- Isolates synchronous code from async context
- Allows blocking operations without affecting event loop
- Standard asyncio pattern via `run_in_executor`

### 4. PPO Training Thread
- Runs existing synchronous training code
- Calls patched `sync_adaptive_yield()`
- Blocks during yield but doesn't block event loop

## Communication Flow

1. **PPO calls sync_adaptive_yield()**
   ```python
   # In thread context
   sync_adaptive_yield(delay=0.5)
   ```

2. **Detects thread context**
   ```python
   try:
       loop = asyncio.get_running_loop()
       # In async context
   except RuntimeError:
       # In thread context - use thread-safe yield
   ```

3. **Sends yield request**
   ```python
   yielder.yield_queue.put((thread_id, delay))
   ```

4. **Main loop processes request**
   ```python
   # In yield processor
   thread_id, delay = yield_queue.get_nowait()
   await asyncio.sleep(delay)  # Actual yield!
   ```

5. **Thread resumes**
   ```python
   # Signal completion back to thread
   response_queues[thread_id].put(True)
   ```

## Key Design Decisions

### Why Thread Pool Executor?
- **Isolation**: Separates sync code from async task context
- **Standard**: Uses asyncio's recommended pattern
- **Compatible**: Works with existing synchronous code

### Why Queue-Based Communication?
- **Thread-safe**: Python's queue.Queue handles synchronization
- **Non-blocking**: Main loop checks queue without blocking
- **Reliable**: Clear request/response pattern

### Why Patch sync_adaptive_yield?
- **Transparency**: Existing code doesn't need changes
- **Flexibility**: Can revert to original behavior easily
- **Compatibility**: Works with all rl_games_dnne code

## Performance Characteristics

- **Yield latency**: ~1-2ms overhead for thread communication
- **Yield accuracy**: Actual yield duration very close to requested
- **CPU usage**: Minimal overhead from yield processor
- **Scalability**: Single yielder handles multiple threads

## Error Handling

1. **No event loop in thread**: Handled by detection logic
2. **Yield timeout**: Thread continues if response delayed
3. **Thread termination**: Cleanup of response queues
4. **Exception propagation**: Errors in thread reported to main

This architecture enables true concurrent execution of independent subgraphs while maintaining asyncio's execution model constraints.