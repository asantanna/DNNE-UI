# Yield Test Development Status

## Overview
The Yield_Test workflow contains two independent subgraphs:
1. **MNIST subgraph**: ML training pipeline (nodes 37-56)
2. **PPO subgraph**: RL training with Isaac Gym (node 58)

The goal is to demonstrate concurrent execution where PPO's `sync_adaptive_yield()` allows MNIST nodes to execute during RL training pauses.

## Current Status: Race Condition Preventing MNIST Execution

### What Works ✅
1. **Initial node startup**: All 13 nodes start correctly
2. **MNIST data flow**: 
   - MNISTDataset emits dataset/schema
   - BatchSampler receives and creates dataloader
   - GetBatch receives dataloader/schema
   - SGDOptimizer connects to Network
   - TrainingStep receives optimizer
3. **PPO execution**: PPO agent starts and runs Isaac Gym training
4. **sync_adaptive_yield**: Correctly yields with 500ms delays

### The Problem ❌
**GetBatch never processes the trigger signal**, even though:
- TrainingStep sends the ready signal
- The signal IS in GetBatch's trigger queue
- But GetBatch has 0 waiting getters when checked

### Root Cause: Task Scheduling Issue (Not a Race Condition)

**Initial incorrect theory**: I thought this was a race condition where GetBatch hadn't registered as a getter before the trigger was sent.

**Actual problem**: GetBatch's task/coroutine isn't being scheduled by the event loop after PPO starts yielding.

### How Asyncio Queues Work
When you call `await queue.get()`:
1. If items in queue → returns immediately (no waiting needed)
2. If queue empty → creates a Future and waits

Since the trigger IS in the queue, GetBatch's `get()` should complete immediately. The fact that it doesn't means **GetBatch's coroutine never gets CPU time to execute**.

### Key Debug Observations
From queue state dumps BEFORE PPO starts:
```
[QUEUE_DEBUG] Input 'trigger' state (node 50):
  Items in queue: 1  <-- Trigger IS there!
  Max size: 100
  First items: [{'signal_type': 'ready', ...}]
  Waiting getters: 0  <-- GetBatch hasn't called get() yet
```

Later, after PPO is yielding:
```
[QUEUE_DEBUG] Input 'trigger' state (node 50):
  Items in queue: 1  <-- Still there!
  Waiting getters: 1  <-- Now GetBatch IS waiting
    [0] Future done=False, cancelled=False
```

This proves GetBatch DID call `get()` and is waiting, but the `get()` isn't completing even though there's an item available.

### The Real Issue: Cannot Call _run_once() From Within a Task

**Root Cause Found**: The sync_adaptive_yield approach is fundamentally flawed.

When PPO (running as TaskC) calls `loop._run_once()`, it attempts to execute other tasks (like GetBatch/TaskB) while still inside PPO's task context. This violates asyncio's execution model and causes:

```
RuntimeError: Cannot enter into task <TaskB> while another task <TaskC> is being executed.
```

**Why This Happens**:
1. PPOAgent runs as an async task
2. It uses runpy to execute sync training code
3. The sync code calls `sync_adaptive_yield()` which calls `loop._run_once()`
4. `_run_once()` tries to run other tasks (MNIST nodes)
5. Asyncio prevents this because we're still inside PPO's task context

**This explains why**:
- GetBatch has the trigger in its queue but never processes it
- MNIST nodes never execute during PPO yields
- Only PPO-related code runs

The fundamental issue is architectural: **sync_adaptive_yield cannot work when called from within an async task**.

### Current Yield Behavior
- PPO calls `sync_adaptive_yield()` with 500ms delays
- The event loop runs (`_run_once` called many times)
- But only PPO-related tasks execute
- MNIST tasks remain unscheduled

## Solution: Thread-Safe Yielding ✅

After discovering that sync_adaptive_yield cannot work from within an async task context, we implemented a **thread-safe yielding mechanism** that allows PPO training (running in a thread via executor) to yield control back to the main event loop.

### How It Works

1. **ThreadSafeYielder**: A singleton that manages yield requests from threads
   - Runs a yield processor task in the main event loop
   - Uses thread-safe queues for communication between threads and the event loop
   - Processes yield requests by performing actual `await asyncio.sleep()` in the main loop

2. **Modified PPO Node**: 
   - Runs training in `run_in_executor` (thread pool)
   - Patches `Global.sync_adaptive_yield` to use thread-safe version
   - Maintains compatibility with existing rl_games_dnne code

3. **Thread-Safe sync_adaptive_yield**:
   - Detects if running in async context or thread
   - In async context: Uses original `loop._run_once()` approach
   - In thread context: Delegates to ThreadSafeYielder

### Results

✅ **Both subgraphs execute concurrently!**
- PPO yields control every 500ms as configured
- MNIST nodes receive execution time during yields
- GetBatch processes triggers successfully
- No asyncio violations or errors

### Key Insights

1. The fundamental issue was architectural - calling `loop._run_once()` from within a task violates asyncio's execution model
2. Running PPO in a thread (via executor) allows proper isolation
3. Thread-safe communication enables yielding without violating asyncio rules
4. This solution maintains compatibility with existing code while enabling concurrent execution

## Technical Details

### File Locations
- Workflow: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/user/default/workflows/Yield_Test.json`
- Export: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Yield_Test/`
- Key files:
  - `framework/globals.py` - Contains `sync_adaptive_yield()`
  - `nodes/getbatchnode_50.py` - The stuck node
  - `nodes/trainingstepnode_45.py` - Sends the trigger
  - `nodes/ppoagentnode_58.py` - Runs Isaac Gym training

### Debug Output Files
- `/tmp/queue_debug.log` - Full debug with event loop instrumentation
- `/tmp/yield_flow.log` - Filtered node execution flow
- `/tmp/yield_test_complete.log` - Complete run with all debug

### Current Debug Settings
- `sync_adaptive_yield` delay: 500ms (increased from 50ms)
- Debug prints added to all MNIST nodes
- Event loop `_run_once` instrumentation active