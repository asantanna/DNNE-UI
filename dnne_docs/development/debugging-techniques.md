# Non-Obvious Debugging Techniques

*The hidden debugging powers in DNNE that you'd never discover from `--help`.*

## Subsystem + Node Targeted Debugging

**The Hidden Power**: You can debug specific subsystems AND specific nodes in one command.

```bash
# Debug queue subsystem + nodes 42 and 56 + initialization
python runner.py --debug queue,42,56,init

# What this shows:
# - All queue operations (subsystem)
# - Everything from nodes 42 and 56 (specific nodes)  
# - Initialization sequence (init subsystem)
```

**Why This Matters**: In a graph with 50 nodes, you can laser-focus on the problem area without drowning in logs.

**Subsystem Options**:
- `queue` - Queue operations, data flow
- `init` - Node registration, barrier, connections
- `checkpoint` - Save/load operations
- `telemetry` - Metrics collection
- `heartbeat` - Node activity monitoring

**Pro Tip**: Node IDs are strings, so `--debug 42` targets node "42", not debug level 42.

## Queue Pressure Analysis

**The Heartbeat Monitor**: Every 5 seconds with `--debug` or `--verbose`, you get:

```
💓 Heartbeat: 12/18 nodes active | Queued: 47 msgs | Queues: node_42.input_d:45, node_33.input:2
```

**How to Read This**:
- `12/18 nodes active` - 12 nodes currently processing (6 might be waiting)
- `Queued: 47 msgs` - Total messages waiting in all queues
- `node_42.input_d:45` - Node 42's input_d has 45 messages backed up (PROBLEM!)

**What Queue Pressure Tells You**:
- **Growing queue**: Node is too slow or stuck
- **Empty queues + idle nodes**: Deadlock or missing data source
- **One huge queue**: Found your bottleneck
- **All queues growing**: System overloaded

**The Investigation Pattern**:
```bash
# 1. See which queue is growing
python runner.py --debug heartbeat --timeout 30

# 2. Debug that specific node
python runner.py --debug 42 --timeout 10

# 3. Check what it's waiting for
python runner.py --debug queue,42 --timeout 5
```

## Deadlock Detection Pattern

**The Symptoms**:
- Some nodes show 0 computations
- Heartbeat shows nodes idle for long periods
- Queues aren't filling (different from bottleneck)

**The Detective Process**:

### Step 1: Get the Full Picture
```bash
python runner.py --debug heartbeat --timeout 15
```

Look for:
```
Idle: 40(14.5s), 49(14.5s), 59(14.5s)  # These haven't run at all!
```

### Step 2: Trace Dependencies
Check what the idle nodes are waiting for:
```bash
python runner.py --debug 40,49,59 --timeout 5
```

You'll see:
```
Node 40 waiting for input 'model'
Node 49 waiting for input 'model'  
Node 59 waiting for input 'model'
```

### Step 3: Find the Source
The model comes from NetworkNode. Check it:
```bash
python runner.py --debug 33,54,62 --timeout 5
```

If these are also waiting, you've found a circular dependency.

### Step 4: Find the Break Point
Usually it's a bootstrap issue (like Franka needing action before observation).

## Template Tracing

**The Problem**: Generated code is broken. Which template created it?

**The Breadcrumb Trail**:

### 1. Error in Generated Code
```python
# In exports/MyWorkflow/nodes/networknode_33.py
TypeError: missing required argument 'device'
```

### 2. Find the Template  
Look at the class name: `NetworkNode_33`
- Node type: `NetworkNode`
- Template: `templates/nodes/network_queue.tpl` (maybe `network_queue.py`)

### 3. Find the UI Node
```bash
grep -r "NetworkNode" custom_nodes/
# Returns: custom_nodes/network_visnode.py
```

### 4. Check the Export Code
```bash
grep -r "NetworkNode" export_system/node_exporters/
# Returns: export_system/node_exporters/ml_nodes.py
```

**The Fix Path**: UI Node → Node Exporter → Template → Generated Code

## Async Profiling

**The Challenge**: Normal Python profilers don't understand async tasks well.

**The DNNE Way**:

### Option 1: Built-in DNNE Profiling
```bash
python runner.py --dnne-profiling --timeout 30
```

This tracks:
- C++ operation timing (if you have custom ops)
- Queue wait times
- Node compute times

### Option 2: Node-Specific Profiling
```python
# Add to any node temporarily
import time

class MyNode(QueueNode):
    async def compute(self, input):
        start = time.perf_counter()
        result = await self.expensive_operation(input)
        elapsed = time.perf_counter() - start
        if elapsed > 0.1:  # Log slow operations
            self.logger.warning(f"Slow compute: {elapsed:.3f}s")
        return result
```

### Option 3: Queue Depth Profiling
```python
# In your node
if self.node_id == "42" and self.compute_count % 100 == 0:
    for name, queue in self.input_queues.items():
        print(f"Queue {name}: {queue.qsize()} waiting")
```

## The asyncio.sleep(0) Probe

**The Diagnostic Tool**: When things are stuck, add `await asyncio.sleep(0)` to yield control.

**Where to Add for Diagnosis**:
```python
# After sending data
await self.send_output(result, "output")
await asyncio.sleep(0)  # <-- Diagnostic yield
print(f"Node {self.node_id}: Sent output")  # This tells you it got here

# Before waiting for input
await asyncio.sleep(0)  # <-- Let others run first
print(f"Node {self.node_id}: About to wait for input")
input = await self.get_input()
```

**What This Reveals**:
- If the print appears, the node reached that point
- If adding sleep fixes the issue, you have an event loop monopolization problem
- If it doesn't help, the problem is elsewhere

**Remember**: Remove these after debugging!

## Connection Validation

**The Question**: Is node A actually connected to node B?

**Method 1: Check the Wiring Log**
```bash
python runner.py --debug init 2>&1 | grep "Connected"
```

Shows:
```
Connected 33.model -> 40.model
Connected 40.optimizer -> 41.optimizer
```

**Method 2: Check Node's Connection Info**
```python
# Temporarily add to your node:
def set_connections(self, connections):
    super().set_connections(connections)
    print(f"Node {self.node_id} connections: {connections}")
```

**Method 3: Validate in Template**
```python
if not self.input_queues:
    raise RuntimeError(f"Node {self.node_id} has no input connections!")
```

## The Nuclear Option: Execution Trace

When everything else fails, trace EVERYTHING:

```bash
# Create a trace script
cat > trace_execution.py << 'EOF'
import sys
import os
os.environ['PYTHONASYNCIODEBUG'] = '1'  # Enable async debug mode

# Run with maximum verbosity
import subprocess
subprocess.run([
    sys.executable, 'runner.py',
    '--debug', 'all',
    '--timeout', '5'
], env={**os.environ, 'PYTHONTRACEMALLOC': '1'})
EOF

python trace_execution.py 2>&1 | tee full_trace.log
```

Then search the log:
```bash
# Find where node 42 stops
grep "node.42" full_trace.log | tail -20

# Find queue operations
grep "Queue" full_trace.log | grep -v "empty"

# Find the last thing before hang
tail -100 full_trace.log
```

## Debugging Checklist

When debugging a workflow issue:

1. **Check the basics**:
   - [ ] Are all nodes registered? (check init logs)
   - [ ] Are connections wired? (check init logs)
   - [ ] Did initialization complete? (look for "System ready")

2. **Check data flow**:
   - [ ] Is data being produced? (check source nodes)
   - [ ] Are queues filling? (check heartbeat)
   - [ ] Are nodes computing? (check stats at end)

3. **Check for deadlocks**:
   - [ ] Any circular dependencies?
   - [ ] Bootstrap patterns in place?
   - [ ] Nodes waiting for each other?

4. **Check for race conditions**:
   - [ ] Task creation/cancellation patterns?
   - [ ] Event loop monopolization?
   - [ ] Missing asyncio.sleep(0)?

## The Meta-Debugging Technique

**The Ultimate Pattern**: When you solve a tricky bug, immediately:

1. Add debug logging to make it obvious next time:
```python
if self.debug:
    self.logger.debug(f"Waiting for {input_name}, queue size: {queue.qsize()}")
```

2. Add a validation to catch it earlier:
```python
if not self.bootstrap_complete:
    raise RuntimeError("Node started before bootstrap!")
```

3. Document it in gotchas.md

Because you WILL hit this bug again in six months and forget how you solved it.