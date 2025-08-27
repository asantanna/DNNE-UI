# Deadlock Analysis System

## Overview

The deadlock analysis system provides tools for detecting and analyzing deadlocks in complex DNNE workflows, particularly those using synchronization nodes like Barrier and Eat_N. It consists of two main components:

1. **Simple Detection**: Heartbeat monitoring with basic deadlock warnings
2. **Full Analysis**: Detailed data collection and offline analysis tools

## Background Reading

To understand the implementation and deadlock patterns in DNNE, read these documents:

1. **`dnne_docs/architecture/queue_framework.md`** - Understanding the async queue-based architecture
2. **`dnne_docs/development/debugging-techniques.md`** - Current debugging capabilities and heartbeat system
3. **`dnne_docs/nodes/utility/eat_n_node.md`** - Eat_N node for temporal synchronization
4. **`dnne_docs/nodes/utility/barrier_node.md`** - Barrier node for holding/releasing data
5. **`dnne_docs/development/gotchas.md`** - Common pitfalls including double-getter deadlocks
6. **`dnne_docs/patterns/temporal_alignment_rl.md`** - Complex synchronization patterns

## Quick Start

### Simple Deadlock Detection

Enable heartbeat monitoring to detect when all nodes become idle:

```bash
# Run with heartbeat monitoring
python runner.py --heartbeat --timeout 30

# Or combine with debug output
python runner.py --heartbeat --debug queue,init --timeout 30
```

The heartbeat displays every 5 seconds and warns if all nodes are idle for > 10 seconds.

### Full Deadlock Analysis

For detailed analysis of deadlock causes:

```bash
# Step 1: Run workflow with deadlock data collection
python runner.py --debug-deadlock --timeout 15

# For workflows with gradient retention issues (like Franka_Coop_Nodes)
python runner.py --debug-deadlock --timeout 15 --override all:retain_graph=True

# Step 2: Analyze the captured data
python /path/to/DNNE-UI/claude_scripts/analyze_deadlock.py

# Or specify custom data location
python analyze_deadlock.py --data-dir /path/to/deadlock_data
```

## Feature Details

### Heartbeat Monitoring (--heartbeat)

The heartbeat monitor shows:
- Active/idle node counts
- Queue depths and pressure points
- Compute counts for active nodes
- Idle duration for stuck nodes

Example output:
```
💓 Heartbeat: 12/18 nodes active | Queued: 47 msgs | Queues: node_42.input_d:45, node_33.input:2
💓 Heartbeat: 0/18 nodes active | ⚠️ POTENTIAL DEADLOCK - all nodes idle for 10.5s
```

### Deadlock Data Collection (--debug-deadlock)

When enabled, the system:
1. Creates `/tmp/dnne_deadlock_data/` directory (overwritten each run)
2. Logs all queue operations and node state changes
3. Exports graph structure from runner.py's perspective
4. Maintains < 0.5% performance overhead

#### Data Files

**`/tmp/dnne_deadlock_data/data_flow.log`** - Event stream (one JSON line per event):
```json
{"ts": 1234567.890123, "type": "NODE_START", "node": "42", "class": "NetworkNode"}
{"ts": 1234567.891234, "type": "QUEUE_GET_WAIT", "node": "42", "queue": "input"}
{"ts": 1234567.892345, "type": "QUEUE_GET_SUCCESS", "node": "42", "queue": "input", "wait_time": 0.001111}
{"ts": 1234567.893456, "type": "NODE_COMPUTE_START", "node": "42"}
{"ts": 1234567.894567, "type": "NODE_COMPUTE_END", "node": "42", "duration": 0.001111}
{"ts": 1234567.895678, "type": "QUEUE_PUT", "node": "42", "output": "output", "subscribers": 3}
```

**`/tmp/dnne_deadlock_data/graph_structure.json`** - Static graph information:
```json
{
  "nodes": {
    "42": {"class": "NetworkNode", "type": "network"},
    "73": {"class": "Eat_NNode", "type": "synchronization"},
    "74": {"class": "BarrierNode", "type": "synchronization"}
  },
  "connections": [
    ["25", "observation", "73", "input"],
    ["73", "trigger", "74", "release"],
    ["74", "output", "42", "input"]
  ]
}
```

**`/tmp/dnne_deadlock_data/node_configs.json`** - Node-specific configurations:
```json
{
  "73": {"num_to_eat": 1, "trigger_mode": "every_eat"},
  "74": {"hold_mode": "FIFO"},
  "25": {"task": "FrankaDNNE", "num_envs": 256}
}
```

### Analysis Tool Output

The `analyze_deadlock.py` script provides:

```
DNNE Deadlock Analysis Report
============================
Data from: /tmp/dnne_deadlock_data/
Time range: 0.00s - 14.52s
Total events: 8,432

Node Activity Summary:
  Node 42 (NetworkNode): Last activity 14.52s ago - STUCK WAITING on 'model'
  Node 73 (Eat_NNode): Last activity 14.51s ago - Consumed 1/1, now passthrough
  Node 74 (BarrierNode): Last activity 14.50s ago - Holding 0 items, 0 pending releases

Detected Issues:
  ❌ DEADLOCK: Circular dependency detected
     - Node 42 waiting for 'model' from node 33
     - Node 33 waiting for 'optimizer' from node 40
     - Node 40 waiting for 'loss' from node 42
  
  ⚠️ BOOTSTRAP: Node 25 (IsaacGymSimNode) never received initial 'action'
     Suggestion: Add null_action bootstrap or Eat_N trigger pattern

Queue Analysis:
  Empty queues: 45/48 (93.8%)
  Largest queue: node_55.input_c (12 items)
  
Recommendations:
  1. Add bootstrap action for IsaacGymSimNode (node 25)
  2. Check SGDOptimizer (node 40) one-time input handling
  3. Review Barrier release connections for nodes 74, 75, 76
```

## Common Deadlock Patterns

### 1. Missing Bootstrap

**Problem**: Simulator needs initial action before producing first observation

**Symptom**: 
- IsaacGymSim node never starts
- All downstream nodes waiting

**Solution**:
```python
# Add null_action bootstrap
null_action = torch.zeros(num_envs, action_dim)
await sim_node.send_initial_action(null_action)
```

### 2. Double-Getter Deadlock

**Problem**: MultiWaiter and manual get() compete for same queue

**Symptom**:
- Node stuck in get_config_inputs()
- MultiWaiter also trying to read same queue

**Solution**:
```python
# DON'T include one-time inputs in setup_inputs
self.setup_inputs(required=["loss"])  # Repeating input only

# Manually create queue for one-time input
self.input_queues["optimizer"] = Queue(maxsize=1)
```

### 3. Barrier Without Release

**Problem**: Barrier holds data but never receives release trigger

**Symptom**:
- Barrier queue depth growing
- No release triggers arriving

**Solution**:
- Verify Eat_N trigger connection
- Check SGD step_complete output
- Ensure proper wiring in workflow

### 4. Synchronization Mismatch

**Problem**: Eat_N consumes wrong number of inputs

**Symptom**:
- Some networks process obs(t), others process obs(t+1)
- Gradient computation uses misaligned data

**Solution**:
- Verify num_to_eat matches expected bootstrap count
- Check trigger_mode setting
- Review temporal alignment pattern

## Performance Considerations

### Overhead Measurements

| Mode | Performance Impact | Use Case |
|------|-------------------|----------|
| Normal | 0% | Production runs |
| --heartbeat | < 0.1% | Live monitoring |
| --debug-deadlock | < 0.5% | Debugging sessions |

### Data Volume

Typical data sizes for 1-minute run:
- Small workflow (10 nodes): ~5 MB
- Medium workflow (50 nodes): ~25 MB  
- Large workflow (100+ nodes): ~50 MB

The `/tmp/dnne_deadlock_data/` directory is overwritten on each run to prevent disk space issues.

## Implementation Details

### Core Components

1. **`framework/deadlock_utils.py`**
   - DeadlockLogger class for event logging
   - Graph structure export
   - Minimal overhead design

2. **`framework/base_nodes.py`**
   - Logging hooks in get_input(), send_output()
   - Conditional compilation when disabled
   - Thread-safe file operations

3. **`framework/graph_runner.py`**
   - Heartbeat deadlock detection
   - Data directory initialization
   - Integration with existing monitoring

4. **`claude_scripts/analyze_deadlock.py`**
   - Event timeline reconstruction
   - Dependency graph analysis
   - Pattern matching for common issues

### Event Types

| Event | Description | Key Fields |
|-------|-------------|------------|
| NODE_START | Node task begins | node, class |
| QUEUE_GET_WAIT | Started waiting for input | node, queue |
| QUEUE_GET_SUCCESS | Received input | node, queue, wait_time |
| NODE_WAIT | Waiting for specific input | node, queue |
| NODE_COMPUTE_START | Computation begins | node |
| NODE_COMPUTE_END | Computation complete | node, duration |
| QUEUE_PUT | Sent output | node, output, subscribers |
| BARRIER_HOLD | Barrier received data | node, queue_depth |
| BARRIER_RELEASE | Barrier released data | node, items_released |
| EAT_N_CONSUME | Eat_N consumed input | node, count, remaining |
| EAT_N_TRIGGER | Eat_N sent trigger | node, trigger_type |

## Future Enhancements

Planned improvements (not yet implemented):

1. **Real-time Detection**
   - Detect partial/regional deadlocks
   - Automatic recovery attempts
   - Live visualization

2. **Visual Analysis**
   - Graphviz wait-for graphs
   - Interactive timeline viewer
   - Queue depth animations

3. **Automatic Fixes**
   - Generate bootstrap code
   - Suggest connection changes
   - Create fixed workflow JSON

4. **Integration**
   - VS Code problem matcher
   - Web UI deadlock indicators
   - Export system warnings

## Troubleshooting

### No Data Generated

If `/tmp/dnne_deadlock_data/` is empty:
- Verify --debug-deadlock flag is set
- Check file permissions on /tmp
- Ensure workflow actually starts (not failing during import)

### Analysis Tool Errors

If `analyze_deadlock.py` fails:
- Check data files exist and aren't corrupted
- Verify Python has json module available
- Try with --verbose flag for detailed errors

### Performance Impact Too High

If overhead exceeds 1%:
- Check disk I/O speed (SSD recommended)
- Reduce node count if possible
- Use sampling mode (future feature)

## Related Documentation

- [Queue Framework](../architecture/queue_framework.md) - Core async architecture
- [Debugging Techniques](../development/debugging-techniques.md) - General debugging guide
- [Temporal Alignment](../patterns/temporal_alignment_rl.md) - RL synchronization patterns
- [Gotchas](../development/gotchas.md) - Common pitfalls and solutions