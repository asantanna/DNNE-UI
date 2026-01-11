# Deadlock Analysis System

## Overview

The deadlock analysis system provides tools for detecting and analyzing deadlocks in complex DNNE workflows, particularly those using synchronization nodes like Barrier and Eat_N. It consists of two main components:

1. **Simple Detection**: Heartbeat monitoring with basic deadlock warnings
2. **Full Analysis**: Detailed data collection and offline analysis tools in `deadlock_tool/`

## Tool Architecture

The analysis tool (`deadlock_tool/`) is a modular Python system that processes event logs to identify deadlock root causes:

### Core Components
- **`analyze_deadlock.py`**: Entry point, orchestrates analysis
- **`data_parser.py`**: Parses JSON event logs and graph structure  
- **`pattern_analyzer.py`**: Identifies patterns (stuck nodes, missing inputs, cycles)
- **`root_cause_analyzer.py`**: Traces dependency chains to find root blockers
- **`report_generator.py`**: Creates human-readable analysis reports
- **`node_behaviors.json`**: Knowledge base defining node types and behaviors

### Node Behaviors Knowledge Base

The `node_behaviors.json` file categorizes nodes to enable intelligent root cause analysis:

```json
{
  "node_types": {
    "IsaacGymSimNode": {
      "category": "processor",
      "stream_inputs": ["action", "reset"],
      "stream_outputs": ["observation", "done"],
      "notes": "Self-bootstraps first observation with null_action"
    },
    "SGDOptimizerNode": {
      "category": "bootstrap_provider",
      "stream_inputs": ["loss"],
      "virtual_inputs": ["model"],  // Not a real queue connection
      "bootstrap_outputs": ["step_complete"]
    }
  }
}
```

Key categories:
- **free_running**: Generates output without input (TensorNode)
- **processor**: Standard input→output transformation
- **bootstrap_dependent**: Needs initial config to start
- **bootstrap_provider**: Sends initial triggers after config
- **synchronization**: Complex timing (Barrier, Eat_N)

### Known Issues & Limitations

1. **Custom Queue Handling**: Some nodes (e.g., IsaacGymSimNode) bypass the standard MultiWaiter and directly call `queue.get()` without logging, making their inputs invisible to analysis

2. **Virtual Connections**: SGDOptimizer receives model as Python object reference, not through queues. These "virtual" connections appear in the graph but have no queue activity

3. **Queue State Analysis**: For wait_all nodes (like Concat), the tool now logs queue depths via QUEUE_STATE events to identify which specific inputs are blocking

4. **NODE_START Logging**: Fixed by moving logging to non-overridable `_call_run_when_ready()` method

## Background Reading

To understand the implementation and deadlock patterns in DNNE, read these documents:

1. **`dnne_docs/architecture/queue_framework.md`** - Understanding the async queue-based architecture
2. **`dnne_docs/development/debugging-techniques.md`** - Current debugging capabilities and heartbeat system
3. **`dnne_docs/nodes/utility/eat_n_node.md`** - Eat_N node for temporal synchronization
4. **`dnne_docs/nodes/utility/barrier_node.md`** - Barrier node for holding/releasing data
5. **`dnne_docs/development/gotchas.md`** - Common pitfalls including double-getter deadlocks
6. **`dnne_docs/nodes/utility/README.md`** - Temporal synchronization patterns and diagrams

## Data Collection & Analysis

### Event Types Logged

When `--debug-deadlock` is enabled, the following events are written to `/tmp/dnne_deadlock_data/data_flow.log`:

- **NODE_START**: Node initialization (includes class name)
- **QUEUE_GET_WAIT**: Node starts waiting for input
- **QUEUE_GET_SUCCESS**: Node receives input (includes wait time)
- **QUEUE_PUT**: Node sends output (includes subscriber count)
- **QUEUE_STATE**: Snapshot of all input queue depths (for wait_all analysis)
- **NODE_COMPUTE_START/END**: Computation timing

### Analysis Output Sections

The analyzer generates reports with these sections:

1. **ROOT CAUSE ANALYSIS**: Primary blockers identified through dependency tracing
2. **NODES THAT NEVER RECEIVED INPUT**: Complete starvation (no inputs received)
3. **NODES THAT NEVER PRODUCED OUTPUT**: May indicate sink nodes or failures
4. **STUCK NODES**: Waiting > 1 second (potential deadlock)
5. **WAITING FOR DATA**: Normal async operation

### Critical Discoveries

- **IsaacGymSimNode Issue**: Uses custom queue handling that bypasses logging, making it appear to never receive actions even when it does
- **Virtual Connections**: Model→SGD connections are object references, not queue-based
- **Double Triggers**: Both Eat_N and SGDOptimizer can send triggers to Barriers (design issue)

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
python runner.py --debug-deadlock --timeout 15 --override all:retain_graph=True --override all:no_bootstrap_trigger=True

# Step 2: Analyze the captured data
cd /path/to/DNNE-UI/deadlock_tool
python analyze_deadlock.py

# Or specify custom data location
python analyze_deadlock.py --data-dir /path/to/deadlock_data

# For detailed analysis with verbose output
python analyze_deadlock.py --verbose
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

### Node-Aware Analysis

The analysis tool uses a knowledge base of node behaviors (`deadlock_tool/node_behaviors.json`) to understand:

- **Free-running nodes**: Generate output without needing input (e.g., TensorNode, dataset nodes)
- **Bootstrap-dependent nodes**: Need initial configuration to start (e.g., IsaacGymSim needs initial action)
- **Bootstrap providers**: Nodes that provide initial triggers after receiving config (e.g., SGDOptimizer)
- **Synchronization nodes**: Complex timing behaviors (e.g., Eat_N, Barrier)

This allows the analyzer to distinguish between:
- Root causes (e.g., missing bootstrap inputs)
- Symptoms (e.g., downstream nodes waiting because upstream is blocked)

### Analysis Tool Output

The `analyze_deadlock.py` script provides:

```
DNNE Deadlock Analysis Report
============================================================
Data from: /tmp/dnne_deadlock_data
Time range: 0.00s - 14.52s
Total events: 8,432
Total nodes: 26
⚠️  16 nodes never started
⚠️  1 nodes never produced output

🔍 ROOT CAUSE ANALYSIS:
----------------------------------------

1. PRIMARY ISSUE: Node 25 (IsaacGymSimNode_25)
   Type: missing_bootstrap
   Problem: Node needs bootstrap inputs: action to start
   Blocks 24 downstream nodes:
     → 125 (ConcatNode_125)
     → 132 (SimulationTracker_132)
     → 33 (NetworkNode_33)
     → 40 (SGDOptimizerNode_40)
     → 42 (ConcatNode_42)
     ... and 19 more
   💡 Suggested fix: Provide required bootstrap inputs at workflow start

2. PRIMARY ISSUE: Node 40 (SGDOptimizerNode_40)
   Type: missing_bootstrap
   Problem: Node needs bootstrap inputs: model to start
   Blocks 14 downstream nodes:
     → 74 (BarrierNode_74)
     → 75 (BarrierNode_75)
     → 76 (BarrierNode_76)
     ... and 11 more
   💡 Suggested fix: Provide required bootstrap inputs at workflow start

📋 OTHER ISSUES:
----------------------------------------
• Node 75 (BarrierNode_75) missing inputs: input, release
• Node 74 (BarrierNode_74) missing inputs: input, release
• Node 76 (BarrierNode_76) missing inputs: input, release

💡 RECOMMENDATIONS:
----------------------------------------
  1. Provide bootstrap inputs for IsaacGymSimNode_25: action input to start
  2. Verify synchronization nodes (Barrier/Eat_N) have proper trigger patterns
  3. Review the visual workflow for missing connections
  4. Check that all required node parameters are configured
  5. Consider adding debug logging to track data flow
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
- [Temporal Alignment](../nodes/utility/README.md) - RL synchronization patterns
- [Gotchas](../development/gotchas.md) - Common pitfalls and solutions