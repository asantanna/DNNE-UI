# DNNE Deadlock Analysis Tool

A comprehensive tool for analyzing deadlocks in DNNE workflows by replaying execution traces through node simulators.

## Overview

This tool helps debug distributed control system deadlocks by:
1. Loading workflow graphs and execution events
2. Simulating dataflow through node simulators
3. Detecting deadlock patterns and root causes
4. Suggesting solutions

## Usage

### Basic Analysis

```bash
python analyze_deadlock.py
```

This automatically:
- Loads graph structure from `/tmp/dnne_deadlock_data/graph_structure.json`
- Converts `data_flow.log` to `events.json` if needed
- Runs deadlock analysis
- Shows results with root cause analysis

### Generating Deadlock Data

To capture deadlock data from a workflow:

```bash
# Run with deadlock monitoring enabled
python runner.py --override all:retain_graph=True

# Data is saved to /tmp/dnne_deadlock_data/
```

## How It Works

### Event Replay
The simulator replays the exact sequence of events that occurred during execution:
- `QUEUE_PUT`: Node produced output
- `QUEUE_GET_SUCCESS`: Node received input
- `QUEUE_GET_WAIT`: Node waiting for input
- `QUEUE_PUT_BLOCKED`: Node blocked on output

### Node Simulators
Each node type has a simulator that models its behavior:
- **BarrierNode**: Waits for data and trigger signal
- **SGDOptimizer**: Can bootstrap with null gradient signal
- **IsaacGymSim**: Can bootstrap with null action
- **Network/Concat/Split**: Standard dataflow operations

### Deadlock Detection
The system detects deadlock when:
1. More than 80% of nodes are waiting
2. The event trace has ended (no more progress possible)
3. Circular dependencies exist between waiting nodes

### Pattern Break Analysis
When deadlock is detected, the tool automatically analyzes execution patterns to find:
- **Execution Cycles**: Repeating patterns marked by IsaacGym simulation steps
- **Incomplete Cycles**: Identifies when the last cycle didn't complete normally
- **Missing Nodes**: Shows which nodes failed to execute in the final cycle
- **Critical Failure Point**: Pinpoints the exact node and time where the pattern broke

## Common Deadlock Patterns

### SGD-Barrier Circular Dependency
**Pattern**: Networks wait for Barriers → Barriers wait for SGD triggers → SGDs wait for loss from Networks

**Solution**: Enable SGD bootstrap signals:
```bash
python runner.py --override all:no_bootstrap_trigger=False
```

### Missing Bootstrap
**Pattern**: System needs initial action/gradient to start

**Solution**: Enable appropriate bootstrap:
- SGD nodes: `no_bootstrap_trigger=False`
- IsaacGym: Automatic null action bootstrap

## Architecture

```
deadlock_tool/
├── analyze_deadlock.py       # Main analysis entry point
├── dataflow_simulator.py     # Core simulation engine
├── node_simulators/          # Node-specific simulators
│   ├── __init__.py
│   ├── base_node_sim.py     # Base simulator class
│   ├── barrier_node_queue_sim.py
│   ├── sgd_optimizer_queue_sim.py
│   ├── isaac_gym_sim_queue_sim.py
│   └── ...                  # Other node simulators
└── test_scripts/             # Test suites
    ├── test_fail_fast.py    # Verify fail-fast behavior
    └── test_detection.py    # Test detection accuracy
```

## Fail-Fast Design

The tool follows fail-fast principles:
- **Missing simulators**: Raises `ValueError` immediately
- **Unknown node types**: Fails during graph construction
- **Execution errors**: Propagates exceptions with context

This ensures meaningful results - no silent failures or incorrect analysis.

## Testing

Run the test suite:
```bash
# Test fail-fast behavior
python test_scripts/test_fail_fast.py

# Test detection accuracy
python test_scripts/test_detection.py
```

## Implementation Notes

- Simulators only replay recorded events - they don't synthesize new behavior
- Bootstrap detection identifies capable nodes but doesn't trigger them
- Relative timestamps are shown for better readability
- The 80% waiting threshold prevents false positives on small graphs