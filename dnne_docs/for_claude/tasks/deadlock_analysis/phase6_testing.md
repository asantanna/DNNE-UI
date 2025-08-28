# Phase 6: Testing & Validation

## Testing Strategy

### Development Testing
During development, use local test scripts for rapid iteration:
- **Location**: `/deadlock_tool/test_scripts/`
- **Purpose**: Quick validation without test suite overhead
- **Approach**: Simple scripts that can be run directly

### Test Script Organization
```
/deadlock_tool/test_scripts/
├── test_basic_nodes.py          # Test individual node simulators
├── test_simple_deadlock.py      # Test simple circular dependency
├── test_barrier_sync.py         # Test barrier synchronization
├── test_eat_n_behavior.py       # Test Eat_N consume/passthrough
├── test_franka_coop.py          # Full Franka_Coop_Nodes test
├── generate_test_data.py        # Create synthetic test cases
└── README.md                     # How to run tests
```

## Test Cases

### 1. Individual Node Simulators
```python
# test_scripts/test_basic_nodes.py
import sys
sys.path.append('..')  # Add parent dir to path

from node_simulators.barrier_node_queue_sim import BarrierNodeSimulator
from node_simulators.eat_n_node_queue_sim import EatNNodeSimulator

def test_barrier_node():
    """Test barrier holds data until triggered"""
    barrier = BarrierNodeSimulator('barrier_1', {'class': 'BarrierNode_1'})
    
    # Send data - should not be ready
    barrier.process_input('input', {'data': 'test'})
    assert not barrier.can_execute()
    
    # Send trigger - should now be ready
    barrier.process_input('release', {'signal': 'trigger'})
    assert barrier.can_execute()
    
    # Execute and check output
    output = barrier.execute()
    assert output['output']['data'] == 'test'
    print("✓ Barrier node test passed")

def test_eat_n_node():
    """Test Eat_N consumes then passes through"""
    eat_n = EatNNodeSimulator('eat_n_1', {'class': 'Eat_NNode_1', 'n': 2})
    
    # First input - consume, no output
    eat_n.process_input('input', {'data': 'first'})
    assert eat_n.can_execute()
    output = eat_n.execute()
    assert output == {}  # No output yet
    
    # Second input - consume and switch to passthrough
    eat_n.process_input('input', {'data': 'second'})
    output = eat_n.execute()
    assert 'output' in output
    assert 'trigger' in output
    
    # Third input - passthrough mode
    eat_n.process_input('input', {'data': 'third'})
    output = eat_n.execute()
    assert output['output']['data'] == 'third'
    assert 'trigger' not in output  # No trigger in passthrough
    print("✓ Eat_N node test passed")

if __name__ == "__main__":
    test_barrier_node()
    test_eat_n_node()
    print("\nAll basic node tests passed!")
```

### 2. Simple Deadlock Detection
```python
# test_scripts/test_simple_deadlock.py
import json
from pathlib import Path

def create_simple_deadlock():
    """Create a simple A→B→C→A circular dependency"""
    graph = {
        "nodes": {
            "A": {"class": "NetworkNode_A", "type": "network"},
            "B": {"class": "NetworkNode_B", "type": "network"},
            "C": {"class": "NetworkNode_C", "type": "network"}
        },
        "connections": [
            ["A", "output", "B", "input"],
            ["B", "output", "C", "input"],
            ["C", "output", "A", "input"]
        ]
    }
    
    events = [
        {"event_type": "QUEUE_GET_WAIT", "node_id": "A", "input_name": "input", "timestamp": 0.1},
        {"event_type": "QUEUE_GET_WAIT", "node_id": "B", "input_name": "input", "timestamp": 0.2},
        {"event_type": "QUEUE_GET_WAIT", "node_id": "C", "input_name": "input", "timestamp": 0.3}
    ]
    
    return graph, events

def test_simple_deadlock():
    """Test detection of simple circular dependency"""
    graph, events = create_simple_deadlock()
    
    # Run deadlock analysis
    from dataflow_simulator import DeadlockSimulator
    
    sim = DeadlockSimulator(graph)
    result = sim.replay_events(events)
    
    assert result['deadlock_detected'] == True
    assert result['cycle'] == ['A', 'B', 'C', 'A']
    print("✓ Simple deadlock detected correctly")

if __name__ == "__main__":
    test_simple_deadlock()
```

### 3. Franka Coop Test
```python
# test_scripts/test_franka_coop.py
import json
from pathlib import Path

def test_franka_coop_deadlock():
    """Test actual Franka_Coop_Nodes deadlock data"""
    # Load real data
    graph_path = Path('/tmp/dnne_deadlock_data/graph_structure.json')
    events_path = Path('/tmp/dnne_deadlock_data/events.json')
    
    with open(graph_path) as f:
        graph = json.load(f)
    with open(events_path) as f:
        events = json.load(f)
        
    # Run analysis
    from dataflow_simulator import DeadlockSimulator
    
    sim = DeadlockSimulator(graph)
    result = sim.replay_events(events)
    
    # Check results
    assert result['deadlock_detected'] == True
    assert result['deadlock_time'] > 4.0  # Should detect after ~4.6s
    
    print(f"Deadlock detected at t={result['deadlock_time']:.3f}s")
    print(f"Cycle: {' → '.join(result['cycle'])}")
    print(f"Root cause: {result['root_cause']}")

if __name__ == "__main__":
    test_franka_coop_deadlock()
```

## Running Tests During Development

```bash
# From deadlock_tool directory
cd deadlock_tool

# Test individual components
python test_scripts/test_basic_nodes.py

# Test deadlock detection
python test_scripts/test_simple_deadlock.py

# Test with real data
python test_scripts/test_franka_coop.py

# Run all tests
python test_scripts/run_all.py
```

## Migration to Test Suite

Once development is complete and tests are stable:

1. **Refactor tests** to use unittest or pytest framework
2. **Move to test suite** at `/dnne_test_suite/deadlock_analysis/`
3. **Add integration** with main test runner
4. **Document** in main testing documentation
5. **CI integration** if applicable

## Success Criteria

### Development Phase
- [ ] All node simulators have basic tests
- [ ] Simple deadlock cases detected correctly
- [ ] Barrier synchronization patterns work
- [ ] Eat_N behavior correctly simulated
- [ ] Franka_Coop deadlock identified

### Production Phase
- [ ] Tests integrated into main suite
- [ ] Performance benchmarks established
- [ ] Edge cases covered
- [ ] Documentation complete