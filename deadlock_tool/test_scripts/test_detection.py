#!/usr/bin/env python3
"""
Test the deadlock detection with various scenarios.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow_simulator import DataflowSimulator

def test_deadlock_scenario():
    """Test detection with actual deadlock data"""
    print("\n=== Testing Actual Deadlock Detection ===")
    
    # Load the actual deadlock data
    data_dir = Path('/tmp/dnne_deadlock_data')
    
    with open(data_dir / 'graph_structure.json', 'r') as f:
        graph = json.load(f)
    
    with open(data_dir / 'events.json', 'r') as f:
        events = json.load(f)
    
    # Create simulator
    sim = DataflowSimulator(graph)
    
    # Run simulation
    results = sim.replay_events(events)
    
    # Check results
    if results['deadlock_detected']:
        print(f"✓ Deadlock correctly detected at {results['deadlock_time']:.3f}s")
        print(f"  {len(results['waiting_nodes'])} nodes waiting")
        return True
    else:
        print(f"❌ Failed to detect deadlock")
        return False

def test_no_deadlock_scenario():
    """Test that normal execution doesn't trigger false positives"""
    print("\n=== Testing False Positive Prevention ===")
    
    # Create a workflow with multiple nodes, some idle
    graph = {
        "nodes": {
            "tensor_1": {"class": "TensorNode_1", "type": "tensor"},
            "tensor_2": {"class": "TensorNode_2", "type": "tensor"},
            "split_1": {"class": "SplitNode_1", "type": "split"},
            "concat_1": {"class": "ConcatNode_1", "type": "concat"},
            "network_1": {"class": "NetworkNode_1", "type": "network"},
            "network_2": {"class": "NetworkNode_2", "type": "network"}
        },
        "connections": [
            ["tensor_1", "tensor", "split_1", "input"],
            ["split_1", "output_a", "concat_1", "input_a"],
            ["split_1", "output_b", "concat_1", "input_b"],
            ["concat_1", "output", "network_1", "input"]
        ]
    }
    
    # Create events that show normal execution with some nodes idle
    events = [
        {"timestamp": 0.0, "event_type": "QUEUE_PUT", "node_id": "tensor_1", 
         "output_name": "tensor", "data": {"value": "test"}},
        {"timestamp": 0.1, "event_type": "QUEUE_GET_SUCCESS", "node_id": "split_1", 
         "input_name": "input"},
        {"timestamp": 0.2, "event_type": "QUEUE_PUT", "node_id": "split_1", 
         "output_name": "output_a", "data": {"part": "a"}},
        {"timestamp": 0.2, "event_type": "QUEUE_PUT", "node_id": "split_1", 
         "output_name": "output_b", "data": {"part": "b"}},
        {"timestamp": 0.3, "event_type": "QUEUE_GET_SUCCESS", "node_id": "concat_1", 
         "input_name": "input_a"},
        {"timestamp": 0.3, "event_type": "QUEUE_GET_SUCCESS", "node_id": "concat_1", 
         "input_name": "input_b"},
        {"timestamp": 0.4, "event_type": "QUEUE_PUT", "node_id": "concat_1", 
         "output_name": "output", "data": {"merged": "data"}},
        {"timestamp": 0.5, "event_type": "QUEUE_GET_SUCCESS", "node_id": "network_1",
         "input_name": "input"},
        {"timestamp": 0.6, "event_type": "QUEUE_PUT", "node_id": "network_1",
         "output_name": "output", "data": {"processed": "data"}},
        # network_2 and tensor_2 are idle but that's OK - workflow completed
    ]
    
    # Create simulator
    sim = DataflowSimulator(graph)
    
    # Run simulation
    results = sim.replay_events(events)
    
    # Check results
    if not results['deadlock_detected']:
        print(f"✓ No false positive - correctly identified as non-deadlocked")
        return True
    else:
        print(f"❌ False positive detected at {results['deadlock_time']:.3f}s")
        return False

def test_partial_waiting():
    """Test when some nodes are waiting but system is still progressing"""
    print("\n=== Testing Partial Waiting (Not Deadlock) ===")
    
    # Create a larger graph where only some nodes are waiting
    graph = {
        "nodes": {
            "source_1": {"class": "TensorNode_1", "type": "tensor"},
            "source_2": {"class": "TensorNode_2", "type": "tensor"},
            "network_1": {"class": "NetworkNode_1", "type": "network"},
            "network_2": {"class": "NetworkNode_2", "type": "network"},
            "network_3": {"class": "NetworkNode_3", "type": "network"},
            "network_4": {"class": "NetworkNode_4", "type": "network"},
            "split_1": {"class": "SplitNode_1", "type": "split"},
            "concat_1": {"class": "ConcatNode_1", "type": "concat"}
        },
        "connections": [
            ["source_1", "tensor", "network_1", "input"],
            ["network_1", "output", "split_1", "input"],
            ["split_1", "output_a", "network_2", "input"],
            ["split_1", "output_b", "network_3", "input"]
        ]
    }
    
    # Events show partial execution - some nodes working, some idle
    events = [
        {"timestamp": 0.0, "event_type": "QUEUE_GET_WAIT", "node_id": "network_4", 
         "input_name": "input"},
        {"timestamp": 0.0, "event_type": "QUEUE_GET_WAIT", "node_id": "concat_1", 
         "input_name": "input_a"},
        {"timestamp": 0.1, "event_type": "QUEUE_PUT", "node_id": "source_1", 
         "output_name": "tensor", "data": {"value": "test"}},
        {"timestamp": 0.2, "event_type": "QUEUE_GET_SUCCESS", "node_id": "network_1", 
         "input_name": "input"},
        {"timestamp": 0.3, "event_type": "QUEUE_PUT", "node_id": "network_1", 
         "output_name": "output", "data": {"processed": "data"}},
        {"timestamp": 0.4, "event_type": "QUEUE_GET_SUCCESS", "node_id": "split_1",
         "input_name": "input"},
        {"timestamp": 0.5, "event_type": "QUEUE_PUT", "node_id": "split_1",
         "output_name": "output_a", "data": {"part": "a"}},
        {"timestamp": 0.5, "event_type": "QUEUE_PUT", "node_id": "split_1",
         "output_name": "output_b", "data": {"part": "b"}},
        {"timestamp": 0.6, "event_type": "QUEUE_GET_SUCCESS", "node_id": "network_2",
         "input_name": "input"},
        {"timestamp": 0.7, "event_type": "QUEUE_GET_SUCCESS", "node_id": "network_3",
         "input_name": "input"},
        # System completes - some nodes never ran but that's OK
    ]
    
    sim = DataflowSimulator(graph)
    results = sim.replay_events(events)
    
    # Should NOT detect deadlock since system made progress
    if not results['deadlock_detected']:
        print(f"✓ Correctly identified as progressing despite waiting node")
        print(f"  {len(results['waiting_nodes'])} nodes waiting (but not deadlocked)")
        return True
    else:
        print(f"❌ Incorrectly detected deadlock")
        return False

def main():
    """Run all detection tests"""
    print("="*60)
    print("DEADLOCK DETECTION TESTS")
    print("="*60)
    
    tests = [
        test_deadlock_scenario,
        test_no_deadlock_scenario,
        test_partial_waiting
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    if failed == 0:
        print(f"✅ SUCCESS: All {passed} detection tests passed!")
    else:
        print(f"❌ FAILURE: {failed} tests failed, {passed} passed")
    print("="*60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())