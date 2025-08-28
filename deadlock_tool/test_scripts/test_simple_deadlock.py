#!/usr/bin/env python3
"""
Test deadlock detection with a simple artificial deadlock scenario.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow_simulator import DataflowSimulator

def create_barrier_deadlock_scenario():
    """
    Create a simple deadlock with barriers and SGD optimizers.
    
    Scenario:
    - Network waits for Barrier output
    - Barrier waits for SGD trigger
    - SGD waits for loss from Network
    - Circular dependency!
    """
    graph = {
        "nodes": {
            "network_1": {"class": "NetworkNode_1", "type": "network"},
            "barrier_1": {"class": "BarrierNode_1", "type": "synchronization"},
            "sgd_1": {"class": "SGDOptimizerNode_1", "type": "optimizer", "no_bootstrap_trigger": True},
            "loss_1": {"class": "CustomComputationNode_1", "type": "computation"}
        },
        "connections": [
            # Barrier output feeds network
            ["barrier_1", "output", "network_1", "input"],
            # Network output goes to loss computation
            ["network_1", "output", "loss_1", "input"],
            # Loss goes to SGD
            ["loss_1", "output", "sgd_1", "loss"],
            # SGD triggers barrier
            ["sgd_1", "step_complete", "barrier_1", "release"]
        ]
    }
    
    # Create events showing the deadlock
    events = [
        # Barrier waits for data (which we provide)
        {"event_type": "QUEUE_PUT", "node_id": "external", "output_name": "data", 
         "timestamp": 0.1, "data": {"value": "initial_data"}},
        # But barrier also needs trigger from SGD
        {"event_type": "QUEUE_GET_WAIT", "node_id": "barrier_1", "input_name": "release", "timestamp": 0.2},
        # Network waits for barrier output
        {"event_type": "QUEUE_GET_WAIT", "node_id": "network_1", "input_name": "input", "timestamp": 0.3},
        # SGD waits for loss
        {"event_type": "QUEUE_GET_WAIT", "node_id": "sgd_1", "input_name": "loss", "timestamp": 0.4},
        # Loss waits for network output
        {"event_type": "QUEUE_GET_WAIT", "node_id": "loss_1", "input_name": "input", "timestamp": 0.5}
    ]
    
    # Add manual connection for external data to barrier
    graph["connections"].append(["external", "data", "barrier_1", "input"])
    
    return graph, events

def create_concat_starvation_scenario():
    """
    Create a scenario where Concat node waits forever for missing input.
    
    Scenario:
    - Concat needs 3 inputs
    - Only 2 sources provide data
    - Third source is deadlocked elsewhere
    """
    graph = {
        "nodes": {
            "source_a": {"class": "NetworkNode_A", "type": "network"},
            "source_b": {"class": "NetworkNode_B", "type": "network"},
            "source_c": {"class": "NetworkNode_C", "type": "network"},
            "concat_1": {"class": "ConcatNode_1", "type": "tensor_ops"},
            "barrier_1": {"class": "BarrierNode_1", "type": "synchronization"}
        },
        "connections": [
            # Sources feed concat
            ["source_a", "output", "concat_1", "input_a"],
            ["source_b", "output", "concat_1", "input_b"],
            ["source_c", "output", "concat_1", "input_c"],
            # Source C is stuck waiting for barrier that never triggers
            ["barrier_1", "output", "source_c", "input"]
        ]
    }
    
    events = [
        # Source A produces output
        {"event_type": "QUEUE_PUT", "node_id": "source_a", "output_name": "output",
         "timestamp": 0.1, "data": {"from": "A"}},
        # Source B produces output  
        {"event_type": "QUEUE_PUT", "node_id": "source_b", "output_name": "output",
         "timestamp": 0.2, "data": {"from": "B"}},
        # Concat receives A and B but waits for C
        {"event_type": "QUEUE_GET_SUCCESS", "node_id": "concat_1", "input_name": "input_a", "timestamp": 0.3},
        {"event_type": "QUEUE_GET_SUCCESS", "node_id": "concat_1", "input_name": "input_b", "timestamp": 0.4},
        {"event_type": "QUEUE_GET_WAIT", "node_id": "concat_1", "input_name": "input_c", "timestamp": 0.5},
        # Source C waits for barrier
        {"event_type": "QUEUE_GET_WAIT", "node_id": "source_c", "input_name": "input", "timestamp": 0.6},
        # Barrier waits for trigger that never comes
        {"event_type": "QUEUE_GET_WAIT", "node_id": "barrier_1", "input_name": "release", "timestamp": 0.7}
    ]
    
    return graph, events

def test_scenario(name, graph, events):
    """Test a deadlock scenario"""
    print(f"\n{'='*60}")
    print(f"TESTING: {name}")
    print(f"{'='*60}")
    
    # Create simulator
    simulator = DataflowSimulator(graph)
    
    # Run simulation
    results = simulator.replay_events(events)
    
    # Show results
    print(f"\n✓ Simulation complete")
    print(f"  Deadlock detected: {results['deadlock_detected']}")
    if results['deadlock_detected']:
        print(f"  Deadlock time: {results['deadlock_time']:.3f}s")
    
    # Show waiting nodes
    if results['waiting_nodes']:
        print(f"\n  Waiting nodes:")
        for node_id, info in results['waiting_nodes'].items():
            print(f"    {node_id}: waiting for {info['waiting_for']}")
            
    return results['deadlock_detected']

def main():
    """Run all test scenarios"""
    print("="*60)
    print("SIMPLE DEADLOCK DETECTION TESTS")
    print("="*60)
    
    passed = 0
    total = 0
    
    # Test 1: Barrier circular dependency
    graph, events = create_barrier_deadlock_scenario()
    total += 1
    if test_scenario("Barrier Circular Dependency", graph, events):
        passed += 1
        
    # Test 2: Concat starvation
    graph, events = create_concat_starvation_scenario()
    total += 1
    if test_scenario("Concat Starvation", graph, events):
        passed += 1
        
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} scenarios detected deadlocks")
    print(f"{'='*60}")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())