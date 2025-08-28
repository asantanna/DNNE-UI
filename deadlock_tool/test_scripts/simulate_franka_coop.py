#!/usr/bin/env python3
"""
Simulate Franka_Coop_Nodes execution to understand deadlock pattern.
This generates synthetic events based on node behaviors.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow_simulator import DataflowSimulator

def load_graph():
    """Load the Franka_Coop_Nodes graph structure"""
    graph_path = Path('/tmp/dnne_deadlock_data/graph_structure.json')
    if not graph_path.exists():
        print(f"❌ Graph not found at {graph_path}")
        return None
        
    with open(graph_path, 'r') as f:
        return json.load(f)

def simulate_with_bootstrap(graph):
    """Simulate with SGD bootstrap enabled (should work)"""
    print("\n" + "="*60)
    print("SIMULATION WITH BOOTSTRAP (Should Work)")
    print("="*60)
    
    # Enable bootstrap for SGD nodes
    for node_id, node_info in graph['nodes'].items():
        if 'SGDOptimizerNode' in node_info['class']:
            node_info['no_bootstrap_trigger'] = False
            print(f"  ✓ Bootstrap enabled for {node_id}")
            
    # Create simulator
    simulator = DataflowSimulator(graph)
    
    # Check bootstrap capability
    bootstrap_nodes = simulator.check_bootstrap_nodes()
    print(f"\n  Bootstrap nodes: {bootstrap_nodes}")
    
    # Create synthetic events to start the system
    events = []
    
    # SGD nodes send bootstrap signals
    for node_id in ['40', '49', '59', '81']:
        if node_id in graph['nodes']:
            events.append({
                "event_type": "QUEUE_PUT",
                "node_id": node_id,
                "output_name": "step_complete",
                "timestamp": 0.1,
                "data": {"signal": "bootstrap", "step": 0}
            })
            
    # IsaacGym bootstraps with null action
    events.append({
        "event_type": "QUEUE_PUT",
        "node_id": "25",
        "output_name": "observation",
        "timestamp": 0.2,
        "data": {"obs": "initial", "bootstrap": True}
    })
    
    # Run for a few cycles
    results = simulator.replay_events(events)
    
    print(f"\n  Results:")
    print(f"    Deadlock: {results['deadlock_detected']}")
    print(f"    Events processed: {results['events_processed']}")
    print(f"    Nodes waiting: {len(results['waiting_nodes'])}")
    
    return results

def simulate_without_bootstrap(graph):
    """Simulate without SGD bootstrap (should deadlock)"""
    print("\n" + "="*60)
    print("SIMULATION WITHOUT BOOTSTRAP (Should Deadlock)")
    print("="*60)
    
    # Disable bootstrap for SGD nodes
    for node_id, node_info in graph['nodes'].items():
        if 'SGDOptimizerNode' in node_info['class']:
            node_info['no_bootstrap_trigger'] = True
            print(f"  ✗ Bootstrap disabled for {node_id}")
            
    # Create simulator with shorter deadlock timeout
    simulator = DataflowSimulator(graph)
    simulator.deadlock_timeout = 1.0  # Detect deadlock faster
    
    # Check bootstrap capability  
    bootstrap_nodes = simulator.check_bootstrap_nodes()
    print(f"\n  Bootstrap nodes: {bootstrap_nodes}")
    
    # Create events - IsaacGym still bootstraps
    events = []
    
    # IsaacGym bootstraps with null action
    events.append({
        "event_type": "QUEUE_PUT",
        "node_id": "25",
        "output_name": "observation",
        "timestamp": 0.1,
        "data": {"obs": "initial", "bootstrap": True}
    })
    
    # Observation goes to Split nodes and Eat_N
    events.append({
        "event_type": "QUEUE_GET_SUCCESS",
        "node_id": "45",  # SplitNode
        "input_name": "input",
        "timestamp": 0.2
    })
    events.append({
        "event_type": "QUEUE_GET_SUCCESS",
        "node_id": "56",  # SplitNode
        "input_name": "input",
        "timestamp": 0.2
    })
    events.append({
        "event_type": "QUEUE_GET_SUCCESS",
        "node_id": "73",  # Eat_N
        "input_name": "input",
        "timestamp": 0.2
    })
    
    # Split nodes produce outputs
    for split_id in ["45", "56"]:
        for output in ["output_a", "output_b", "output_c"]:
            events.append({
                "event_type": "QUEUE_PUT",
                "node_id": split_id,
                "output_name": output,
                "timestamp": 0.3,
                "data": {"split": split_id, "output": output}
            })
            
    # Concat nodes receive split outputs but wait for all inputs
    concat_nodes = ["47", "55", "57"]
    for concat_id in concat_nodes:
        events.append({
            "event_type": "QUEUE_GET_WAIT",
            "node_id": concat_id,
            "input_name": "input_c",  # Still waiting for one input
            "timestamp": 0.4
        })
        
    # Barriers wait for both data and triggers
    barrier_nodes = ["74", "75", "76"]
    for barrier_id in barrier_nodes:
        # Data arrives at barriers from concat
        events.append({
            "event_type": "QUEUE_GET_SUCCESS",
            "node_id": barrier_id,
            "input_name": "input",
            "timestamp": 0.5
        })
        # But barriers wait for release trigger from SGD
        events.append({
            "event_type": "QUEUE_GET_WAIT",
            "node_id": barrier_id,
            "input_name": "release",
            "timestamp": 0.6
        })
        
    # Networks wait for barrier outputs
    network_nodes = ["33", "54", "62"]
    for net_id in network_nodes:
        events.append({
            "event_type": "QUEUE_GET_WAIT",
            "node_id": net_id,
            "input_name": "input",
            "timestamp": 0.7
        })
        
    # SGD optimizers wait for loss
    sgd_nodes = ["40", "49", "59", "81"]
    for sgd_id in sgd_nodes:
        if sgd_id in graph['nodes']:
            events.append({
                "event_type": "QUEUE_GET_WAIT",
                "node_id": sgd_id,
                "input_name": "loss",
                "timestamp": 0.8
            })
            
    # This creates the deadlock:
    # - Networks wait for barriers
    # - Barriers wait for SGD triggers
    # - SGDs wait for loss from networks
    # Circular dependency!
    
    # Add a long wait to trigger deadlock detection
    events.append({
        "event_type": "QUEUE_GET_WAIT",
        "node_id": "25",  # IsaacGym waits for action
        "input_name": "action",
        "timestamp": 2.0  # After timeout
    })
    
    # Run simulation
    results = simulator.replay_events(events)
    
    print(f"\n  Results:")
    print(f"    Deadlock: {results['deadlock_detected']}")
    if results['deadlock_detected']:
        print(f"    Deadlock time: {results['deadlock_time']:.3f}s")
    print(f"    Events processed: {results['events_processed']}")
    print(f"    Nodes waiting: {len(results['waiting_nodes'])}")
    
    if results['waiting_nodes']:
        print(f"\n  Key waiting patterns:")
        # Show barriers
        barriers_waiting = [nid for nid in results['waiting_nodes'] 
                           if 'Barrier' in results['waiting_nodes'][nid]['class']]
        if barriers_waiting:
            print(f"    Barriers waiting for triggers: {barriers_waiting}")
            
        # Show networks
        networks_waiting = [nid for nid in results['waiting_nodes']
                          if 'Network' in results['waiting_nodes'][nid]['class']]
        if networks_waiting:
            print(f"    Networks waiting for barriers: {networks_waiting}")
            
        # Show SGDs
        sgds_waiting = [nid for nid in results['waiting_nodes']
                       if 'SGD' in results['waiting_nodes'][nid]['class']]
        if sgds_waiting:
            print(f"    SGDs waiting for loss: {sgds_waiting}")
    
    return results

def main():
    """Main simulation"""
    print("="*60)
    print("FRANKA COOP NODES DEADLOCK SIMULATION")
    print("="*60)
    
    # Load graph
    graph = load_graph()
    if not graph:
        return 1
        
    print(f"✓ Loaded graph with {len(graph['nodes'])} nodes")
    
    # Test both scenarios
    
    # 1. With bootstrap - should work
    results_with = simulate_with_bootstrap(graph.copy())
    
    # 2. Without bootstrap - should deadlock
    results_without = simulate_without_bootstrap(graph.copy())
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"With Bootstrap: {'✓ No deadlock' if not results_with['deadlock_detected'] else '✗ Deadlocked'}")
    print(f"Without Bootstrap: {'✓ Deadlocked as expected' if results_without['deadlock_detected'] else '✗ Did not deadlock'}")
    
    # Analysis
    if results_without['deadlock_detected']:
        print("\n🎯 Root Cause Analysis:")
        print("  The deadlock occurs because:")
        print("  1. Networks (33,54,62) wait for Barriers (74,75,76)")
        print("  2. Barriers wait for triggers from SGD optimizers (40,49,59)")
        print("  3. SGD optimizers wait for loss computed from Network outputs")
        print("  4. This creates a circular dependency!")
        print("\n  Solution: Enable SGD bootstrap signals to break the cycle")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())