#!/usr/bin/env python3
"""
Test deadlock analysis on actual Franka_Coop_Nodes data.
"""

import sys
import os
import json
from pathlib import Path
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataflow_simulator import DataflowSimulator

def convert_logs_if_needed():
    """Convert data_flow.log to events.json if needed"""
    data_dir = Path('/tmp/dnne_deadlock_data')
    log_path = data_dir / 'data_flow.log'
    events_path = data_dir / 'events.json'
    
    # Check if conversion is needed
    if events_path.exists() and events_path.stat().st_mtime > log_path.stat().st_mtime:
        return True  # events.json is up to date
        
    if not log_path.exists():
        return False
        
    print(f"📝 Converting data_flow.log to events.json...")
    
    events = []
    
    # Read JSON Lines format
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                
                # Convert to our event format
                event = {
                    'timestamp': entry.get('ts', 0),
                    'node_id': entry.get('node', ''),
                }
                
                # Map event types
                entry_type = entry.get('type', '')
                
                if entry_type == 'QUEUE_GET_WAIT':
                    event['event_type'] = 'QUEUE_GET_WAIT'
                    event['input_name'] = entry.get('queue', 'input')
                    
                elif entry_type == 'QUEUE_GET_SUCCESS':
                    event['event_type'] = 'QUEUE_GET_SUCCESS'
                    event['input_name'] = entry.get('queue', 'input')
                    event['wait_time'] = entry.get('wait_time', 0)
                    
                elif entry_type == 'QUEUE_PUT':
                    event['event_type'] = 'QUEUE_PUT'
                    event['output_name'] = entry.get('queue', 'output')
                    event['data'] = entry.get('data', {})
                    
                elif entry_type == 'QUEUE_PUT_BLOCKED':
                    event['event_type'] = 'QUEUE_PUT_BLOCKED'
                    event['output_name'] = entry.get('queue', 'output')
                    
                elif entry_type == 'NODE_START':
                    event['event_type'] = 'NODE_START'
                    event['class_name'] = entry.get('class_name', '')
                    
                elif entry_type == 'NODE_EXECUTE':
                    event['event_type'] = 'NODE_EXECUTE'
                    
                elif entry_type == 'QUEUE_STATE':
                    event['event_type'] = 'QUEUE_STATE'
                    event['queue_depths'] = entry.get('queue_depths', {})
                    
                else:
                    # Keep original type for unmapped events
                    event['event_type'] = entry_type
                    event['data'] = entry
                    
                events.append(event)
                
            except json.JSONDecodeError:
                continue
                
    # Save as JSON array
    with open(events_path, 'w') as f:
        json.dump(events, f, indent=2)
        
    print(f"✓ Converted {len(events)} events")
    return True

def load_deadlock_data():
    """Load the graph structure and events from tmp directory"""
    data_dir = Path('/tmp/dnne_deadlock_data')
    
    # Load graph structure
    graph_path = data_dir / 'graph_structure.json'
    if not graph_path.exists():
        print(f"❌ Graph structure not found at {graph_path}")
        print("Please run Franka_Coop_Nodes with deadlock monitoring enabled first")
        return None, None
        
    with open(graph_path, 'r') as f:
        graph = json.load(f)
        
    print(f"✓ Loaded graph with {len(graph['nodes'])} nodes")
    
    # Convert logs if needed
    if not convert_logs_if_needed():
        print(f"❌ No event data found")
        print("Please run Franka_Coop_Nodes with deadlock monitoring enabled first")
        return graph, []
    
    # Load events
    events_path = data_dir / 'events.json'
    with open(events_path, 'r') as f:
        events = json.load(f)
        
    print(f"✓ Loaded {len(events)} events")
    
    return graph, events

def analyze_graph_structure(graph):
    """Analyze the graph structure before simulation"""
    print("\n" + "="*60)
    print("GRAPH STRUCTURE ANALYSIS")
    print("="*60)
    
    # Count node types
    node_types = {}
    for node_id, node_info in graph['nodes'].items():
        node_class = node_info.get('class', '')
        base_type = node_class.rsplit('_', 1)[0] if '_' in node_class else node_class
        node_types[base_type] = node_types.get(base_type, 0) + 1
        
    print("\nNode Types:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
        
    # Analyze connections
    print(f"\nTotal Connections: {len(graph['connections'])}")
    
    # Find nodes with multiple inputs (potential deadlock sources)
    multi_input_nodes = {}
    for conn in graph['connections']:
        target = conn[2]
        input_name = conn[3]
        if target not in multi_input_nodes:
            multi_input_nodes[target] = set()
        multi_input_nodes[target].add(input_name)
        
    print("\nNodes with Multiple Inputs (potential deadlock points):")
    for node_id, inputs in multi_input_nodes.items():
        if len(inputs) > 1:
            node_class = graph['nodes'][node_id]['class']
            print(f"  {node_id} ({node_class}): {len(inputs)} inputs - {inputs}")

def run_simulation(graph, events):
    """Run the dataflow simulation"""
    print("\n" + "="*60)
    print("RUNNING DATAFLOW SIMULATION")
    print("="*60)
    
    # Create simulator
    simulator = DataflowSimulator(graph)
    
    # Check initial state
    print("\nInitial State:")
    bootstrap_nodes = simulator.check_bootstrap_nodes()
    if bootstrap_nodes:
        print(f"  Bootstrap-capable nodes: {bootstrap_nodes}")
    else:
        print("  No bootstrap nodes found")
        
    # Run simulation
    print("\nReplaying events...")
    results = simulator.replay_events(events)
    
    return simulator, results

def analyze_results(simulator, results):
    """Analyze and display simulation results"""
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    print(f"\nDeadlock Detected: {results['deadlock_detected']}")
    if results['deadlock_detected']:
        print(f"Deadlock Time: {results['deadlock_time']:.3f}s")
    print(f"Total Simulation Time: {results['simulation_time']:.3f}s")
    print(f"Events Processed: {results['events_processed']}")
    
    # Show waiting nodes
    if results['waiting_nodes']:
        print(f"\nNodes Waiting ({len(results['waiting_nodes'])} nodes):")
        for node_id, info in results['waiting_nodes'].items():
            print(f"  {node_id} ({info['class']}):")
            print(f"    Waiting for: {info['waiting_for']}")
            
    # Show blocked nodes
    if results['blocked_nodes']:
        print(f"\nBlocked Nodes: {results['blocked_nodes']}")
        
    # Show node states summary
    state_counts = {}
    for state in results['node_states'].values():
        state_counts[state] = state_counts.get(state, 0) + 1
    print(f"\nNode States Summary:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} nodes")
        
    # Get detailed state for deadlock analysis
    if results['deadlock_detected']:
        print("\n" + "="*60)
        print("DEADLOCK ANALYSIS")
        print("="*60)
        
        detailed = simulator.get_detailed_state()
        
        # Find critical waiting patterns
        print("\nCritical Waiting Patterns:")
        
        # Look for barriers waiting
        barriers_waiting = []
        eat_n_waiting = []
        concat_waiting = []
        
        for node_id, state_info in detailed['nodes'].items():
            if 'BarrierNode' in state_info['class'] and state_info['state'] == 'WAITING':
                barriers_waiting.append((node_id, state_info))
            elif 'Eat_NNode' in state_info['class'] and state_info['state'] == 'WAITING':
                eat_n_waiting.append((node_id, state_info))
            elif 'ConcatNode' in state_info['class'] and state_info['state'] == 'WAITING':
                concat_waiting.append((node_id, state_info))
                
        if barriers_waiting:
            print(f"\n  Barriers Waiting ({len(barriers_waiting)}):")
            for node_id, info in barriers_waiting:
                waiting_for = info.get('waiting_for', [])
                has_data = info.get('has_data', False)
                has_trigger = info.get('has_trigger', False)
                print(f"    {node_id}: has_data={has_data}, has_trigger={has_trigger}, waiting={waiting_for}")
                
        if eat_n_waiting:
            print(f"\n  Eat_N Nodes Waiting ({len(eat_n_waiting)}):")
            for node_id, info in eat_n_waiting:
                mode = info.get('mode', 'unknown')
                print(f"    {node_id}: mode={mode}, waiting={info.get('waiting_for', [])}")
                
        if concat_waiting:
            print(f"\n  Concat Nodes Waiting ({len(concat_waiting)}):")
            for node_id, info in concat_waiting:
                expected = info.get('inputs_expected', 0)
                received = info.get('inputs_received', 0)
                missing = info.get('missing_inputs', [])
                print(f"    {node_id}: {received}/{expected} inputs, missing={missing}")

def main():
    """Main test function"""
    print("="*60)
    print("FRANKA COOP NODES DEADLOCK ANALYSIS")
    print("="*60)
    
    # Load data
    graph, events = load_deadlock_data()
    if not graph:
        return 1
        
    # Analyze structure
    analyze_graph_structure(graph)
    
    # Run simulation
    if events:
        simulator, results = run_simulation(graph, events)
        analyze_results(simulator, results)
    else:
        print("\n⚠️ No events to replay - cannot perform deadlock analysis")
        print("Please run the Franka_Coop_Nodes workflow to generate event data")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())