#!/usr/bin/env python3
"""
Analyze deadlock data from DNNE workflow execution.

Usage:
    python analyze_deadlock.py
    
Automatically:
- Loads graph and events from /tmp/dnne_deadlock_data/
- Converts data_flow.log to events.json if needed
- Runs deadlock analysis
- Shows results
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

from dataflow_simulator import DataflowSimulator

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors by default
    format='%(message)s'
)

def convert_logs_if_needed() -> bool:
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

def load_deadlock_data() -> Tuple[Dict, List]:
    """Load graph structure and events"""
    data_dir = Path('/tmp/dnne_deadlock_data')
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"❌ No deadlock data found at {data_dir}")
        print("\nTo generate deadlock data, run a workflow with:")
        print("  python runner.py --override all:retain_graph=True")
        return None, None
        
    # Load graph structure
    graph_path = data_dir / 'graph_structure.json'
    if not graph_path.exists():
        print(f"❌ Graph structure not found at {graph_path}")
        return None, None
        
    with open(graph_path, 'r') as f:
        graph = json.load(f)
        
    # Convert logs if needed
    if not convert_logs_if_needed():
        print(f"❌ No event data found")
        return graph, []
    
    # Load events
    events_path = data_dir / 'events.json'
    with open(events_path, 'r') as f:
        events = json.load(f)
        
    return graph, events

def analyze_deadlock(graph: Dict, events: List) -> Tuple:
    """Run deadlock analysis"""
    # Create simulator (suppress debug logs)
    simulator = DataflowSimulator(graph)
    
    # Check for bootstrap nodes
    bootstrap_nodes = simulator.check_bootstrap_nodes()
    
    # Run simulation
    results = simulator.replay_events(events)
    
    return results, simulator, bootstrap_nodes

def print_results(graph: Dict, events: List, results: Dict, simulator, bootstrap_nodes):
    """Display analysis results"""
    
    # Header
    print("\n" + "="*60)
    print("DEADLOCK ANALYSIS RESULTS")
    print("="*60)
    
    # Basic info
    print(f"\n📊 Workflow Info:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Connections: {len(graph['connections'])}")
    print(f"  Events: {len(events)}")
    if events:
        start_time = min(e.get('timestamp', 0) for e in events)
        end_time = max(e.get('timestamp', 0) for e in events)
        duration = end_time - start_time
        print(f"  Duration: {duration:.3f} seconds")
    else:
        start_time = 0
    
    # Bootstrap info
    if bootstrap_nodes:
        print(f"\n🚀 Bootstrap Nodes: {bootstrap_nodes}")
    else:
        print(f"\n⚠️  No bootstrap nodes detected")
    
    # Deadlock status
    print(f"\n🔍 Analysis:")
    if results['deadlock_detected']:
        relative_deadlock_time = results['deadlock_time'] - start_time if events else 0
        print(f"  ❌ DEADLOCK DETECTED at {relative_deadlock_time:.3f}s")
    else:
        print(f"  ✅ No deadlock detected")
    
    # Node states
    state_counts = {}
    for state in results['node_states'].values():
        state_counts[state] = state_counts.get(state, 0) + 1
    
    print(f"\n📈 Node States:")
    for state, count in sorted(state_counts.items()):
        print(f"  {state}: {count} nodes")
    
    # Waiting nodes (critical for deadlock analysis)
    if results['waiting_nodes']:
        print(f"\n⏳ Waiting Nodes ({len(results['waiting_nodes'])}):")
        
        # Group by node type
        by_type = {}
        for node_id, info in results['waiting_nodes'].items():
            node_class = info['class']
            base_type = node_class.rsplit('_', 1)[0]
            if base_type not in by_type:
                by_type[base_type] = []
            by_type[base_type].append((node_id, info))
        
        # Show grouped
        for node_type, nodes in sorted(by_type.items()):
            print(f"\n  {node_type} ({len(nodes)} nodes):")
            for node_id, info in nodes:
                waiting = info.get('waiting_for', [])
                print(f"    {node_id}: waiting for {waiting}")
    
    # Detailed deadlock analysis
    if results['deadlock_detected']:
        print("\n" + "="*60)
        print("DEADLOCK ROOT CAUSE ANALYSIS")
        print("="*60)
        
        # Get detailed state
        detailed = simulator.get_detailed_state()
        
        # Check for common patterns
        barriers = [nid for nid, info in detailed['nodes'].items() 
                   if 'Barrier' in info['class'] and info['state'] == 'WAITING']
        networks = [nid for nid, info in detailed['nodes'].items()
                   if 'Network' in info['class'] and info['state'] == 'WAITING']
        sgds = [nid for nid, info in detailed['nodes'].items()
              if 'SGD' in info['class'] and info['state'] == 'WAITING']
        
        if barriers and networks and sgds:
            print("\n🔄 Circular Dependency Detected:")
            print(f"  - {len(networks)} Networks waiting for Barriers")
            print(f"  - {len(barriers)} Barriers waiting for SGD triggers")
            print(f"  - {len(sgds)} SGDs waiting for loss from Networks")
            print("\n💡 Solution: Enable SGD bootstrap signals with:")
            print("  --override all:no_bootstrap_trigger=False")
        else:
            print("\n🔍 Complex deadlock pattern detected")
            print("  Further investigation needed")

def main():
    """Main entry point"""
    print("="*60)
    print("DNNE DEADLOCK ANALYZER")
    print("="*60)
    
    # Load data
    graph, events = load_deadlock_data()
    if not graph:
        return 1
    
    print(f"✓ Loaded graph with {len(graph['nodes'])} nodes")
    print(f"✓ Loaded {len(events)} events")
    
    # Run analysis
    print("\n🔄 Running deadlock analysis...")
    try:
        results, simulator, bootstrap_nodes = analyze_deadlock(graph, events)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1
    
    # Show results
    print_results(graph, events, results, simulator, bootstrap_nodes)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())