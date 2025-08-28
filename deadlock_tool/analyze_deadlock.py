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
from collections import defaultdict

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
                    event['output_name'] = entry.get('output', 'output')
                    event['data'] = entry.get('data', {})
                    
                elif entry_type == 'QUEUE_PUT_BLOCKED':
                    event['event_type'] = 'QUEUE_PUT_BLOCKED'
                    event['output_name'] = entry.get('output', 'output')
                    
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

def extract_execution_cycles(events: List[Dict], graph: Dict = None) -> List[Dict]:
    """
    Extract execution cycles from events.
    For IsaacGym workflows: uses simulation steps as markers
    For other workflows: uses generic pattern detection
    """
    cycles = []
    current_cycle = {
        'start_time': None,
        'end_time': None,
        'isaac_steps': [],
        'network_forwards': [],
        'sgd_optimizations': [],
        'barrier_releases': [],
        'nodes_executed': set()
    }
    
    start_time = events[0]['timestamp'] if events else 0
    
    # Find IsaacGym node ID from graph
    isaac_node_id = None
    if graph:
        for node_id, config in graph.get('nodes', {}).items():
            if 'IsaacGym' in config.get('class', ''):
                isaac_node_id = node_id
                break
    
    # Also identify other node types from graph
    network_nodes = set()
    sgd_nodes = set()
    barrier_nodes = set()
    if graph:
        for node_id, config in graph.get('nodes', {}).items():
            node_class = config.get('class', '')
            if 'Network' in node_class:
                network_nodes.add(node_id)
            elif 'SGD' in node_class:
                sgd_nodes.add(node_id)
            elif 'Barrier' in node_class:
                barrier_nodes.add(node_id)
    
    for event in events:
        node_id = event.get('node_id', '')
        event_type = event.get('event_type', '')
        rel_time = event['timestamp'] - start_time
        
        # Track IsaacGym simulation steps (marks new cycle)
        if isaac_node_id and node_id == isaac_node_id and event_type == 'QUEUE_PUT':
            if current_cycle['isaac_steps']:
                # Save previous cycle
                cycles.append(current_cycle)
                current_cycle = {
                    'start_time': rel_time,
                    'end_time': rel_time,
                    'isaac_steps': [],
                    'network_forwards': [],
                    'sgd_optimizations': [],
                    'barrier_releases': [],
                    'nodes_executed': set()
                }
            else:
                current_cycle['start_time'] = rel_time
            
            current_cycle['isaac_steps'].append((rel_time, node_id))
            current_cycle['nodes_executed'].add(node_id)
        
        # Track other node executions
        elif event_type == 'QUEUE_PUT':
            current_cycle['nodes_executed'].add(node_id)
            # Categorize by node type using graph info
            if node_id in network_nodes:
                current_cycle['network_forwards'].append((rel_time, node_id))
            elif node_id in sgd_nodes:
                current_cycle['sgd_optimizations'].append((rel_time, node_id))
            elif node_id in barrier_nodes:
                current_cycle['barrier_releases'].append((rel_time, node_id))
            
        current_cycle['end_time'] = rel_time
    
    # Add last cycle if it has content
    if current_cycle['nodes_executed']:
        cycles.append(current_cycle)
    
    # If no IsaacGym node found, try generic cycle detection
    if not isaac_node_id and len(cycles) == 0:
        cycles = extract_generic_cycles(events, graph)
    
    return cycles

def extract_generic_cycles(events: List[Dict], graph: Dict) -> List[Dict]:
    """
    Generic cycle extraction for non-IsaacGym workflows.
    Uses pattern detection to identify repeating execution cycles.
    """
    if not events:
        return []
    
    # Find nodes with regular output patterns
    node_output_times = defaultdict(list)
    start_time = events[0]['timestamp']
    
    for event in events:
        if event['event_type'] == 'QUEUE_PUT':
            node_id = event['node_id']
            rel_time = event['timestamp'] - start_time
            node_output_times[node_id].append(rel_time)
    
    # Find the most regular node to use as cycle marker
    best_marker = None
    best_regularity = float('inf')
    
    for node_id, times in node_output_times.items():
        if len(times) < 3:
            continue
        
        # Calculate variance in intervals
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                regularity = variance / avg_interval
                if regularity < best_regularity:
                    best_regularity = regularity
                    best_marker = node_id
    
    # Build cycles using the marker
    cycles = []
    current_cycle = {
        'start_time': 0,
        'end_time': 0,
        'nodes_executed': set(),
        'isaac_steps': [],  # Empty for compatibility
        'network_forwards': [],
        'sgd_optimizations': [],
        'barrier_releases': []
    }
    
    for event in events:
        node_id = event['node_id']
        event_type = event['event_type']
        rel_time = event['timestamp'] - start_time
        
        # Check for cycle boundary
        if best_marker and node_id == best_marker and event_type == 'QUEUE_PUT':
            if current_cycle['nodes_executed']:
                current_cycle['end_time'] = rel_time
                cycles.append(current_cycle)
                current_cycle = {
                    'start_time': rel_time,
                    'end_time': rel_time,
                    'nodes_executed': set(),
                    'isaac_steps': [],
                    'network_forwards': [],
                    'sgd_optimizations': [],
                    'barrier_releases': []
                }
        
        if event_type == 'QUEUE_PUT':
            current_cycle['nodes_executed'].add(node_id)
        
        current_cycle['end_time'] = rel_time
    
    # Add last cycle
    if current_cycle['nodes_executed']:
        cycles.append(current_cycle)
    
    return cycles

def analyze_pattern_break(events: List[Dict], graph: Dict) -> Dict:
    """
    Analyze where the execution pattern breaks, causing deadlock.
    Returns detailed information about the pattern break.
    """
    if not events:
        return {}
    
    cycles = extract_execution_cycles(events, graph)
    if len(cycles) < 2:
        return {'message': 'Not enough cycles to detect pattern break'}
    
    # Find the critical nodes that stopped
    start_time = events[0]['timestamp']
    
    # Get last events for each node
    last_events = {}
    for event in events:
        node_id = event['node_id']
        last_events[node_id] = event
    
    # Find nodes that didn't complete their pattern
    missing_nodes = []
    if len(cycles) >= 2:
        last_cycle_nodes = cycles[-1]['nodes_executed']
        prev_cycle_nodes = cycles[-2]['nodes_executed']
        missing_nodes = list(prev_cycle_nodes - last_cycle_nodes)
    
    # Find the critical break point
    critical_node = None
    critical_time = None
    
    # Look for IsaacGym node that didn't continue
    for node_id, config in graph['nodes'].items():
        if 'IsaacGym' in config.get('class', ''):
            if node_id in last_events:
                last_event = last_events[node_id]
                if last_event['event_type'] == 'QUEUE_PUT':
                    # IsaacGym produced output but didn't wait for next input
                    critical_node = node_id
                    critical_time = last_event['timestamp'] - start_time
                    break
    
    # Build pattern break analysis
    analysis = {
        'total_cycles': len(cycles),
        'last_cycle_incomplete': len(cycles[-1]['nodes_executed']) < len(cycles[-2]['nodes_executed']) if len(cycles) >= 2 else False,
        'missing_nodes': missing_nodes,
        'critical_node': critical_node,
        'critical_time': critical_time,
        'avg_cycle_duration': sum(c['end_time'] - c['start_time'] for c in cycles[:-1]) / max(1, len(cycles) - 1) if len(cycles) > 1 else 0,
        'last_cycle_duration': cycles[-1]['end_time'] - cycles[-1]['start_time'] if cycles else 0
    }
    
    return analysis

def analyze_deadlock(graph: Dict, events: List) -> Tuple:
    """Run deadlock analysis"""
    # Create simulator (suppress debug logs)
    simulator = DataflowSimulator(graph)
    
    # Check for bootstrap nodes
    bootstrap_nodes = simulator.check_bootstrap_nodes()
    
    # Run simulation
    results = simulator.replay_events(events)
    
    # Add pattern break analysis if deadlock detected
    if results['deadlock_detected']:
        results['pattern_break'] = analyze_pattern_break(events, graph)
    
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
        
        # Pattern break analysis
        if 'pattern_break' in results:
            pb = results['pattern_break']
            if pb.get('total_cycles'):
                print(f"\n🔄 Pattern Analysis:")
                print(f"  Completed cycles: {pb['total_cycles'] - 1}")
                print(f"  Average cycle duration: {pb['avg_cycle_duration']:.3f}s")
                print(f"  Last cycle duration: {pb['last_cycle_duration']:.3f}s")
                
                if pb['last_cycle_incomplete']:
                    print(f"  ❌ Last cycle was incomplete")
                
                if pb['missing_nodes']:
                    print(f"\n  ⚠️ Nodes that didn't execute in final cycle:")
                    for node_id in pb['missing_nodes']:
                        node_class = graph['nodes'].get(node_id, {}).get('class', 'Unknown')
                        print(f"    - Node {node_id} ({node_class})")
                
                if pb['critical_node']:
                    critical_class = graph['nodes'].get(pb['critical_node'], {}).get('class', 'Unknown')
                    print(f"\n  🎯 Critical failure point:")
                    print(f"    Node {pb['critical_node']} ({critical_class}) at t={pb['critical_time']:.3f}s")
                    print(f"    This node produced output but didn't continue its cycle")
        
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