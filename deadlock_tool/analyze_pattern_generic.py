#!/usr/bin/env python3
"""
Generic pattern break analysis for any workflow type.
Detects repeating patterns and identifies where they break.
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional

def find_cycle_markers(events: List[Dict], graph: Dict) -> Set[str]:
    """
    Identify nodes that could mark cycle boundaries.
    These are typically nodes that:
    1. Have periodic execution patterns
    2. Are data sources or sinks
    3. Have the most regular output patterns
    """
    if not events:
        return set()
    
    # Count output frequencies for each node
    node_output_times = defaultdict(list)
    start_time = events[0]['timestamp']
    
    for event in events:
        if event['event_type'] == 'QUEUE_PUT':
            node_id = event['node_id']
            rel_time = event['timestamp'] - start_time
            node_output_times[node_id].append(rel_time)
    
    # Find nodes with regular patterns
    cycle_markers = set()
    
    for node_id, times in node_output_times.items():
        if len(times) < 3:
            continue
            
        # Calculate intervals between outputs
        intervals = []
        for i in range(1, len(times)):
            intervals.append(times[i] - times[i-1])
        
        if not intervals:
            continue
            
        # Check for regularity (low variance in intervals)
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        
        # If variance is low relative to average, it's regular
        if avg_interval > 0 and variance / avg_interval < 0.5:
            cycle_markers.add(node_id)
    
    # If no regular nodes found, use nodes with most outputs
    if not cycle_markers and node_output_times:
        max_outputs = max(len(times) for times in node_output_times.values())
        for node_id, times in node_output_times.items():
            if len(times) >= max_outputs * 0.8:  # Within 80% of max
                cycle_markers.add(node_id)
    
    return cycle_markers

def extract_generic_cycles(events: List[Dict], graph: Dict) -> List[Dict]:
    """
    Extract execution cycles using generic pattern detection.
    Works with any workflow type, not just IsaacGym.
    """
    if not events:
        return []
    
    # Find potential cycle markers
    cycle_markers = find_cycle_markers(events, graph)
    
    if not cycle_markers:
        # No clear cycles, treat entire execution as one cycle
        return [{
            'start_time': 0,
            'end_time': events[-1]['timestamp'] - events[0]['timestamp'],
            'nodes_executed': set(e['node_id'] for e in events if e['event_type'] == 'QUEUE_PUT'),
            'event_count': len(events)
        }]
    
    # Extract cycles based on marker nodes
    cycles = []
    current_cycle = {
        'start_time': 0,
        'end_time': 0,
        'nodes_executed': set(),
        'event_count': 0,
        'marker_node': None
    }
    
    start_time = events[0]['timestamp']
    
    for event in events:
        node_id = event['node_id']
        event_type = event['event_type']
        rel_time = event['timestamp'] - start_time
        
        # Check if this is a cycle boundary
        if node_id in cycle_markers and event_type == 'QUEUE_PUT':
            if current_cycle['nodes_executed'] and current_cycle['marker_node'] == node_id:
                # Same marker again - new cycle
                current_cycle['end_time'] = rel_time
                cycles.append(current_cycle)
                current_cycle = {
                    'start_time': rel_time,
                    'end_time': rel_time,
                    'nodes_executed': set(),
                    'event_count': 0,
                    'marker_node': node_id
                }
            elif not current_cycle['marker_node']:
                # First marker
                current_cycle['marker_node'] = node_id
                current_cycle['start_time'] = rel_time
        
        # Track all node executions
        if event_type == 'QUEUE_PUT':
            current_cycle['nodes_executed'].add(node_id)
        
        current_cycle['event_count'] += 1
        current_cycle['end_time'] = rel_time
    
    # Add last cycle
    if current_cycle['nodes_executed']:
        cycles.append(current_cycle)
    
    return cycles

def detect_pattern_anomalies(cycles: List[Dict]) -> Dict:
    """
    Detect anomalies in cycle patterns.
    Generic approach that works for any workflow.
    """
    if len(cycles) < 2:
        return {'has_anomaly': False, 'reason': 'Not enough cycles'}
    
    # Calculate statistics for normal cycles (exclude last one)
    normal_cycles = cycles[:-1]
    if not normal_cycles:
        return {'has_anomaly': False, 'reason': 'No complete cycles'}
    
    # Average nodes per cycle
    avg_nodes = sum(len(c['nodes_executed']) for c in normal_cycles) / len(normal_cycles)
    
    # Average duration
    avg_duration = sum(c['end_time'] - c['start_time'] for c in normal_cycles) / len(normal_cycles)
    
    # Check last cycle for anomalies
    last_cycle = cycles[-1]
    last_duration = last_cycle['end_time'] - last_cycle['start_time']
    last_nodes = len(last_cycle['nodes_executed'])
    
    anomalies = []
    
    # Check for incomplete execution (fewer nodes)
    if last_nodes < avg_nodes * 0.8:  # 20% fewer nodes
        anomalies.append(f"Incomplete execution: {last_nodes} nodes vs normal {avg_nodes:.0f}")
    
    # Check for timing anomaly
    if avg_duration > 0 and last_duration < avg_duration * 0.5:  # 50% shorter
        anomalies.append(f"Short cycle: {last_duration:.3f}s vs normal {avg_duration:.3f}s")
    
    # Find missing nodes
    if len(normal_cycles) > 0:
        typical_nodes = normal_cycles[-1]['nodes_executed'] if normal_cycles else set()
        missing_nodes = typical_nodes - last_cycle['nodes_executed']
        if missing_nodes:
            anomalies.append(f"Missing nodes: {missing_nodes}")
    
    return {
        'has_anomaly': len(anomalies) > 0,
        'anomalies': anomalies,
        'avg_duration': avg_duration,
        'avg_nodes': avg_nodes,
        'last_duration': last_duration,
        'last_nodes': last_nodes,
        'missing_nodes': list(missing_nodes) if 'missing_nodes' in locals() else []
    }

def analyze_generic_pattern_break(events: List[Dict], graph: Dict) -> Dict:
    """
    Generic pattern break analysis that works for any workflow type.
    """
    cycles = extract_generic_cycles(events, graph)
    
    if not cycles:
        return {'message': 'No execution cycles detected'}
    
    anomalies = detect_pattern_anomalies(cycles)
    
    # Find critical failure point (last node to successfully execute)
    last_success = None
    if events:
        for event in reversed(events):
            if event['event_type'] == 'QUEUE_PUT':
                last_success = event['node_id']
                break
    
    return {
        'total_cycles': len(cycles),
        'cycle_info': {
            'count': len(cycles),
            'avg_duration': anomalies.get('avg_duration', 0),
            'avg_nodes': anomalies.get('avg_nodes', 0)
        },
        'anomaly_detected': anomalies['has_anomaly'],
        'anomalies': anomalies.get('anomalies', []),
        'missing_nodes': anomalies.get('missing_nodes', []),
        'last_successful_node': last_success,
        'is_generic_analysis': True  # Flag to indicate this is generic
    }

def main():
    """Test the generic analysis"""
    print("="*60)
    print("GENERIC PATTERN ANALYSIS TEST")
    print("="*60)
    
    # Load test data
    data_dir = Path('/tmp/dnne_deadlock_data')
    
    if not data_dir.exists():
        print("No test data found")
        return
    
    with open(data_dir / 'graph_structure.json', 'r') as f:
        graph = json.load(f)
    
    with open(data_dir / 'events.json', 'r') as f:
        events = json.load(f)
    
    print(f"\nAnalyzing {len(events)} events from {len(graph['nodes'])} nodes...")
    
    # Run generic analysis
    result = analyze_generic_pattern_break(events, graph)
    
    print(f"\n📊 Generic Pattern Analysis:")
    print(f"  Total cycles detected: {result['total_cycles']}")
    
    if result.get('cycle_info'):
        info = result['cycle_info']
        print(f"  Average cycle duration: {info['avg_duration']:.3f}s")
        print(f"  Average nodes per cycle: {info['avg_nodes']:.0f}")
    
    if result.get('anomaly_detected'):
        print(f"\n⚠️  Anomalies detected:")
        for anomaly in result.get('anomalies', []):
            print(f"  - {anomaly}")
    
    if result.get('missing_nodes'):
        print(f"\n❌ Nodes that didn't execute in last cycle:")
        for node_id in result['missing_nodes']:
            node_class = graph['nodes'].get(node_id, {}).get('class', 'Unknown')
            print(f"  - {node_id} ({node_class})")
    
    if result.get('last_successful_node'):
        node_class = graph['nodes'].get(result['last_successful_node'], {}).get('class', 'Unknown')
        print(f"\n🎯 Last successful execution: {result['last_successful_node']} ({node_class})")

if __name__ == "__main__":
    main()