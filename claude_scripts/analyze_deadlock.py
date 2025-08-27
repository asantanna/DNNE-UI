#!/usr/bin/env python3
"""
DNNE Deadlock Analysis Tool

Analyzes deadlock data collected with --debug-deadlock flag to identify
stuck nodes, circular dependencies, and bootstrap issues.

Usage:
    python analyze_deadlock.py [--data-dir /path/to/data] [--verbose]
"""

import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque


class DeadlockAnalyzer:
    """Analyzes DNNE deadlock data to identify issues"""
    
    def __init__(self, data_dir: str = "/tmp/dnne_deadlock_data", verbose: bool = False):
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.events = []
        self.graph = {}
        self.connections = []
        self.node_configs = {}
        self.node_last_activity = {}
        self.node_wait_status = {}
        self.node_classes = {}
        
    def load_data(self) -> bool:
        """Load all data files"""
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            print("    Make sure to run workflow with --debug-deadlock flag first")
            return False
        
        # Load event log
        log_file = self.data_dir / "data_flow.log"
        if not log_file.exists():
            print(f"❌ Event log not found: {log_file}")
            return False
        
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        if self.verbose:
                            print(f"Warning: Failed to parse line: {e}")
        
        if not self.events:
            print("❌ No events found in log file")
            return False
        
        # Load graph structure
        graph_file = self.data_dir / "graph_structure.json"
        if graph_file.exists():
            with open(graph_file) as f:
                data = json.load(f)
                self.graph = data.get("nodes", {})
                self.connections = data.get("connections", [])
        
        # Load node configs
        config_file = self.data_dir / "node_configs.json"
        if config_file.exists():
            with open(config_file) as f:
                self.node_configs = json.load(f)
        
        return True
    
    def analyze(self):
        """Perform complete analysis"""
        print("\nDNNE Deadlock Analysis Report")
        print("=" * 60)
        print(f"Data from: {self.data_dir}")
        
        # Basic statistics
        self.analyze_events()
        
        # Node activity analysis
        self.analyze_node_activity()
        
        # Queue analysis
        self.analyze_queue_operations()
        
        # Detect specific issues
        issues = []
        
        # Check for bootstrap issues
        bootstrap_issues = self.check_bootstrap_issues()
        if bootstrap_issues:
            issues.extend(bootstrap_issues)
        
        # Check for circular dependencies
        circular = self.check_circular_dependencies()
        if circular:
            issues.extend(circular)
        
        # Check for stuck nodes
        stuck = self.check_stuck_nodes()
        if stuck:
            issues.extend(stuck)
        
        # Report issues
        if issues:
            print("\nDetected Issues:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("\n✅ No obvious deadlock issues detected")
        
        # Recommendations
        self.generate_recommendations()
    
    def analyze_events(self):
        """Analyze event timeline"""
        if not self.events:
            return
        
        start_time = self.events[0]["ts"]
        end_time = self.events[-1]["ts"]
        duration = end_time - start_time
        
        print(f"Time range: 0.00s - {duration:.2f}s")
        print(f"Total events: {len(self.events):,}")
        
        # Count event types
        event_counts = defaultdict(int)
        for event in self.events:
            event_counts[event["type"]] += 1
        
        if self.verbose:
            print("\nEvent breakdown:")
            for event_type, count in sorted(event_counts.items()):
                print(f"  {event_type}: {count}")
    
    def analyze_node_activity(self):
        """Analyze node activity patterns"""
        # Track last activity and wait status for each node
        for event in self.events:
            node_id = event.get("node")
            if node_id:
                self.node_last_activity[node_id] = event["ts"]
                
                # Track node class
                if event["type"] == "NODE_START":
                    self.node_classes[node_id] = event.get("class_name", "Unknown")
                
                # Track wait status
                if event["type"] == "QUEUE_GET_WAIT":
                    self.node_wait_status[node_id] = f"waiting for '{event['queue']}'"
                elif event["type"] == "QUEUE_GET_SUCCESS":
                    self.node_wait_status[node_id] = "active"
                elif event["type"] == "NODE_COMPUTE_START":
                    self.node_wait_status[node_id] = "computing"
                elif event["type"] == "NODE_COMPUTE_END":
                    self.node_wait_status[node_id] = "active"
        
        # Calculate time since last activity
        if self.events:
            current_time = self.events[-1]["ts"]
            
            print("\nNode Activity Summary:")
            inactive_nodes = []
            
            for node_id in sorted(self.node_last_activity.keys()):
                last_activity = self.node_last_activity[node_id]
                time_since = current_time - last_activity
                class_name = self.node_classes.get(node_id, "Unknown")
                status = self.node_wait_status.get(node_id, "unknown")
                
                if time_since > 1.0:  # More than 1 second inactive
                    inactive_nodes.append((node_id, time_since, class_name, status))
            
            # Show most inactive nodes
            if inactive_nodes:
                for node_id, idle_time, class_name, status in sorted(inactive_nodes, key=lambda x: x[1], reverse=True)[:10]:
                    status_str = f"STUCK {status.upper()}" if "waiting" in status else status
                    print(f"  Node {node_id} ({class_name}): Last activity {idle_time:.2f}s ago - {status_str}")
    
    def analyze_queue_operations(self):
        """Analyze queue patterns"""
        queue_gets = defaultdict(list)  # node -> list of (queue, wait_time)
        queue_puts = defaultdict(int)    # node -> count
        
        for event in self.events:
            if event["type"] == "QUEUE_GET_SUCCESS":
                node = event["node"]
                queue = event["queue"]
                wait_time = event.get("wait_time", 0)
                queue_gets[node].append((queue, wait_time))
            elif event["type"] == "QUEUE_PUT":
                node = event["node"]
                queue_puts[node] += 1
        
        # Find nodes that never got input
        never_received = []
        for node_id in self.node_classes:
            if node_id not in queue_gets:
                never_received.append(node_id)
        
        if never_received and self.verbose:
            print("\nNodes that never received input:")
            for node_id in never_received:
                class_name = self.node_classes.get(node_id, "Unknown")
                print(f"  {node_id} ({class_name})")
    
    def check_bootstrap_issues(self) -> List[str]:
        """Check for nodes that never received any input"""
        issues = []
        
        # Find all nodes that are supposed to receive input
        nodes_with_inputs = set()
        for conn in self.connections:
            if len(conn) >= 4:
                to_node = conn[2]
                nodes_with_inputs.add(to_node)
        
        # Check which nodes never got any input
        nodes_that_got_input = set()
        for event in self.events:
            if event["type"] == "QUEUE_GET_SUCCESS":
                nodes_that_got_input.add(event["node"])
        
        # Find nodes that should have gotten input but didn't
        for node_id in nodes_with_inputs:
            if node_id not in nodes_that_got_input:
                class_name = self.graph.get(node_id, {}).get("class", "Unknown")
                # Check what inputs this node was waiting for
                waiting_for = set()
                for conn in self.connections:
                    if len(conn) >= 4 and conn[2] == node_id:
                        waiting_for.add(conn[3])
                
                if waiting_for:
                    inputs_str = ", ".join(sorted(waiting_for))
                    issues.append(f"⚠️ NO INPUT: Node {node_id} ({class_name}) never received any input on: {inputs_str}")
                else:
                    issues.append(f"⚠️ NO INPUT: Node {node_id} ({class_name}) never received any input")
        
        return issues
    
    def check_circular_dependencies(self) -> List[str]:
        """Check for circular wait dependencies"""
        issues = []
        
        # Build wait-for graph
        wait_for = {}  # node -> what it's waiting for
        
        for event in self.events:
            if event["type"] == "QUEUE_GET_WAIT":
                node = event["node"]
                queue = event["queue"]
                # Find who should produce this
                for conn in self.connections:
                    if len(conn) >= 4 and conn[2] == node and conn[3] == queue:
                        producer = conn[0]
                        wait_for[node] = producer
                        break
        
        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in wait_for:
                next_node = wait_for[node]
                if next_node not in visited:
                    if has_cycle(next_node, visited, rec_stack, path):
                        return True
                elif next_node in rec_stack:
                    # Found cycle
                    cycle_start = path.index(next_node)
                    cycle = path[cycle_start:] + [next_node]
                    return cycle
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node in wait_for:
            if node not in visited:
                rec_stack = set()
                path = []
                result = has_cycle(node, visited, rec_stack, path)
                if result and isinstance(result, list):
                    cycle_str = " → ".join(result)
                    issues.append(f"❌ DEADLOCK: Circular dependency detected: {cycle_str}")
        
        return issues
    
    def check_stuck_nodes(self) -> List[str]:
        """Check for nodes stuck waiting"""
        issues = []
        
        # Find nodes in permanent wait state
        for node_id, status in self.node_wait_status.items():
            if "waiting" in status:
                # Check how long they've been waiting
                last_activity = self.node_last_activity.get(node_id, 0)
                if self.events:
                    current_time = self.events[-1]["ts"]
                    wait_time = current_time - last_activity
                    
                    if wait_time > 5.0:  # Stuck for > 5 seconds
                        class_name = self.node_classes.get(node_id, "Unknown")
                        issues.append(f"⚠️ STUCK: Node {node_id} ({class_name}) {status} for {wait_time:.1f}s")
        
        return issues[:5]  # Limit to top 5 to avoid spam
    
    def generate_recommendations(self):
        """Generate generic recommendations based on detected patterns"""
        print("\nRecommendations:")
        
        recommendations = []
        
        # Check if any nodes never received input
        nodes_no_input = 0
        for node_id in self.graph:
            if node_id not in self.node_last_activity and node_id in self.node_classes:
                nodes_no_input += 1
        
        if nodes_no_input > 0:
            recommendations.append(f"• {nodes_no_input} node(s) never started - check for missing bootstrap or initial triggers")
        
        # Check for nodes waiting for input
        waiting_nodes = [nid for nid, status in self.node_wait_status.items() if "waiting" in status]
        if waiting_nodes:
            recommendations.append(f"• {len(waiting_nodes)} node(s) stuck waiting - verify all required connections exist")
        
        # Check for potential synchronization issues
        if any("hold" in str(self.graph.get(nid, {}).get("type", "")).lower() or
               "barrier" in str(self.graph.get(nid, {}).get("class", "")).lower() 
               for nid in self.graph):
            recommendations.append("• Synchronization nodes detected - ensure proper trigger/release patterns")
        
        # Check for one-way data flow (nodes that only consume)
        nodes_that_never_output = []
        for node_id in self.node_classes:
            outputs_data = False
            for event in self.events:
                if event.get("node") == node_id and event["type"] == "QUEUE_PUT":
                    outputs_data = True
                    break
            if not outputs_data and node_id in self.node_last_activity:
                nodes_that_never_output.append(node_id)
        
        if nodes_that_never_output:
            recommendations.append(f"• {len(nodes_that_never_output)} node(s) never produced output - may be sinks or incorrectly configured")
        
        # Generic recommendations
        recommendations.extend([
            "• Review the visual workflow for missing connections",
            "• Check if workflow needs bootstrap data or initial triggers",
            "• Verify all nodes are properly configured with required parameters"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze DNNE deadlock data")
    parser.add_argument("--data-dir", default="/tmp/dnne_deadlock_data",
                       help="Path to deadlock data directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed analysis")
    args = parser.parse_args()
    
    analyzer = DeadlockAnalyzer(args.data_dir, args.verbose)
    
    if not analyzer.load_data():
        sys.exit(1)
    
    analyzer.analyze()


if __name__ == "__main__":
    main()