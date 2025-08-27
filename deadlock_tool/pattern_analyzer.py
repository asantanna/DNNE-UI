"""
Pattern analyzer for DNNE deadlock analysis.
Identifies common patterns and issues in the workflow execution.
"""

from typing import List, Dict, Any, Set
from dataclasses import dataclass


@dataclass
class AnalysisPattern:
    """Container for analysis patterns"""
    nodes_never_started: Set[str]
    nodes_never_received_input: Dict[str, List[str]]  # node_id -> list of missing inputs
    nodes_stuck_waiting: Dict[str, float]  # node_id -> wait_time
    nodes_never_output: Set[str]
    circular_dependencies: List[List[str]]  # list of cycles
    queue_pressures: Dict[str, int]  # queue_name -> depth


class PatternAnalyzer:
    """Analyzes execution patterns to identify issues"""
    
    def __init__(self, events: List[Dict], graph: Dict, connections: List, node_configs: Dict):
        self.events = events
        self.graph = graph
        self.connections = connections
        self.node_configs = node_configs
        
        # Load node behaviors to identify virtual inputs
        import json
        from pathlib import Path
        behaviors_file = Path(__file__).parent / "node_behaviors.json"
        self.node_behaviors = {}
        if behaviors_file.exists():
            with open(behaviors_file, 'r') as f:
                data = json.load(f)
                self.node_behaviors = data.get("node_types", {})
    
    def _is_virtual_input(self, node_class: str, input_name: str) -> bool:
        """Check if an input is virtual (not a real queue input)"""
        # Remove node ID suffix if present (e.g., "SGDOptimizerNode_40" -> "SGDOptimizerNode")
        if "_" in node_class:
            base_class = node_class.rsplit("_", 1)[0]
        else:
            base_class = node_class
            
        # Check if this class has virtual inputs defined
        if base_class in self.node_behaviors:
            virtual_inputs = self.node_behaviors[base_class].get("virtual_inputs", [])
            return input_name in virtual_inputs
        return False
    
    def analyze(self) -> AnalysisPattern:
        """Perform all pattern analysis"""
        return AnalysisPattern(
            nodes_never_started=self._find_never_started(),
            nodes_never_received_input=self._find_never_received_input(),
            nodes_stuck_waiting=self._find_stuck_waiting(),
            nodes_never_output=self._find_never_output(),
            circular_dependencies=self._find_circular_dependencies(),
            queue_pressures=self._analyze_queue_pressures()
        )
    
    def _find_never_started(self) -> Set[str]:
        """Find nodes that never started executing"""
        all_nodes = set(self.graph.keys())
        started_nodes = set()
        
        for event in self.events:
            if event["type"] == "NODE_START":
                started_nodes.add(event["node"])
        
        return all_nodes - started_nodes
    
    def _find_never_received_input(self) -> Dict[str, List[str]]:
        """Find nodes that never received expected inputs (excluding virtual inputs)"""
        # Build map of what inputs each node expects
        expected_inputs = {}
        for conn in self.connections:
            if len(conn) >= 4:
                to_node = conn[2]
                input_name = conn[3]
                
                # Get node class to check for virtual inputs
                node_class = self.graph.get(to_node, {}).get("class", "Unknown")
                
                # Skip virtual inputs (they don't use the queue system)
                if self._is_virtual_input(node_class, input_name):
                    continue
                    
                if to_node not in expected_inputs:
                    expected_inputs[to_node] = set()
                expected_inputs[to_node].add(input_name)
        
        # Track what inputs were actually received
        received_inputs = {}
        for event in self.events:
            if event["type"] == "QUEUE_GET_SUCCESS":
                node = event["node"]
                queue = event["queue"]
                if node not in received_inputs:
                    received_inputs[node] = set()
                received_inputs[node].add(queue)
        
        # Find missing inputs
        missing_inputs = {}
        for node, expected in expected_inputs.items():
            received = received_inputs.get(node, set())
            missing = expected - received
            if missing:
                missing_inputs[node] = sorted(list(missing))
        
        return missing_inputs
    
    def _find_stuck_waiting(self) -> Dict[str, float]:
        """Find nodes stuck waiting and how long"""
        if not self.events:
            return {}
        
        current_time = self.events[-1]["ts"]
        stuck_nodes = {}
        
        # Track last wait start for each node
        wait_starts = {}
        
        for event in self.events:
            node = event.get("node")
            if not node:
                continue
            
            if event["type"] == "QUEUE_GET_WAIT":
                wait_starts[node] = event["ts"]
            elif event["type"] == "QUEUE_GET_SUCCESS":
                # Clear wait state
                if node in wait_starts:
                    del wait_starts[node]
        
        # Any node still in wait_starts is stuck
        for node, start_time in wait_starts.items():
            wait_time = current_time - start_time
            if wait_time > 1.0:  # Only report if waiting > 1 second
                stuck_nodes[node] = wait_time
        
        return stuck_nodes
    
    def _find_never_output(self) -> Set[str]:
        """Find nodes that never produced output"""
        nodes_with_output = set()
        
        for event in self.events:
            if event["type"] == "QUEUE_PUT":
                nodes_with_output.add(event["node"])
        
        # Check which nodes ran but never output
        nodes_that_ran = set()
        for event in self.events:
            if event["type"] in ["NODE_COMPUTE_END", "NODE_START"]:
                nodes_that_ran.add(event["node"])
        
        return nodes_that_ran - nodes_with_output
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependency cycles"""
        # Build wait-for graph from current wait states
        wait_for = {}
        
        # Get latest wait state for each node
        for event in self.events:
            if event["type"] == "QUEUE_GET_WAIT":
                node = event["node"]
                queue = event["queue"]
                
                # Find who produces this queue
                for conn in self.connections:
                    if len(conn) >= 4 and conn[2] == node and conn[3] == queue:
                        producer = conn[0]
                        wait_for[node] = producer
                        break
        
        # Find cycles using DFS
        cycles = []
        visited = set()
        
        def find_cycle(node, path, rec_stack):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                return [cycle]
            
            if node in visited:
                return []
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            found_cycles = []
            if node in wait_for:
                next_node = wait_for[node]
                found_cycles.extend(find_cycle(next_node, path.copy(), rec_stack.copy()))
            
            return found_cycles
        
        for node in wait_for:
            if node not in visited:
                cycles.extend(find_cycle(node, [], set()))
        
        return cycles
    
    def _analyze_queue_pressures(self) -> Dict[str, int]:
        """Analyze queue depths to find bottlenecks"""
        # This would require queue depth events which we don't currently log
        # For now, return empty dict
        return {}