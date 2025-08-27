"""
Data parser for DNNE deadlock analysis.
Loads and parses the raw event logs and graph structure.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


class DeadlockDataParser:
    """Parses deadlock data files into structured format"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.events = []
        self.graph = {}
        self.connections = []
        self.node_configs = {}
        
        # Derived data
        self.node_classes = {}  # node_id -> class_name
        self.node_last_activity = {}  # node_id -> timestamp
        self.node_wait_status = {}  # node_id -> status string
        
    def load_data(self) -> bool:
        """Load all data files and return success status"""
        if not self.data_dir.exists():
            return False
        
        # Load event log
        if not self._load_events():
            return False
        
        # Load graph structure
        self._load_graph_structure()
        
        # Load node configs
        self._load_node_configs()
        
        # Process events to extract metadata
        self._process_events()
        
        return True
    
    def _load_events(self) -> bool:
        """Load the event log file"""
        log_file = self.data_dir / "data_flow.log"
        if not log_file.exists():
            return False
        
        with open(log_file) as f:
            for line in f:
                if line.strip():
                    try:
                        self.events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Skip malformed lines
        
        return len(self.events) > 0
    
    def _load_graph_structure(self):
        """Load the graph structure file"""
        graph_file = self.data_dir / "graph_structure.json"
        if graph_file.exists():
            with open(graph_file) as f:
                data = json.load(f)
                self.graph = data.get("nodes", {})
                self.connections = data.get("connections", [])
    
    def _load_node_configs(self):
        """Load node configuration file"""
        config_file = self.data_dir / "node_configs.json"
        if config_file.exists():
            with open(config_file) as f:
                self.node_configs = json.load(f)
    
    def _process_events(self):
        """Process events to extract node metadata"""
        for event in self.events:
            node_id = event.get("node")
            if not node_id:
                continue
            
            # Track last activity
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
                self.node_wait_status[node_id] = "idle"
    
    def get_time_range(self) -> tuple:
        """Get the time range of events"""
        if not self.events:
            return (0, 0)
        return (self.events[0]["ts"], self.events[-1]["ts"])
    
    def get_event_counts(self) -> Dict[str, int]:
        """Get counts by event type"""
        counts = {}
        for event in self.events:
            event_type = event["type"]
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def get_node_input_map(self) -> Dict[str, List[str]]:
        """Get mapping of node_id to list of input names"""
        node_inputs = {}
        for conn in self.connections:
            if len(conn) >= 4:
                to_node = conn[2]
                input_name = conn[3]
                if to_node not in node_inputs:
                    node_inputs[to_node] = []
                if input_name not in node_inputs[to_node]:
                    node_inputs[to_node].append(input_name)
        return node_inputs
    
    def get_node_output_map(self) -> Dict[str, List[str]]:
        """Get mapping of node_id to list of output names"""
        node_outputs = {}
        for conn in self.connections:
            if len(conn) >= 4:
                from_node = conn[0]
                output_name = conn[1]
                if from_node not in node_outputs:
                    node_outputs[from_node] = []
                if output_name not in node_outputs[from_node]:
                    node_outputs[from_node].append(output_name)
        return node_outputs
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get dependency graph: node -> list of nodes it depends on"""
        dependencies = {}
        for conn in self.connections:
            if len(conn) >= 4:
                from_node = conn[0]
                to_node = conn[2]
                if to_node not in dependencies:
                    dependencies[to_node] = []
                if from_node not in dependencies[to_node]:
                    dependencies[to_node].append(from_node)
        return dependencies