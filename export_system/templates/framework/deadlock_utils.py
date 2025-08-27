"""
Deadlock debugging utilities for DNNE workflows.
Provides lightweight data collection for offline analysis.
"""

import json
import time
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional


class DeadlockLogger:
    """
    Lightweight logger for deadlock analysis.
    Writes events directly to file with minimal overhead.
    """
    
    def __init__(self, output_dir: str = "/tmp/dnne_deadlock_data"):
        """Initialize the deadlock logger"""
        self.output_dir = Path(output_dir)
        self.start_time = time.time()
        
        # Clean and create output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open data flow log file
        self.log_file = open(self.output_dir / "data_flow.log", "w", buffering=1)  # Line buffered
        
        # Track node configurations
        self.node_configs = {}
        
    def log_event(self, event_type: str, **kwargs):
        """Log an event to the data flow log"""
        event = {
            "ts": time.time(),
            "type": event_type,
            **kwargs
        }
        # Write as single line JSON
        self.log_file.write(json.dumps(event) + "\n")
    
    def export_graph_structure(self, nodes: Dict[str, Any], connections: List[List[str]]):
        """Export the graph structure for analysis"""
        # Extract node information
        node_info = {}
        for node_id, node in nodes.items():
            node_info[node_id] = {
                "class": node.__class__.__name__,
                "type": self._get_node_type(node.__class__.__name__)
            }
            
            # Store node configurations if available
            if hasattr(node, 'get_config'):
                self.node_configs[node_id] = node.get_config()
        
        # Write graph structure
        graph_data = {
            "nodes": node_info,
            "connections": connections
        }
        
        with open(self.output_dir / "graph_structure.json", "w") as f:
            json.dump(graph_data, f, indent=2)
        
        # Write node configurations
        if self.node_configs:
            with open(self.output_dir / "node_configs.json", "w") as f:
                json.dump(self.node_configs, f, indent=2)
    
    def _get_node_type(self, class_name: str) -> str:
        """Categorize node type based on class name"""
        if "Network" in class_name:
            return "network"
        elif "SGD" in class_name or "Optimizer" in class_name:
            return "optimizer"
        elif "Loss" in class_name:
            return "loss"
        elif "Dataset" in class_name or "Batch" in class_name:
            return "data"
        elif "Barrier" in class_name or "Eat_N" in class_name:
            return "synchronization"
        elif "Isaac" in class_name or "Sim" in class_name:
            return "simulation"
        elif "Split" in class_name or "Concat" in class_name:
            return "tensor_ops"
        else:
            return "other"
    
    def close(self):
        """Close the log file"""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()
    
    def __del__(self):
        """Ensure file is closed on deletion"""
        self.close()


# Convenience functions for logging from nodes

def log_queue_state(node_id: str, queue_states: Dict[str, int]):
    """Log the current state of all input queues for a node"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "QUEUE_STATE",
            node=node_id,
            queue_depths=queue_states
        )


def log_queue_get_wait(node_id: str, queue_name: str):
    """Log that a node started waiting for input"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "QUEUE_GET_WAIT",
            node=node_id,
            queue=queue_name
        )


def log_queue_get_success(node_id: str, queue_name: str, wait_time: float):
    """Log that a node successfully got input"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "QUEUE_GET_SUCCESS", 
            node=node_id,
            queue=queue_name,
            wait_time=wait_time
        )


def log_queue_put(node_id: str, output_name: str, num_subscribers: int):
    """Log that a node sent output"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "QUEUE_PUT",
            node=node_id,
            output=output_name,
            subscribers=num_subscribers
        )


def log_node_wait(node_id: str, waiting_for: str):
    """Log that a node is waiting for specific input"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "NODE_WAIT",
            node=node_id,
            queue=waiting_for
        )


def log_node_compute_start(node_id: str):
    """Log that a node started computing"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "NODE_COMPUTE_START",
            node=node_id
        )


def log_node_compute_end(node_id: str, duration: float):
    """Log that a node finished computing"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "NODE_COMPUTE_END",
            node=node_id,
            duration=duration
        )


def log_node_start(node_id: str, class_name: str):
    """Log that a node task started"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "NODE_START",
            node=node_id,
            class_name=class_name
        )


def log_barrier_hold(node_id: str, queue_depth: int):
    """Log that a barrier received data to hold"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "BARRIER_HOLD",
            node=node_id,
            queue_depth=queue_depth
        )


def log_barrier_release(node_id: str, items_released: int):
    """Log that a barrier released held data"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "BARRIER_RELEASE",
            node=node_id,
            items_released=items_released
        )


def log_eat_n_consume(node_id: str, count: int, remaining: int):
    """Log that Eat_N consumed an input"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "EAT_N_CONSUME",
            node=node_id,
            count=count,
            remaining=remaining
        )


def log_eat_n_trigger(node_id: str, trigger_type: str):
    """Log that Eat_N sent a trigger"""
    from .globals import Global as g
    if g.deadlock_logger:
        g.deadlock_logger.log_event(
            "EAT_N_TRIGGER",
            node=node_id,
            trigger_type=trigger_type
        )