"""
SimulationTracker Node Exporter
Exports the SimulationTracker node for RL/robotics workflows.
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_ROBOTICS


class SimulationTrackerExporter(ExportableNode):
    """Exporter for SimulationTracker nodes"""
    
    @classmethod
    def get_template_name(cls):
        """Get the template file for this node type"""
        return "nodes/simulation_tracker_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        """Prepare template variables for SimulationTracker node"""
        
        # Extract widget values
        widgets = node_data.get("widgets_values", [])
        
        # Map widget values based on node definition
        # From SimulationTrackerNode.INPUT_TYPES:
        # Optional widgets in order:
        # 1. max_episodes (INT, default 1000)
        # 2. success_threshold (FLOAT, default 0.95)
        # 3. telemetry_mode (COMBO, default "time")
        # 4. telemetry_interval (STRING, default "10s")
        # 5. telemetry_stats (BOOLEAN, default True)
        
        max_episodes = widgets[0] if len(widgets) > 0 else 1000
        success_threshold = widgets[1] if len(widgets) > 1 else 0.95
        telemetry_mode = widgets[2] if len(widgets) > 2 else "time"
        telemetry_interval = widgets[3] if len(widgets) > 3 else "10s"
        telemetry_stats = widgets[4] if len(widgets) > 4 else True
        
        # Escape string values for Python code generation
        telemetry_interval_escaped = repr(telemetry_interval)
        
        return {
            "NODE_ID": node_id,
            "MAX_EPISODES": max_episodes,
            "SUCCESS_THRESHOLD": success_threshold,
            "TELEMETRY_MODE": repr(telemetry_mode),
            "TELEMETRY_INTERVAL": telemetry_interval_escaped,
            "TELEMETRY_STATS": telemetry_stats,
        }
    
    @classmethod  
    def get_input_names(cls):
        """Get the ordered list of input names for this node type"""
        # From SimulationTrackerNode.INPUT_TYPES
        return ["observation", "done", "loss", "reward", "custom_metrics"]
    
    @classmethod
    def get_output_names(cls):
        """Get the ordered list of output names for this node type"""
        # From SimulationTrackerNode.RETURN_NAMES
        return ["control_metrics"]
    
    @classmethod
    def get_imports(cls):
        """Get the list of imports needed for this node"""
        return [
            "import time",
            "import statistics",
            "import numpy as np",
            "from typing import Dict, Any, Optional",
            "from framework.time_utils import parse_duration",
        ]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Get the initial output schema for this node"""
        # Return initial control metrics structure
        return {
            "control_metrics": {
                "episode": 0,
                "timestep": 0,
                "done": False,
                "episode_done": False,
                "episode_reward": 0.0,
                "avg_reward": 0.0,
                "success_rate": 0.0,
                "improvement_rate": 0.0,
                "best_reward": float('-inf'),
                "avg_episode_length": 0.0,
                "episodes_since_improvement": 0,
            }
        }

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_ROBOTICS