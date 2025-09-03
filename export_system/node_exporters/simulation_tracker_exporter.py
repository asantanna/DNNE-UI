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
        # 3. telemetry_interval (STRING, default "100_steps")
        # 4. telemetry_level (COMBO, default "off")
        
        max_episodes = widgets[0] if len(widgets) > 0 else 1000
        success_threshold = widgets[1] if len(widgets) > 1 else 0.95
        telemetry_interval = widgets[2] if len(widgets) > 2 else "100_steps"
        telemetry_level = widgets[3] if len(widgets) > 3 else "off"
        
        # Escape string values for Python code generation
        telemetry_interval_escaped = repr(telemetry_interval)
        telemetry_level_escaped = repr(telemetry_level)
        
        return {
            "NODE_ID": node_id,
            "MAX_EPISODES": max_episodes,
            "SUCCESS_THRESHOLD": success_threshold,
            "TELEMETRY_INTERVAL": telemetry_interval_escaped,
            "TELEMETRY_LEVEL": telemetry_level_escaped,
        }
    
    @classmethod  
    def get_input_names(cls):
        """Get the ordered list of input names for this node type"""
        # From SimulationTrackerNode.INPUT_TYPES - all optional now
        return ["step_done", "episode_done", "loss", "custom_metrics"]
    
    @classmethod
    def get_output_names(cls):
        """Get the ordered list of output names for this node type"""
        return []  # No outputs - all metrics go through telemetry
    
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
        return {}  # No outputs - all metrics go through telemetry

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_ROBOTICS