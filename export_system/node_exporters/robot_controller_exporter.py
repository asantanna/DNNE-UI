#!/usr/bin/env python3
"""
Exporter for RobotController node using queue-based template
"""

from ..graph_exporter import ExportableNode

class RobotControllerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/robot_controller_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        
        # Parse joint limits
        joint_limits = params.get("joint_limits", [-3.14, 3.14])
        if isinstance(joint_limits, str):
            joint_limits = eval(joint_limits)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "RobotControllerNode",
            "JOINT_LIMITS_MIN": joint_limits[0],
            "JOINT_LIMITS_MAX": joint_limits[1],
            "CONTROL_TYPE": params.get("control_type", "position"),
            "NUM_JOINTS": params.get("num_joints", 7)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]


# IsaacGymStepExporter removed - no corresponding node or template exists