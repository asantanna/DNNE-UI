#!/usr/bin/env python3
"""
Exporter for ORNode node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_CONTROL

class ORExporter(ExportableNode):
    """Exporter for the OR/ANY routing node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/or_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # OR node doesn't require any widget parameters, just return the basic template vars
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "ORNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input_a", "input_b", "input_c"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_CONTROL