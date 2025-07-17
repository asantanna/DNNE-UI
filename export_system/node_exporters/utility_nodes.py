#!/usr/bin/env python3
"""
Exporters for utility nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode


class ORNodeExporter(ExportableNode):
    """Exporter for the OR/ANY routing node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/or_node_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
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


# Registration function
def register_utility_exporters(exporter):
    """Register all utility node exporters"""
    exporter.register_node("ORNode", ORNodeExporter)