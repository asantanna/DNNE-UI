#!/usr/bin/env python3
"""
Exporter for Display node using queue-based template
"""

from ..graph_exporter import ExportableNode

class DisplayExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/display_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DisplayNode",
            "DISPLAY_TYPE": params.get("display_type", "tensor_stats"),
            "LOG_INTERVAL": params.get("log_interval", 10)  # Log every N inputs
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return []  # Display has no outputs
    
    @classmethod
    def get_input_names(cls):
        return ["input_0"]