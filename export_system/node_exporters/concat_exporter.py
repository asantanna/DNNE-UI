#!/usr/bin/env python3
"""
Exporter for ConcatNode using queue-based template
"""

from ..graph_exporter import ExportableNode

class ConcatExporter(ExportableNode):
    """Exporter for the Concat tensor concatenation node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/concat_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget values for mode and pad_mode
        widgets = node_data.get("widgets_values", [])
        
        # Default values if not provided
        mode = "wait for all"
        pad_mode = "pad with zeros"
        
        # Extract from widgets array based on expected order
        if len(widgets) >= 1:
            mode = widgets[0]
        if len(widgets) >= 2:
            pad_mode = widgets[1]
            
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "ConcatNode",
            "MODE": mode,
            "PAD_MODE": pad_mode
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "from typing import Dict, Any, Optional, List",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input_a", "input_b", "input_c", "input_d"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]