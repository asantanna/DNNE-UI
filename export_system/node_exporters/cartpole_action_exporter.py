#!/usr/bin/env python3
"""
Exporter for CartpoleAction node using queue-based template
"""

from ..graph_exporter import ExportableNode

class CartpoleActionExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cartpole_action_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [10.0])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CartpoleActionNode",
            "MAX_PUSH_EFFORT": widget_values[0] if len(widget_values) > 0 else 10.0
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["policy"]
    
    @classmethod
    def get_output_names(cls):
        return ["action"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "action": {
                    "type": "tensor",
                    "shape": [1],  # Single action value
                    "dtype": "float32"
                }
            }
        }