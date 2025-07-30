#!/usr/bin/env python3
"""
Exporter for CartpoleReward node using queue-based template
"""

from ..graph_exporter import ExportableNode

class CartpoleRewardExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cartpole_reward_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [2.0, True])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CartpoleRewardNode",
            "RESET_DIST": widget_values[0] if len(widget_values) > 0 else 2.0,
            "INVERT_FOR_LOSS": widget_values[1] if len(widget_values) > 1 else True
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import numpy as np",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["observations"]
    
    @classmethod
    def get_output_names(cls):
        return ["reward_or_loss", "done", "info_dict"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "reward_or_loss": {
                    "type": "scalar",
                    "dtype": "float32"
                },
                "done": {
                    "type": "boolean",
                    "dtype": "bool"
                },
                "info_dict": {
                    "type": "dict",
                    "dtype": "dict"
                }
            }
        }