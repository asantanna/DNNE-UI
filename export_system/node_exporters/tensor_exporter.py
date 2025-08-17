#!/usr/bin/env python3
"""
Exporter for TensorNode using queue-based template
"""

from ..graph_exporter import ExportableNode

class TensorExporter(ExportableNode):
    """Exporter for the Tensor constant generation node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/tensor_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget values from node data
        widgets = node_data.get("widgets_values", {})
        
        # Get values with defaults
        tensor_dims = widgets.get("tensor_dims", "10")
        fill_mode = widgets.get("fill_mode", "zeros")
        custom_fill = widgets.get("custom_fill", 0.0)
        dtype = widgets.get("dtype", "float32")
        seed = widgets.get("seed", -1)
        
        # Normalize dimension string for template
        # Ensure it's in a format that can be parsed
        tensor_dims_str = str(tensor_dims).strip()
        if not tensor_dims_str.startswith('[') and not tensor_dims_str.endswith(']'):
            # Add brackets if not present for consistency
            if ',' in tensor_dims_str:
                tensor_dims_str = f"[{tensor_dims_str}]"
            else:
                # Single dimension
                tensor_dims_str = f"[{tensor_dims_str}]"
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "TensorNode",
            "TENSOR_DIMS": tensor_dims_str,
            "FILL_MODE": fill_mode,
            "CUSTOM_FILL": str(custom_fill),
            "DTYPE": dtype,
            "SEED": str(seed)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn.init as init",
            "import asyncio",
            "from typing import Dict, Any",
        ]
    
    @classmethod
    def get_input_names(cls):
        # Tensor node has no inputs
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["tensor"]