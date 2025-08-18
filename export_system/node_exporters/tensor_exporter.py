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
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'tensor_dims', 'widget_index': 0},
            {'name': 'fill_mode', 'widget_index': 1},
            {'name': 'custom_fill', 'widget_index': 2},
            {'name': 'dtype', 'widget_index': 3},
            {'name': 'seed', 'widget_index': 4}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['tensor_dims', 'fill_mode', 'custom_fill', 'dtype', 'seed']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"TensorNode {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all tensor configuration parameters."
            )
        
        tensor_dims = params['tensor_dims']
        fill_mode = params['fill_mode']
        custom_fill = params['custom_fill']
        dtype = params['dtype']
        seed = params['seed']
        
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