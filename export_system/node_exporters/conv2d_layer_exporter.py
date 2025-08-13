#!/usr/bin/env python3
"""
Exporter for Conv2DLayer node using queue-based template
"""

from ..graph_exporter import ExportableNode

class Conv2DLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/conv2d_layer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'in_channels', 'widget_index': 1},
            {'name': 'out_channels', 'widget_index': 2},
            {'name': 'kernel_size', 'widget_index': 3},
            {'name': 'stride', 'widget_index': 4},
            {'name': 'padding', 'widget_index': 5},
            {'name': 'activation', 'widget_index': 6},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'activation']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"Conv2DLayer node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all convolution layer configuration parameters."
            )
        
        # Get input connections
        input_connections = connections.get('inputs', {})
        input_conn_info = input_connections.get("input", None)
        input_conn = input_conn_info["from_node"] if input_conn_info else None
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "Conv2DLayerNode",
            "IN_CHANNELS": params["in_channels"],
            "OUT_CHANNELS": params["out_channels"],
            "KERNEL_SIZE": params["kernel_size"],
            "STRIDE": params["stride"],
            "PADDING": params["padding"],
            "ACTIVATION": params["activation"],
            "INPUT_QUEUE": f"{input_conn}_queue" if input_conn else None,
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ['output']
    
    @classmethod
    def get_input_names(cls):
        return ['input']
