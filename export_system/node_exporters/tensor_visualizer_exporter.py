#!/usr/bin/env python3
"""
Exporter for TensorVisualizer node using queue-based template
"""

from ..graph_exporter import ExportableNode

class TensorVisualizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/tensor_visualizer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader
        param_specs = [
            {'name': 'name', 'widget_index': 1, 'default': 'tensor'},
            {'name': 'bins', 'widget_index': 2, 'default': 50},
            {'name': 'sample_size', 'widget_index': 3, 'default': 1000},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Get input connections
        input_connections = connections.get('inputs', {})
        tensor_conn_info = input_connections.get("tensor", None)
        tensor_conn = tensor_conn_info["from_node"] if tensor_conn_info else None
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "TensorVisualizerNode",
            "NAME": params["name"],
            "BINS": params["bins"],
            "SAMPLE_SIZE": params["sample_size"],
            "TENSOR_QUEUE": f"{tensor_conn}_queue" if tensor_conn else None,
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
        return ['tensor', 'visualization']
    
    @classmethod
    def get_input_names(cls):
        return ['tensor']
