#!/usr/bin/env python3
"""
Exporter for BatchNorm node using queue-based template
"""

from ..graph_exporter import ExportableNode

class BatchNormExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/batchnorm_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader
        param_specs = [
            {'name': 'num_features', 'widget_index': 1, 'default': 128},
            {'name': 'norm_type', 'widget_index': 2, 'default': 'BatchNorm1d'},
            {'name': 'eps', 'widget_index': 3, 'default': 1e-05},
            {'name': 'momentum', 'widget_index': 4, 'default': 0.1},
            {'name': 'training', 'widget_index': 5, 'default': True},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Get input connections
        input_connections = connections.get('inputs', {})
        input_conn_info = input_connections.get("input", None)
        input_conn = input_conn_info["from_node"] if input_conn_info else None
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BatchNormNode",
            "NUM_FEATURES": params["num_features"],
            "NORM_TYPE": params["norm_type"],
            "EPS": params["eps"],
            "MOMENTUM": params["momentum"],
            "TRAINING": params["training"],
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
