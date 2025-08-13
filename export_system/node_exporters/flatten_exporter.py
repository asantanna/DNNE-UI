#!/usr/bin/env python3
"""
Exporter for Flatten node using queue-based template
"""

from ..graph_exporter import ExportableNode

class FlattenExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/flatten_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'start_dim', 'widget_index': 1},
            {'name': 'end_dim', 'widget_index': 2},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['start_dim', 'end_dim']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"Flatten node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all flatten configuration parameters."
            )
        
        # Get input connections
        input_connections = connections.get('inputs', {})
        input_conn_info = input_connections.get("input", None)
        input_conn = input_conn_info["from_node"] if input_conn_info else None
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "FlattenNode",
            "START_DIM": params["start_dim"],
            "END_DIM": params["end_dim"],
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
        return ['output', 'flattened_size']
    
    @classmethod
    def get_input_names(cls):
        return ['input']
