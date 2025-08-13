#!/usr/bin/env python3
"""
Exporter for Accuracy node using queue-based template
"""

from ..graph_exporter import ExportableNode

class AccuracyExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/accuracy_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'top_k', 'widget_index': 2},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        if params.get('top_k') is None:
            raise ValueError(
                f"Accuracy node {node_id} missing required parameter: top_k. "
                f"The UI must provide the top-k value for accuracy calculation."
            )
        
        # Get input connections
        input_connections = connections.get('inputs', {})
        predictions_conn_info = input_connections.get("predictions", None)
        predictions_conn = predictions_conn_info["from_node"] if predictions_conn_info else None
        targets_conn_info = input_connections.get("targets", None)
        targets_conn = targets_conn_info["from_node"] if targets_conn_info else None
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "AccuracyNode",
            "TOP_K": params["top_k"],
            "PREDICTIONS_QUEUE": f"{predictions_conn}_queue" if predictions_conn else None,
            "TARGETS_QUEUE": f"{targets_conn}_queue" if targets_conn else None,
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
        return ['accuracy', 'metrics']
    
    @classmethod
    def get_input_names(cls):
        return ['predictions', 'targets']
