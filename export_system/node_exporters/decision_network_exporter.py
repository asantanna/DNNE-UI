#!/usr/bin/env python3
"""
Exporter for DecisionNetwork node using queue-based template
"""

from ..graph_exporter import ExportableNode

class DecisionNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/decision_network_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        
        # Count input connections to determine input dimension
        num_inputs = len(connections.get("inputs", {}))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DecisionNetworkNode",
            "NUM_INPUTS": num_inputs,
            "ACTION_DIM": params.get("action_dim", 6),
            "HIDDEN_SIZE": params.get("hidden_size", 256),
            "DEVICE": params.get("device", "cuda")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]