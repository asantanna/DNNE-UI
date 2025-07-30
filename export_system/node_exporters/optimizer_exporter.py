#!/usr/bin/env python3
"""
Exporter for Optimizer node using queue-based template
"""

from ..graph_exporter import ExportableNode

class OptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/optimizer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        
        # Extract model nodes from connections
        model_nodes = []
        # TODO: Parse from connections to find upstream model nodes
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "OptimizerNode",
            "OPTIMIZER_TYPE": params.get("optimizer", "adam"),
            "LEARNING_RATE": params.get("learning_rate", 0.001),
            "MODEL_NODES": str(model_nodes)  # Will be a list
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.optim as optim",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["step_complete"]
    
    @classmethod
    def get_input_names(cls):
        return ["loss"]