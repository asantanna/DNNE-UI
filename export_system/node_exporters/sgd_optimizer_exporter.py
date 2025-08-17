#!/usr/bin/env python3
"""
Exporter for SGDOptimizer node using queue-based template
"""

from ..graph_exporter import ExportableNode

class SGDOptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/sgd_optimizer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'learning_rate', 'widget_index': 0},
            {'name': 'momentum', 'widget_index': 1},
            {'name': 'weight_decay', 'widget_index': 2}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['learning_rate', 'momentum', 'weight_decay']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"SGDOptimizer node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all optimizer parameters."
            )
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SGDOptimizerNode",
            "LEARNING_RATE": params['learning_rate'],
            "MOMENTUM": params['momentum'],
            "WEIGHT_DECAY": params['weight_decay']
        }
    
    @classmethod
    def get_imports(cls):
        return ["import torch.optim as optim"]
    
    
    @classmethod
    def get_output_names(cls):
        return ["optimizer"]
    
    @classmethod
    def get_input_names(cls):
        return ["model"]  # Connection from Network node
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "optimizer": {
                    "type": "optimizer",
                    "optimizer_type": "SGD"
                }
            }
        }