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
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'learning_rate', 'widget_index': 0, 'default': 0.01},
            {'name': 'momentum', 'widget_index': 1, 'default': 0.9},
            {'name': 'weight_decay', 'widget_index': 2, 'default': 0.0}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
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