#!/usr/bin/env python3
"""
Exporter for SGDOptimizer node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_TRAINING

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
            {'name': 'weight_decay', 'widget_index': 2},
            {'name': 'enable_bootstrap', 'widget_index': 3}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['learning_rate', 'momentum', 'weight_decay', 'enable_bootstrap']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"SGDOptimizer node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all optimizer parameters."
            )
        
        # Find the Network node ID for virtual connection
        network_node_id = None
        if all_nodes:
            for node in all_nodes:
                node_type = node.get("class_type") or node.get("type")
                if node_type == "Network":
                    network_node_id = str(node['id'])  # Just the ID, no "node_" prefix
                    break
        
        if not network_node_id:
            network_node_id = "network_1"  # Default placeholder
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SGDOptimizerNode",
            "LEARNING_RATE": params['learning_rate'],
            "MOMENTUM": params['momentum'],
            "WEIGHT_DECAY": params['weight_decay'],
            "ENABLE_BOOTSTRAP": params['enable_bootstrap'],  # No default - fail-fast!
            "NETWORK_NODE_ID": network_node_id  # Pass the network node ID for virtual connection
        }
    
    @classmethod
    def get_imports(cls):
        return ["import torch.optim as optim"]
    
    
    @classmethod
    def get_output_names(cls):
        return ["step_complete"]
    
    @classmethod
    def get_input_names(cls):
        return ["model", "loss"]  # Model from Network, loss from loss function
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "step_complete": {
                    "type": "signal",
                    "signal_type": "training_step_complete"
                }
            }
        }
    
    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_TRAINING