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
        
        # Find the Network node ID by tracing the "model" input connection
        # SGDOptimizer.model <- Network.model (virtual connection)
        
        # The model input should tell us which network node is connected
        network_node_id = None
        if "model" in connections.get("inputs", {}):
            model_connections = connections["inputs"]["model"]
            if model_connections and len(model_connections) > 0:
                # Get the first (and should be only) connection to model input
                model_conn = model_connections[0]
                network_node_id = str(model_conn.get("from_node", ""))
                
                # Validate that it's actually a Network node
                if network_node_id and all_nodes:
                    network_node = next((n for n in all_nodes if str(n.get('id')) == network_node_id), None)
                    if network_node:
                        node_type = network_node.get("class_type") or network_node.get("type")
                        if node_type != "Network":
                            raise ValueError(
                                f"SGDOptimizer node {node_id}: Model input connected to {node_type} node {network_node_id}, "
                                f"but expected a Network node. Connect Network.model → SGDOptimizer.model."
                            )
        
        # FAIL-FAST: SGDOptimizer MUST have a Network connected
        if not network_node_id:
            raise ValueError(
                f"SGDOptimizer node {node_id}: No Network node connected to 'model' input! "
                f"SGDOptimizer requires a Network connection for parameter access. "
                f"Connect Network.model → SGDOptimizer.model to establish the link."
            )
        
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