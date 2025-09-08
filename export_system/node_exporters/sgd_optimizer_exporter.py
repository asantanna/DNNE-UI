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
            {'name': 'batch_size', 'widget_index': 3},
            {'name': 'enable_bootstrap', 'widget_index': 4}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['learning_rate', 'momentum', 'weight_decay', 'batch_size', 'enable_bootstrap']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"SGDOptimizer node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all optimizer parameters."
            )
        
        # Find ALL Network node IDs by tracing the "model" input connections
        # SGDOptimizer.model <- Network.model (virtual connection - can be multiple!)
        
        model_node_ids = []
        if "model" in connections.get("inputs", {}):
            model_connections = connections["inputs"]["model"]
            if model_connections and len(model_connections) > 0:
                # Collect all connected model nodes
                for model_conn in model_connections:
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
                            model_node_ids.append(network_node_id)
        
        # FAIL-FAST: SGDOptimizer MUST have at least one Network connected
        if not model_node_ids:
            raise ValueError(
                f"SGDOptimizer node {node_id}: No Network nodes connected to 'model' input! "
                f"SGDOptimizer requires Network connections for parameter access. "
                f"Connect Network.model → SGDOptimizer.model to establish the link."
            )
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SGDOptimizerNode",
            "LEARNING_RATE": params['learning_rate'],
            "MOMENTUM": params['momentum'],
            "WEIGHT_DECAY": params['weight_decay'],
            "BATCH_SIZE": params['batch_size'],
            "ENABLE_BOOTSTRAP": params['enable_bootstrap'],  # No default - fail-fast!
            "MODEL_NODE_IDS": model_node_ids  # Pass the list of model node IDs
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