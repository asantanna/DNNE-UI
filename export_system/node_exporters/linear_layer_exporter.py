#!/usr/bin/env python3
"""
Exporter for Linear Layer node - Virtual node used within Networks
"""

from ..graph_exporter import ExportableNode

class LinearLayerExporter(ExportableNode):
    # LinearLayers are virtual - they only exist within Networks
    # Virtual status is handled by @dnne_node decorator
    
    @classmethod
    def get_template_name(cls):
        # Virtual nodes don't need templates
        return None
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Virtual nodes don't generate code via templates
        return {}
    
    @classmethod
    def get_layer_pytorch_code(cls, node_id, node_data, input_size=None):
        """Generate PyTorch layer definition code for use by Network node.
        
        Args:
            node_id: ID of this layer node
            node_data: Node data from workflow
            input_size: Input size for this layer (provided by Network)
        
        Returns:
            Dict with 'layer_code', 'activation_code', 'dropout_code', and 'output_size'
        """
        # Extract parameters
        param_specs = [
            {'name': 'output_size', 'widget_index': 0},
            {'name': 'bias', 'widget_index': 1},
            {'name': 'activation', 'widget_index': 2},
            {'name': 'dropout', 'widget_index': 3},
            {'name': 'weight_init', 'widget_index': 4}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters
        required_params = ['output_size', 'bias', 'activation', 'dropout']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"LinearLayer node {node_id} missing required parameters: {missing_params}"
            )
        
        if input_size is None:
            raise ValueError(f"LinearLayer node {node_id}: input_size must be provided by Network")
        
        # Generate layer code
        result = {
            'layer_code': f"nn.Linear({input_size}, {params['output_size']}, bias={params['bias']})",
            'activation_code': None,
            'dropout_code': None,
            'output_size': params['output_size']
        }
        
        # Add activation - fail if missing
        if 'activation' not in params:
            raise ValueError(f"LinearLayer node {node_id}: activation parameter missing")
        activation = params['activation']
        if activation == 'relu':
            result['activation_code'] = "nn.ReLU()"
        elif activation == 'tanh':
            result['activation_code'] = "nn.Tanh()"
        elif activation == 'sigmoid':
            result['activation_code'] = "nn.Sigmoid()"
        elif activation == 'elu':
            result['activation_code'] = "nn.ELU()"
        
        # Add dropout - fail if missing
        if 'dropout' not in params:
            raise ValueError(f"LinearLayer node {node_id}: dropout parameter missing")
        dropout = params['dropout']
        if dropout > 0:
            result['dropout_code'] = f"nn.Dropout({dropout})"
        
        return result
    
    @classmethod
    def get_imports(cls):
        # Virtual nodes don't generate files, so no imports needed
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # Get output size for schema
        output_size = cls.get_node_parameter(node_data, 'output_size', widget_index=0)
        
        if output_size is None:
            raise ValueError(f"LinearLayer node missing required widget values. Could not extract output_size parameter.")
        
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "flattened_size": output_size,
                    "dtype": "float32"
                }
            }
        }