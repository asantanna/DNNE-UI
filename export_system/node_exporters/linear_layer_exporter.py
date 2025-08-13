#!/usr/bin/env python3
"""
Exporter for Linear Layer node using queue-based template
"""

from ..graph_exporter import ExportableNode

class LinearLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/linear_layer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'output_size', 'widget_index': 0},
            {'name': 'bias', 'widget_index': 1},
            {'name': 'activation', 'widget_index': 2},
            {'name': 'dropout', 'widget_index': 3},
            {'name': 'weight_init', 'widget_index': 4}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['output_size', 'bias', 'activation', 'dropout', 'weight_init']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"LinearLayer node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all layer configuration parameters."
            )
        
        # Query input size from connected source node
        input_schema = cls.get_input_schema(node_data, connections, 
                                          node_registry, all_nodes, all_links)
        
        if "input" in input_schema and input_schema["input"] and "flattened_size" in input_schema["input"]:
            input_size = input_schema["input"]["flattened_size"]
        else:
            raise ValueError(f"LinearLayer node {node_id}: Could not determine input tensor size")
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LinearLayerNode",
            "INPUT_SIZE": input_size,
            "OUTPUT_SIZE": params['output_size'],
            "ACTIVATION_VALUE": params['activation'],
            "BIAS_VALUE": params['bias'],
            "DROPOUT": params['dropout'],
            "WEIGHT_INIT": params['weight_init']
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
        return ["output"]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # Get output size from widgets_values (ComfyUI workflow format)
        widget_values = node_data.get("widgets_values", [])
        if not widget_values or len(widget_values) < 1:
            raise ValueError(f"LinearLayer node missing required widget values. Expected at least 1, got {len(widget_values)}")
        output_size = widget_values[0]
        
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "flattened_size": output_size,
                    "dtype": "float32"
                }
            }
        }