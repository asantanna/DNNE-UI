#!/usr/bin/env python3
"""
Exporter for Network node using queue-based template
"""

from ..graph_exporter import ExportableNode

class NetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/network_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Detect and analyze the network pattern
        network_layers = cls._detect_network_layers(node_id, all_nodes, all_links)
        
        # Query input size for the first layer if not set
        if network_layers and network_layers[0]["input_size"] is None:
            # Get the input schema to determine tensor size
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if "input" in input_schema and input_schema["input"]:
                input_tensor_schema = input_schema["input"]
                if "flattened_size" in input_tensor_schema:
                    network_layers[0]["input_size"] = input_tensor_schema["flattened_size"]
                else:
                    raise ValueError(f"Network node {node_id}: Could not determine input tensor size from connected node")
            else:
                raise ValueError(f"Network node {node_id}: No input connection found")
        
        # Generate layer definitions code
        layer_definitions = []
        for i, layer in enumerate(network_layers):
            # Add linear layer
            layer_definitions.append(
                f"        layers.append(nn.Linear({layer['input_size']}, {layer['output_size']}, bias={layer['bias']}))"
            )
            
            # Add activation
            if layer["activation"] == "relu":
                layer_definitions.append("        layers.append(nn.ReLU())")
            elif layer["activation"] == "tanh":
                layer_definitions.append("        layers.append(nn.Tanh())")
            elif layer["activation"] == "sigmoid":
                layer_definitions.append("        layers.append(nn.Sigmoid())")
            
            # Add dropout
            if layer["dropout"] > 0:
                layer_definitions.append(f"        layers.append(nn.Dropout({layer['dropout']}))")
        
        # Read checkpoint settings using universal parameter reader
        checkpoint_specs = [
            {'name': 'checkpoint_enabled', 'widget_index': 0, 'default': True},
            {'name': 'checkpoint_trigger_type', 'widget_index': 1, 'default': 'epoch'},
            {'name': 'checkpoint_trigger_value', 'widget_index': 2, 'default': '50'},
            {'name': 'checkpoint_load_on_start', 'widget_index': 3, 'default': False}
        ]
        
        checkpoint_params = cls.get_node_parameters_batch(node_data, checkpoint_specs)
        checkpoint_enabled = checkpoint_params['checkpoint_enabled']
        checkpoint_trigger_type = checkpoint_params['checkpoint_trigger_type']
        checkpoint_trigger_value = checkpoint_params['checkpoint_trigger_value']
        checkpoint_load_on_start = checkpoint_params['checkpoint_load_on_start']
        
        # Validate checkpoint values
        if not isinstance(checkpoint_enabled, bool):
            raise ValueError(f"Network node {node_id}: checkpoint_enabled must be boolean, got {type(checkpoint_enabled)}: {checkpoint_enabled}")
        
        if checkpoint_trigger_type not in ["epoch", "time", "best_metric"]:
            raise ValueError(f"Network node {node_id}: checkpoint_trigger_type must be 'epoch', 'time', or 'best_metric', got: {checkpoint_trigger_type}")
        
        if not isinstance(checkpoint_load_on_start, bool):
            raise ValueError(f"Network node {node_id}: checkpoint_load_on_start must be boolean, got {type(checkpoint_load_on_start)}: {checkpoint_load_on_start}")
        
        # Validate that we have determined input/output sizes
        if not network_layers:
            raise ValueError(f"Network node {node_id}: No layers detected in network")
        
        if network_layers[0]["input_size"] is None:
            raise ValueError(f"Network node {node_id}: Could not determine input size for first layer")
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "NetworkNode",
            "NETWORK_LAYERS": str(network_layers),
            "LAYER_DEFINITIONS": "\n".join(layer_definitions),
            "NUM_LAYERS": len(network_layers),
            "INPUT_SIZE": network_layers[0]["input_size"] if network_layers else None,
            "OUTPUT_SIZE": network_layers[-1]["output_size"] if network_layers else None,
            "CHECKPOINT_ENABLED": checkpoint_enabled,
            "CHECKPOINT_TRIGGER_TYPE": checkpoint_trigger_type,
            "CHECKPOINT_TRIGGER_VALUE": checkpoint_trigger_value,
            "CHECKPOINT_LOAD_ON_START": checkpoint_load_on_start
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
        return ["layers", "output", "model"]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # Network node initial schema
        return {
            "outputs": {
                "layers": {
                    "type": None,  # Will be resolved from input (passthrough)
                    "passthrough_from": "input"
                },
                "output": {
                    "type": None,  # Will be resolved from input (passthrough)
                    "passthrough_from": "input"
                },
                "model": {
                    "type": "model",
                    "contains_layers": True
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Resolve passthrough schema from input"""
        if key == "type" and "passthrough_from" in parent_schema:
            # Get the schema from the specified input
            input_name = parent_schema["passthrough_from"]
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if input_name in input_schema and input_schema[input_name]:
                # Copy all fields from input schema to parent schema
                source_schema = input_schema[input_name]
                for field_key, field_value in source_schema.items():
                    if field_key not in parent_schema:
                        parent_schema[field_key] = field_value
                return source_schema.get("type")
                
        return None
    
    
    @classmethod
    def _detect_network_layers(cls, network_node_id, all_nodes, all_links):
        """Detect the sequence of layers connected to this network node"""
        layers = []
        
        # Find the "layers" output connection from the network node
        layers_connection = None
        if all_links:
            for link in all_links:
                if len(link) >= 5:
                    from_node, from_slot, to_node, to_slot = str(link[1]), link[2], str(link[3]), link[4]
                    if from_node == network_node_id and from_slot == 0:  # "layers" output (slot 0)
                        layers_connection = (to_node, to_slot)
                        break
        
        if not layers_connection:
            return []
        
        # Follow the chain of layer connections
        current_node = layers_connection[0]
        visited = set()
        
        while current_node and current_node not in visited:
            visited.add(current_node)
            
            # Find the node data
            node_data = None
            for node in all_nodes:
                if str(node["id"]) == current_node:
                    node_data = node
                    break
            
            # Check both class_type and type for LinearLayer
            node_type = node_data.get("class_type") or node_data.get("type")
            if not node_data or node_type != "LinearLayer":
                break
            
            # Extract layer information from widgets_values (ComfyUI workflow format)
            widget_values = node_data.get("widgets_values", [128, True, "relu", 0.0])
            layer_info = {
                "node_id": current_node,
                "output_size": widget_values[0] if len(widget_values) > 0 else 128,
                "bias": widget_values[1] if len(widget_values) > 1 else True,
                "activation": widget_values[2] if len(widget_values) > 2 else "none",
                "dropout": widget_values[3] if len(widget_values) > 3 else 0.0
            }
            layers.append(layer_info)
            
            # Find the next layer in the chain
            next_node = None
            if all_links:
                for link in all_links:
                    if len(link) >= 5:
                        from_node, to_node = str(link[1]), str(link[3])
                        if from_node == current_node:
                            # Check if this goes to another LinearLayer or back to network
                            if to_node == network_node_id:
                                # Loop back to network - we're done
                                break
                            else:
                                next_node = to_node
                                break
            
            current_node = next_node
        
        # Determine input sizes based on actual connections
        for i, layer in enumerate(layers):
            if i == 0:
                # First layer input size must be determined from the Network node's input
                # NetworkExporter will query this properly
                layer["input_size"] = None  # To be determined by NetworkExporter
            else:
                layer["input_size"] = layers[i-1]["output_size"]
        
        return layers