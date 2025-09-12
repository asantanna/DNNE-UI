#!/usr/bin/env python3
"""
Exporter for Network node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..utils import export_utils
from ..subsystems import SUBSYSTEM_NETWORK

class NetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/network_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use new architecture to collect layer definitions
        layer_definitions = []
        layers_info = []
        
        # Start at the "layers" output and follow the chain
        current_node_id = export_utils.follow_node_connection(node_id, "layers")
        
        # Determine input size from the Network's input connection
        input_size = None
        input_schema = cls.get_input_schema(node_data, connections, 
                                          node_registry, all_nodes, all_links)
        if "input" in input_schema and input_schema["input"]:
            input_tensor_schema = input_schema["input"]
            if "flattened_size" in input_tensor_schema:
                input_size = input_tensor_schema["flattened_size"]
                if input_size is None:
                    # The connected node couldn't determine size
                    raise ValueError(f"Network node {node_id}: Connected node returned None for flattened_size")
            else:
                raise ValueError(f"Network node {node_id}: Could not determine input tensor size from connected node")
        else:
            # More helpful error message
            if "input" in connections["inputs"]:
                connected_nodes = [conn.get("from_node", "unknown") for conn in connections["inputs"].get("input", [])]
                raise ValueError(
                    f"Network node {node_id}: Input connection has invalid or missing schema. "
                    f"Connected nodes: {connected_nodes}. "
                    f"Check for Label nodes or unrecognized node types in the connection chain."
                )
            else:
                raise ValueError(f"Network node {node_id}: No input connection found")
        
        # Follow the chain of layers
        visited = set()
        while current_node_id and current_node_id != node_id and current_node_id not in visited:
            visited.add(current_node_id)
            
            # Get the current node
            node = export_utils.get_node_by_id(current_node_id)
            if not node:
                break
            
            # Get node type - fail if missing
            if 'class_type' not in node and 'type' not in node:
                raise RuntimeError(f"Network: Connected node {current_node_id} has no type information")
            node_type = node.get('class_type') or node['type']
            
            # Get the exporter for this node type
            exporter = export_utils.get_node_exporter(node_type)
            if exporter:
                # Try to call get_layer_pytorch_code - let AttributeError propagate if missing
                try:
                    # Ask the layer for its PyTorch code
                    layer_info = exporter.get_layer_pytorch_code(current_node_id, node, input_size)
                except AttributeError:
                    # Node doesn't support layer code generation - skip it
                    layer_info = None
                
                if layer_info:
                    # Add layer definition
                    layer_definitions.append(f"        layers.append({layer_info['layer_code']})")
                
                    # Add activation if present
                    if 'activation_code' in layer_info and layer_info['activation_code']:
                        layer_definitions.append(f"        layers.append({layer_info['activation_code']})")
                    
                    # Add dropout if present
                    if 'dropout_code' in layer_info and layer_info['dropout_code']:
                        layer_definitions.append(f"        layers.append({layer_info['dropout_code']})")
                
                    # Store layer info for debugging/reference
                    layers_info.append({
                        'node_id': current_node_id,
                        'input_size': input_size,
                        'output_size': layer_info['output_size']
                    })
                    
                    # Update input size for next layer
                    input_size = layer_info['output_size']
            
            # Follow to the next node
            current_node_id = export_utils.follow_node_connection(current_node_id, "output")
        
        # Read checkpoint settings - FAIL-FAST: no defaults
        checkpoint_specs = [
            {'name': 'checkpoint_enabled', 'widget_index': 0},
            {'name': 'checkpoint_trigger_type', 'widget_index': 1},
            {'name': 'checkpoint_trigger_value', 'widget_index': 2},
            {'name': 'checkpoint_load_on_start', 'widget_index': 3}
        ]
        
        checkpoint_params = cls.get_node_parameters_batch(node_data, checkpoint_specs)
        
        # Validate required checkpoint parameters are present
        required_checkpoint = ['checkpoint_enabled', 'checkpoint_trigger_type', 
                              'checkpoint_trigger_value', 'checkpoint_load_on_start']
        missing_checkpoint = [p for p in required_checkpoint if p not in checkpoint_params or checkpoint_params[p] is None]
        if missing_checkpoint:
            raise ValueError(
                f"Network node {node_id} missing checkpoint parameters: {missing_checkpoint}. "
                f"The UI must provide all checkpoint configuration."
            )
        checkpoint_enabled = checkpoint_params['checkpoint_enabled']
        checkpoint_trigger_type = checkpoint_params['checkpoint_trigger_type']
        checkpoint_trigger_value = checkpoint_params['checkpoint_trigger_value']
        checkpoint_load_on_start = checkpoint_params['checkpoint_load_on_start']
        
        # Validate checkpoint values
        if not isinstance(checkpoint_enabled, bool):
            raise ValueError(f"Network node {node_id}: checkpoint_enabled must be boolean, got {type(checkpoint_enabled)}: {checkpoint_enabled}")
        
        if checkpoint_trigger_type not in ["epoch", "time", "best_metric", "end"]:
            raise ValueError(f"Network node {node_id}: checkpoint_trigger_type must be 'epoch', 'time', 'best_metric', or 'end', got: {checkpoint_trigger_type}")
        
        if not isinstance(checkpoint_load_on_start, bool):
            raise ValueError(f"Network node {node_id}: checkpoint_load_on_start must be boolean, got {type(checkpoint_load_on_start)}: {checkpoint_load_on_start}")
        
        # FAIL-FAST: Validate that we have layers
        if not layer_definitions:
            raise ValueError(f"Network node {node_id}: No layers detected in network")
        
        # FAIL-FAST: Validate we have layer info
        if not layers_info:
            raise ValueError(f"Network node {node_id}: No layer information collected")
            
        # Get final output size
        output_size = layers_info[-1]["output_size"]
        
        # Find the connected optimizer node ID by following the model output
        # Network's model output connects to optimizer's model input
        optimizer_node_id = export_utils.follow_node_connection(node_id, "model")
        
        # FAIL-FAST: Network must have an optimizer connected (except in inference mode)
        if not optimizer_node_id:
            # Check if this might be an inference-only workflow
            # (could add a check for g.inference_mode in the future)
            raise ValueError(
                f"Network node {node_id}: No optimizer connected to 'model' output! "
                f"Networks must be connected to an optimizer for training. "
                f"Connect Network.model → SGDOptimizer.model to enable sync checking."
            )
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "NetworkNode",
            "NETWORK_LAYERS": str(layers_info),  # For debugging/documentation
            "LAYER_DEFINITIONS": "\n".join(layer_definitions),
            "NUM_LAYERS": len(layers_info),
            "INPUT_SIZE": layers_info[0]["input_size"] if layers_info else None,
            "OUTPUT_SIZE": output_size,
            "CHECKPOINT_ENABLED": checkpoint_enabled,
            "CHECKPOINT_TRIGGER_TYPE": checkpoint_trigger_type,
            "CHECKPOINT_TRIGGER_VALUE": checkpoint_trigger_value,
            "CHECKPOINT_LOAD_ON_START": checkpoint_load_on_start,
            "OPTIMIZER_NODE_ID": optimizer_node_id  # Pass the optimizer node ID for sync checking
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
        return ["input", "to_output"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # Network node initial schema
        # NOTE: The "output" should NOT be passthrough - it should be the size of the last layer!
        return {
            "outputs": {
                "layers": {
                    "type": None,  # Will be resolved from input (passthrough)
                    "passthrough_from": "input"
                },
                "output": {
                    "type": "tensor",  # Network output is always a tensor
                    "dtype": "float32",
                    "flattened_size": None  # Will be determined by the last layer
                },
                "model": {
                    "type": "model",
                    "contains_layers": True
                }
            }
        }
    
    @classmethod
    def get_output_schema(cls, node_data, connections=None, node_registry=None, 
                         all_nodes=None, all_links=None):
        """Get output schema with actual network output size"""
        schema = cls.get_initial_output_schema(node_data)
        
        # Determine the actual output size by following the layer chain
        node_id = str(node_data.get('id'))
        current_node_id = export_utils.follow_node_connection(node_id, "layers")
        
        # Track the last layer's output size
        final_output_size = None
        
        # Follow the chain of layers
        visited = set()
        while current_node_id and current_node_id != node_id and current_node_id not in visited:
            visited.add(current_node_id)
            
            # Get the current node
            node = export_utils.get_node_by_id(current_node_id)
            if not node:
                break
            
            # Get node type
            node_type = node.get('class_type') or node.get('type')
            
            # Get the exporter for this node type
            exporter = export_utils.get_node_exporter(node_type)
            if exporter and hasattr(exporter, 'get_layer_pytorch_code'):
                try:
                    # Ask the layer for its configuration (we don't need actual code here)
                    # Pass a dummy input size since we only care about output size
                    layer_info = exporter.get_layer_pytorch_code(current_node_id, node, 1)
                    if layer_info and 'output_size' in layer_info:
                        final_output_size = layer_info['output_size']
                except:
                    pass
            
            # Follow to the next node
            current_node_id = export_utils.follow_node_connection(current_node_id, "output")
        
        # Update the output schema with the actual output size
        if final_output_size is not None:
            schema["outputs"]["output"]["flattened_size"] = final_output_size
        
        return schema
    
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
    def get_subsystem(cls):
        return SUBSYSTEM_NETWORK