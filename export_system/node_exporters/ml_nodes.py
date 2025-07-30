#!/usr/bin/env python3
"""
Exporters for ML nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode

class MNISTDatasetExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/mnist_dataset_simple_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'data_path', 'widget_index': 0, 'default': './data'},
            {'name': 'train', 'widget_index': 1, 'default': True},
            {'name': 'download', 'widget_index': 2, 'default': True}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "MNISTDatasetNode",
            "DATA_PATH": params['data_path'],
            "TRAIN": params['train'],
            "DOWNLOAD": params['download'],
            "BATCH_SIZE": 32,  # Fixed for MNIST
            "EMIT_RATE": 10.0  # Batches per second
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from torch.utils.data import DataLoader",
            "from torchvision import datasets, transforms",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["dataset", "schema"]
    
    @classmethod
    def get_input_names(cls):
        return []  # No inputs
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # MNIST dataset has fixed schema - no resolution needed
        return {
            "outputs": {
                "dataset": {
                    "type": "dataset",
                    "contains": {
                        "images": {
                            "type": "tensor",
                            "shape": (28, 28),
                            "flattened_size": 784,
                            "dtype": "float32"
                        },
                        "labels": {
                            "type": "tensor", 
                            "shape": (),
                            "num_classes": 10,
                            "dtype": "int64"
                        }
                    }
                },
                "schema": {
                    "type": "schema",
                    "value": {
                        "outputs": {
                            "dataset": {
                                "type": "dataset",
                                "contains": {
                                    "images": {
                                        "type": "tensor",
                                        "shape": (28, 28),
                                        "flattened_size": 784,
                                        "dtype": "float32"
                                    },
                                    "labels": {
                                        "type": "tensor",
                                        "shape": (),
                                        "num_classes": 10,
                                        "dtype": "int64"
                                    }
                                }
                            }
                        },
                        "num_samples": 60000
                    }
                }
            },
            "num_samples": 60000
        }


class CIFAR10DatasetExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cifar10_dataset_simple_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'data_path', 'widget_index': 0, 'default': './data'},
            {'name': 'train', 'widget_index': 1, 'default': True},
            {'name': 'download', 'widget_index': 2, 'default': True}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CIFAR10DatasetNode",
            "DATA_PATH": params['data_path'],
            "TRAIN": params['train'],
            "DOWNLOAD": params['download'],
            "BATCH_SIZE": 32,  # Fixed for CIFAR-10
            "EMIT_RATE": 10.0  # Batches per second
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from torch.utils.data import DataLoader",
            "from torchvision import datasets, transforms",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["dataset", "schema"]
    
    @classmethod
    def get_input_names(cls):
        return []  # No inputs
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # CIFAR-10 dataset has fixed schema - no resolution needed
        return {
            "outputs": {
                "dataset": {
                    "type": "dataset",
                    "contains": {
                        "images": {
                            "type": "tensor",
                            "shape": (3, 32, 32),
                            "flattened_size": 3072,
                            "dtype": "float32"
                        },
                        "labels": {
                            "type": "tensor", 
                            "shape": (),
                            "num_classes": 10,
                            "dtype": "int64"
                        }
                    }
                },
                "schema": {
                    "type": "schema",
                    "value": {
                        "outputs": {
                            "dataset": {
                                "type": "dataset",
                                "contains": {
                                    "images": {
                                        "type": "tensor",
                                        "shape": (3, 32, 32),
                                        "flattened_size": 3072,
                                        "dtype": "float32"
                                    },
                                    "labels": {
                                        "type": "tensor",
                                        "shape": (),
                                        "num_classes": 10,
                                        "dtype": "int64"
                                    }
                                }
                            }
                        },
                        "num_samples": 50000
                    }
                }
            },
            "num_samples": 50000  # CIFAR-10 training set size
        }


class LinearLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/linear_layer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'output_size', 'widget_index': 0, 'default': 128},
            {'name': 'bias', 'widget_index': 1, 'default': True},
            {'name': 'activation', 'widget_index': 2, 'default': 'relu'},
            {'name': 'dropout', 'widget_index': 3, 'default': 0.0}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
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
            "DROPOUT": params['dropout']
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
    def get_output_schema(cls, node_data, connections, node_registry, all_nodes, all_links):
        # Get output size from widgets_values (ComfyUI workflow format)
        widget_values = node_data.get("widgets_values", [128, True, "relu", 0.0])
        output_size = widget_values[0] if len(widget_values) > 0 else 128
        
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "flattened_size": output_size,
                    "dtype": "float32"
                }
            },
            "num_samples": 1
        }


class LossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/loss_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LossNode",
            "LOSS_TYPE": params.get("loss_type", "cross_entropy")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["loss", "loss_value"]
    
    @classmethod
    def get_input_names(cls):
        return ["predictions", "labels"]


class OptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/optimizer_queue.py"
    
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


class DisplayExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/display_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DisplayNode",
            "DISPLAY_TYPE": params.get("display_type", "tensor_stats"),
            "LOG_INTERVAL": params.get("log_interval", 10)  # Log every N inputs
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return []  # Display has no outputs
    
    @classmethod
    def get_input_names(cls):
        return ["input_0"]

class GetBatchExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/get_batch_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "GetBatchNode"
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["images", "labels", "epoch_complete", "epoch_stats"]
    
    @classmethod
    def get_input_names(cls):
        return ["dataloader", "schema", "trigger"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # GetBatch initial schema - tensor dimensions will be resolved from schema input
        return {
            "outputs": {
                "images": {
                    "type": "tensor",
                    "flattened_size": None,  # To be resolved from schema input
                    "dtype": "float32"
                },
                "labels": {
                    "type": "tensor",
                    "flattened_size": None,  # To be resolved from schema input
                    "dtype": "int64"
                },
                "epoch_complete": {
                    "type": "boolean",
                    "dtype": "bool"
                },
                "epoch_stats": {
                    "type": "dict",
                    "dtype": "dict"
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Resolve tensor dimensions from the schema input"""
        if key == "flattened_size":
            # Get the schema from our "schema" input
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if "schema" in input_schema and input_schema["schema"]:
                dataset_schema = input_schema["schema"]
                
                # Navigate through parent to determine which output we're resolving
                # parent_schema should be the "images" or "labels" dict
                parent_keys = list(parent_schema.keys())
                
                # Check if we're resolving for images or labels
                if "outputs" in dataset_schema:
                    dataset_outputs = dataset_schema["outputs"]
                    
                    # For datasets, the schema has a "dataset" output containing images/labels
                    if "dataset" in dataset_outputs and "contains" in dataset_outputs["dataset"]:
                        contains = dataset_outputs["dataset"]["contains"]
                        
                        # Determine if we're in images or labels by checking parent
                        if parent_schema.get("dtype") == "float32":  # images
                            if "images" in contains and "flattened_size" in contains["images"]:
                                return contains["images"]["flattened_size"]
                        elif parent_schema.get("dtype") == "int64":  # labels
                            if "labels" in contains and "flattened_size" in contains["labels"]:
                                return contains["labels"].get("flattened_size", 1)
                                
        return None
    
    @classmethod
    def get_output_schema_by_connector(cls, output_slot, node_data, connections,
                                     node_registry, all_nodes, all_links):
        """Return schema for specific output connector"""
        output_names = cls.get_output_names()
        
        if output_slot < len(output_names):
            output_name = output_names[output_slot]
            
            # Get full schema first
            full_schema = cls.get_output_schema(node_data, connections,
                                              node_registry, all_nodes, all_links)
            
            # For GetBatch, we need to resolve the schema from input
            if output_name == "images":
                # Get the schema input to determine tensor dimensions
                input_schema = cls.get_input_schema(node_data, connections,
                                                  node_registry, all_nodes, all_links)
                
                if "schema" in input_schema and input_schema["schema"]:
                    dataset_schema = input_schema["schema"]
                    
                    # Extract image tensor info from dataset schema
                    if "outputs" in dataset_schema and "dataset" in dataset_schema["outputs"]:
                        dataset_info = dataset_schema["outputs"]["dataset"]
                        if "contains" in dataset_info and "images" in dataset_info["contains"]:
                            return dataset_info["contains"]["images"]
            
            elif output_name == "labels":
                # Similar logic for labels
                input_schema = cls.get_input_schema(node_data, connections,
                                                  node_registry, all_nodes, all_links)
                
                if "schema" in input_schema and input_schema["schema"]:
                    dataset_schema = input_schema["schema"]
                    
                    if "outputs" in dataset_schema and "dataset" in dataset_schema["outputs"]:
                        dataset_info = dataset_schema["outputs"]["dataset"]
                        if "contains" in dataset_info and "labels" in dataset_info["contains"]:
                            return dataset_info["contains"]["labels"]
            
            # For other outputs, return from initial schema
            elif "outputs" in full_schema and output_name in full_schema["outputs"]:
                return full_schema["outputs"][output_name]
        
        return None

class SGDOptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/sgd_optimizer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'learning_rate', 'widget_index': 0, 'default': 0.01},
            {'name': 'momentum', 'widget_index': 1, 'default': 0.9}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SGDOptimizerNode",
            "LEARNING_RATE": params['learning_rate'],
            "MOMENTUM": params['momentum'],
            "WEIGHT_DECAY": 0.0  # Not configurable in this node type
        }
    
    @classmethod
    def get_imports(cls):
        return ["import torch.optim as optim"]
    
    
    @classmethod
    def get_output_names(cls):
        return ["optimizer"]
    
    @classmethod
    def get_input_names(cls):
        return ["network"]  # Connection from Network node


class CrossEntropyLossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cross_entropy_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LossNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["loss", "accuracy"]
    
    @classmethod
    def get_input_names(cls):
        return ["predictions", "labels"]


class TrainingStepExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/training_step_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "TrainingStepNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio"
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["ready", "step_complete"]
    
    @classmethod
    def get_input_names(cls):
        return ["loss", "optimizer"]


class EpochTrackerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/epoch_tracker_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        max_epochs = cls.get_node_parameter(node_data, 'max_epochs', default_value=None, widget_index=0)
        
        if max_epochs is None:
            raise ValueError(f"EpochTracker node {node_id}: missing max_epochs parameter. "
                           f"Available in node_data: inputs={node_data.get('inputs', {}).keys()}, "
                           f"widgets_values={node_data.get('widgets_values', [])}")
        
        if not isinstance(max_epochs, (int, float)) or max_epochs <= 0:
            raise ValueError(f"EpochTracker node {node_id}: max_epochs must be a positive number, got: {max_epochs}")
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "EpochTrackerNode",
            "MAX_EPOCHS": int(max_epochs)
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["training_summary"]
    
    @classmethod
    def get_input_names(cls):
        return ["epoch_stats", "loss", "accuracy", "max_epochs"]


class NetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/network_queue.py"
    
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
        # Network node initial schema - actual layer info comes from detected layers
        return {
            "outputs": {
                "layers": {
                    "type": "layers",
                    "value": None  # Will be resolved from detected layers
                },
                "output": {
                    "type": "tensor",
                    "flattened_size": None,  # Will be resolved from final layer
                    "dtype": "float32"
                },
                "model": {
                    "type": "model",
                    "contains_layers": True
                }
            }
        }
    
    @classmethod
    def get_output_schema_by_connector(cls, output_slot, node_data, connections,
                                     node_registry, all_nodes, all_links):
        """Return schema for specific output connector"""
        output_names = cls.get_output_names()
        
        if output_slot < len(output_names):
            output_name = output_names[output_slot]
            
            # For "layers" connector, return the schema from our input
            # This allows LinearLayers connected to "layers" to know the input size
            if output_name == "layers":
                # Get the input schema to pass through
                input_schema = cls.get_input_schema(node_data, connections,
                                                  node_registry, all_nodes, all_links)
                
                if "input" in input_schema and input_schema["input"]:
                    return input_schema["input"]
            
            # For "output" connector, also return the schema from our input
            elif output_name == "output":
                # Get the input schema to pass through
                input_schema = cls.get_input_schema(node_data, connections,
                                                  node_registry, all_nodes, all_links)
                
                if "input" in input_schema and input_schema["input"]:
                    return input_schema["input"]
            
            # For other outputs, use default behavior
            return super().get_output_schema_by_connector(output_slot, node_data, connections,
                                                        node_registry, all_nodes, all_links)
        
        return {"type": "unknown", "shape": None}
    
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


class BatchSamplerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/batch_sampler_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'batch_size', 'widget_index': 0, 'default': 32},
            {'name': 'shuffle', 'widget_index': 1, 'default': True},
            {'name': 'seed', 'widget_index': 2, 'default': 42}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BatchSamplerNode",
            "BATCH_SIZE": params['batch_size'],
            "SHUFFLE": params['shuffle'],
            "SEED": params['seed']
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from torch.utils.data import DataLoader",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["dataloader", "schema"]
    
    @classmethod
    def get_input_names(cls):
        return ["dataset", "schema"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """BatchSampler passes through dataset schema but wraps data in DataLoader"""
        return {
            "outputs": {
                "dataloader": {
                    "type": "dataloader",
                    "batch_size": node_data.get("widgets_values", [32])[0] if node_data.get("widgets_values") else 32,
                    "shuffle": node_data.get("widgets_values", [32, True])[1] if len(node_data.get("widgets_values", [])) > 1 else True,
                    "contains_schema": True  # Indicates this contains schema information
                },
                "schema": {
                    "type": "schema",
                    "value": None  # Will be resolved from input
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Pass through the schema from input"""
        if key == "value" and parent_schema.get("type") == "schema":
            # Get the schema from our "schema" input
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if "schema" in input_schema and input_schema["schema"]:
                return input_schema["schema"]
                
        return None
    

# Registration function
def register_ml_exporters(exporter):
    """Register all ML node exporters"""
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("CIFAR10Dataset", CIFAR10DatasetExporter)
    exporter.register_node("LinearLayer", LinearLayerExporter)
    exporter.register_node("Loss", LossExporter)
    exporter.register_node("Optimizer", OptimizerExporter)
    exporter.register_node("Display", DisplayExporter)
    exporter.register_node("GetBatch", GetBatchExporter)
    exporter.register_node("SGDOptimizer", SGDOptimizerExporter)
    exporter.register_node("TrainingStep", TrainingStepExporter)
    exporter.register_node("EpochTracker", EpochTrackerExporter)
    exporter.register_node("BatchSampler", BatchSamplerExporter)
    exporter.register_node("CrossEntropyLoss", CrossEntropyLossExporter)
    exporter.register_node("Network", NetworkExporter)
    # Aliases for compatibility
    exporter.register_node("Linear", LinearLayerExporter)