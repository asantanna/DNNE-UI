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
        # ComfyUI workflow format uses widgets_values list, not inputs dict
        widget_values = node_data.get("widgets_values", ["./data", True, True])
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "MNISTDatasetNode",
            "DATA_PATH": widget_values[0] if len(widget_values) > 0 else "./data",
            "TRAIN": widget_values[1] if len(widget_values) > 1 else True,
            "DOWNLOAD": widget_values[2] if len(widget_values) > 2 else True,
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
    def get_output_schema(cls, node_data):
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
                    "description": "Dataset schema information"
                }
            },
            "num_samples": 60000
        }


class LinearLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/linear_layer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [128, True, "relu", 0.0])
        
        # Query input size from connected source node
        input_size = cls.query_input_tensor_size("input", connections, node_registry, all_nodes, all_links)
        
        # LinearLayer widget values from widgets_values list
        output_size = widget_values[0] if len(widget_values) > 0 else 128
        bias_value = widget_values[1] if len(widget_values) > 1 else True
        activation = widget_values[2] if len(widget_values) > 2 else "relu"
        dropout = widget_values[3] if len(widget_values) > 3 else 0.0
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LinearLayerNode",
            "INPUT_SIZE": input_size,
            "OUTPUT_SIZE": output_size,
            "ACTIVATION_VALUE": activation,
            "BIAS_VALUE": bias_value,
            "DROPOUT": dropout
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
    def get_output_schema(cls, node_data):
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
    def get_output_schema(cls, node_data):
        # GetBatch passes through the tensor dimensions from the schema input
        # The schema input tells us the dataset structure, and GetBatch adds batch dimension
        return {
            "outputs": {
                "images": {
                    "type": "tensor",
                    "flattened_size": 784,  # 28*28 for MNIST images
                    "dtype": "float32"
                },
                "labels": {
                    "type": "tensor",
                    "flattened_size": 1,  # Single label per sample
                    "dtype": "int64"
                },
                "epoch_complete": {
                    "type": "boolean",
                    "dtype": "bool"
                }
            },
            "num_samples": 1  # Per batch
        }

class SGDOptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/sgd_optimizer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [0.01, 0.9])
        
        # SGD Optimizer widget values from widgets_values list
        learning_rate = widget_values[0] if len(widget_values) > 0 else 0.01
        momentum = widget_values[1] if len(widget_values) > 1 else 0.9
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SGDOptimizerNode",
            "LEARNING_RATE": learning_rate,
            "MOMENTUM": momentum,
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
        # Get max epochs from inputs (ComfyUI processed format) or widgets_values (workflow format)
        max_epochs = None
        
        # Try ComfyUI processed format first (inputs dict)
        inputs = node_data.get("inputs", {})
        if "max_epochs" in inputs:
            max_epochs = inputs["max_epochs"]
        
        # Fall back to workflow format (widgets_values)
        if max_epochs is None:
            widget_values = node_data.get("widgets_values")
            if widget_values and len(widget_values) >= 1:
                max_epochs = widget_values[0]
        
        if max_epochs is None:
            raise ValueError(f"EpochTracker node {node_id}: missing max_epochs in node_data: {node_data}")
        
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
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "NetworkNode",
            "NETWORK_LAYERS": str(network_layers),
            "LAYER_DEFINITIONS": "\n".join(layer_definitions),
            "NUM_LAYERS": len(network_layers),
            "INPUT_SIZE": network_layers[0]["input_size"] if network_layers else 784,
            "OUTPUT_SIZE": network_layers[-1]["output_size"] if network_layers else 10,
            "CHECKPOINT_ENABLED": False,
            "CHECKPOINT_TRIGGER_TYPE": "epoch",
            "CHECKPOINT_TRIGGER_VALUE": "50",
            "CHECKPOINT_LOAD_ON_START": False
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
    def get_output_schema(cls, node_data):
        # Network node acts as a pass-through for schema queries
        # The "layers" output should query the "input" connection for its schema
        # The "network_output" schema will be determined by the final layer
        return "pass_through"  # Special marker indicating this node passes through queries
    
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
        
        # Determine input sizes based on schema or previous layer
        for i, layer in enumerate(layers):
            if i == 0:
                layer["input_size"] = 784  # Default for MNIST, could be determined by schema
            else:
                layer["input_size"] = layers[i-1]["output_size"]
        
        return layers


class BatchSamplerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/batch_sampler_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # ComfyUI workflow format uses widgets_values list
        widget_values = node_data.get("widgets_values", [32, True, 42, "randomize"])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BatchSamplerNode",
            "BATCH_SIZE": widget_values[0] if len(widget_values) > 0 else 32,
            "SHUFFLE": widget_values[1] if len(widget_values) > 1 else True,
            "SEED": widget_values[2] if len(widget_values) > 2 else 42
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
    def get_output_schema(cls, node_data):
        """BatchSampler passes through dataset schema but wraps data in DataLoader"""
        return {
            "outputs": {
                "dataloader": {
                    "type": "dataloader",
                    "batch_size": node_data.get("inputs", {}).get("batch_size", 32),
                    "shuffle": node_data.get("inputs", {}).get("shuffle", True),
                    "contains_schema": True  # Indicates this contains schema information
                },
                "schema": {
                    "type": "schema",
                    "description": "Dataset schema passed through from input"
                }
            }
        }
    

# Registration function
def register_ml_exporters(exporter):
    """Register all ML node exporters"""
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
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