#!/usr/bin/env python3
"""
Exporters for ML nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode

class MNISTDatasetExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/mnist_dataset_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "MNISTDatasetNode",
            "DATA_PATH": params.get("data_path", "./data"),
            "TRAIN": params.get("train", True),
            "DOWNLOAD": params.get("download", True),
            "BATCH_SIZE": params.get("batch_size", 32),
            "EMIT_RATE": params.get("emit_rate", 10.0)  # Batches per second
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from torch.utils.data import DataLoader",
            "from torchvision import datasets, transforms",
        ]


class LinearLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/linear_layer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LinearLayerNode",
            "INPUT_SIZE": params.get("input_size", 784),  # Default for MNIST
            "OUTPUT_SIZE": params.get("output_size", 128),
            "ACTIVATION": params.get("activation", "relu"),
            "DROPOUT": params.get("dropout", 0.0),
            "BIAS": params.get("bias", True)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        ]


class LossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/loss_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
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


class OptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/optimizer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
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


class DisplayExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/display_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
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


# Registration function
def register_ml_exporters(exporter):
    """Register all ML node exporters"""
    exporter.register_node("MNISTDataset", MNISTDatasetExporter)
    exporter.register_node("LinearLayer", LinearLayerExporter)
    exporter.register_node("Loss", LossExporter)
    exporter.register_node("Optimizer", OptimizerExporter)
    exporter.register_node("Display", DisplayExporter)
    
    # Aliases for compatibility
    exporter.register_node("Linear", LinearLayerExporter)
    exporter.register_node("CrossEntropyLoss", LossExporter)
    exporter.register_node("BatchSampler", MNISTDatasetExporter)  # Combined with dataset
