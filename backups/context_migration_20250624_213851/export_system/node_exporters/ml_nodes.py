"""
Exporters for ML nodes
"""

import sys
from pathlib import Path

# Handle imports whether run as module or directly
try:
    from ..graph_exporter import ExportableNode
except ImportError:
    # If relative import fails, try absolute
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from graph_exporter import ExportableNode

class LinearLayerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/linear_layer_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        return {
            "NODE_ID": f"node_{node_id}",
            "INPUT_SIZE": -1,  # Infer from input
            "OUTPUT_SIZE": params.get("output_size", 128),
            "ACTIVATION": params.get("activation", "relu"),
            "DROPOUT": params.get("dropout", 0.0),
            "BIAS": params.get("bias", True),
            "WEIGHT_INIT": params.get("weight_init", "xavier")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch.nn as nn",
            "import torch.nn.functional as F"
        ]

class MNISTDatasetExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/mnist_dataset_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": f"node_{node_id}",
            "DATA_PATH": params.get("data_path", "./data"),
            "TRAIN": params.get("train", True),
            "DOWNLOAD": params.get("download", True)
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "from torchvision import datasets, transforms"
        ]

class BatchSamplerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/batch_sampler_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        # Get dataset variable from connections
        dataset_var = "dataset"  # default
        if node_id in connections and "input_0" in connections[node_id]:
            source_info = connections[node_id]["input_0"]
            dataset_var = source_info["source_var"]
        
        return {
            "NODE_ID": f"node_{node_id}",
            "DATASET_VAR": dataset_var,
            "BATCH_SIZE": params.get("batch_size", 32),
            "SHUFFLE": params.get("shuffle", True),
            "NUM_WORKERS": 0
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "from torch.utils.data import DataLoader"
        ]

class GetBatchExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/get_batch_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        # Get dataloader from connections
        dataloader_var = "dataloader"
        if node_id in connections and "input_0" in connections[node_id]:
            source_info = connections[node_id]["input_0"]
            dataloader_var = source_info["source_var"]
        
        return {
            "NODE_ID": f"node_{node_id}",
            "DATALOADER_VAR": dataloader_var,
            "CONTEXT_VAR": "context"
        }
    
    @classmethod
    def get_imports(cls):
        return []

class CrossEntropyLossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cross_entropy_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        return {
            "NODE_ID": f"node_{node_id}",
            "PREDICTIONS_VAR": "predictions",
            "LABELS_VAR": "labels"
        }
    
    @classmethod
    def get_imports(cls):
        return ["import torch.nn.functional as F"]

class SGDOptimizerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/sgd_optimizer_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        params = node_data.get("inputs", {})
        
        return {
            "NODE_ID": f"node_{node_id}",
            "LEARNING_RATE": params.get("learning_rate", 0.01),
            "MOMENTUM": params.get("momentum", 0.9),
            "WEIGHT_DECAY": params.get("weight_decay", 0.0)
        }
    
    @classmethod
    def get_imports(cls):
        return ["import torch.optim"]

class AccuracyExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/accuracy_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        return {
            "NODE_ID": f"node_{node_id}",
            "PREDICTIONS_VAR": "predictions",
            "LABELS_VAR": "labels"
        }
    
    @classmethod
    def get_imports(cls):
        return []

class TrainingStepExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/training_step_template.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections):
        return {
            "NODE_ID": f"node_{node_id}",
            "LOSS_VAR": "loss",
            "OPTIMIZER_VAR": "optimizer"
        }
    
    @classmethod
    def get_imports(cls):
        return []