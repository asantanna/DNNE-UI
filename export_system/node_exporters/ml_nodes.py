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
        # TODO: Extract parameters from node_data
        # TODO: Determine input variable names from connections
        return {
            "NODE_ID": f"node_{node_id}",
            "INPUT_VAR": "input_tensor",
            "OUTPUT_SIZE": 128,
            "ACTIVATION": "relu",
            "DROPOUT": 0.0,
            "BIAS": True,
            "WEIGHT_INIT": "xavier"
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

# TODO: Add more exporters for other node types