#!/usr/bin/env python3
"""
Exporter for MNIST Dataset node using queue-based template
"""

from ..graph_exporter import ExportableNode

class MNISTDatasetExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/mnist_dataset_simple_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'data_path', 'widget_index': 0},
            {'name': 'train', 'widget_index': 1},
            {'name': 'download', 'widget_index': 2}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['data_path', 'train', 'download']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"MNISTDataset node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all dataset configuration parameters."
            )
        
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