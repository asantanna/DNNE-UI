"""
MNIST Dataset Node
Loads the MNIST handwritten digit dataset for training or testing.
"""

import torch
from torch.utils.data import DataLoader
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class MNISTDatasetNode(RoboticsNodeBase):
    """MNIST Dataset Node
    Loads the MNIST handwritten digit dataset for training or testing."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("data")["color"]
    BGCOLOR = get_node_colors("data")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_path": ("STRING", {
                    "default": "./data",
                    "tooltip": "Directory path where MNIST dataset will be stored or loaded from. Creates the directory if it doesn't exist."
                }),
                "train": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to load training set (True) or test set (False). Training set has 60,000 samples, test set has 10,000 samples."
                }),
                "download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to automatically download the MNIST dataset if not found in data_path. Set to False if dataset is already downloaded."
                }),
            }
        }

    RETURN_TYPES = ("DATASET", "SCHEMA")
    RETURN_NAMES = ("dataset", "schema")
    FUNCTION = "load_dataset"
    CATEGORY = "ml"

    def load_dataset(self, data_path, train, download):
        # Import here to avoid dependency if not used
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
        
        # Create schema describing the dataset
        schema = {
            "outputs": {
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
            },
            "num_samples": len(dataset)
        }

        return (dataset, schema)


# Node registration
NODE_CLASS_MAPPINGS = {
    "MNISTDataset": MNISTDatasetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MNISTDataset": "MNIST Dataset"
}