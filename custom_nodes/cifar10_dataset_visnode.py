"""
CIFAR-10 Dataset Node
Loads the CIFAR-10 dataset containing 60,000 32x32 color images in 10 classes.
"""

import torch
from torch.utils.data import DataLoader
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class CIFAR10DatasetNode(RoboticsNodeBase):
    """CIFAR-10 Dataset Node
    Loads the CIFAR-10 dataset containing 60,000 32x32 color images in 10 classes."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("data")["color"]
    BGCOLOR = get_node_colors("data")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_path": ("STRING", {
                    "default": "./data",
                    "tooltip": "Directory path where CIFAR-10 dataset will be stored or loaded from. Creates the directory if it doesn't exist."
                }),
                "train": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to load training set (True) or test set (False). Training set has 50,000 samples, test set has 10,000 samples."
                }),
                "download": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to automatically download the CIFAR-10 dataset if not found in data_path. Set to False if dataset is already downloaded."
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

        # CIFAR-10 specific normalization values
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010))
        ])

        dataset = datasets.CIFAR10(
            root=data_path,
            train=train,
            download=download,
            transform=transform
        )
        
        # Create schema describing the dataset
        # CIFAR-10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        schema = {
            "outputs": {
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
            },
            "num_samples": len(dataset)
        }

        return (dataset, schema)

# Node registration
NODE_CLASS_MAPPINGS = {
    "CIFAR10Dataset": CIFAR10DatasetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CIFAR10Dataset": "CIFAR-10 Dataset"
}