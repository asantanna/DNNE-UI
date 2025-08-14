"""
CIFAR-10 Dataset Node
Loads the CIFAR-10 dataset containing 60,000 32x32 color images in 10 classes.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


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

    RETURN_TYPES = ("CIFAR10_DATASET", "CIFAR10_DATASET_SCHEMA")
    RETURN_NAMES = ("dataset", "schema")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "CIFAR10Dataset": CIFAR10DatasetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CIFAR10Dataset": "CIFAR-10 Dataset"
}