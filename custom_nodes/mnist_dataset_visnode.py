"""
MNIST Dataset Node
Loads the MNIST handwritten digit dataset for training or testing.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


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

    RETURN_TYPES = ("MNIST_DATASET_OBJ", "MNIST_DATASET_SCHEMA_PYDICT")
    RETURN_NAMES = ("dataset", "schema")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"


# Node registration
NODE_CLASS_MAPPINGS = {
    "MNISTDataset": MNISTDatasetNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MNISTDataset": "MNIST Dataset"
}