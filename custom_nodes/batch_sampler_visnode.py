"""
Batch Sampler Node
Creates a DataLoader that provides batched samples from a dataset with configurable batch size and shuffling.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class BatchSamplerNode(RoboticsNodeBase):
    """Batch Sampler Node
    Creates a DataLoader that provides batched samples from a dataset with configurable batch size and shuffling."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("data")["color"]
    BGCOLOR = get_node_colors("data")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("*DATASET", {
                    "tooltip": "Input dataset to create batches from. Should be a PyTorch dataset object (e.g., from MNISTDatasetNode)."
                }),
                "schema": ("*SCHEMA", {
                    "tooltip": "Dataset schema containing metadata about data shapes, types, and structure. Used for validation and downstream processing."
                }),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 512,
                    "tooltip": "Number of samples per batch. Larger batches use more memory but can improve training stability. Common values: 16, 32, 64, 128."
                }),
                "shuffle": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to randomly shuffle the dataset order each epoch. Generally True for training, False for evaluation."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "tooltip": "Random seed for reproducible shuffling. Set to -1 for random seed, or any positive integer for deterministic shuffling."
                }),
                "seed_control": (["fixed", "randomize"], {
                    "default": "fixed",
                    "tooltip": "How to handle seed between epochs. 'fixed' uses same seed, 'randomize' generates new seed each epoch."
                }),
            }
        }

    RETURN_TYPES = ("SAMPLER_BATCH_DATALOADER", "SAMPLER_BATCH_SCHEMA")
    RETURN_NAMES = ("dataloader", "schema")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "BatchSampler": BatchSamplerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchSampler": "Batch Sampler"
}