"""
Epoch Tracker Node
Tracks training progress across epochs and provides statistics and stopping conditions.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class EpochTrackerNode(RoboticsNodeBase):
    """Epoch Tracker Node
    Tracks training progress across epochs and provides statistics and stopping conditions."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("training")["color"]
    BGCOLOR = get_node_colors("training")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "epoch_stats": ("DICT", {
                    "tooltip": "Dictionary containing current epoch statistics like batch count, running totals, etc. Usually from GetBatch or similar nodes."
                }),
                "loss": ("TENSOR", {
                    "tooltip": "Current batch loss tensor for tracking training progress. Used to compute epoch averages and convergence metrics."
                }),
                "accuracy": ("*", {
                    "tooltip": "Current batch accuracy (float) or accuracy metrics. Can be from CrossEntropyLoss. Used for epoch averaging."
                }),
            },
            "optional": {
                "max_epochs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Maximum number of training epochs. Training will stop when this limit is reached or manually interrupted."
                }),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("training_summary",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "EpochTracker": EpochTrackerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EpochTracker": "Epoch Tracker"
}