"""
Epoch Tracker Node
Tracks training progress across epochs and provides statistics and stopping conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


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
                    "tooltip": "Current batch accuracy (float) or accuracy metrics. Can be from CrossEntropyLoss or AccuracyNode. Used for epoch averaging."
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
    FUNCTION = "track_progress"
    CATEGORY = "ml"

    def track_progress(self, epoch_stats, loss, accuracy, max_epochs=10):
        # This is a placeholder for UI - actual logic is in the template
        return ({"epoch": 0, "avg_loss": 0.0, "avg_accuracy": 0.0},)

# Node registration
NODE_CLASS_MAPPINGS = {
    "EpochTracker": EpochTrackerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EpochTracker": "Epoch Tracker"
}