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
                "epoch_complete": ("BOOLEAN", {
                    "tooltip": "Signal from GetBatch node indicating an epoch has completed"
                }),
                "epoch_stats": ("DICT", {
                    "tooltip": "Statistics dictionary from GetBatch node with epoch information"
                }),
                "loss_value": ("FLOAT", {
                    "tooltip": "Current loss value to track across epochs"
                }),
                "accuracy": ("FLOAT", {
                    "tooltip": "Current accuracy to track across epochs"
                }),
                "max_epochs": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 1000,
                    "tooltip": "Maximum number of epochs to train. Training stops when this is reached."
                }),
                "early_stop_patience": ("INT", {
                    "default": 10,
                    "min": 0,
                    "max": 100,
                    "tooltip": "Stop if no improvement for this many epochs. Set to 0 to disable early stopping."
                })
            }
        }

    RETURN_TYPES = ("BOOLEAN", "DICT")
    RETURN_NAMES = ("continue_training", "training_stats")
    FUNCTION = "track_epoch"
    CATEGORY = "ml"

    def track_epoch(self, epoch_complete, epoch_stats, loss_value, accuracy, max_epochs, early_stop_patience):
        context = get_context()
        
        # Initialize tracking if needed
        if "epoch_tracker" not in context.memory:
            context.memory["epoch_tracker"] = {
                "best_loss": float('inf'),
                "best_accuracy": 0.0,
                "best_epoch": 0,
                "epochs_without_improvement": 0,
                "loss_history": [],
                "accuracy_history": []
            }
        
        tracker = context.memory["epoch_tracker"]
        
        # Update metrics
        tracker["loss_history"].append(loss_value)
        tracker["accuracy_history"].append(accuracy)
        
        # Check for improvement
        if accuracy > tracker["best_accuracy"]:
            tracker["best_accuracy"] = accuracy
            tracker["best_loss"] = loss_value
            tracker["best_epoch"] = epoch_stats.get("epoch", 0)
            tracker["epochs_without_improvement"] = 0
        else:
            tracker["epochs_without_improvement"] += 1
        
        # Determine if we should continue
        current_epoch = epoch_stats.get("epoch", 0)
        continue_training = True
        stop_reason = None
        
        if current_epoch >= max_epochs:
            continue_training = False
            stop_reason = "max_epochs_reached"
        elif early_stop_patience > 0 and tracker["epochs_without_improvement"] >= early_stop_patience:
            continue_training = False
            stop_reason = "early_stopping"
        
        # Prepare stats
        training_stats = {
            "epoch": current_epoch,
            "loss": loss_value,
            "accuracy": accuracy,
            "best_loss": tracker["best_loss"],
            "best_accuracy": tracker["best_accuracy"],
            "best_epoch": tracker["best_epoch"],
            "epochs_without_improvement": tracker["epochs_without_improvement"],
            "continue_training": continue_training,
            "stop_reason": stop_reason
        }
        
        return (continue_training, training_stats)

# Node registration
NODE_CLASS_MAPPINGS = {
    "EpochTracker": EpochTrackerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EpochTracker": "Epoch Tracker"
}