"""
Cross Entropy Loss Node
Computes cross-entropy loss for multi-class classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class CrossEntropyLossNode(RoboticsNodeBase):
    """Cross Entropy Loss Node
    Computes cross-entropy loss for multi-class classification tasks."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("loss")["color"]
    BGCOLOR = get_node_colors("loss")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR", {
                    "tooltip": "Model predictions (logits). Shape: (batch_size, num_classes). Raw outputs before softmax."
                }),
                "targets": ("TENSOR", {
                    "tooltip": "Ground truth labels. Shape: (batch_size,) with class indices as integers."
                }),
                "reduction": (["mean", "sum", "none"], {
                    "default": "mean",
                    "tooltip": "How to reduce the loss: 'mean' averages over batch, 'sum' totals, 'none' returns per-sample losses."
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "FLOAT")
    RETURN_NAMES = ("loss", "loss_value")
    FUNCTION = "compute_loss"
    CATEGORY = "ml"

    def compute_loss(self, predictions, targets, reduction):
        # Compute cross entropy loss
        loss = F.cross_entropy(predictions, targets, reduction=reduction)
        
        # Get scalar value for monitoring
        loss_value = loss.item() if loss.dim() == 0 else loss.mean().item()
        
        return (loss, loss_value)

# Node registration
NODE_CLASS_MAPPINGS = {
    "CrossEntropyLoss": CrossEntropyLossNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossEntropyLoss": "Cross Entropy Loss"
}