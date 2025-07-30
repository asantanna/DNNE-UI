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
                    "tooltip": "Model predictions/logits tensor with shape (batch_size, num_classes). Raw output from neural network before softmax."
                }),
                "labels": ("TENSOR", {
                    "tooltip": "Ground truth class labels tensor with shape (batch_size,). Integer values representing correct class indices (0 to num_classes-1)."
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "FLOAT")
    RETURN_NAMES = ("loss", "accuracy")
    FUNCTION = "compute_loss"
    CATEGORY = "ml"

    def compute_loss(self, predictions, labels):
        loss = F.cross_entropy(predictions, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(predictions, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        
        return (loss, accuracy)

# Node registration
NODE_CLASS_MAPPINGS = {
    "CrossEntropyLoss": CrossEntropyLossNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossEntropyLoss": "Cross Entropy Loss"
}