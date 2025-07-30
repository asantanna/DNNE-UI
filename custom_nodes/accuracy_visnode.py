"""
Accuracy Node
Computes classification accuracy metrics for model evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class AccuracyNode(RoboticsNodeBase):
    """Accuracy Node
    Computes classification accuracy metrics for model evaluation."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("metric")["color"]
    BGCOLOR = get_node_colors("metric")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "predictions": ("TENSOR", {
                    "tooltip": "Model predictions. Can be logits or probabilities. Shape: (batch_size, num_classes)"
                }),
                "targets": ("TENSOR", {
                    "tooltip": "Ground truth labels. Shape: (batch_size,) with class indices"
                }),
                "top_k": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Calculate top-k accuracy. Set to 1 for standard accuracy, 5 for top-5 accuracy."
                })
            }
        }

    RETURN_TYPES = ("FLOAT", "DICT")
    RETURN_NAMES = ("accuracy", "metrics")
    FUNCTION = "compute_accuracy"
    CATEGORY = "ml"

    def compute_accuracy(self, predictions, targets, top_k):
        # Get predicted classes
        if top_k == 1:
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == targets).float()
            accuracy = correct.mean().item()
        else:
            # Top-k accuracy
            _, predicted = predictions.topk(top_k, 1, True, True)
            predicted = predicted.t()
            correct = predicted.eq(targets.view(1, -1).expand_as(predicted))
            correct_k = correct[:top_k].reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(100.0 / predictions.size(0)).item()
        
        # Additional metrics
        metrics = {
            "accuracy": accuracy,
            "correct": int(correct.sum().item()) if top_k == 1 else int(correct_k.item()),
            "total": len(targets),
            "top_k": top_k
        }
        
        return (accuracy, metrics)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Accuracy": AccuracyNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Accuracy": "Accuracy"
}