"""
BatchNorm Node
Applies batch normalization to stabilize and accelerate neural network training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class BatchNormNode(RoboticsNodeBase):
    """BatchNorm Node
    Applies batch normalization to stabilize and accelerate neural network training."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to normalize. Shape depends on norm_type."}),
                "num_features": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of features to normalize. For BatchNorm1d: size of last dim. For BatchNorm2d: number of channels."
                }),
                "norm_type": (["BatchNorm1d", "BatchNorm2d"], {
                    "default": "BatchNorm1d",
                    "tooltip": "Type of batch normalization. Use BatchNorm1d for fully connected layers, BatchNorm2d for convolutional layers."
                }),
                "eps": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-7,
                    "max": 1e-3,
                    "tooltip": "Small constant for numerical stability. Default 1e-5 works for most cases."
                }),
                "momentum": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Momentum for running mean and variance computation. Lower values = more stable statistics."
                }),
                "training": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether in training mode. Affects whether to update running statistics."
                })
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_batchnorm"
    CATEGORY = "ml"

    def __init__(self):
        super().__init__()
        self.bn_layers = {}

    def apply_batchnorm(self, input, num_features, norm_type, eps, momentum, training):
        # Create unique key for this configuration
        key = f"{norm_type}_{num_features}"
        
        # Create batch norm layer if needed
        if key not in self.bn_layers:
            if norm_type == "BatchNorm1d":
                self.bn_layers[key] = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
            else:  # BatchNorm2d
                self.bn_layers[key] = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
            
            # Move to same device as input
            device = input.device if isinstance(input, torch.Tensor) else torch.device("cpu")
            self.bn_layers[key] = self.bn_layers[key].to(device)

        # Apply batch normalization
        bn_layer = self.bn_layers[key]
        bn_layer.train(training)
        output = bn_layer(input)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BatchNorm": BatchNormNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchNorm": "Batch Normalization"
}