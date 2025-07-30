"""
Linear Layer Node
Represents a fully connected (dense) layer in a neural network with optional activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class LinearLayerNode(RoboticsNodeBase):
    """Linear Layer Node
    Represents a fully connected (dense) layer in a neural network with optional activation."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor from previous layer or data source"}),
                "in_features": ("INT", {
                    "default": 784,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of input features. Must match the size of the input tensor's last dimension."
                }),
                "out_features": ("INT", {
                    "default": 128,
                    "min": 1, 
                    "max": 10000,
                    "tooltip": "Number of output features. This becomes the size of the output tensor's last dimension."
                }),
                "activation": (["none", "relu", "tanh", "sigmoid", "leaky_relu"], {
                    "default": "relu",
                    "tooltip": "Activation function to apply after the linear transformation. 'none' for no activation (linear output)."
                }),
                "bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to include a learnable bias term. Set to False for certain normalization schemes."
                }),
                "weight_init": (["auto", "kaiming_normal", "kaiming_uniform", "xavier_normal", "xavier_uniform", "normal", "uniform", "none"], {
                    "default": "auto",
                    "widget": {"name": "weight_init"},
                    "tooltip": "Weight initialization method. 'auto' chooses based on activation function: Kaiming for ReLU/LeakyReLU, Xavier for tanh/sigmoid"
                })
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_layer"
    CATEGORY = "ml"

    def __init__(self):
        super().__init__()
        self.layer = None

    def apply_layer(self, input, in_features, out_features, activation, bias, weight_init):
        # Create layer if not exists
        if self.layer is None or self.layer.in_features != in_features or self.layer.out_features != out_features:
            self.layer = nn.Linear(in_features, out_features, bias=bias)
            
            # Move to same device as input
            device = input.device if isinstance(input, torch.Tensor) else torch.device("cpu")
            self.layer = self.layer.to(device)

        # Apply linear transformation
        output = self.layer(input)

        # Apply activation
        if activation == "relu":
            output = F.relu(output)
        elif activation == "tanh":
            output = torch.tanh(output)
        elif activation == "sigmoid":
            output = torch.sigmoid(output)
        elif activation == "leaky_relu":
            output = F.leaky_relu(output, negative_slope=0.01)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "LinearLayer": LinearLayerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearLayer": "Linear Layer"
}