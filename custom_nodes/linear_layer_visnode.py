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
                "input": ("TENSOR", {"tooltip": "Input tensor to transform (automatically flattened if > 2D)"}),
                "output_size": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 4096,
                    "tooltip": "Number of output features (neurons) in this layer"
                }),
                "bias": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether to include learnable bias parameters"
                }),
                "activation": (["none", "relu", "tanh", "sigmoid", "leaky_relu"], {
                    "default": "relu",
                    "tooltip": "Activation function to apply after linear transformation"
                }),
                "dropout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.9,
                    "tooltip": "Dropout probability for regularization (0.0 = no dropout)"
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

    def apply_layer(self, input, output_size, bias, activation, dropout, weight_init):
        # Flatten input if needed
        if len(input.shape) > 2:
            input = input.view(input.size(0), -1)

        input_size = input.shape[1]
        
        # Create layer if not exists or size changed
        if self.layer is None or self.layer.in_features != input_size or self.layer.out_features != output_size:
            self.layer = nn.Linear(input_size, output_size, bias=bias)
            
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

        # Apply dropout if in training mode
        if dropout > 0 and self.training:
            output = F.dropout(output, p=dropout, training=True)

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "LinearLayer": LinearLayerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LinearLayer": "Linear Layer"
}