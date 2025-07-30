"""
Activation Node
Applies various activation functions to input tensors for introducing non-linearity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class ActivationNode(RoboticsNodeBase):
    """Activation Node
    Applies various activation functions to input tensors for introducing non-linearity."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to apply activation function to"}),
                "activation": (["relu", "tanh", "sigmoid", "softmax", "leaky_relu", "elu", "selu", "gelu"], {
                    "default": "relu",
                    "tooltip": "Activation function type. ReLU for hidden layers, Sigmoid/Tanh for bounded output, Softmax for multi-class classification."
                }),
                "dim": ("INT", {
                    "default": -1,
                    "min": -4,
                    "max": 3,
                    "tooltip": "Dimension along which to apply activation (only for softmax). Default -1 means last dimension."
                })
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_activation"
    CATEGORY = "ml"

    def apply_activation(self, input, activation, dim):
        if activation == "relu":
            output = F.relu(input)
        elif activation == "tanh":
            output = torch.tanh(input)
        elif activation == "sigmoid":
            output = torch.sigmoid(input)
        elif activation == "softmax":
            output = F.softmax(input, dim=dim)
        elif activation == "leaky_relu":
            output = F.leaky_relu(input, negative_slope=0.01)
        elif activation == "elu":
            output = F.elu(input)
        elif activation == "selu":
            output = F.selu(input)
        elif activation == "gelu":
            output = F.gelu(input)
        else:
            output = input

        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Activation": ActivationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Activation": "Activation"
}