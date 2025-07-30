"""
Dropout Node
Applies dropout regularization during training to prevent overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class DropoutNode(RoboticsNodeBase):
    """Dropout Node
    Applies dropout regularization during training to prevent overfitting."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("layer")["color"]
    BGCOLOR = get_node_colors("layer")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to apply dropout to"}),
                "dropout_rate": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": "Probability of dropping each neuron. 0.5 means 50% chance. Common values: 0.2-0.5 for hidden layers."
                }),
                "training": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether model is in training mode. Dropout only applied during training, not inference."
                })
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "apply_dropout"
    CATEGORY = "ml"

    def apply_dropout(self, input, dropout_rate, training):
        if training and dropout_rate > 0:
            output = F.dropout(input, p=dropout_rate, training=True)
        else:
            output = input
            
        return (output,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "Dropout": DropoutNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dropout": "Dropout"
}