"""
Cross Entropy Loss Node
Computes cross-entropy loss for multi-class classification tasks.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class CrossEntropyLossNode(RoboticsNodeBase):
    """Cross Entropy Loss Node
    Computes cross-entropy loss for multi-class classification tasks."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("training")["color"]
    BGCOLOR = get_node_colors("training")["bgcolor"]

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
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "CrossEntropyLoss": CrossEntropyLossNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossEntropyLoss": "Cross Entropy Loss"
}