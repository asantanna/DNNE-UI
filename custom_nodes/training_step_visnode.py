"""
Training Step Node
Executes a single training step: forward pass, loss computation, backpropagation, and parameter update.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class TrainingStepNode(RoboticsNodeBase):
    """Training Step Node
    Executes a single training step: forward pass, loss computation, backpropagation, and parameter update."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("training")["color"]
    BGCOLOR = get_node_colors("training")["bgcolor"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loss": ("TENSOR", {
                    "tooltip": "Loss tensor to backpropagate. Scalar tensor (single value) computed from loss function like CrossEntropyLoss."
                }),
                "optimizer": ("OPTIMIZER", {
                    "tooltip": "Optimizer instance (SGD, Adam, etc.) that will update model parameters. Connect from SGDOptimizer or similar node."
                })
            }
        }

    RETURN_TYPES = ("SYNC",)
    RETURN_NAMES = ("ready",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    CATEGORY = "ml"

# Node registration
NODE_CLASS_MAPPINGS = {
    "TrainingStep": TrainingStepNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainingStep": "Training Step"
}