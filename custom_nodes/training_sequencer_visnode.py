#!/usr/bin/env python3
"""
Training Sequencer node for orchestrating multiple optimizer backward passes
Prevents gradient conflicts when multiple optimizers share network parameters
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from .utils.dnne_decorator import dnne_node

@dnne_node(is_virtual=False)
class TrainingSequencerNode(RoboticsNodeBase):
    """
    Orchestrates training for multiple optimizers to prevent gradient conflicts.
    Accepts multiple losses and coordinates backward passes in specified order.
    """
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("training")["color"]
    BGCOLOR = get_node_colors("training")["bgcolor"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "loss1": ("*LOSS_SCALAR", {}),
                "loss2": ("*LOSS_SCALAR", {}),
                "loss3": ("*LOSS_SCALAR", {}),
                "loss4": ("*LOSS_SCALAR", {}),
                "order": ("STRING", {
                    "default": "1,2,3,4",
                    "multiline": False,
                    "display": "input",
                    "tooltip": "Execution order for optimizers (e.g., '2,1,3' for loss2→loss1→loss3)"
                }),
                "retain_graph": ("BOOLEAN", {
                    "default": True,
                    "display": "toggle",
                    "tooltip": "Automatically retain graph for all but last backward pass"
                })
            }
        }
    
    RETURN_TYPES = ("SEQ_LOSS_SCALAR", "SEQ_LOSS_SCALAR", "SEQ_LOSS_SCALAR", "SEQ_LOSS_SCALAR")
    RETURN_NAMES = ("to_opt1", "to_opt2", "to_opt3", "to_opt4")
    FUNCTION = None  # DNNE doesn't execute nodes locally
    OUTPUT_NODE = True
    CATEGORY = "ml"
    

# Node registration
NODE_CLASS_MAPPINGS = {
    "TrainingSequencer": TrainingSequencerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainingSequencer": "Training Sequencer"
}