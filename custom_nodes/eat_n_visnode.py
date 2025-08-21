"""
Eat_N Node
Synchronization primitive that consumes the first N inputs then becomes passthrough.
"""

from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors
from custom_nodes.utils.dnne_decorator import dnne_node


@dnne_node(is_virtual=False)
class Eat_NNode(RoboticsNodeBase):
    """
    Eat_N Node
    Consumes the first N inputs and then becomes a passthrough for all subsequent inputs.
    
    Features:
    - Bootstrap reinforcement learning pipelines
    - Generate triggers to release held data in Barrier nodes
    - Create temporal shifts in data streams
    - Stateful counter tracking consumed inputs
    """
    
    CATEGORY = "utility"
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_to_eat": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of inputs to consume before becoming passthrough"
                }),
                "trigger_mode": (["every_eat", "last_only"], {
                    "default": "every_eat",
                    "tooltip": "When to send triggers: 'every_eat' or 'last_only'"
                })
            },
            "optional": {
                "input": ("*TENSOR", {
                    "tooltip": "Any tensor input to consume or pass through"
                })
            }
        }
    
    RETURN_TYPES = ("TENSOR", "EAT_N_TRIGGER")
    RETURN_NAMES = ("output", "trigger")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    DESCRIPTION = cleandoc(__doc__)


# Node registration
NODE_CLASS_MAPPINGS = {
    "Eat_N": Eat_NNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Eat_N": "Eat_N"
}