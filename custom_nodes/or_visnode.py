"""
Generic OR/ANY Node for routing inputs
Outputs when ANY input becomes available - useful for routing between multiple data sources
"""

from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


class ORNode(RoboticsNodeBase):
    """
    OR/ANY Node for routing inputs
    Outputs when ANY input becomes available - useful for routing between multiple data sources
    
    Common use cases:
    - RL training loops: routing initial state vs ongoing state
    - Data pipeline: selecting between different data sources
    - Conditional routing: switching between inputs based on availability
    """
    
    CATEGORY = "utility"
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_a": ("TENSOR",),
                "input_b": ("TENSOR",),
                "input_c": ("TENSOR",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = None  # DNNE nodes don't execute in UI, only export
    DESCRIPTION = cleandoc(__doc__)


# Node registration
NODE_CLASS_MAPPINGS = {
    "OR": ORNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OR": "OR"
}