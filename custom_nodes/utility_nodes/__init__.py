# __init__.py
"""
Utility nodes for DNNE
Generic utility nodes that can be used across different domains
"""

from .or_node import ORNode
from .balancing_node import BalancingNode
from .balancing_config import BalancingConfig

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ORNode": ORNode,
    "BalancingNode": BalancingNode,
    "BalancingConfig": BalancingConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ORNode": "OR/ANY Router",
    "BalancingNode": "Balancing Node",
    "BalancingConfig": "Balancing Config (Virtual)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']