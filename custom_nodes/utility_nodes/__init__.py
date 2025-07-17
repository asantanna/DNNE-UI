# __init__.py
"""
Utility nodes for DNNE
Generic utility nodes that can be used across different domains
"""

from .or_node import ORNode

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ORNode": ORNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ORNode": "OR/ANY Router",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']