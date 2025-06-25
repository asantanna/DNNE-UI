"""
Control nodes for managing context and training mode
"""

import torch.nn as nn
from .base import RoboticsNodeBase, get_context, Context


class CreateContextNode(RoboticsNodeBase):
    """Create a new context"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "create"
    CATEGORY = "ml/control"

    def create(self, reset):
        import custom_nodes.ml_nodes.base as base
        if reset or base.context is None:
            base.context = Context()
        return ()


class SetModeNode(RoboticsNodeBase):
    """Set training/eval mode"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["train", "eval"], {"default": "train"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "set_mode"
    CATEGORY = "ml/control"

    def set_mode(self, mode):
        context = get_context()
        context.training = (mode == "train")
        
        # Update all stored modules
        for key, module in context.memory.items():
            if isinstance(module, nn.Module):
                module.train(context.training)
        
        return ()