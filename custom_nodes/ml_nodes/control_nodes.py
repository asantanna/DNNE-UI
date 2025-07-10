"""
Control nodes for managing context and training mode
"""

import torch.nn as nn
from inspect import cleandoc
from .base import RoboticsNodeBase, get_context, Context


class CreateContextNode(RoboticsNodeBase):
    """Create Context Node
    Creates or resets the global execution context for neural network operations."""
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset": ("BOOLEAN", {"default": False, "tooltip": "Force reset the global context. If True, creates a new context even if one already exists. If False, only creates context if none exists."}),
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
    """Set Mode Node
    Sets training or evaluation mode for all neural network modules in the context."""
    DESCRIPTION = cleandoc(__doc__)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["train", "eval"], {"default": "train", "tooltip": "Set the training mode for all neural network modules. 'train' enables training mode (gradients, dropout, batch norm training), 'eval' enables evaluation mode (no gradients, deterministic behavior)."}),
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