"""
Training Step Node
Executes a single training step: forward pass, loss computation, backpropagation, and parameter update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


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
                    "tooltip": "Loss tensor from loss node (e.g., CrossEntropyLoss). Must be scalar or reducible to scalar."
                }),
                "optimizer": ("OPTIMIZER", {
                    "tooltip": "Optimizer instance (e.g., from SGDOptimizer) that will update model parameters."
                }),
                "gradient_clip": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Max norm for gradient clipping. 0 disables clipping. Use 1.0-5.0 to prevent exploding gradients."
                })
            }
        }

    RETURN_TYPES = ("SYNC", "FLOAT")
    RETURN_NAMES = ("trigger", "grad_norm")
    FUNCTION = "train_step"
    CATEGORY = "ml"

    def train_step(self, loss, optimizer, gradient_clip):
        # Zero gradients from previous step
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if requested
        grad_norm = 0.0
        if gradient_clip > 0:
            # Get all parameters from optimizer
            params = []
            for group in optimizer.param_groups:
                params.extend(group['params'])
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(params, gradient_clip)
            grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
        
        # Update parameters
        optimizer.step()
        
        # Return trigger for next batch
        return (True, grad_norm)

# Node registration
NODE_CLASS_MAPPINGS = {
    "TrainingStep": TrainingStepNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrainingStep": "Training Step"
}