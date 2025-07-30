"""
Network Node
Consolidates multiple LinearLayer nodes into a single PyTorch Sequential model with checkpoint support.
For checkpoint debugging: check console logs or exported code for actual node ID.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase, get_context
from custom_nodes.node_colors import get_node_colors


class NetworkNode(RoboticsNodeBase):
    """
    Network Node
    Consolidates multiple LinearLayer nodes into a single PyTorch Sequential model with checkpoint support.
    For checkpoint debugging: check console logs or exported code for actual node ID.
    """

    def __init__(self):
        super().__init__()
        self.checkpoint_manager = None
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("TENSOR", {"tooltip": "Input tensor to process through the neural network"}),
                "to_output": ("TENSOR", {"tooltip": "Loop-back connection from the last layer output"}),
                # Checkpoint parameters - must be widgets to save to widgets_values
                "checkpoint_enabled": ("BOOLEAN", {"default": True, "widget": {"name": "checkpoint_enabled"}, "tooltip": "Enable automatic checkpoint saving for this network. Checkpoints saved to 'node_<ID>' subdirectories."}),
                "checkpoint_trigger_type": (["epoch", "time", "best_metric"], {"default": "epoch", "widget": {"name": "checkpoint_trigger_type"}, "tooltip": "When to save checkpoints: every N steps, time intervals, or metric improvements"}),
                "checkpoint_trigger_value": ("STRING", {"default": "50", "widget": {"name": "checkpoint_trigger_value"}, "tooltip": "Trigger value: number (steps), time format (1h30m), or 'min'/'max' (metrics)"}),
                "checkpoint_load_on_start": ("BOOLEAN", {"default": False, "widget": {"name": "checkpoint_load_on_start"}, "tooltip": "Automatically load saved checkpoint when network starts"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "MODEL")
    RETURN_NAMES = ("layers", "output", "model")
    FUNCTION = "forward"
    CATEGORY = "ml"
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("network")["color"]
    BGCOLOR = get_node_colors("network")["bgcolor"]

    def forward(self, input, to_output, unique_id=None, checkpoint_enabled=True, 
                checkpoint_trigger_type="epoch", checkpoint_trigger_value="50", 
                checkpoint_load_on_start=False, **kwargs):
        
        # Handle checkpoint configuration
        if checkpoint_enabled and unique_id:
            # Initialize checkpoint manager if needed
            if self.checkpoint_manager is None:
                try:
                    from export_system.templates.base.run_utils import CheckpointManager
                    self.checkpoint_manager = CheckpointManager(
                        node_id=unique_id,
                        trigger_type=checkpoint_trigger_type,
                        trigger_value=checkpoint_trigger_value
                    )
                except ImportError:
                    # Checkpoint manager not available
                    self.checkpoint_manager = None
        
        # If model doesn't exist, we're just passing through
        # The actual model is built during the linearization pass
        if self.model is None:
            # During runtime, look for pre-built model in context
            context = get_context()
            if hasattr(context, 'network_models') and unique_id in context.network_models:
                self.model = context.network_models[unique_id]
                
                # Load checkpoint if requested
                if checkpoint_load_on_start and self.checkpoint_manager:
                    self.checkpoint_manager.try_load_checkpoint(self.model)
        
        # Forward pass through model if available
        if self.model is not None:
            output = self.model(input)
            
            # Check if we should save checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.check_and_save(self.model)
            
            return (to_output, output, self.model)
        else:
            # Passthrough mode during graph construction
            return (to_output, to_output, None)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always execute to maintain state
        return True

# Node registration
NODE_CLASS_MAPPINGS = {
    "Network": NetworkNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Network": "Neural Network"
}