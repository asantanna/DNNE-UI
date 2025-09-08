"""
Example custom computation script that augments actions with debug visualization.
Uses the config dictionary to control debug behavior.
"""

import torch

# Global config storage
config = {}

def set_config_info(cfg):
    """Called by CustomComputation node to set configuration."""
    global config
    config = cfg
    print(f"Debug augmenter config set: {config}")

def compute(action):
    """
    Augment action tensor with debug visualization data.
    
    Args:
        action: Action tensor from control network
        
    Returns:
        Action tensor with extra_args attached if DEBUG is enabled
    """
    # Check if debug is enabled in config
    if config.get("DEBUG", False):
        # Get debug position from config (or use default)
        debug_pos = config.get("debug_pos", [0.0, 0.0, 1.5])
        
        # Attach debug visualization data to action
        action.extra_args = {
            "debug_sphere_pos": debug_pos
        }
        
        # Optional: scale action if scale factor provided
        if "scale" in config:
            action = action * config["scale"]
    
    return action

def get_output_type():
    """Return the output type for this computation."""
    return "TENSOR"

def get_script_output_schema(initial=True, input_schema=None):
    """Return output schema information."""
    return {
        "outputs": {
            "output": {
                "type": "TENSOR",
                "description": "Action tensor, possibly with debug augmentation"
            }
        }
    }