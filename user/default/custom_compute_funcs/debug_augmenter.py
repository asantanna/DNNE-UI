"""
Example custom computation script that augments actions with debug visualization.
Uses extra_args tensor for dynamic position data and config for static settings.
FAIL-FAST: No defaults - all required config values must be provided.
"""

# Global config storage
config = {}

def set_config_info(cfg):
    """Called by CustomComputation node to set configuration."""
    global config
    config = cfg
    print(f"Debug augmenter config set: {config}")

def compute(action, extra_args=None):
    """
    Augment action tensor with debug visualization data.
    
    Args:
        action: Action tensor from control network
        extra_args: Tensor containing dynamic data (e.g., predicted positions)
        
    Returns:
        Action tensor with extra_args dictionary attached if DEBUG is enabled
    """
    # FAIL-FAST: Check required config keys
    if "DEBUG" not in config:
        raise RuntimeError("config must contain 'DEBUG' key")
    
    # Check if debug is enabled in config
    if config["DEBUG"]:
        # FAIL-FAST: Require extra_args when DEBUG is True
        if extra_args is None:
            raise RuntimeError("extra_args tensor must be provided when DEBUG=True")
        
        # FAIL-FAST: Require extra_args_len when DEBUG is True
        if "extra_args_len" not in config:
            raise RuntimeError("config must contain 'extra_args_len' when DEBUG=True")
        
        # Get the key name from config (default to 'data' if not specified)
        key_name = config.get("key_name", "data")
        
        # Get expected tensor length from config
        extra_args_len = config["extra_args_len"]
        
        # Convert tensor to list
        import torch
        if isinstance(extra_args, torch.Tensor):
            if extra_args.numel() != extra_args_len:
                raise RuntimeError(f"extra_args tensor must have exactly {extra_args_len} elements, got shape: {extra_args.shape}")
            debug_data = extra_args.flatten().tolist()
        else:
            raise RuntimeError(f"extra_args must be a tensor, got: {type(extra_args)}")
        
        # Create extra_args dictionary on action with the specified key
        action.extra_args = {
            key_name: debug_data
        }
    
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