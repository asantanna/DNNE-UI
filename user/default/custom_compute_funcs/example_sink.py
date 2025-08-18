"""
Example Sink Function for Custom Computation Node

This example acts as a data sink - consumes all inputs without producing outputs.
Useful for terminating data flows or for side-effect operations like logging.
"""

import torch
from typing import Optional

#
# These functions are for node configuration
#

def get_output_type():
    """Return the DNNE type for this node's output.
    Sink nodes technically output nothing, but we use VOID type."""
    return "VOID"

def get_script_output_schema(initial=True, input_schema=None):
    """Return the output schema.
    Sink nodes don't produce output, so schema is minimal."""
    
    # Sink always has the same schema - no output
    return {
        "outputs": {
            "output": {
                "type": "void",
                "shape": None,
                "flattened_size": 0,
                "dtype": None
            }
        }
    }

#
# This function gets called at runtime
#

def compute(input: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Sink function - consumes input without producing output.
    
    This demonstrates how to use the Custom Computation node as a sink.
    Always returns None, effectively consuming all data.
    
    Can be used for:
    - Terminating data flows
    - Logging/monitoring without passing data forward
    - Side-effect operations
    
    Args:
        input: Input tensor to consume
        
    Returns:
        Always None (no output)
    """
    # You could add logging or other side effects here
    # For example:
    # print(f"Sink consumed tensor with shape: {input.shape}, mean: {input.mean().item():.4f}")
    
    # Always return None - consume everything
    return None