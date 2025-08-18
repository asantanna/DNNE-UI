"""
Example Identity Function for Custom Computation Node

This simple example returns the input tensor unchanged.
It serves as a template for creating custom compute functions.
"""

import torch

#
# These functions are for node configuration
#

def get_output_type():
    """Return the DNNE type for this node's output.
    Since this is an identity function, we accept any tensor type."""
    return "*TENSOR"

def get_script_output_schema(initial=True, input_schema=None):
    """Return the output schema.
    For identity function, output schema matches input schema."""
    
    if initial or not input_schema:
        # Initial call - return partial schema
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "shape": None,  # Will be resolved from input
                    "flattened_size": None,  # Will be resolved from input
                    "dtype": None  # Will be resolved from input
                }
            }
        }
    else:
        # Resolution call - pass through input schema
        return {
            "outputs": {
                "output": input_schema["input"]  # Output matches input exactly
            }
        }

#
# This function gets called at runtime
#

def compute(input: torch.Tensor) -> torch.Tensor:
    """
    Identity function - returns input unchanged.
    
    This is the simplest possible compute function.
    It can be used for testing or as a passthrough.
    
    Args:
        input: Any input tensor
        
    Returns:
        The same tensor, unchanged
    """
    return input