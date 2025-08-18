"""
Example Filter Function for Custom Computation Node

This example filters tensors based on their mean value.
Only tensors with positive mean are passed through.
"""

import torch
from typing import Optional

#
# These functions are for node configuration
#

def get_output_type():
    """Return the DNNE type for this node's output.
    Filter passes through any tensor type when condition is met."""
    return "*TENSOR"

def get_script_output_schema(initial=True, input_schema=None):
    """Return the output schema.
    Filter outputs match input schema when the condition is met."""
    
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
        # Resolution call - output matches input when passed through
        return {
            "outputs": {
                "output": input_schema["input"]  # Output matches input when filter passes
            }
        }

#
# This function gets called at runtime
#

def compute(input: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Filter function - only passes through tensors with positive mean.
    
    This demonstrates how to use the Custom Computation node as a filter.
    When None is returned, no output is emitted.
    
    Args:
        input: Input tensor to evaluate
        
    Returns:
        The input tensor if mean > 0, None otherwise
    """
    mean_value = input.mean().item()
    
    if mean_value > 0:
        # Pass through tensors with positive mean
        return input
    else:
        # Filter out tensors with non-positive mean
        return None