"""
Example Filter Function for Custom Computation Node

This example filters tensors based on their mean value.
Only tensors with positive mean are passed through.
"""

import torch
from typing import Optional


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