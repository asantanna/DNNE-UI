"""
Example Identity Function for Custom Computation Node

This simple example returns the input tensor unchanged.
It serves as a template for creating custom compute functions.
"""

import torch


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