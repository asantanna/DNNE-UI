# or_node.py
"""
Generic OR/ANY Node for routing inputs
Outputs when ANY input becomes available - useful for routing between multiple data sources
"""

import torch
from typing import Optional
from inspect import cleandoc

# Import base node from robotics nodes
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from robotics_nodes.base_node import RoboticsNodeBase


class ORNode(RoboticsNodeBase):
    """
    OR/ANY Node for routing inputs
    Outputs when ANY input becomes available - useful for routing between multiple data sources
    
    Common use cases:
    - RL training loops: routing initial state vs ongoing state
    - Data pipeline: selecting between different data sources
    - Conditional routing: switching between inputs based on availability
    """
    
    CATEGORY = "utility"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_a": ("TENSOR",),
                "input_b": ("TENSOR",),
                "input_c": ("TENSOR",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "route_input"
    DESCRIPTION = cleandoc(__doc__)
    
    def __init__(self):
        super().__init__()
        self.last_input_source = None
        self.output_count = 0
    
    def route_input(self, input_a: Optional[torch.Tensor] = None, 
                   input_b: Optional[torch.Tensor] = None, 
                   input_c: Optional[torch.Tensor] = None):
        """Route the first available input to output"""
        
        # Check inputs in order of priority (A, B, C)
        if input_a is not None:
            self.last_input_source = "A"
            self.output_count += 1
            print(f"OR Node: Routing input A (shape: {input_a.shape}) - output #{self.output_count}")
            return (input_a,)
        
        elif input_b is not None:
            self.last_input_source = "B"
            self.output_count += 1
            print(f"OR Node: Routing input B (shape: {input_b.shape}) - output #{self.output_count}")
            return (input_b,)
        
        elif input_c is not None:
            self.last_input_source = "C"
            self.output_count += 1
            print(f"OR Node: Routing input C (shape: {input_c.shape}) - output #{self.output_count}")
            return (input_c,)
        
        else:
            # No input available - should not happen in normal operation
            raise ValueError("OR Node: No inputs available. At least one input must be connected.")
    
    @classmethod
    def IS_CHANGED(cls, **inputs):
        """Always execute when any input is available"""
        return float("nan")  # This ensures node always runs when inputs change