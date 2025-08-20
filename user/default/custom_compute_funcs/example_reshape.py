"""
Example Reshape Function for Custom Computation Node

This example reshapes the input tensor to double its first dimension
and halve its second dimension (if possible).
Demonstrates actual computation and schema transformation.
"""

import torch

#
# These functions are for node configuration
#

def get_output_type():
    """Return the DNNE type for this node's output.
    We output a reshaped tensor."""
    print("[RESHAPE] get_output_type() called - returning 'RESHAPED_TENSOR'")
    return "RESHAPED_TENSOR"

def get_script_output_schema(initial=True, input_schema=None):
    """Return the output schema.
    This reshape doubles the first dimension and halves the second."""
    
    if initial or not input_schema:
        # Initial call - return partial schema
        print("[RESHAPE] get_script_output_schema(initial=True) - returning partial schema with None values")
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "shape": None,  # Will be computed from input
                    "flattened_size": None,  # Will be same as input
                    "dtype": None  # Will match input
                }
            }
        }
    else:
        # Resolution call - compute new shape
        input_info = input_schema["input"]
        
        # Get input shape and properties
        input_shape = input_info.get("shape", None)
        input_size = input_info.get("flattened_size", None)
        input_dtype = input_info.get("dtype", "float32")
        
        print(f"[RESHAPE] get_script_output_schema(initial=False) - resolving schema")
        print(f"  Input shape: {input_shape}")
        print(f"  Input flattened_size: {input_size}")
        print(f"  Input dtype: {input_dtype}")
        
        # Compute output shape
        if input_shape and len(input_shape) >= 2:
            # Double first dim, halve second dim
            output_shape = list(input_shape)
            output_shape[0] = output_shape[0] * 2
            output_shape[1] = output_shape[1] // 2
            
            # If second dimension becomes 0, make it 1
            if output_shape[1] == 0:
                output_shape[1] = 1
            
            print(f"  Computed output shape: {output_shape}")
        else:
            # Can't reshape properly, keep same shape
            output_shape = input_shape
            print(f"  Cannot reshape (not 2D+), keeping shape: {output_shape}")
        
        return {
            "outputs": {
                "output": {
                    "type": "tensor",
                    "shape": output_shape,
                    "flattened_size": input_size,  # Total size stays same
                    "dtype": input_dtype
                }
            }
        }

#
# This function gets called at runtime
#

# Counter for testing - exit after 5 calls
_compute_counter = 0

def compute(input: torch.Tensor) -> torch.Tensor:
    """
    Reshape function - doubles first dimension, halves second dimension.
    
    This demonstrates a custom computation that transforms tensor shape.
    The total number of elements remains the same.
    
    Args:
        input: Input tensor to reshape
        
    Returns:
        Reshaped tensor with modified dimensions
    """
    global _compute_counter
    _compute_counter += 1
    
    original_shape = input.shape
    print(f"[RESHAPE] compute() called at runtime (call #{_compute_counter})")
    print(f"  Input tensor shape: {original_shape}")
    print(f"  Input tensor dtype: {input.dtype}")
    
    if len(original_shape) < 2:
        # Can't reshape 1D tensor this way, return as-is
        print(f"  Cannot reshape 1D tensor, returning unchanged")
        return input
    
    # Calculate new shape
    new_shape = list(original_shape)
    new_shape[0] = new_shape[0] * 2
    new_shape[1] = new_shape[1] // 2
    
    # Check if reshape is valid
    original_elements = torch.prod(torch.tensor(original_shape)).item()
    new_elements = torch.prod(torch.tensor(new_shape)).item()
    
    if new_elements != original_elements:
        # If exact reshape isn't possible, keep original shape
        # (Can't double first and halve second while preserving elements for odd shapes)
        print(f"  Cannot reshape {original_shape} by doubling/halving dimensions")
        print(f"  Keeping original shape")
        new_shape = list(original_shape)
    
    try:
        # Reshape the tensor
        output = input.reshape(*new_shape)
        print(f"  Successfully reshaped to: {output.shape}")
        print(f"  Output tensor dtype: {output.dtype}")
        
        # For testing: exit after 5 calls
        if _compute_counter >= 5:
            print(f"\n[RESHAPE] Test complete - exiting after {_compute_counter} calls")
            # Use CauseExitException for graceful completion
            # Import from framework for clean async exit
            from framework.exceptions import CauseExitException
            raise CauseExitException("2", "Test complete - 5 iterations done", exit_code=0)
        
        return output
    except RuntimeError as e:
        # If reshape fails, return original
        print(f"  ERROR: Could not reshape from {original_shape} to {new_shape}")
        print(f"  Error: {e}")
        print(f"  Returning original tensor unchanged")
        return input