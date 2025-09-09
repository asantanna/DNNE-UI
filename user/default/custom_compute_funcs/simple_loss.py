
import torch

def compute(input_tensor, extra_args=None):
    """Simple MSE loss against zeros
    
    Args:
        input_tensor: Input tensor to compute loss for
        extra_args: Optional additional arguments (unused in this function)
        
    Returns:
        MSE loss value
    """
    target = torch.zeros_like(input_tensor)
    return torch.nn.functional.mse_loss(input_tensor, target)
