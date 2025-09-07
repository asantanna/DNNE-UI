
import torch

def compute(input_tensor):
    """Simple MSE loss against zeros"""
    target = torch.zeros_like(input_tensor)
    return torch.nn.functional.mse_loss(input_tensor, target)
