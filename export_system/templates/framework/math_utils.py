"""
Mathematical utility functions for DNNE nodes
Provides common metrics and distance functions used across various loss and metric nodes
"""

import torch
import torch.nn.functional as F
from typing import Optional


def max_abs_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute maximum absolute error (L∞ norm) between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor with maximum absolute difference
    """
    return torch.max(torch.abs(predictions - targets))


def euclidean_distance(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance (L2 norm) between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor with Euclidean distance
    """
    return torch.norm(predictions - targets, p=2)


def manhattan_distance(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Manhattan distance (L1 norm) between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor with sum of absolute differences
    """
    return torch.sum(torch.abs(predictions - targets))


def kl_divergence(predictions: torch.Tensor, targets: torch.Tensor, 
                  normalize: bool = False, eps: float = 1e-10) -> torch.Tensor:
    """
    Compute KL divergence between two distributions.
    
    Args:
        predictions: Predicted distribution (will be normalized to sum to 1)
        targets: Target distribution (will be normalized to sum to 1)
        normalize: If True, normalize by log(n) to get 0-1 range
        eps: Small epsilon to avoid numerical issues with log(0)
        
    Returns:
        Scalar tensor with KL divergence (or normalized KL divergence if normalize=True)
    """
    # Flatten tensors
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Ensure positive values for probability interpretation
    pred_positive = torch.abs(pred_flat) + eps
    target_positive = torch.abs(target_flat) + eps
    
    # Normalize to sum to 1 (probability distributions)
    pred_norm = pred_positive / pred_positive.sum()
    target_norm = target_positive / target_positive.sum()
    
    # Compute KL divergence: sum(p * log(p/q))
    # Using F.kl_div expects log probabilities for first argument
    kl_div = F.kl_div(torch.log(pred_norm), target_norm, reduction='sum')
    
    if normalize:
        # Normalize by log(n) where n is the vector size
        # This gives a value between 0 and 1
        n = pred_flat.numel()
        if n > 1:
            max_kl = torch.log(torch.tensor(n, dtype=torch.float32, device=pred_flat.device))
            return kl_div / max_kl
        else:
            return torch.tensor(0.0, device=pred_flat.device)  # Single element has no divergence
    
    return kl_div


def squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute squared error (squared L2 norm) between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor with sum of squared differences
    """
    return torch.sum((predictions - targets) ** 2)


def mean_squared_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean squared error between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor with mean of squared differences
    """
    return torch.mean((predictions - targets) ** 2)


def mean_absolute_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean absolute error between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor with mean of absolute differences
    """
    return torch.mean(torch.abs(predictions - targets))


def cosine_similarity(predictions: torch.Tensor, targets: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute cosine similarity between predictions and targets.
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        dim: Dimension along which to compute similarity
        
    Returns:
        Tensor with cosine similarity values
    """
    return F.cosine_similarity(predictions, targets, dim=dim)


def wasserstein_distance_1d(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance) between two distributions.
    Note: This is a simplified 1D implementation. For higher dimensions, use specialized libraries.
    
    Args:
        predictions: Predicted distribution
        targets: Target distribution
        
    Returns:
        Scalar tensor with Wasserstein distance
    """
    # Sort both distributions
    pred_sorted, _ = torch.sort(predictions.flatten())
    target_sorted, _ = torch.sort(targets.flatten())
    
    # Compute L1 distance between sorted distributions
    return torch.mean(torch.abs(pred_sorted - target_sorted))