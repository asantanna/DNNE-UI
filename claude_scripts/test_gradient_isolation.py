#!/usr/bin/env python3
"""
Test script for ContextVar-based gradient isolation mechanism.

This script validates that:
1. Authorized optimizers can update network weights
2. Unauthorized optimizers cannot update network weights but gradients flow through
3. Multiple networks can have different authorized optimizers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from contextvars import ContextVar
import numpy as np

# Global context variable to track current optimizer
CURRENT_OPTIMIZER_ID = ContextVar("CURRENT_OPTIMIZER_ID", default=None)

class OptimizerContext:
    """Context manager to set current optimizer ID during backward pass"""
    def __init__(self, optimizer_id: str):
        self.optimizer_id = optimizer_id
        self.token = None
    
    def __enter__(self):
        self.token = CURRENT_OPTIMIZER_ID.set(self.optimizer_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        CURRENT_OPTIMIZER_ID.reset(self.token)

def zero_grad_if_unauthorized(module: nn.Module, authorized_id: str):
    """Register gradient hooks that zero gradients from unauthorized optimizers"""
    handles = []
    for param in module.parameters():
        def make_hook():
            def hook(grad):
                current_opt = CURRENT_OPTIMIZER_ID.get()
                if current_opt == authorized_id:
                    return grad  # Allow gradient update
                else:
                    return torch.zeros_like(grad)  # Block update but pass through
            return hook
        handles.append(param.register_hook(make_hook()))
    return handles

def test_gradient_isolation():
    """Test gradient isolation between multiple networks and optimizers"""
    
    print("=" * 60)
    print("Testing ContextVar-based Gradient Isolation")
    print("=" * 60)
    
    # Create three simple networks
    control_net_1 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    control_net_2 = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    
    shadow_net = nn.Sequential(
        nn.Linear(20, 30),  # Takes concatenated input
        nn.ReLU(),
        nn.Linear(30, 10)   # Predicts next observation
    )
    
    # Register gradient isolation
    # Control nets are authorized by their respective optimizers
    control_1_handles = zero_grad_if_unauthorized(control_net_1, "SGD_Control_1")
    control_2_handles = zero_grad_if_unauthorized(control_net_2, "SGD_Control_2")
    shadow_handles = zero_grad_if_unauthorized(shadow_net, "SGD_Shadow")
    
    # Create optimizers
    opt_control_1 = optim.SGD(control_net_1.parameters(), lr=0.1)
    opt_control_2 = optim.SGD(control_net_2.parameters(), lr=0.1)
    opt_shadow = optim.SGD(shadow_net.parameters(), lr=0.1)
    
    # Save initial weights to verify updates
    initial_control_1 = control_net_1[0].weight.data.clone()
    initial_control_2 = control_net_2[0].weight.data.clone()
    initial_shadow = shadow_net[0].weight.data.clone()
    
    # Test data
    obs = torch.randn(4, 10)  # Batch of 4 observations
    target = torch.randn(4, 10)  # Target for prediction
    
    print("\n1. Testing Shadow Network Training (authorized update)")
    print("-" * 40)
    
    # Forward pass through control networks
    action_1 = control_net_1(obs)
    action_2 = control_net_2(obs)
    
    # Concatenate observations and actions for shadow network
    shadow_input = torch.cat([action_1, action_2], dim=1)
    pred_obs = shadow_net(shadow_input)
    
    # Shadow loss (prediction error)
    shadow_loss = nn.MSELoss()(pred_obs, target)
    
    # Shadow optimizer updates shadow network
    opt_shadow.zero_grad()
    with OptimizerContext("SGD_Shadow"):
        shadow_loss.backward(retain_graph=True)  # Retain for control training
    opt_shadow.step()
    
    shadow_updated = not torch.allclose(initial_shadow, shadow_net[0].weight.data)
    control_1_unchanged = torch.allclose(initial_control_1, control_net_1[0].weight.data)
    control_2_unchanged = torch.allclose(initial_control_2, control_net_2[0].weight.data)
    
    print(f"Shadow network updated: {shadow_updated} ✓" if shadow_updated else "Shadow network NOT updated ✗")
    print(f"Control net 1 unchanged: {control_1_unchanged} ✓" if control_1_unchanged else "Control net 1 CHANGED ✗")
    print(f"Control net 2 unchanged: {control_2_unchanged} ✓" if control_2_unchanged else "Control net 2 CHANGED ✗")
    
    print("\n2. Testing Control Network 1 Training (gradient flows through shadow)")
    print("-" * 40)
    
    # Reset for control training
    initial_control_1 = control_net_1[0].weight.data.clone()
    initial_shadow_2 = shadow_net[0].weight.data.clone()
    
    # Forward pass again
    action_1 = control_net_1(obs)
    action_2 = control_net_2(obs)
    shadow_input = torch.cat([action_1, action_2], dim=1)
    pred_obs = shadow_net(shadow_input)
    
    # Control loss (uses prediction from shadow network)
    control_loss = nn.MSELoss()(pred_obs, target)
    
    # Control optimizer 1 updates control network 1
    opt_control_1.zero_grad()
    with OptimizerContext("SGD_Control_1"):
        control_loss.backward(retain_graph=True)
    opt_control_1.step()
    
    control_1_updated = not torch.allclose(initial_control_1, control_net_1[0].weight.data)
    shadow_unchanged = torch.allclose(initial_shadow_2, shadow_net[0].weight.data)
    
    print(f"Control net 1 updated: {control_1_updated} ✓" if control_1_updated else "Control net 1 NOT updated ✗")
    print(f"Shadow network unchanged: {shadow_unchanged} ✓" if shadow_unchanged else "Shadow network CHANGED ✗")
    
    # Check that gradients flowed through shadow network
    has_gradients = control_net_1[0].weight.grad is not None and torch.any(control_net_1[0].weight.grad != 0)
    print(f"Gradients flowed to control net 1: {has_gradients} ✓" if has_gradients else "No gradients ✗")
    
    print("\n3. Testing Unauthorized Updates are Blocked")
    print("-" * 40)
    
    # Save current weights
    initial_control_1 = control_net_1[0].weight.data.clone()
    initial_control_2 = control_net_2[0].weight.data.clone()
    
    # Control optimizer 1 tries to update control network 2 (unauthorized)
    opt_control_1.zero_grad()
    opt_control_2.zero_grad()
    
    # Add control_net_2 parameters to opt_control_1 temporarily
    temp_optimizer = optim.SGD(list(control_net_1.parameters()) + list(control_net_2.parameters()), lr=0.1)
    
    action_1 = control_net_1(obs)
    action_2 = control_net_2(obs)
    loss = nn.MSELoss()(action_1 + action_2, target)
    
    temp_optimizer.zero_grad()
    with OptimizerContext("SGD_Control_1"):  # Only authorized for control_net_1
        loss.backward()
    temp_optimizer.step()
    
    control_1_updated = not torch.allclose(initial_control_1, control_net_1[0].weight.data)
    control_2_unchanged = torch.allclose(initial_control_2, control_net_2[0].weight.data)
    
    print(f"Control net 1 updated (authorized): {control_1_updated} ✓" if control_1_updated else "Control net 1 NOT updated ✗")
    print(f"Control net 2 unchanged (unauthorized): {control_2_unchanged} ✓" if control_2_unchanged else "Control net 2 CHANGED ✗")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_tests_passed = (
        shadow_updated and control_1_unchanged and control_2_unchanged and  # Test 1
        control_1_updated and shadow_unchanged and has_gradients and  # Test 2
        control_2_unchanged  # Test 3
    )
    
    if all_tests_passed:
        print("✅ All gradient isolation tests PASSED!")
        print("\nKey findings:")
        print("1. Networks only accept weight updates from authorized optimizers")
        print("2. Gradients flow through networks even when updates are blocked")
        print("3. Multiple networks can coexist with different authorizations")
        print("\nThis confirms the gradient isolation mechanism works correctly")
        print("for the shadow environment architecture.")
    else:
        print("❌ Some tests FAILED - gradient isolation not working correctly")
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_gradient_isolation()
    exit(0 if success else 1)