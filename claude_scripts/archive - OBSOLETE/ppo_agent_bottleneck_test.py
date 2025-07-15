#!/usr/bin/env python3
"""
PPO Agent Bottleneck Test
Tests the actual forward pass performance of the PPO agent in isolation
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import time
import numpy as np

class SimplePPOAgent:
    """Minimal PPO agent matching DNNE configuration"""
    
    def __init__(self, obs_dim=4, hidden_sizes=[32, 32], action_dim=1, device="cuda"):
        self.device = device
        self.action_dim = action_dim
        
        # Build network matching DNNE config
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ELU()
            ])
            prev_size = hidden_size
            
        self.shared_layers = nn.Sequential(*layers).to(device)
        self.policy_mean = nn.Linear(prev_size, action_dim).to(device)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim)).to(device)
        self.value_head = nn.Linear(prev_size, 1).to(device)
        
    def forward(self, observations):
        """Forward pass matching DNNE implementation"""
        # Ensure correct device
        if observations.device != torch.device(self.device):
            observations = observations.to(self.device)
            
        # Forward through shared layers
        features = self.shared_layers(observations)
        
        # Compute value
        value = self.value_head(features).squeeze(-1)
        
        # Compute policy
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std)
        
        # Create distribution and sample
        policy_dist = dist.Normal(action_mean, action_std)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action).sum(dim=-1)
        
        return action, value, log_prob

def test_forward_pass_performance(batch_size=512, num_iterations=1000):
    """Test forward pass performance with different configurations"""
    
    print(f"Testing PPO Agent Forward Pass Performance")
    print(f"Batch size: {batch_size}")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)
    
    # Create agent
    agent = SimplePPOAgent(obs_dim=4, hidden_sizes=[32, 32], action_dim=1)
    
    # Test input
    observations = torch.randn(batch_size, 4, device="cuda")
    
    # Warmup
    for _ in range(10):
        agent.forward(observations)
    
    # Synchronize CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Time the forward passes
    start_time = time.time()
    
    for _ in range(num_iterations):
        action, value, log_prob = agent.forward(observations)
        
    # Synchronize CUDA for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    time_per_iteration = total_time / num_iterations
    ms_per_iteration = time_per_iteration * 1000
    iterations_per_second = num_iterations / total_time
    
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Time per forward pass: {ms_per_iteration:.3f} ms")
    print(f"Forward passes per second: {iterations_per_second:.0f}")
    print()
    
    # Test with different batch sizes
    print("Testing different batch sizes:")
    print("-" * 40)
    batch_sizes = [1, 16, 64, 256, 512, 1024]
    
    for bs in batch_sizes:
        obs = torch.randn(bs, 4, device="cuda")
        
        # Time 100 iterations
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            agent.forward(obs)
            
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        ms_per_pass = (elapsed / 100) * 1000
        print(f"Batch size {bs:4d}: {ms_per_pass:6.3f} ms per forward pass")
    
    print()
    print("Analysis:")
    if ms_per_iteration < 1.0:
        print("✅ Forward pass is fast (<1ms) - bottleneck is elsewhere")
    elif ms_per_iteration < 10.0:
        print("⚠️  Forward pass is moderately slow (1-10ms)")
    else:
        print("❌ Forward pass is very slow (>10ms) - this is the bottleneck!")
    
    return ms_per_iteration

if __name__ == "__main__":
    # Test with DNNE's batch size
    ms_time = test_forward_pass_performance(batch_size=512, num_iterations=1000)
    
    print("\nConclusion:")
    print(f"DNNE reports 60ms forward pass, but direct test shows {ms_time:.3f}ms")
    if ms_time < 10:
        print("The bottleneck is NOT in the PyTorch forward pass itself.")
        print("It must be in the async wrapper or queue coordination.")
    else:
        print("The forward pass itself is the bottleneck.")