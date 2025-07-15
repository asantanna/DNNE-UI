#!/usr/bin/env python3
"""
Benchmark raw PyTorch performance vs DNNE performance
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import time
import numpy as np
import asyncio

# Configuration matching DNNE
BATCH_SIZE = 512
OBS_DIM = 4
ACTION_DIM = 1
HIDDEN_SIZES = [32, 32]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_ITERATIONS = 1000

class SimplePPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        
        # Build shared layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ELU()
            ])
            prev_size = hidden_size
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head (continuous)
        self.policy_mean = nn.Linear(prev_size, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value_head = nn.Linear(prev_size, 1)
        
    def forward(self, obs):
        features = self.shared_layers(obs)
        
        # Value
        value = self.value_head(features).squeeze(-1)
        
        # Policy
        action_mean = self.policy_mean(features)
        action_std = torch.exp(self.policy_log_std)
        
        # Sample action
        policy_dist = dist.Normal(action_mean, action_std)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action).sum(dim=-1)
        
        return action, value, log_prob

def benchmark_raw_pytorch():
    """Benchmark raw PyTorch performance"""
    print(f"🔧 Benchmarking raw PyTorch on {DEVICE}")
    
    # Create model
    model = SimplePPONetwork(OBS_DIM, ACTION_DIM, HIDDEN_SIZES).to(DEVICE)
    model.eval()
    
    # Create dummy data
    observations = torch.randn(BATCH_SIZE, OBS_DIM, device=DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(observations)
    
    # Benchmark
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(NUM_ITERATIONS):
            action, value, log_prob = model(observations)
    
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / NUM_ITERATIONS) * 1000
    fps = NUM_ITERATIONS / total_time
    
    print(f"✅ Raw PyTorch Results:")
    print(f"   Total time: {total_time:.3f}s for {NUM_ITERATIONS} iterations")
    print(f"   Average time per forward pass: {avg_time_ms:.3f}ms")
    print(f"   Throughput: {fps:.1f} FPS")
    print(f"   Batch throughput: {fps * BATCH_SIZE:.1f} samples/second")
    
    return avg_time_ms

def benchmark_with_async_overhead():
    """Benchmark with simulated async/queue overhead"""
    print(f"\n🔧 Benchmarking with async simulation")
    
    async def async_forward(model, observations):
        """Simulate async forward pass"""
        with torch.no_grad():
            action, value, log_prob = model(observations)
        return action, value, log_prob
    
    async def run_benchmark():
        # Create model
        model = SimplePPONetwork(OBS_DIM, ACTION_DIM, HIDDEN_SIZES).to(DEVICE)
        model.eval()
        
        # Create dummy data
        observations = torch.randn(BATCH_SIZE, OBS_DIM, device=DEVICE)
        
        # Warmup
        for _ in range(10):
            await async_forward(model, observations)
        
        # Benchmark
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        start_time = time.perf_counter()
        
        for _ in range(NUM_ITERATIONS):
            await async_forward(model, observations)
        
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / NUM_ITERATIONS) * 1000
        fps = NUM_ITERATIONS / total_time
        
        print(f"✅ Async Simulation Results:")
        print(f"   Total time: {total_time:.3f}s for {NUM_ITERATIONS} iterations")
        print(f"   Average time per forward pass: {avg_time_ms:.3f}ms")
        print(f"   Throughput: {fps:.1f} FPS")
        print(f"   Batch throughput: {fps * BATCH_SIZE:.1f} samples/second")
        
        return avg_time_ms
    
    return asyncio.run(run_benchmark())

def main():
    print("=" * 60)
    print("🏁 DNNE vs Raw PyTorch Performance Benchmark")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Network: {HIDDEN_SIZES}")
    print(f"  Iterations: {NUM_ITERATIONS}")
    print("=" * 60)
    
    # Run benchmarks
    raw_time = benchmark_raw_pytorch()
    async_time = benchmark_with_async_overhead()
    
    # Compare with DNNE reported time
    dnne_time = 69.0  # 0.069s = 69ms from DNNE logs
    
    print("\n" + "=" * 60)
    print("📊 Performance Comparison:")
    print(f"  Raw PyTorch: {raw_time:.3f}ms per forward pass")
    print(f"  With Async: {async_time:.3f}ms per forward pass")
    print(f"  DNNE Actual: {dnne_time:.3f}ms per forward pass")
    print(f"\n  DNNE Overhead: {dnne_time/raw_time:.1f}x slower than raw PyTorch")
    print(f"  Async Overhead: {async_time/raw_time:.1f}x slower than raw PyTorch")
    print("=" * 60)

if __name__ == "__main__":
    main()