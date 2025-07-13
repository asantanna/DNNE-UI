#!/usr/bin/env python3
"""
Compare PPO Agent performance in different contexts
"""

import sys
import time
import asyncio
from pathlib import Path

# Add export directory to path
sys.path.insert(0, str(Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")))

async def test_in_dnne_context():
    """Test PPO Agent in full DNNE context"""
    print("Test 1: PPO Agent in DNNE Context")
    print("=" * 60)
    
    # Import Isaac Gym first
    import isaacgym
    import torch  # Now safe to import torch
    
    # Import the actual node
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    
    # Create node exactly as DNNE does
    agent = PPOAgentNode_3("3")
    
    # Create observations similar to what DNNE provides
    observations = torch.randn(512, 4, device="cuda")
    
    # Warmup
    print("Warming up...")
    for i in range(10):
        result = await agent.compute(observations)
        if i == 0:
            print(f"First call (includes model build)")
    
    # Time post-warmup calls
    print("\nTiming post-warmup calls:")
    times = []
    
    # Test with CUDA synchronization (which might be happening in DNNE)
    for i in range(20):
        torch.cuda.synchronize()  # Ensure previous operations complete
        start = time.perf_counter()
        
        result = await agent.compute(observations)
        
        torch.cuda.synchronize()  # Ensure this operation completes
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if i < 5:  # Print first few
            print(f"  Call {i+1}: {elapsed:.2f}ms")
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms")
    print(f"  Max: {max_time:.2f}ms")
    
    return avg_time

async def test_isolated():
    """Test just the PyTorch operations"""
    print("\n\nTest 2: Isolated PyTorch Operations")
    print("=" * 60)
    
    import torch
    import torch.nn as nn
    import torch.distributions as dist
    
    # Build network matching DNNE config
    device = "cuda"
    obs_dim = 4
    hidden_sizes = [32, 32]
    action_dim = 1
    
    # Build layers
    layers = []
    prev_size = obs_dim
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.ELU()
        ])
        prev_size = hidden_size
    
    shared_layers = nn.Sequential(*layers).to(device)
    policy_mean = nn.Linear(prev_size, action_dim).to(device)
    policy_log_std = nn.Parameter(torch.zeros(action_dim)).to(device)
    value_head = nn.Linear(prev_size, 1).to(device)
    
    # Test observations
    observations = torch.randn(512, 4, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        features = shared_layers(observations)
        value = value_head(features).squeeze(-1)
        action_mean = policy_mean(features)
        action_std = torch.exp(policy_log_std)
        policy_dist = dist.Normal(action_mean, action_std)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action).sum(dim=-1)
    
    # Time the operations
    print("\nTiming operations:")
    times = []
    
    for i in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        features = shared_layers(observations)
        value = value_head(features).squeeze(-1)
        action_mean = policy_mean(features)
        action_std = torch.exp(policy_log_std)
        policy_dist = dist.Normal(action_mean, action_std)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action).sum(dim=-1)
        
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        
        if i < 5:
            print(f"  Call {i+1}: {elapsed:.2f}ms")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage: {avg_time:.2f}ms")
    
    return avg_time

async def test_memory_state():
    """Check if memory/cache state affects performance"""
    print("\n\nTest 3: Memory and Cache Effects")
    print("=" * 60)
    
    import isaacgym
    import torch
    import gc
    
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    
    observations = torch.randn(512, 4, device="cuda")
    
    # Test 1: Fresh agent
    agent1 = PPOAgentNode_3("test1")
    await agent1.compute(observations)  # Build model
    
    times1 = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        await agent1.compute(observations)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times1.append(elapsed)
    
    avg1 = sum(times1) / len(times1)
    print(f"Fresh agent: {avg1:.2f}ms average")
    
    # Test 2: After creating many tensors (memory pressure)
    print("\nCreating memory pressure...")
    tensors = []
    for _ in range(100):
        tensors.append(torch.randn(1024, 1024, device="cuda"))
    
    times2 = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        await agent1.compute(observations)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times2.append(elapsed)
    
    avg2 = sum(times2) / len(times2)
    print(f"With memory pressure: {avg2:.2f}ms average")
    print(f"Difference: {avg2 - avg1:.2f}ms")
    
    # Clean up
    del tensors
    gc.collect()
    torch.cuda.empty_cache()

async def test_import_overhead():
    """Test if imports in compute() add overhead"""
    print("\n\nTest 4: Import Overhead in compute()")
    print("=" * 60)
    
    import isaacgym
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    
    # Create modified agent that skips imports
    class FastPPOAgent(PPOAgentNode_3):
        async def compute(self, observations):
            # Pre-import everything
            import torch
            import torch.nn as nn
            import torch.distributions as dist
            import builtins
            from framework.base import _global_profiler
            
            # Now time the actual compute without imports
            start = time.perf_counter()
            
            # Copy the compute logic without the import statements
            profile_enabled = getattr(builtins, 'PROFILE_MODE', False)
            
            if isinstance(observations, torch.Tensor):
                if observations.device != torch.device(self.device):
                    observations = observations.to(self.device)
            else:
                observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
            
            if observations.dim() == 1:
                observations = observations.unsqueeze(0)
                single_sample = True
            else:
                single_sample = False
            
            batch_size, obs_dim = observations.shape
            
            if self.model is None:
                self.build_model(obs_dim)
            
            # Forward pass
            features = self.shared_layers(observations)
            value = self.value_head(features).squeeze(1)
            
            action_mean = self.policy_mean(features)
            action_std = torch.exp(self.policy_log_std)
            policy_dist = dist.Normal(action_mean, action_std)
            action = policy_dist.sample()
            log_prob = policy_dist.log_prob(action).sum(dim=-1)
            action_params = torch.cat([action_mean, action_std.expand_as(action_mean)], dim=-1)
            
            if single_sample:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                action_params = action_params.squeeze(0)
            
            policy_output = {
                "action": action,
                "value": value,
                "log_prob": log_prob,
                "action_params": action_params
            }
            
            compute_time = (time.perf_counter() - start) * 1000
            
            return {
                "policy_output": policy_output,
                "model": self.model
            }
    
    # Compare original vs optimized
    observations = torch.randn(512, 4, device="cuda")
    
    original = PPOAgentNode_3("original")
    fast = FastPPOAgent("fast")
    
    # Warmup both
    for _ in range(5):
        await original.compute(observations)
        await fast.compute(observations)
    
    # Time both
    original_times = []
    fast_times = []
    
    for _ in range(20):
        # Original
        torch.cuda.synchronize()
        start = time.perf_counter()
        await original.compute(observations)
        torch.cuda.synchronize()
        original_times.append((time.perf_counter() - start) * 1000)
        
        # Fast
        torch.cuda.synchronize()
        start = time.perf_counter()
        await fast.compute(observations)
        torch.cuda.synchronize()
        fast_times.append((time.perf_counter() - start) * 1000)
    
    original_avg = sum(original_times) / len(original_times)
    fast_avg = sum(fast_times) / len(fast_times)
    
    print(f"Original (with imports): {original_avg:.2f}ms")
    print(f"Optimized (imports outside): {fast_avg:.2f}ms")
    print(f"Import overhead: {original_avg - fast_avg:.2f}ms")

async def main():
    """Run all tests"""
    
    # Import Isaac Gym first before any test
    import isaacgym
    
    # Test in DNNE context
    dnne_time = await test_in_dnne_context()
    
    # Test isolated
    isolated_time = await test_isolated()
    
    # Memory effects
    await test_memory_state()
    
    # Import overhead
    await test_import_overhead()
    
    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"DNNE Context: {dnne_time:.2f}ms")
    print(f"Isolated PyTorch: {isolated_time:.2f}ms")
    print(f"Overhead: {dnne_time - isolated_time:.2f}ms ({(dnne_time/isolated_time - 1)*100:.1f}%)")

if __name__ == "__main__":
    # Run in the DNNE environment
    import subprocess
    import os
    
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("Activating conda environment...")
        cmd = [
            "bash", "-c",
            "source /home/asantanna/miniconda/bin/activate DNNE_PY38 && python " + __file__
        ]
        subprocess.run(cmd)
    else:
        asyncio.run(main())