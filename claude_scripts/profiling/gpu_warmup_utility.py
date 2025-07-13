#!/usr/bin/env python3
"""
GPU Warmup Utility for DNNE Performance Testing

This utility ensures the GPU is fully active before performance measurements.
GPU sleep mode can cause significant performance regression in initial measurements.
"""

import time
import torch
import torch.nn as nn

def warmup_gpu(duration_seconds: float = 3.0, verbose: bool = True) -> dict:
    """
    Warm up the GPU with compute operations to exit sleep mode
    
    Args:
        duration_seconds: How long to run warmup operations
        verbose: Whether to print progress messages
        
    Returns:
        dict: Warmup statistics including timings
    """
    if verbose:
        print(f"🔥 Warming up GPU for {duration_seconds:.1f} seconds...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cpu":
        if verbose:
            print("⚠️  No CUDA GPU available, skipping warmup")
        return {"device": "cpu", "warmup_time": 0.0}
    
    # Create a simple neural network for warmup
    warmup_net = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    ).to(device)
    
    # Track warmup timing
    start_time = time.perf_counter()
    end_time = start_time + duration_seconds
    
    iterations = 0
    first_iteration_time = None
    last_iteration_time = None
    
    if verbose:
        print("  Running GPU warmup operations...")
    
    while time.perf_counter() < end_time:
        iteration_start = time.perf_counter()
        
        # Create random input data
        batch_size = 256
        input_data = torch.randn(batch_size, 1024, device=device)
        
        # Forward pass
        output = warmup_net(input_data)
        
        # Backward pass to fully engage GPU
        loss = output.sum()
        loss.backward()
        
        # Clear gradients
        warmup_net.zero_grad()
        
        # Ensure operation completes
        torch.cuda.synchronize()
        
        iteration_time = (time.perf_counter() - iteration_start) * 1000
        
        if first_iteration_time is None:
            first_iteration_time = iteration_time
        last_iteration_time = iteration_time
        
        iterations += 1
        
        # Progress indicator
        if verbose and iterations % 50 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"    {iterations} iterations, {elapsed:.1f}s elapsed")
    
    total_warmup_time = time.perf_counter() - start_time
    
    # GPU memory info
    if hasattr(torch.cuda, 'memory_allocated'):
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    else:
        memory_allocated = memory_reserved = 0
    
    stats = {
        "device": str(device),
        "warmup_time": total_warmup_time,
        "iterations": iterations,
        "first_iteration_ms": first_iteration_time,
        "last_iteration_ms": last_iteration_time,
        "speedup_ratio": first_iteration_time / last_iteration_time if last_iteration_time else 1.0,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved
    }
    
    if verbose:
        print(f"✅ GPU warmup complete:")
        print(f"    Duration: {total_warmup_time:.2f}s")
        print(f"    Iterations: {iterations}")
        print(f"    First iteration: {first_iteration_time:.2f}ms")
        print(f"    Last iteration: {last_iteration_time:.2f}ms")
        if first_iteration_time and last_iteration_time:
            speedup = first_iteration_time / last_iteration_time
            print(f"    Speedup: {speedup:.1f}x (first→last)")
        print(f"    GPU memory: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved")
        print()
    
    # Clean up
    del warmup_net
    torch.cuda.empty_cache()
    
    return stats

def ensure_gpu_ready(min_speedup: float = 1.5, max_attempts: int = 3, verbose: bool = True) -> bool:
    """
    Ensure GPU is fully warmed up by checking speedup from first to last iteration
    
    Args:
        min_speedup: Minimum speedup ratio required (first_time / last_time)
        max_attempts: Maximum warmup attempts
        verbose: Whether to print messages
        
    Returns:
        bool: True if GPU appears fully warmed up
    """
    for attempt in range(max_attempts):
        if verbose and attempt > 0:
            print(f"🔄 GPU warmup attempt {attempt + 1}/{max_attempts}")
        
        stats = warmup_gpu(duration_seconds=2.0, verbose=verbose)
        
        if stats["device"] == "cpu":
            return True  # No GPU to warm up
        
        speedup = stats.get("speedup_ratio", 1.0)
        
        if speedup >= min_speedup:
            if verbose:
                print(f"✅ GPU fully warmed up (speedup: {speedup:.1f}x ≥ {min_speedup:.1f}x)")
            return True
        else:
            if verbose:
                print(f"⚠️  GPU may still be warming up (speedup: {speedup:.1f}x < {min_speedup:.1f}x)")
    
    if verbose:
        print(f"⚠️  GPU warmup may be incomplete after {max_attempts} attempts")
    return False

def test_warmup_effectiveness():
    """Test the warmup utility effectiveness"""
    print("🧪 Testing GPU Warmup Effectiveness")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("No CUDA GPU available for testing")
        return
    
    # Test without warmup
    print("Test 1: Cold GPU performance")
    torch.cuda.empty_cache()
    time.sleep(1)  # Let GPU potentially enter sleep
    
    # Simple timing test
    test_tensor = torch.randn(1024, 1024, device=device)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = torch.mm(test_tensor, test_tensor.T)
    torch.cuda.synchronize()
    cold_time = (time.perf_counter() - start) * 1000
    
    print(f"  Cold GPU matrix multiply: {cold_time:.2f}ms")
    
    # Test with warmup
    print("\nTest 2: Warmed GPU performance")
    warmup_gpu(duration_seconds=2.0, verbose=False)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = torch.mm(test_tensor, test_tensor.T)
    torch.cuda.synchronize()
    warm_time = (time.perf_counter() - start) * 1000
    
    print(f"  Warm GPU matrix multiply: {warm_time:.2f}ms")
    
    # Analysis
    if cold_time > warm_time:
        speedup = cold_time / warm_time
        print(f"\n📊 Warmup effectiveness: {speedup:.1f}x speedup")
        if speedup > 2.0:
            print("✅ Significant GPU warmup benefit detected")
        else:
            print("⚠️  Minor GPU warmup benefit")
    else:
        print("\n🤔 No clear warmup benefit detected")

if __name__ == "__main__":
    # Test the warmup utility
    test_warmup_effectiveness()
    
    print("\n" + "=" * 50)
    print("🔄 Full GPU warmup test")
    print("=" * 50)
    
    # Full warmup test
    ensure_gpu_ready(verbose=True)