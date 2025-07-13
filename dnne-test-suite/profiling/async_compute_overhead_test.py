#!/usr/bin/env python3
"""
Test async function call overhead for compute-heavy operations
"""

import asyncio
import time
import torch
import torch.nn as nn

# Simple compute function that does matrix multiplication
def sync_compute(x):
    """Synchronous compute function"""
    # Simulate PPO agent computation
    layer1 = nn.Linear(512, 256).cuda()
    layer2 = nn.Linear(256, 128).cuda()
    layer3 = nn.Linear(128, 64).cuda()
    
    y = layer1(x)
    y = torch.relu(y)
    y = layer2(y)
    y = torch.relu(y)
    y = layer3(y)
    
    return y

async def async_compute(x):
    """Async wrapper around sync compute"""
    return sync_compute(x)

async def async_with_await_compute(x):
    """Async with unnecessary await"""
    # This simulates what might be happening in DNNE
    await asyncio.sleep(0)  # Yield control
    return sync_compute(x)

async def test_overhead():
    """Test different calling patterns"""
    print("Testing async overhead for compute-heavy operations")
    print("=" * 60)
    
    # Test data
    x = torch.randn(512, 512).cuda()
    iterations = 1000
    
    # Test 1: Direct synchronous calls
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        result = sync_compute(x)
        
    torch.cuda.synchronize()
    sync_time = time.time() - start
    sync_ms = (sync_time / iterations) * 1000
    print(f"Synchronous compute: {sync_ms:.3f} ms per call")
    
    # Test 2: Async function but called with await
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        result = await async_compute(x)
        
    torch.cuda.synchronize()
    async_time = time.time() - start
    async_ms = (async_time / iterations) * 1000
    print(f"Async compute (direct): {async_ms:.3f} ms per call")
    
    # Test 3: Async with unnecessary await (simulating potential DNNE issue)
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        result = await async_with_await_compute(x)
        
    torch.cuda.synchronize()
    async_await_time = time.time() - start
    async_await_ms = (async_await_time / iterations) * 1000
    print(f"Async compute (with await): {async_await_ms:.3f} ms per call")
    
    # Test 4: Event loop scheduling overhead
    torch.cuda.synchronize()
    start = time.time()
    
    async def scheduled_compute():
        return sync_compute(x)
    
    for _ in range(iterations):
        # Schedule as a task (simulating queue-based execution)
        task = asyncio.create_task(scheduled_compute())
        result = await task
        
    torch.cuda.synchronize()
    scheduled_time = time.time() - start
    scheduled_ms = (scheduled_time / iterations) * 1000
    print(f"Scheduled as task: {scheduled_ms:.3f} ms per call")
    
    print("\nOverhead Analysis:")
    print(f"Async direct overhead: {async_ms - sync_ms:.3f} ms ({(async_ms/sync_ms - 1)*100:.1f}%)")
    print(f"Async with await overhead: {async_await_ms - sync_ms:.3f} ms ({(async_await_ms/sync_ms - 1)*100:.1f}%)")
    print(f"Task scheduling overhead: {scheduled_ms - sync_ms:.3f} ms ({(scheduled_ms/sync_ms - 1)*100:.1f}%)")
    
    print("\nConclusion:")
    if scheduled_ms > sync_ms * 10:
        print("❌ Massive overhead from async task scheduling!")
        print("   This could explain DNNE's 60ms vs 1.87ms issue")
    elif async_await_ms > sync_ms * 10:
        print("❌ Massive overhead from unnecessary awaits!")
    else:
        print("✅ Async overhead is reasonable")

if __name__ == "__main__":
    asyncio.run(test_overhead())