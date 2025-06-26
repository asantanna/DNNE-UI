#!/usr/bin/env python3
"""
Timing diagnostic for slow startup
"""
import time

def timed_import(module_name, globals_dict=None):
    """Import a module and print timing"""
    print(f"[{time.time():.3f}] Importing {module_name}...", end='', flush=True)
    start = time.time()
    
    if globals_dict is None:
        globals_dict = globals()
    
    # Do the import
    exec(f"import {module_name}", globals_dict)
    
    elapsed = time.time() - start
    print(f" done in {elapsed:.3f}s")
    return elapsed

print(f"[{time.time():.3f}] === Starting timing diagnostic ===")
start_time = time.time()

# Time each import separately
total_import_time = 0

print("\n--- Basic imports ---")
total_import_time += timed_import("asyncio")
total_import_time += timed_import("logging")
total_import_time += timed_import("json")
total_import_time += timed_import("random")

print("\n--- PyTorch imports (this is likely the slow part) ---")
torch_start = time.time()

# Break down torch import
print(f"[{time.time():.3f}] Starting torch import sequence...")

# First, numpy (torch depends on it)
print(f"[{time.time():.3f}]   Importing numpy...", end='', flush=True)
numpy_start = time.time()
import numpy as np
numpy_time = time.time() - numpy_start
print(f" done in {numpy_time:.3f}s")

# Then torch itself
print(f"[{time.time():.3f}]   Importing torch...", end='', flush=True)
torch_core_start = time.time()
import torch
torch_core_time = time.time() - torch_core_start
print(f" done in {torch_core_time:.3f}s")

# Check CUDA
print(f"[{time.time():.3f}]   Checking CUDA...", end='', flush=True)
cuda_check_start = time.time()
cuda_available = torch.cuda.is_available()
cuda_check_time = time.time() - cuda_check_start
print(f" done in {cuda_check_time:.3f}s (available: {cuda_available})")

torch_total_time = time.time() - torch_start
print(f"[{time.time():.3f}] Total PyTorch import time: {torch_total_time:.3f}s")

print("\n--- Creating a simple tensor ---")
tensor_start = time.time()
x = torch.randn(3, 240, 320)  # Small image size
tensor_time = time.time() - tensor_start
print(f"[{time.time():.3f}] Created tensor in {tensor_time:.3f}s")

print("\n--- Summary ---")
total_time = time.time() - start_time
print(f"Total diagnostic time: {total_time:.3f}s")
print(f"  - NumPy import: {numpy_time:.3f}s")
print(f"  - PyTorch import: {torch_core_time:.3f}s")
print(f"  - CUDA check: {cuda_check_time:.3f}s")
print(f"  - Other imports: {total_import_time:.3f}s")

# Now test a minimal queue node
print("\n--- Testing minimal queue execution ---")
import asyncio
from typing import Dict, Any

class MinimalNode:
    def __init__(self):
        self.start_time = time.time()
        print(f"[{self.start_time:.3f}] Node initialized")
    
    async def run(self):
        print(f"[{time.time():.3f}] Node starting (init took {time.time() - self.start_time:.3f}s)")
        for i in range(3):
            print(f"[{time.time():.3f}] Processing {i}")
            await asyncio.sleep(0.1)
        print(f"[{time.time():.3f}] Node done")

async def test_minimal():
    print(f"[{time.time():.3f}] Creating node...")
    node = MinimalNode()
    print(f"[{time.time():.3f}] Running node...")
    await node.run()

print(f"\n[{time.time():.3f}] Starting async test...")
asyncio.run(test_minimal())
print(f"[{time.time():.3f}] All done!")
