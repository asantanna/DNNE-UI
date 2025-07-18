#!/usr/bin/env python3
"""
Test DNNE with potential CUDA fixes
"""

import subprocess
import sys
import os
import time

def test_dnne_fresh():
    """Test DNNE in a fresh subprocess with clean environment"""
    
    # Clean environment
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    # Force CUDA device
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Disable CUDA caching that might cause issues
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Create a wrapper script that ensures clean imports
    wrapper = '''
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PPO_CYCLE_DEBUG"] = "1"

# Import isaacgym FIRST
import isaacgym

# Now run the DNNE runner
import sys
sys.path.insert(0, "/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")

# Import and run
from runner import main
import asyncio

# Run with minimal epochs
import builtins
builtins.EPOCHS_OVERRIDE = 1

print("[WRAPPER] Starting DNNE runner...")
try:
    asyncio.run(main())
except SystemExit as e:
    print(f"[WRAPPER] Exited with code: {e.code}")
except Exception as e:
    print(f"[WRAPPER] Error: {e}")
    import traceback
    traceback.print_exc()
'''
    
    with open("/tmp/dnne_wrapper.py", "w") as f:
        f.write(wrapper)
    
    print("[TEST] Running DNNE with clean wrapper...\n")
    
    # Add command line args
    cmd = [sys.executable, "/tmp/dnne_wrapper.py", "--fixed-seed=42", "--epochs=1", "--headless"]
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env
    )
    
    output_lines = []
    ppo_lines = []
    cuda_error = False
    
    try:
        while True:
            if time.time() - start_time > 15:
                print("\n[TEST] Timeout after 15s")
                process.terminate()
                break
                
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.rstrip()
                output_lines.append(line)
                
                # Print relevant lines
                if any(keyword in line for keyword in ["[PPO_CYCLE]", "CUDA", "error", "Starting", "[WRAPPER]"]):
                    print(f"  {line}")
                
                if "[PPO_CYCLE]" in line:
                    ppo_lines.append(line)
                    
                if "cuda error" in line.lower() or "invalid resource handle" in line:
                    cuda_error = True
                    
    except KeyboardInterrupt:
        process.terminate()
    
    process.wait()
    
    print(f"\n[TEST] Results:")
    print(f"  - Return code: {process.returncode}")
    print(f"  - PPO_CYCLE lines: {len(ppo_lines)}")
    print(f"  - CUDA error: {cuda_error}")
    print(f"  - Total output: {len(output_lines)} lines")
    
    if ppo_lines:
        print("\n[TEST] ✅ PPO cycle logging detected!")
        for line in ppo_lines[:5]:
            print(f"    {line}")
    elif cuda_error:
        print("\n[TEST] ❌ CUDA error persists")
        # Try to find more context
        for i, line in enumerate(output_lines):
            if "cuda error" in line.lower():
                print("\nContext around CUDA error:")
                for j in range(max(0, i-2), min(len(output_lines), i+3)):
                    print(f"    {output_lines[j]}")
                break

def main():
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    print("Testing DNNE with CUDA fixes...\n")
    
    # Kill any hanging processes first
    subprocess.run(["pkill", "-f", "runner.py"], capture_output=True)
    time.sleep(1)
    
    test_dnne_fresh()

if __name__ == "__main__":
    main()