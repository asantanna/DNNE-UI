#!/usr/bin/env python3
"""
Test DNNE by running runner.py directly in a subprocess
"""

import subprocess
import sys
import os
import signal
import time

def run_dnne_with_timeout(timeout=10):
    """Run DNNE runner with timeout and capture output"""
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["PPO_CYCLE_DEBUG"] = "1"
    
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--fixed-seed=42",
        "--epochs=1",
        "--headless"
    ]
    
    print(f"[TEST] Running: {' '.join(cmd)}")
    print(f"[TEST] Timeout: {timeout}s")
    
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
        preexec_fn=os.setsid
    )
    
    output_lines = []
    ppo_lines = []
    error_lines = []
    
    try:
        while True:
            if time.time() - start_time > timeout:
                print(f"\n[TEST] Timeout after {timeout}s")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                break
                
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            if line:
                line = line.rstrip()
                output_lines.append(line)
                
                # Print key lines
                if any(keyword in line for keyword in ["[PPO_CYCLE]", "Error", "error", "Starting", "initialized"]):
                    print(f"  {line}")
                
                if "[PPO_CYCLE]" in line:
                    ppo_lines.append(line)
                
                if "error" in line.lower() or "Error" in line:
                    error_lines.append(line)
                    
    except Exception as e:
        print(f"[TEST] Exception: {e}")
    finally:
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            process.kill()
    
    elapsed = time.time() - start_time
    print(f"\n[TEST] Process ended after {elapsed:.1f}s")
    print(f"[TEST] Total output lines: {len(output_lines)}")
    print(f"[TEST] PPO_CYCLE lines: {len(ppo_lines)}")
    print(f"[TEST] Error lines: {len(error_lines)}")
    
    if error_lines and not ppo_lines:
        print("\n[TEST] First few errors:")
        for err in error_lines[:3]:
            print(f"  {err}")
    
    return process.returncode, ppo_lines, error_lines

def main():
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    print("[TEST] Testing DNNE Cartpole PPO runner\n")
    
    # Try with a short timeout first
    code, ppo_lines, errors = run_dnne_with_timeout(10)
    
    print(f"\n[TEST] Return code: {code}")
    
    if ppo_lines:
        print("\n[TEST] ✅ PPO_CYCLE logging detected!")
        print("PPO_CYCLE lines:")
        for line in ppo_lines[:10]:
            print(f"  {line}")
    else:
        print("\n[TEST] ❌ No PPO_CYCLE logs found")
        
        if errors:
            print("\nKey errors:")
            # Look for the main error
            for err in errors:
                if "ImportError" in err or "RuntimeError" in err:
                    print(f"  → {err}")

if __name__ == "__main__":
    main()