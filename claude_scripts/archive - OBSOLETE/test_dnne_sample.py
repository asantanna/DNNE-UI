#!/usr/bin/env python3
"""
Quick test to collect sample DNNE performance data
"""
import subprocess
import time
import re

def collect_dnne_sample():
    """Collect a sample of DNNE performance data"""
    print("🔧 Collecting DNNE sample data...")
    
    cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38 && cd /mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO && timeout 15s python runner.py --headless --verbose"
    
    start_time = time.time()
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    output = result.stdout + "\n" + result.stderr
    
    # Parse metrics
    or_outputs = 0
    training_steps = 0
    
    lines = output.split('\n')
    print(f"📊 Total output lines: {len(lines)}")
    print(f"⏱️  Elapsed time: {elapsed:.1f}s")
    
    for line in lines:
        if 'OR Node: Routing' in line and 'output #' in line:
            or_outputs += 1
        elif 'PPO training step' in line and 'complete' in line:
            training_steps += 1
    
    print(f"🔄 OR Node outputs: {or_outputs}")
    print(f"🧠 PPO training steps: {training_steps}")
    
    if or_outputs > 0:
        fps = or_outputs / elapsed
        print(f"📈 Calculated FPS: {fps:.1f}")
        return fps, or_outputs, training_steps, elapsed
    
    # Show some sample lines for debugging
    print("\n📝 Sample output lines:")
    for i, line in enumerate(lines[-10:]):  # Last 10 lines
        if line.strip():
            print(f"  {i}: {line[:100]}")
    
    return None, or_outputs, training_steps, elapsed

if __name__ == "__main__":
    collect_dnne_sample()