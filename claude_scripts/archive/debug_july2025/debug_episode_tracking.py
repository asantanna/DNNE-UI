#!/usr/bin/env python3
"""Debug episode tracking in DNNE Cartpole PPO"""

import subprocess
import sys
import time

def run_dnne_debug():
    """Run DNNE and capture episode tracking output"""
    print("Running DNNE Cartpole PPO with episode tracking debug...")
    
    # First export
    subprocess.run([sys.executable, "claude_scripts/programmatic_export.py", "Cartpole_PPO"], 
                   check=True, capture_output=True)
    
    # Run with short timeout
    cmd = [
        sys.executable,
        "export_system/exports/Cartpole_PPO/runner.py",
        "--epochs=10",
        "--timeout=15s",
        "--headless"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1)
    
    episode_count = 0
    episode_returns = []
    
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            
            # Look for episode completions
            if "Episode" in line and "episode return" in line:
                episode_count += 1
                try:
                    # Extract return value
                    parts = line.split("episode return = ")
                    if len(parts) > 1:
                        return_val = float(parts[1].strip())
                        episode_returns.append(return_val)
                except:
                    pass
    
    process.wait()
    
    print("\n" + "="*80)
    print(f"Total episodes completed: {episode_count}")
    if episode_returns:
        print(f"Episode returns: {episode_returns[:10]}")  # First 10
        print(f"Average return: {sum(episode_returns)/len(episode_returns):.2f}")
    else:
        print("No episode returns captured!")
    
    return episode_count, episode_returns

if __name__ == "__main__":
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    episode_count, returns = run_dnne_debug()
    
    if episode_count == 0:
        print("\n❌ PROBLEM: No episodes are completing!")
        print("Possible issues:")
        print("1. Termination conditions not being checked")
        print("2. Reset not happening when episodes should end")
        print("3. Progress buffer not incrementing")
        sys.exit(1)
    else:
        print(f"\n✅ Found {episode_count} completed episodes")
        if returns and sum(returns)/len(returns) > 100:
            print("✅ Learning appears to be working!")
        else:
            print("⚠️  Low episode returns - check PPO algorithm")