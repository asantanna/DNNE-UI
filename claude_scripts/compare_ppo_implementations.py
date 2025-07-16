#!/usr/bin/env python3
"""
Comprehensive comparison of DNNE and IsaacGymEnvs PPO implementations
"""

import subprocess
import sys
import os
import json
import re
import time

def extract_debug_values(output_lines, system_name):
    """Extract debug values from output lines"""
    debug_data = {
        "system": system_name,
        "observations": [],
        "observation_normalization": {},
        "network_weights": [],
        "forward_pass": {},
        "ppo_update": {},
        "environment": {}
    }
    
    for line in output_lines:
        # Initial observations
        if "Raw observations (first" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["observations"].append({"raw": values, "line": line})
        
        # Observation normalization
        elif "RunningMeanStd initialized" in line or "Initialized observation normalization" in line:
            debug_data["observation_normalization"]["initialized"] = line
        elif "Obs mean:" in line or "Initial mean:" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["observation_normalization"]["mean"] = values
        elif "Obs var:" in line or "Initial var:" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["observation_normalization"]["var"] = values
        
        # Network weights
        elif "Initial model weights" in line or "weight shape:" in line:
            debug_data["network_weights"].append(line)
        elif "shared.0.weight:" in line or "actor_mlp" in line and "weight" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["network_weights"].append({"layer": "first", "values": values})
        
        # Forward pass values
        elif "Action mean" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["forward_pass"]["action_mean"] = values
        elif "Action std" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["forward_pass"]["action_std"] = values
        elif "Value output" in line or "Value (first env)" in line:
            match = re.search(r'[-\d.e]+$', line)
            if match:
                debug_data["forward_pass"]["value"] = float(match.group())
        elif "Sampled action" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["forward_pass"]["sampled_action"] = values
        
        # PPO update values
        elif "Computing GAE" in line or "First PPO update" in line:
            debug_data["ppo_update"]["started"] = True
        elif "Rewards (first 5)" in line or "Rewards:" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["ppo_update"]["rewards"] = values
        elif "Advantages (first 5)" in line or "Computed advantages:" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["ppo_update"]["advantages"] = values
        elif "Returns (first 5)" in line or "Computed returns:" in line:
            match = re.search(r'\[([-\d., e]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                debug_data["ppo_update"]["returns"] = values
    
    return debug_data

def run_dnne_debug(seed=42, timeout=5):
    """Run DNNE with debug output"""
    print("\n" + "="*80)
    print(f"Running DNNE with seed {seed}")
    print("="*80)
    
    # First export the workflow
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    export_cmd = [sys.executable, "claude_scripts/programmatic_export.py", "Cartpole_PPO"]
    subprocess.run(export_cmd, check=True, capture_output=True)
    
    # Run with debug output
    os.chdir("export_system/exports/Cartpole_PPO")
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    cmd = [
        sys.executable,
        "runner.py",
        f"--fixed-seed={seed}",
        "--epochs=1",
        f"--timeout={timeout}s",
        "--headless",
        "--verbose"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            output_lines.append(line.rstrip())
            if "Debug" in line:
                print(f"DNNE: {line.rstrip()}")
    
    process.wait()
    os.chdir("/mnt/e/ALS-Projects/DNNE/DNNE-UI")
    
    return extract_debug_values(output_lines, "DNNE")

def run_isaacgymenvs_debug(seed=42, timeout=5):
    """Run IsaacGymEnvs with debug output"""
    print("\n" + "="*80)
    print(f"Running IsaacGymEnvs with seed {seed}")
    print("="*80)
    
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    cmd = [
        sys.executable,
        "claude_scripts/isaacgymenvs_debug_runner.py",
        f"--seed={seed}",
        "--num-envs=512",
        "--horizon-length=16",
        "--max-iterations=1"
    ]
    
    # Start process with timeout
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, bufsize=1, env=env)
    
    output_lines = []
    start_time = time.time()
    
    while True:
        line = process.stdout.readline()
        if line:
            output_lines.append(line.rstrip())
            if "Debug" in line:
                print(f"IsaacGymEnvs: {line.rstrip()}")
        
        # Check for completion or timeout
        if process.poll() is not None:
            break
        if time.time() - start_time > timeout:
            process.terminate()
            break
    
    # Get any remaining output
    remaining, _ = process.communicate(timeout=1)
    if remaining:
        output_lines.extend(remaining.strip().split('\n'))
    
    return extract_debug_values(output_lines, "IsaacGymEnvs")

def compare_values(dnne_data, igenv_data):
    """Compare extracted values and identify differences"""
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    differences = []
    
    # Compare initial observations
    print("\n1. Initial Observations:")
    if dnne_data["observations"] and igenv_data["observations"]:
        dnne_obs = dnne_data["observations"][0]["raw"] if dnne_data["observations"] else None
        igenv_obs = igenv_data["observations"][0]["raw"] if igenv_data["observations"] else None
        
        if dnne_obs and igenv_obs:
            print(f"   DNNE:        {dnne_obs}")
            print(f"   IsaacGymEnvs: {igenv_obs}")
            if dnne_obs != igenv_obs:
                differences.append(("Initial observations", dnne_obs, igenv_obs))
    
    # Compare observation normalization
    print("\n2. Observation Normalization:")
    if dnne_data["observation_normalization"] and igenv_data["observation_normalization"]:
        dnne_mean = dnne_data["observation_normalization"].get("mean", [])
        igenv_mean = igenv_data["observation_normalization"].get("mean", [])
        print(f"   DNNE mean:        {dnne_mean}")
        print(f"   IsaacGymEnvs mean: {igenv_mean}")
        
        dnne_var = dnne_data["observation_normalization"].get("var", [])
        igenv_var = igenv_data["observation_normalization"].get("var", [])
        print(f"   DNNE var:         {dnne_var}")
        print(f"   IsaacGymEnvs var:  {igenv_var}")
    
    # Compare network outputs
    print("\n3. Network Forward Pass:")
    if dnne_data["forward_pass"] and igenv_data["forward_pass"]:
        for key in ["action_mean", "action_std", "value", "sampled_action"]:
            dnne_val = dnne_data["forward_pass"].get(key)
            igenv_val = igenv_data["forward_pass"].get(key)
            if dnne_val is not None and igenv_val is not None:
                print(f"   {key}:")
                print(f"     DNNE:        {dnne_val}")
                print(f"     IsaacGymEnvs: {igenv_val}")
                if dnne_val != igenv_val:
                    differences.append((key, dnne_val, igenv_val))
    
    # Compare PPO update values
    print("\n4. PPO Update Values:")
    if dnne_data["ppo_update"] and igenv_data["ppo_update"]:
        for key in ["rewards", "advantages", "returns"]:
            dnne_val = dnne_data["ppo_update"].get(key)
            igenv_val = igenv_data["ppo_update"].get(key)
            if dnne_val is not None and igenv_val is not None:
                print(f"   {key}:")
                print(f"     DNNE:        {dnne_val[:5] if len(dnne_val) > 5 else dnne_val}")
                print(f"     IsaacGymEnvs: {igenv_val[:5] if len(igenv_val) > 5 else igenv_val}")
    
    # Summary
    print(f"\n📊 Found {len(differences)} differences")
    if differences:
        print("\nKey differences:")
        for name, dnne_val, igenv_val in differences[:5]:  # Show first 5
            print(f"  - {name}: DNNE={dnne_val}, IsaacGymEnvs={igenv_val}")
    
    return differences

def main():
    # Activate conda environment
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    seed = 42
    
    # Run both systems
    dnne_data = run_dnne_debug(seed, timeout=10)
    igenv_data = run_isaacgymenvs_debug(seed, timeout=10)
    
    # Compare results
    differences = compare_values(dnne_data, igenv_data)
    
    # Save detailed results
    results = {
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dnne": dnne_data,
        "isaacgymenvs": igenv_data,
        "differences": [
            {"name": d[0], "dnne": d[1], "isaacgymenvs": d[2]} 
            for d in differences
        ]
    }
    
    output_file = "/mnt/e/ALS-Projects/DNNE/DNNE-UI/claude_scripts/ppo_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Print actionable summary
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if differences:
        first_diff = differences[0]
        print(f"\n🎯 First divergence found in: {first_diff[0]}")
        print(f"   This is where the implementations start to differ.")
        print(f"   Focus debugging efforts on this component.")
    else:
        print("\n✅ No differences found in captured values!")
        print("   The implementations appear to be producing identical results.")
        print("   Consider capturing more detailed traces or running longer.")

if __name__ == "__main__":
    main()