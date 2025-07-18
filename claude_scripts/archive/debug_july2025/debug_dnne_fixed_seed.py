#!/usr/bin/env python3
"""
DNNE wrapper script with fixed seed support for debugging PPO differences
"""

import subprocess
import sys
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run DNNE with fixed seed for debugging")
    parser.add_argument("--fixed-seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--epochs", type=int, default=1, help="Max epochs (1 for single cycle)")
    parser.add_argument("--timeout", type=str, default="30s", help="Timeout duration")
    parser.add_argument("--workflow", type=str, default="IsaacGym-CartpoleTask", help="Workflow to export")
    args = parser.parse_args()
    
    # Activate conda environment
    print("🔧 Activating conda environment...")
    activate_cmd = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
    
    # Change to DNNE directory
    dnne_path = "/mnt/e/ALS-Projects/DNNE/DNNE-UI"
    os.chdir(dnne_path)
    print(f"📁 Changed directory to: {os.getcwd()}")
    
    # First, export the workflow
    print("\n📦 Exporting workflow...")
    export_cmd = [
        sys.executable,
        "claude_scripts/programmatic_export.py",
        "--workflow", args.workflow,
        "--export-path", f"export_system/exports/{args.workflow}_debug"
    ]
    
    subprocess.run(export_cmd, check=True)
    
    # Change to exported directory
    export_dir = f"export_system/exports/{args.workflow}_debug"
    os.chdir(export_dir)
    print(f"📁 Changed to export directory: {os.getcwd()}")
    
    # Build DNNE run command with fixed seed
    cmd = [
        sys.executable,
        "runner.py",
        f"--fixed-seed={args.fixed_seed}",
        f"--epochs={args.epochs}",
        f"--timeout={args.timeout}",
        "--verbose"  # Enable verbose mode to see debug output
    ]
    
    if args.headless:
        cmd.append("--headless")
    
    print(f"\n🚀 Running DNNE with fixed seed: {args.fixed_seed}")
    print(f"📊 Command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Run and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                             universal_newlines=True, bufsize=1)
    
    debug_lines = []
    
    # Read output line by line
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line.rstrip())
            
            # Capture debug lines
            if "[PPO Agent Debug]" in line or "[PPO Trainer Debug]" in line:
                debug_lines.append(line.rstrip())
    
    process.wait()
    
    # Save debug output
    if debug_lines:
        debug_file = f"dnne_debug_seed_{args.fixed_seed}.log"
        with open(debug_file, "w") as f:
            f.write("\n".join(debug_lines))
        print(f"\n💾 Saved debug output to: {debug_file}")
    
    print("\n✅ DNNE run complete")

if __name__ == "__main__":
    main()