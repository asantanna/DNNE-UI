#!/usr/bin/env python3
"""
Environment Step Performance Comparison

Measures env.step() calls per second for both IsaacGymEnvs and DNNE
"""

import time
import subprocess
import sys
import os
from pathlib import Path
import json
import re

def run_isaacgymenvs_test(num_iterations=30):
    """Run IsaacGymEnvs and extract performance metrics"""
    print("🔬 RUNNING ISAACGYMENVS TEST")
    print("=" * 60)
    
    # Run IsaacGymEnvs training
    cmd = [
        sys.executable, "-m", "isaacgymenvs.train",
        "task=Cartpole",
        "num_envs=512",
        f"max_iterations={num_iterations}",
        "headless=True"
    ]
    
    print(f"Running IsaacGymEnvs for {num_iterations} iterations...")
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs",
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Parse output for fps measurements
            fps_step_values = []
            lines = result.stdout.split('\n')
            
            for line in lines:
                # Look for "fps step: XXX"
                match = re.search(r'fps step:\s*(\d+)', line)
                if match:
                    fps_step_values.append(int(match.group(1)))
            
            if fps_step_values:
                avg_fps = sum(fps_step_values) / len(fps_step_values)
                
                # Calculate env.step() calls per second
                # fps_step = num_envs * steps_per_batch / time
                # For PPO default: steps_per_batch = horizon_length = 16
                steps_per_batch = 16
                num_envs = 512
                env_steps_per_fps = num_envs * steps_per_batch
                
                # Convert to actual env.step() calls per second
                step_calls_per_sec = avg_fps / env_steps_per_fps
                
                print(f"\n📊 ISAACGYMENVS RESULTS:")
                print(f"Average reported fps_step: {avg_fps:.0f}")
                print(f"Environment transitions per fps: {env_steps_per_fps}")
                print(f"Calculated env.step() calls/sec: {step_calls_per_sec:.1f}")
                
                return {
                    'system': 'IsaacGymEnvs',
                    'fps_step': avg_fps,
                    'env_steps_per_fps': env_steps_per_fps,
                    'step_calls_per_sec': step_calls_per_sec,
                    'num_envs': num_envs,
                    'steps_per_batch': steps_per_batch
                }
            else:
                print("❌ No FPS measurements found")
                
        else:
            print(f"❌ Error running IsaacGymEnvs: {result.stderr[:500]}")
            
    except Exception as e:
        print(f"❌ Failed to run IsaacGymEnvs: {e}")
    
    return None

def run_dnne_raw_test(num_steps=500):
    """Run raw DNNE Isaac Gym test"""
    print("\n🔬 RUNNING DNNE RAW ISAAC GYM TEST")
    print("=" * 60)
    
    # Use our existing control frequency test
    from control_frequency_test import benchmark_with_control_frequency
    
    print(f"Running DNNE for {num_steps} steps...")
    
    try:
        # Run with control_freq_inv=1 for direct step measurement
        result = benchmark_with_control_frequency(
            control_freq_inv=1,
            num_steps=num_steps,
            num_envs=512
        )
        
        if result:
            step_calls_per_sec = result['control_fps']
            
            print(f"\n📊 DNNE RESULTS:")
            print(f"env.step() calls/sec: {step_calls_per_sec:.1f}")
            print(f"Average step time: {result['avg_control_step_ms']:.2f}ms")
            
            return {
                'system': 'DNNE',
                'step_calls_per_sec': step_calls_per_sec,
                'avg_step_ms': result['avg_control_step_ms'],
                'num_envs': 512
            }
            
    except Exception as e:
        print(f"❌ Failed to run DNNE test: {e}")
    
    return None

def generate_comparison_table(isaacgym_results, dnne_results):
    """Generate performance comparison table"""
    print("\n📊 PERFORMANCE COMPARISON: env.step() calls per second")
    print("=" * 60)
    
    if not isaacgym_results or not dnne_results:
        print("❌ Missing results for comparison")
        return
    
    # Calculate values
    ig_steps = isaacgym_results['step_calls_per_sec']
    dnne_steps = dnne_results['step_calls_per_sec']
    
    # Generate table
    print(f"{'Metric':<35} {'IsaacGymEnvs':>15} {'DNNE':>15}")
    print("=" * 60)
    print(f"{'env.step() per second':<35} {ig_steps:>15.1f} {dnne_steps:>15.1f}")
    print(f"{'Number of environments':<35} {isaacgym_results['num_envs']:>15} {dnne_results['num_envs']:>15}")
    print("-" * 60)
    
    # Performance ratio
    if ig_steps > 0:
        ratio = dnne_steps / ig_steps
        print(f"{'Relative Performance:':<35} {'1.0x':>15} {f'{ratio:.2f}x':>15}")
        
        if ratio > 1:
            print(f"\n✅ DNNE is {ratio:.2f}x faster at env.step() calls!")
        else:
            print(f"\n❌ DNNE is {1/ratio:.2f}x slower at env.step() calls")
    
    print("=" * 60)
    
    # Additional context
    print("\n📝 MEASUREMENT NOTES:")
    print(f"- IsaacGymEnvs reports {isaacgym_results['fps_step']:.0f} 'fps_step'")
    print(f"- This equals {isaacgym_results['env_steps_per_fps']} environment transitions")
    print(f"- Which translates to {ig_steps:.1f} actual env.step() calls/sec")
    print(f"- DNNE directly measures {dnne_steps:.1f} env.step() calls/sec")
    
    # Save results
    comparison = {
        'isaacgymenvs': isaacgym_results,
        'dnne': dnne_results,
        'comparison': {
            'metric': 'env.step() calls per second',
            'isaacgymenvs_value': ig_steps,
            'dnne_value': dnne_steps,
            'ratio': ratio if ig_steps > 0 else 0
        }
    }
    
    with open("env_step_comparison_results.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n💾 Results saved to env_step_comparison_results.json")

def main():
    """Run environment step comparison"""
    print("🚀 ENVIRONMENT STEP PERFORMANCE COMPARISON")
    print("=" * 70)
    print("Comparing actual env.step() calls per second")
    print()
    
    # Check environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("❌ Please activate DNNE_PY38 conda environment first:")
        print("   source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        return
    
    # Run tests
    isaacgym_results = run_isaacgymenvs_test(num_iterations=30)
    dnne_results = run_dnne_raw_test(num_steps=500)
    
    # Generate comparison
    generate_comparison_table(isaacgym_results, dnne_results)
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()