#!/usr/bin/env python3
"""
DNNE Cartpole Performance Benchmark

Measures and compares performance metrics for our DNNE Cartpole implementation
against IsaacGymEnvs baseline.

Key metrics:
- Steps per second
- Training throughput 
- Memory usage
- Environment count verification
"""

import os
import sys
import time
import re
import subprocess
from pathlib import Path

def run_dnne_benchmark(duration_seconds=180):
    """Run DNNE Cartpole training and measure performance"""
    
    print(f"🔧 Running DNNE benchmark for {duration_seconds}s")
    
    # Change to export directory
    export_dir = "/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO_Updated"
    os.chdir(export_dir)
    
    # Prepare environment
    conda_activate = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    
    # Run training with performance monitoring
    cmd = f"{conda_activate} && timeout {duration_seconds} python runner.py --headless --timeout {duration_seconds}s"
    
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for completion
    stdout, stderr = process.communicate()
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Parse output for performance metrics
    computations = {}
    actual_num_envs = None
    
    # Look in both stdout and stderr
    all_output = stdout + "\n" + stderr
    
    for line in all_output.split('\n'):
        if 'computations, avg time:' in line:
            parts = line.split()
            if len(parts) >= 2:
                node_id = parts[0].rstrip(':')
                try:
                    comp_count = int(parts[1])
                    computations[node_id] = comp_count
                except ValueError:
                    pass
        elif 'environments' in line and ('Created' in line or 'initialized' in line):
            # Try to extract actual environment count
            try:
                match = re.search(r'(\d+)\s+.*environments', line)
                if match:
                    actual_num_envs = int(match.group(1))
            except:
                pass
    
    total_computations = sum(computations.values())
    
    return {
        'duration': actual_duration,
        'total_computations': total_computations,
        'computations_per_second': total_computations / actual_duration if actual_duration > 0 else 0,
        'num_envs': actual_num_envs or "Unknown",
        'computations_by_node': computations,
        'stdout': stdout,
        'stderr': stderr
    }

def analyze_isaacgymenvs_baseline():
    """Analyze the IsaacGymEnvs baseline we just ran"""
    
    # Metrics from the interrupted run we observed
    baseline_metrics = {
        'fps_step': 10500,  # Average from 10,000-12,000 range
        'fps_total': 9500,  # Average from 9,000-10,000 range  
        'num_envs': 512,
        'epochs_per_5min': 35,  # Reached epoch 35 in ~5 minutes
        'steps_per_epoch': 8192,  # From horizon_length * num_envs / minibatch_size
        'network_size': [32, 32],  # From config
        'learning_rate': 0.0003
    }
    
    return baseline_metrics

def compare_performance(dnne_results, baseline_metrics):
    """Compare DNNE vs IsaacGymEnvs performance"""
    
    print("\n" + "="*80)
    print("🏁 PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n📊 DNNE Results ({dnne_results['duration']:.1f}s):")
    print(f"  • Total computations: {dnne_results['total_computations']:,}")
    print(f"  • Computations/sec: {dnne_results['computations_per_second']:,.0f}")
    print(f"  • Environments: {dnne_results['num_envs']}")
    
    print(f"\n📊 IsaacGymEnvs Baseline:")
    print(f"  • FPS step: {baseline_metrics['fps_step']:,}")
    print(f"  • FPS total: {baseline_metrics['fps_total']:,}")
    print(f"  • Environments: {baseline_metrics['num_envs']}")
    print(f"  • Epochs in 5min: {baseline_metrics['epochs_per_5min']}")
    
    # Calculate relative performance
    if baseline_metrics['fps_total'] > 0 and dnne_results['computations_per_second'] > 0:
        relative_speed = (dnne_results['computations_per_second'] / baseline_metrics['fps_total']) * 100
        print(f"\n🎯 Relative Performance:")
        print(f"  • DNNE vs IsaacGymEnvs: {relative_speed:.2f}%")
        
        if relative_speed > 80:
            print("  ✅ Excellent performance (>80% of baseline)")
        elif relative_speed > 60:
            print("  ⚠️  Good performance (60-80% of baseline)")
        elif relative_speed > 40:
            print("  ⚠️  Moderate performance (40-60% of baseline)")
        else:
            print("  ❌ Low performance (<40% of baseline)")
    
    # Environment count comparison
    if isinstance(dnne_results['num_envs'], int):
        env_ratio = dnne_results['num_envs'] / baseline_metrics['num_envs']
        print(f"\n🌍 Environment Scaling:")
        print(f"  • Environment ratio: {env_ratio:.2f}x ({dnne_results['num_envs']} vs {baseline_metrics['num_envs']})")
        
        if dnne_results['computations_per_second'] > 0:
            per_env_performance = dnne_results['computations_per_second'] / dnne_results['num_envs']
            baseline_per_env = baseline_metrics['fps_total'] / baseline_metrics['num_envs']
            per_env_ratio = (per_env_performance / baseline_per_env) * 100
            print(f"  • Per-environment performance: {per_env_ratio:.2f}% of baseline")
    
    print(f"\n🔍 Analysis:")
    print(f"  • DNNE uses queue-based async architecture")
    print(f"  • IsaacGymEnvs uses direct vectorized operations")
    print(f"  • Both use GPU PhysX and similar physics parameters")
    
    # Environment comparison
    print(f"\n🌍 Environment Configuration:")
    print(f"  • Both: dt=0.0166s, substeps=2, maxEffort=400N")
    print(f"  • Both: GPU PhysX with identical physics settings")
    print(f"  • Both: Same Cartpole URDF asset")

def main():
    """Main benchmark execution"""
    
    print("🚀 DNNE vs IsaacGymEnvs Performance Benchmark")
    print("=" * 60)
    
    # Run DNNE benchmark
    dnne_results = run_dnne_benchmark(duration_seconds=60)
    
    # Get baseline metrics
    baseline_metrics = analyze_isaacgymenvs_baseline()
    
    # Compare results
    compare_performance(dnne_results, baseline_metrics)
    
    # Detailed breakdown
    print(f"\n📋 Detailed DNNE Node Performance:")
    for node_id, count in dnne_results['computations_by_node'].items():
        print(f"  • Node {node_id}: {count:,} computations")
    
    print(f"\n🔍 Raw Output Analysis:")
    print(f"  • Stdout lines: {len(dnne_results['stdout'].split())}")
    print(f"  • Stderr lines: {len(dnne_results['stderr'].split())}")

if __name__ == "__main__":
    main()