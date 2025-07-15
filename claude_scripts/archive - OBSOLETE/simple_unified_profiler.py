#!/usr/bin/env python3
"""
Simple Unified Performance Profiler - Compares IsaacGymEnvs and DNNE

Runs both systems using cProfile and generates performance comparison.
"""

import subprocess
import time
import json
import pstats
import os
from pathlib import Path
import asyncio

def profile_isaacgymenvs(num_iterations=10, num_envs=512):
    """Profile IsaacGymEnvs using subprocess with cProfile"""
    print("\n🔬 PROFILING ISAACGYMENVS")
    print("=" * 60)
    
    # Change to IsaacGymEnvs directory
    original_dir = os.getcwd()
    os.chdir('/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs')
    
    try:
        # Build the command
        cmd = [
            'python', '-m', 'cProfile',
            '-o', '/tmp/isaacgymenvs_training.prof',
            'isaacgymenvs/train.py',
            'task=Cartpole',
            f'task.env.numEnvs={num_envs}',
            f'train.params.config.max_epochs={num_iterations}',
            'train.params.config.horizon_length=16',
            'train.params.config.minibatch_size=8192',
            'headless=True',
            'test=False'
        ]
        
        print(f"Running {num_iterations} iterations with {num_envs} environments...")
        
        # Track time
        start_time = time.time()
        
        # Run with profiling
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ IsaacGymEnvs completed in {total_time:.2f}s")
            
            # Extract metrics from profile
            metrics = analyze_isaacgymenvs_profile(num_iterations, num_envs, total_time)
            
            # Save metrics
            with open('/tmp/isaacgymenvs_metrics.json', 'w') as f:
                json.dump(metrics, f)
                
            return metrics
        else:
            print(f"❌ IsaacGymEnvs failed with code {result.returncode}")
            if result.stderr:
                print("Error:", result.stderr[:500])
            return None
            
    finally:
        os.chdir(original_dir)

def profile_dnne(num_iterations=10, num_envs=512):
    """Profile DNNE using subprocess with cProfile"""
    print("\n🔬 PROFILING DNNE")
    print("=" * 60)
    
    export_dir = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")
    if not export_dir.exists():
        print("❌ DNNE export not found. Please export Cartpole_PPO workflow first.")
        return None
    
    # Build the command
    cmd = [
        'python', '-m', 'cProfile',
        '-o', '/tmp/dnne_training.prof',
        str(export_dir / 'runner.py'),
        '--timeout', f'{num_iterations * 3}s'  # Rough estimate: 3 seconds per iteration
    ]
    
    print(f"Running {num_iterations} iterations with {num_envs} environments...")
    
    # Track time
    start_time = time.time()
    
    # Run with profiling
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(export_dir))
    
    total_time = time.time() - start_time
    
    if result.returncode == 0 or "Completed" in result.stdout:
        print(f"✅ DNNE completed in {total_time:.2f}s")
        
        # Extract metrics from profile
        metrics = analyze_dnne_profile(num_iterations, num_envs, total_time)
        
        # Save metrics
        with open('/tmp/dnne_metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        return metrics
    else:
        print(f"❌ DNNE failed with code {result.returncode}")
        if result.stderr:
            print("Error:", result.stderr[:500])
        return None

def analyze_isaacgymenvs_profile(num_iterations, num_envs, total_time):
    """Analyze IsaacGymEnvs profile to extract metrics"""
    metrics = {
        'system': 'IsaacGymEnvs',
        'num_iterations': num_iterations,
        'num_envs': num_envs,
        'total_time': total_time,
        'step_count': num_iterations * 16,  # horizon_length
        'steps_per_sec': (num_iterations * 16) / total_time if total_time > 0 else 0,
        'env_step_time': 0,
        'ppo_time': 0,
        'overhead_time': 0
    }
    
    # Analyze profile if it exists
    prof_path = Path('/tmp/isaacgymenvs_training.prof')
    if prof_path.exists():
        stats = pstats.Stats(str(prof_path))
        
        # Extract timing for key functions
        for func_info in stats.stats.items():
            func_name = func_info[0][2]  # function name
            cumtime = func_info[1][3]     # cumulative time
            
            if 'step' in func_name.lower() and 'env' in func_name.lower():
                metrics['env_step_time'] += cumtime
            elif 'ppo' in func_name.lower() or 'train' in func_name.lower():
                metrics['ppo_time'] += cumtime
    
    return metrics

def analyze_dnne_profile(num_iterations, num_envs, total_time):
    """Analyze DNNE profile to extract metrics"""
    metrics = {
        'system': 'DNNE',
        'num_iterations': num_iterations,
        'num_envs': num_envs,
        'total_time': total_time,
        'step_count': num_iterations * 16,  # Estimated
        'steps_per_sec': (num_iterations * 16) / total_time if total_time > 0 else 0,
        'queue_time': 0,
        'compute_time': 0,
        'overhead_time': 0
    }
    
    # Analyze profile if it exists
    prof_path = Path('/tmp/dnne_training.prof')
    if prof_path.exists():
        stats = pstats.Stats(str(prof_path))
        
        # Extract timing for key functions
        for func_info in stats.stats.items():
            func_name = func_info[0][2]  # function name
            cumtime = func_info[1][3]     # cumulative time
            
            if 'queue' in func_name.lower():
                metrics['queue_time'] += cumtime
            elif 'compute' in func_name.lower():
                metrics['compute_time'] += cumtime
    
    return metrics

def generate_comparison_table(igenv_metrics, dnne_metrics):
    """Generate performance comparison table"""
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if not igenv_metrics or not dnne_metrics:
        print("❌ Missing results for comparison")
        return
    
    # Header
    print(f"{'Metric':30} {'IsaacGymEnvs':>15} {'DNNE':>15}")
    print("-" * 60)
    
    # Main metrics
    print(f"{'Total Time (s)':<30} {igenv_metrics['total_time']:>15.2f} {dnne_metrics['total_time']:>15.2f}")
    print(f"{'Steps/sec':<30} {igenv_metrics['steps_per_sec']:>15.1f} {dnne_metrics['steps_per_sec']:>15.1f}")
    print(f"{'Total Steps':<30} {igenv_metrics['step_count']:>15} {dnne_metrics['step_count']:>15}")
    print(f"{'Iterations/sec':<30} {igenv_metrics['num_iterations']/igenv_metrics['total_time']:>15.2f} {dnne_metrics['num_iterations']/dnne_metrics['total_time']:>15.2f}")
    
    print("=" * 60)
    
    # Performance ratio
    if igenv_metrics['steps_per_sec'] > 0:
        ratio = dnne_metrics['steps_per_sec'] / igenv_metrics['steps_per_sec']
        print(f"\nRelative Performance: {ratio:.2f}x")
        
        if ratio > 1.1:
            print(f"✅ DNNE is {ratio:.1f}x faster")
        elif ratio < 0.9:
            print(f"❌ DNNE is {1/ratio:.1f}x slower")
        else:
            print("✅ Performance is comparable")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Unified Performance Profiler')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations (default: 10)')
    parser.add_argument('--num-envs', type=int, default=512,
                       help='Number of parallel environments (default: 512)')
    args = parser.parse_args()
    
    print("🚀 SIMPLE UNIFIED PERFORMANCE PROFILER")
    print("=" * 70)
    print(f"Configuration: {args.iterations} iterations, {args.num_envs} environments")
    
    # Profile both systems
    igenv_metrics = profile_isaacgymenvs(args.iterations, args.num_envs)
    dnne_metrics = profile_dnne(args.iterations, args.num_envs)
    
    # Generate comparison
    generate_comparison_table(igenv_metrics, dnne_metrics)
    
    # Save combined results
    results = {
        'isaacgymenvs': igenv_metrics,
        'dnne': dnne_metrics
    }
    
    with open('/tmp/performance_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to /tmp/performance_comparison.json")

if __name__ == "__main__":
    main()