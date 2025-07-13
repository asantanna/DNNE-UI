#!/usr/bin/env python3
"""
Environment Step Timing Analysis

Simplified profiler that analyzes the existing DNNE export to understand
why environment steps run at 127-175 FPS vs IsaacGymEnvs 32,000 FPS.
"""

import time
import subprocess
import re
import statistics
from pathlib import Path

def run_dnne_with_timing(duration_seconds=15):
    """Run DNNE export and capture detailed timing information"""
    print("🔍 Environment Step Timing Analysis")
    print("=" * 50)
    print(f"Running DNNE for {duration_seconds} seconds with detailed timing...")
    print()
    
    # Change to export directory
    export_dir = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")
    
    # Command to run with timing
    conda_activate = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    cmd = f"{conda_activate} && cd {export_dir} && python runner.py --headless --timeout {duration_seconds}s"
    
    print(f"🚀 Running: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True
    )
    actual_duration = time.time() - start_time
    
    print(f"✅ Completed in {actual_duration:.1f} seconds")
    print(f"   Return code: {result.returncode}")
    print()
    
    # Parse output for timing information
    all_output = result.stdout + "\n" + result.stderr
    
    # Extract key metrics
    metrics = extract_timing_metrics(all_output, actual_duration)
    
    return metrics, all_output

def extract_timing_metrics(output, actual_duration):
    """Extract timing metrics from DNNE output"""
    metrics = {
        'actual_duration': actual_duration,
        'environment_steps': 0,
        'training_updates': 0,
        'node_computations': {},
        'step_times': [],
        'performance_lines': []
    }
    
    # Parse the output line by line
    lines = output.split('\n')
    
    for line in lines:
        # Count computations by node
        if 'computations, avg time:' in line:
            # Parse: "  2: 124 computations, avg time: 0.000s"
            parts = line.split()
            if len(parts) >= 2:
                node_id = parts[0].rstrip(':')
                try:
                    comp_count = int(parts[1])
                    metrics['node_computations'][node_id] = comp_count
                    
                    # Extract timing
                    time_match = re.search(r'avg time:\s*([\d.]+)s', line)
                    if time_match:
                        avg_time_s = float(time_match.group(1))
                        metrics['node_computations'][f'{node_id}_avg_time_ms'] = avg_time_s * 1000
                except ValueError:
                    pass
        
        # Look for specific performance-related lines
        if any(keyword in line.lower() for keyword in ['fps', 'step', 'time', 'ms', 'performance']):
            metrics['performance_lines'].append(line.strip())
    
    # Calculate environment step rate
    if "2" in metrics['node_computations']:  # OR node is typically node 2
        metrics['environment_steps'] = metrics['node_computations']["2"]
    elif "9" in metrics['node_computations']:  # Isaac Gym Step node is typically node 9
        metrics['environment_steps'] = metrics['node_computations']["9"]
    
    # Calculate FPS
    if metrics['environment_steps'] > 0 and actual_duration > 0:
        metrics['calculated_fps'] = metrics['environment_steps'] / actual_duration
    else:
        metrics['calculated_fps'] = 0
    
    return metrics

def analyze_environment_bottleneck(metrics, output):
    """Analyze where the environment step bottleneck occurs"""
    print("📊 ENVIRONMENT STEP ANALYSIS")
    print("=" * 50)
    
    # Overall performance
    print(f"Execution duration: {metrics['actual_duration']:.1f}s")
    print(f"Environment steps: {metrics['environment_steps']}")
    print(f"Calculated FPS: {metrics['calculated_fps']:.1f}")
    print(f"Target FPS: 32,000 (IsaacGymEnvs baseline)")
    print(f"Performance gap: {32000/metrics['calculated_fps']:.0f}x slower" if metrics['calculated_fps'] > 0 else "N/A")
    print()
    
    # Node-by-node analysis
    print("Node Computation Analysis:")
    print("-" * 30)
    
    node_names = {
        "2": "OR Node",
        "3": "PPO Agent", 
        "6": "PPO Trainer",
        "7": "Isaac Gym Env",
        "9": "Isaac Gym Step",
        "11": "Cartpole Action"
    }
    
    total_computations = 0
    for node_id, count in metrics['node_computations'].items():
        if isinstance(count, int) and count > 0:
            name = node_names.get(node_id, f"Node {node_id}")
            avg_time_key = f"{node_id}_avg_time_ms"
            avg_time = metrics['node_computations'].get(avg_time_key, 0)
            
            print(f"{name:15} | {count:5} steps | {avg_time:6.2f}ms avg")
            total_computations += count
    
    print("-" * 30)
    print(f"{'TOTAL':15} | {total_computations:5} steps")
    print()
    
    # Environment step frequency analysis
    isaac_gym_steps = metrics['node_computations'].get("9", 0)
    or_node_steps = metrics['node_computations'].get("2", 0)
    
    print("Environment Step Frequency Analysis:")
    print("-" * 40)
    if isaac_gym_steps > 0:
        isaac_fps = isaac_gym_steps / metrics['actual_duration']
        print(f"Isaac Gym Step Node: {isaac_gym_steps} steps in {metrics['actual_duration']:.1f}s = {isaac_fps:.1f} FPS")
    
    if or_node_steps > 0:
        or_fps = or_node_steps / metrics['actual_duration']
        print(f"OR Node (data flow): {or_node_steps} steps in {metrics['actual_duration']:.1f}s = {or_fps:.1f} FPS")
    
    # Timing breakdown per step
    if metrics['calculated_fps'] > 0:
        step_time_ms = 1000 / metrics['calculated_fps']
        print(f"\nTime per environment step: {step_time_ms:.2f}ms")
        print(f"IsaacGymEnvs baseline: {1000/32000:.3f}ms per step")
        print(f"DNNE overhead: {step_time_ms - (1000/32000):.2f}ms per step")
    
    # Search for specific bottleneck indicators
    print(f"\n🔍 Bottleneck Indicators:")
    
    # Check Isaac Gym Step timing
    isaac_avg_time = metrics['node_computations'].get("9_avg_time_ms", 0)
    if isaac_avg_time > 3.0:
        print(f"⚠️  Isaac Gym Step Node avg time: {isaac_avg_time:.2f}ms (HIGH)")
        print("   This suggests environment simulation is the bottleneck")
    elif isaac_avg_time > 0:
        print(f"✓ Isaac Gym Step Node avg time: {isaac_avg_time:.2f}ms (reasonable)")
    
    # Check training timing
    ppo_trainer_avg = metrics['node_computations'].get("6_avg_time_ms", 0)
    if ppo_trainer_avg > 5.0:
        print(f"⚠️  PPO Trainer avg time: {ppo_trainer_avg:.2f}ms (HIGH)")
        print("   Training operations may be causing slowdown")
    elif ppo_trainer_avg > 0:
        print(f"✓ PPO Trainer avg time: {ppo_trainer_avg:.2f}ms (reasonable)")
    
    # Check OR node timing (should be very fast)
    or_avg_time = metrics['node_computations'].get("2_avg_time_ms", 0)
    if or_avg_time > 0.1:
        print(f"⚠️  OR Node avg time: {or_avg_time:.2f}ms (unexpectedly high)")
        print("   Queue coordination overhead may be significant")
    elif or_avg_time > 0:
        print(f"✓ OR Node avg time: {or_avg_time:.3f}ms (fast as expected)")

def compare_with_baseline():
    """Compare current results with known baseline performance"""
    print(f"\n📈 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    print("Known Performance Levels:")
    print("- IsaacGymEnvs baseline: 32,000 FPS")
    print("- DNNE current (from tests): 127-175 FPS") 
    print("- PPO forward pass: 2.0ms (good)")
    print("- Training efficiency: 1.0x overhead (optimal)")
    print()
    
    print("Investigation Focus Areas:")
    print("1. Environment simulation frequency (primary bottleneck)")
    print("2. Queue coordination overhead")
    print("3. GPU synchronization patterns")
    print("4. Memory allocation during environment steps")

def main():
    """Main analysis function"""
    print("🚀 DNNE Environment Step Timing Analysis")
    print("=" * 60)
    print("Goal: Understand why environment steps run at 127-175 FPS vs 32,000 FPS")
    print()
    
    # Run DNNE with timing analysis
    metrics, output = run_dnne_with_timing(duration_seconds=10)
    
    # Analyze the results
    analyze_environment_bottleneck(metrics, output)
    
    # Compare with baseline
    compare_with_baseline()
    
    # Save results
    results_file = Path(__file__).parent / "environment_step_timing_results.txt"
    with open(results_file, 'w') as f:
        f.write("DNNE Environment Step Timing Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Metrics: {metrics}\n\n")
        f.write("Performance Lines:\n")
        for line in metrics['performance_lines']:
            f.write(f"  {line}\n")
        f.write(f"\nFull Output:\n{output}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return metrics

if __name__ == "__main__":
    main()