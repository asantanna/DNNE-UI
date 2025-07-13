#!/usr/bin/env python3
"""
Simple IsaacGymEnvs Baseline Test

This script runs the standard IsaacGymEnvs training to measure
actual performance and verify the 32,000 FPS baseline claim.
"""

import subprocess
import time
import re
from pathlib import Path

def run_isaacgymenvs_cartpole_test(duration_seconds=30):
    """Run IsaacGymEnvs Cartpole with performance measurement"""
    print("🚀 IsaacGymEnvs Cartpole Baseline Test")
    print("=" * 60)
    print(f"Running IsaacGymEnvs Cartpole for {duration_seconds} seconds...")
    print("This will measure actual FPS performance of the original implementation")
    print()
    
    # Change to IsaacGymEnvs directory and run training
    isaacgymenvs_dir = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
    conda_activate = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    
    # Run with performance monitoring
    cmd = f"{conda_activate} && cd {isaacgymenvs_dir} && timeout {duration_seconds}s python -m isaacgymenvs.train task=Cartpole headless=True num_envs=512 max_iterations=100000"
    
    print(f"🔧 Command: python -m isaacgymenvs.train task=Cartpole headless=True num_envs=512")
    print(f"⏱️  Duration: {duration_seconds} seconds")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=duration_seconds + 10  # Extra timeout buffer
        )
        actual_duration = time.time() - start_time
        
        print(f"✅ Completed in {actual_duration:.1f} seconds")
        print(f"   Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        actual_duration = time.time() - start_time
        print(f"⏱️  Timed out after {actual_duration:.1f} seconds (expected)")
        result = subprocess.CompletedProcess(
            args=["timeout"], 
            returncode=124,  # timeout return code
            stdout="",
            stderr=""
        )
    
    # Combine output
    full_output = result.stdout + "\n" + result.stderr
    
    return analyze_isaacgymenvs_output(full_output, actual_duration)

def analyze_isaacgymenvs_output(output, duration):
    """Extract performance metrics from IsaacGymEnvs output"""
    print("\n📊 ANALYZING ISAACGYMENVS OUTPUT")
    print("=" * 60)
    
    metrics = {
        'duration': duration,
        'fps_measurements': [],
        'iteration_times': [],
        'mean_fps': 0,
        'total_iterations': 0
    }
    
    lines = output.split('\n')
    
    for line in lines:
        # Look for IsaacGymEnvs FPS measurements
        # Format: "fps step: 12345 fps step and policy inference: 6789 fps total: 4567 epoch: 1/100000 frames: 0"
        
        # Parse fps step (environment simulation)
        fps_step_match = re.search(r'fps step:\s*(\d+)', line)
        if fps_step_match:
            fps = int(fps_step_match.group(1))
            metrics['fps_measurements'].append(fps)
            print(f"📈 Environment Step FPS: {fps}")
        
        # Parse fps total (overall training FPS)
        fps_total_match = re.search(r'fps total:\s*(\d+)', line)
        if fps_total_match:
            fps = int(fps_total_match.group(1))
            if 'fps_total' not in metrics:
                metrics['fps_total'] = []
            metrics['fps_total'].append(fps)
            print(f"📊 Total Training FPS: {fps}")
        
        # Parse fps step and policy inference
        fps_policy_match = re.search(r'fps step and policy inference:\s*(\d+)', line)
        if fps_policy_match:
            fps = int(fps_policy_match.group(1))
            if 'fps_policy' not in metrics:
                metrics['fps_policy'] = []
            metrics['fps_policy'].append(fps)
            print(f"🧠 Step + Policy FPS: {fps}")
            
        # Look for epoch counts
        epoch_match = re.search(r'epoch:\s*(\d+)/\d+', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            metrics['total_iterations'] = max(metrics['total_iterations'], epoch)
    
    # Calculate statistics
    import statistics
    
    if metrics['fps_measurements']:
        metrics['avg_fps_step'] = statistics.mean(metrics['fps_measurements'])
        metrics['max_fps_step'] = max(metrics['fps_measurements'])
        metrics['min_fps_step'] = min(metrics['fps_measurements'])
        print(f"📊 Environment Step FPS: avg={metrics['avg_fps_step']:.0f}, range={metrics['min_fps_step']}-{metrics['max_fps_step']}")
    
    if 'fps_total' in metrics and metrics['fps_total']:
        metrics['avg_fps_total'] = statistics.mean(metrics['fps_total'])
        metrics['max_fps_total'] = max(metrics['fps_total'])
        metrics['min_fps_total'] = min(metrics['fps_total'])
        print(f"📊 Total Training FPS: avg={metrics['avg_fps_total']:.0f}, range={metrics['min_fps_total']}-{metrics['max_fps_total']}")
        # Use total FPS as the main metric for comparison
        metrics['avg_fps'] = metrics['avg_fps_total']
    
    if 'fps_policy' in metrics and metrics['fps_policy']:
        metrics['avg_fps_policy'] = statistics.mean(metrics['fps_policy'])
        print(f"📊 Step + Policy FPS: avg={metrics['avg_fps_policy']:.0f}")
    
    # Fallback if no FPS found
    if 'avg_fps' not in metrics:
        if metrics['total_iterations'] > 0:
            estimated_fps = metrics['total_iterations'] / duration
            metrics['avg_fps'] = estimated_fps
            print(f"📈 Estimated FPS from iterations: {estimated_fps:.1f}")
        else:
            metrics['avg_fps'] = 0
    
    return metrics, output

def compare_with_dnne_performance(isaacgymenvs_metrics):
    """Compare IsaacGymEnvs results with DNNE performance"""
    print(f"\n🔬 PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Known DNNE performance
    dnne_fps = 166  # From our recent tests
    raw_isaac_fps = 166  # From raw_isaac_gym_test.py
    claimed_baseline = 32000
    
    isaacgymenvs_fps = isaacgymenvs_metrics.get('avg_fps', 0)
    
    print(f"Performance Results:")
    print(f"  IsaacGymEnvs actual: {isaacgymenvs_fps:.1f} FPS")
    print(f"  DNNE current: {dnne_fps} FPS") 
    print(f"  Raw Isaac Gym: {raw_isaac_fps} FPS")
    print(f"  Claimed baseline: {claimed_baseline} FPS")
    print()
    
    if isaacgymenvs_fps > 0:
        # Calculate actual gaps
        claimed_vs_actual = claimed_baseline / isaacgymenvs_fps
        isaacgymenvs_vs_dnne = isaacgymenvs_fps / dnne_fps
        
        print(f"Performance Gaps:")
        print(f"  Claimed vs Actual IsaacGymEnvs: {claimed_vs_actual:.1f}x")
        print(f"  IsaacGymEnvs vs DNNE: {isaacgymenvs_vs_dnne:.1f}x") 
        print(f"  IsaacGymEnvs vs Raw Isaac: {isaacgymenvs_fps / raw_isaac_fps:.1f}x")
        print()
        
        # Determine conclusion
        if isaacgymenvs_fps < 1000:
            print("🚨 CRITICAL FINDING:")
            print("   IsaacGymEnvs actual performance is much lower than claimed!")
            print("   The 32,000 FPS baseline appears to be incorrect.")
            print("   DNNE's performance gap is much smaller than initially thought.")
        elif isaacgymenvs_fps < 5000:
            print("⚠️  IMPORTANT FINDING:")
            print("   IsaacGymEnvs runs significantly slower than claimed baseline.")
            print("   DNNE performance gap is smaller than expected.")
        else:
            print("✅ IsaacGymEnvs performance verified close to baseline.")
            print("   DNNE performance investigation should continue.")
    else:
        print("❌ Could not extract performance metrics from IsaacGymEnvs output.")
        print("   Manual analysis of output may be required.")
    
    return {
        'isaacgymenvs_fps': isaacgymenvs_fps,
        'dnne_fps': dnne_fps,
        'raw_isaac_fps': raw_isaac_fps,
        'claimed_baseline': claimed_baseline
    }

def save_results(metrics, output, comparison):
    """Save results for future analysis"""
    results_file = Path(__file__).parent / "isaacgymenvs_baseline_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("IsaacGymEnvs Baseline Performance Test Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Test Duration: {metrics['duration']:.1f} seconds\n")
        f.write(f"Total Iterations: {metrics['total_iterations']}\n")
        f.write(f"FPS Measurements: {metrics['fps_measurements']}\n")
        f.write(f"Average FPS: {metrics.get('avg_fps', 'N/A')}\n")
        f.write(f"Mean FPS (reported): {metrics['mean_fps']}\n")
        f.write("\n")
        
        f.write("Performance Comparison:\n")
        for key, value in comparison.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        f.write("Full Output:\n")
        f.write("-" * 30 + "\n")
        f.write(output)
    
    print(f"\n💾 Results saved to: {results_file}")
    return results_file

def main():
    """Main test execution"""
    
    # Check environment
    import os
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("❌ Please activate DNNE_PY38 conda environment first:")
        print("   source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        return
    
    print("🚀 IsaacGymEnvs Baseline Performance Verification")
    print("=" * 60)
    print("Goal: Verify claimed 32,000 FPS baseline with actual measurements")
    print()
    
    try:
        # Run the test
        metrics, output = run_isaacgymenvs_cartpole_test(duration_seconds=30)
        
        # Analyze results
        comparison = compare_with_dnne_performance(metrics)
        
        # Save results
        results_file = save_results(metrics, output, comparison)
        
        # Final summary
        print(f"\n💡 SUMMARY:")
        if metrics.get('avg_fps', 0) > 0:
            print(f"   IsaacGymEnvs measured: {metrics['avg_fps']:.1f} FPS")
            if metrics['avg_fps'] < 1000:
                print("   ✨ MAJOR DISCOVERY: Baseline claim appears incorrect!")
                print("   ✅ DNNE performance gap is much smaller than thought.")
            else:
                print("   ✅ Baseline verified, continue DNNE optimization.")
        else:
            print("   ❓ Results unclear, manual output analysis needed.")
        
        return metrics
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        print("Consider running IsaacGymEnvs manually to verify installation:")
        print("  cd /home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs")
        print("  python -m isaacgymenvs.train task=Cartpole headless=True num_envs=64 max_iterations=100")
        return None

if __name__ == "__main__":
    main()