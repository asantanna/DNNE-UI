#!/usr/bin/env python3
"""
Original IsaacGymEnvs Performance Profiler

This script profiles the original IsaacGymEnvs Cartpole implementation
to verify the claimed 32,000 FPS baseline performance and understand
where the 200x performance gap with DNNE comes from.
"""

import time
import cProfile
import pstats
import io
from pathlib import Path
import sys
import os

def setup_isaacgymenvs_environment():
    """Setup the original IsaacGymEnvs environment for profiling"""
    print("🔧 Setting up Original IsaacGymEnvs Environment...")
    
    # Add IsaacGymEnvs to Python path
    isaacgymenvs_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
    if isaacgymenvs_path not in sys.path:
        sys.path.insert(0, isaacgymenvs_path)
    
    # Import Isaac Gym first to avoid import order issues
    import isaacgym
    from isaacgym import gymapi
    
    # Now import IsaacGymEnvs modules
    from isaacgymenvs.tasks.cartpole import Cartpole
    
    print("✅ IsaacGymEnvs imports successful")
    
    # Create Cartpole environment configuration
    cfg = {
        'name': 'Cartpole',
        'physics_engine': 'physx',
        'env': {
            'numEnvs': 512,
            'envSpacing': 4.0,
            'resetDist': 3.0,
            'maxEpisodeLength': 500
        },
        'sim': {
            'dt': 0.0166,  # 60 FPS
            'substeps': 2,
            'up_axis': 'z',
            'use_gpu_pipeline': True,
            'gravity': [0.0, 0.0, -9.81],
            'physx': {
                'num_threads': 4,
                'solver_type': 1,
                'use_gpu': True,
                'num_position_iterations': 4,
                'num_velocity_iterations': 0,
                'contact_offset': 0.02,
                'rest_offset': 0.001,
                'bounce_threshold_velocity': 0.2,
                'max_depenetration_velocity': 100.0,
                'default_buffer_size_multiplier': 2.0,
                'max_gpu_contact_pairs': 1048576,
                'contact_collection': 0  # CC_NEVER
            }
        },
        'task': {
            'randomize': False
        },
        'graphics_device_id': -1,  # Headless
        'headless': True
    }
    
    # Create the environment
    print(f"🏗️  Creating Cartpole environment with {cfg['env']['numEnvs']} environments...")
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self, cfg_dict):
            for key, value in cfg_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)
    
    config = SimpleConfig(cfg)
    
    # Create environment
    env = Cartpole(
        cfg=config,
        rl_device="cuda",
        sim_device="cuda",
        graphics_device_id=-1,
        headless=True
    )
    
    print(f"✅ Environment created successfully")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Number of environments: {env.num_envs}")
    
    return env

def benchmark_original_isaacgymenvs(num_steps=1000):
    """Benchmark the original IsaacGymEnvs Cartpole implementation"""
    print(f"⚡ Original IsaacGymEnvs Performance Benchmark")
    print("=" * 60)
    print(f"Testing {num_steps} steps with original IsaacGymEnvs implementation")
    print()
    
    # Setup environment
    env = setup_isaacgymenvs_environment()
    
    # Import torch for actions
    import torch
    
    print(f"🚀 Starting {num_steps} step benchmark...")
    
    # Reset environment
    obs = env.reset()
    print(f"✅ Environment reset complete, starting timed benchmark...")
    
    # Warmup steps
    for _ in range(50):
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device)
        obs, rewards, dones, info = env.step(actions)
    
    print("✅ Warmup complete, starting timed benchmark...")
    
    # Timing measurement
    step_times = []
    total_start = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Generate random actions
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device)
        
        # Step environment
        obs, rewards, dones, info = env.step(actions)
        
        step_time = (time.perf_counter() - step_start) * 1000
        step_times.append(step_time)
        
        # Progress indicator
        if step % 200 == 0:
            print(f"  Step {step}: {step_time:.3f}ms")
    
    total_time = time.perf_counter() - total_start
    
    # Calculate statistics
    import statistics
    avg_step_time = statistics.mean(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    std_step_time = statistics.stdev(step_times)
    
    original_fps = 1000 / avg_step_time
    total_fps = num_steps / total_time
    
    print(f"\n📊 ORIGINAL ISAACGYMENVS PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Steps completed: {num_steps}")
    print(f"Environments: {env.num_envs}")
    print()
    
    print(f"Step timing statistics:")
    print(f"  Average: {avg_step_time:.3f}ms")
    print(f"  Minimum: {min_step_time:.3f}ms") 
    print(f"  Maximum: {max_step_time:.3f}ms")
    print(f"  Std dev: {std_step_time:.3f}ms")
    print()
    
    print(f"Performance metrics:")
    print(f"  Original FPS (from avg step): {original_fps:.0f}")
    print(f"  Total FPS (wall clock): {total_fps:.0f}")
    print(f"  Claimed baseline: 32,000 FPS")
    print(f"  Actual vs claimed: {32000/original_fps:.1f}x difference")
    print()
    
    return {
        'avg_step_time_ms': avg_step_time,
        'original_fps': original_fps,
        'total_fps': total_fps,
        'claimed_vs_actual': 32000/original_fps,
        'step_times': step_times,
        'total_time': total_time,
        'num_envs': env.num_envs
    }

def profile_original_isaacgymenvs(num_steps=500):
    """Profile the original IsaacGymEnvs with cProfile"""
    print(f"\n🔬 Profiling Original IsaacGymEnvs Implementation")
    print("=" * 60)
    
    # Setup profiler
    profiler = cProfile.Profile()
    
    # Setup environment outside profiling
    env = setup_isaacgymenvs_environment()
    import torch
    obs = env.reset()
    
    print(f"🔍 Starting profiled run of {num_steps} steps...")
    
    # Profile the actual stepping
    profiler.enable()
    
    for step in range(num_steps):
        actions = torch.rand((env.num_envs, env.num_actions), device=env.device)
        obs, rewards, dones, info = env.step(actions)
    
    profiler.disable()
    
    # Analyze profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    profile_output = s.getvalue()
    
    print("📊 Top 20 Functions by Cumulative Time:")
    print("-" * 50)
    print(profile_output)
    
    return profile_output

def compare_with_dnne_results():
    """Compare original IsaacGymEnvs results with DNNE performance"""
    print(f"\n🔬 COMPARISON WITH DNNE PERFORMANCE")
    print("=" * 60)
    
    # Known performance values
    dnne_fps = 129  # From environment_step_timing.py
    raw_isaac_fps = 166  # From raw_isaac_gym_test.py
    
    print(f"Known Performance Metrics:")
    print(f"  DNNE FPS: {dnne_fps}")
    print(f"  Raw Isaac Gym FPS: {raw_isaac_fps}")
    print(f"  Claimed IsaacGymEnvs: 32,000 FPS")
    print()
    
    return dnne_fps, raw_isaac_fps

def main():
    """Main profiling execution"""
    
    # Check environment
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("❌ Please activate DNNE_PY38 conda environment first:")
        print("   source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        return
    
    print("🚀 Original IsaacGymEnvs Performance Analysis")
    print("=" * 60)
    print("Goal: Verify claimed 32,000 FPS baseline and identify performance gap")
    print()
    
    try:
        # Run performance benchmark
        results = benchmark_original_isaacgymenvs(num_steps=1000)
        
        # Run detailed profiling
        profile_output = profile_original_isaacgymenvs(num_steps=500)
        
        # Compare with DNNE
        dnne_fps, raw_isaac_fps = compare_with_dnne_results()
        
        # Final analysis
        print(f"\n📈 FINAL PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"Original IsaacGymEnvs actual FPS: {results['original_fps']:.0f}")
        print(f"Claimed IsaacGymEnvs FPS: 32,000")
        print(f"DNNE system FPS: {dnne_fps}")
        print(f"Raw Isaac Gym FPS: {raw_isaac_fps}")
        print()
        
        print(f"Performance gaps:")
        print(f"  Claimed vs actual IsaacGymEnvs: {results['claimed_vs_actual']:.1f}x")
        print(f"  Original vs DNNE: {results['original_fps']/dnne_fps:.1f}x")
        print(f"  Original vs Raw Isaac Gym: {results['original_fps']/raw_isaac_fps:.1f}x")
        print()
        
        # Determine if baseline is correct
        if results['original_fps'] < 1000:
            print("🚨 CRITICAL FINDING: Original IsaacGymEnvs runs much slower than claimed!")
            print("   The 32,000 FPS baseline appears to be incorrect.")
            print("   Our DNNE performance gap is much smaller than initially thought.")
        else:
            print("✅ Original IsaacGymEnvs performance confirmed.")
            print("   DNNE performance gap investigation should continue.")
        
        # Save results
        results_file = Path(__file__).parent / "original_isaacgymenvs_results.txt"
        with open(results_file, 'w') as f:
            f.write("Original IsaacGymEnvs Performance Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Performance Results: {results}\n\n")
            f.write("Profiling Output:\n")
            f.write(profile_output)
            f.write(f"\nComparison:\n")
            f.write(f"  Original: {results['original_fps']:.0f} FPS\n")
            f.write(f"  DNNE: {dnne_fps} FPS\n")
            f.write(f"  Raw Isaac: {raw_isaac_fps} FPS\n")
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        print("This might indicate IsaacGymEnvs installation issues.")
        print("Try checking the IsaacGymEnvs installation:")
        print("  cd /home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs")
        print("  python -m isaacgymenvs.train task=Cartpole headless=True num_envs=64")
        return None

if __name__ == "__main__":
    main()