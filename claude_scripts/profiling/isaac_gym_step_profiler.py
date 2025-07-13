#!/usr/bin/env python3
"""
Isaac Gym Step Node Performance Profiler

This script analyzes the performance bottleneck in Isaac Gym Step Node execution.
Focus: Understanding why environment steps run at 127-175 FPS vs IsaacGymEnvs 32,000 FPS.
"""

import asyncio
import time
import sys
import json
import statistics
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")))

# Import Isaac Gym first to avoid import order issues
import isaacgym

# Import GPU warmup utility
sys.path.insert(0, str(Path(__file__).parent))
from gpu_warmup_utility import ensure_gpu_ready

class IsaacGymStepProfiler:
    """Detailed profiler for Isaac Gym Step Node operations"""
    
    def __init__(self):
        self.step_timings = []
        self.operation_timings = {
            'action_generation': [],
            'simulation_step': [],
            'observation_extraction': [],
            'reward_computation': [],
            'done_checking': [],
            'queue_operations': [],
            'total_step': []
        }
        
    def time_operation(self, operation_name: str):
        """Context manager for timing individual operations"""
        class Timer:
            def __init__(self, profiler, op_name):
                self.profiler = profiler
                self.op_name = op_name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = (time.perf_counter() - self.start_time) * 1000
                self.profiler.operation_timings[self.op_name].append(duration)
                
        return Timer(self, operation_name)
        
    def get_summary(self) -> Dict:
        """Get timing summary statistics"""
        summary = {}
        for op_name, timings in self.operation_timings.items():
            if timings:
                summary[op_name] = {
                    'count': len(timings),
                    'avg_ms': statistics.mean(timings),
                    'min_ms': min(timings),
                    'max_ms': max(timings),
                    'std_ms': statistics.stdev(timings) if len(timings) > 1 else 0,
                    'total_ms': sum(timings)
                }
            else:
                summary[op_name] = {
                    'count': 0,
                    'avg_ms': 0,
                    'min_ms': 0,
                    'max_ms': 0,
                    'std_ms': 0,
                    'total_ms': 0
                }
        return summary

async def profile_isaac_gym_step_node():
    """Profile the Isaac Gym Step Node in detail"""
    print("🔍 Isaac Gym Step Node Performance Profiler")
    print("=" * 60)
    print("Analyzing environment step bottleneck (127-175 FPS vs 32,000 FPS)")
    print()
    
    # Import Isaac Gym first
    import isaacgym
    import torch
    
    # Import the Isaac Gym Step Node
    from nodes.isaacgymstepnode_9 import IsaacGymStepNode_9
    from nodes.isaacgymenvnode_7 import IsaacGymEnvNode_7
    from nodes.cartpoleactionnode_11 import CartpoleActionNode_11
    
    # Create profiler
    profiler = IsaacGymStepProfiler()
    
    print("✅ Creating Isaac Gym environment...")
    
    # Create environment node
    env_node = IsaacGymEnvNode_7("7")
    
    # Initialize the environment (this might take time)
    init_start = time.perf_counter()
    env_result = await env_node.compute()
    init_time = (time.perf_counter() - init_start) * 1000
    
    print(f"✅ Environment initialized in {init_time:.1f}ms")
    
    sim_handle = env_result["sim_handle"]
    initial_observations = env_result["observations"]
    
    # Create step node
    step_node = IsaacGymStepNode_9("9")
    action_node = CartpoleActionNode_11("11")
    
    print(f"✅ Created Isaac Gym Step Node")
    print(f"   Environment: {initial_observations.shape}")
    print()
    
    # Profile environment steps
    num_steps = 200
    print(f"🚀 Profiling {num_steps} environment steps...")
    
    # Create dummy actions for testing
    batch_size = initial_observations.shape[0]
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # Create dummy policy output for action generation
        dummy_policy = {
            "action": torch.randn(batch_size, 1, device="cuda"),
            "value": torch.randn(batch_size, device="cuda"),
            "log_prob": torch.randn(batch_size, device="cuda"),
            "action_params": torch.randn(batch_size, 2, device="cuda")
        }
        
        with profiler.time_operation('action_generation'):
            # Generate actions using action node
            action_result = await action_node.compute(dummy_policy)
            actions = action_result["action"]
        
        with profiler.time_operation('simulation_step'):
            # This is where the actual Isaac Gym step happens
            step_result = await step_node.compute(sim_handle, actions, trigger=True)
        
        with profiler.time_operation('observation_extraction'):
            observations = step_result["observations"]
            
        with profiler.time_operation('reward_computation'):
            rewards = step_result["rewards"]
            
        with profiler.time_operation('done_checking'):
            done = step_result["done"]
        
        step_duration = (time.perf_counter() - step_start) * 1000
        profiler.operation_timings['total_step'].append(step_duration)
        
        # Progress indicator
        if step % 50 == 0:
            print(f"  Step {step}: {step_duration:.2f}ms")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ISAAC GYM STEP NODE PROFILING RESULTS")
    print("=" * 60)
    
    summary = profiler.get_summary()
    
    # Overall metrics
    total_steps = summary['total_step']['count']
    avg_step_time = summary['total_step']['avg_ms']
    calculated_fps = 1000 / avg_step_time if avg_step_time > 0 else 0
    
    print(f"Total steps profiled: {total_steps}")
    print(f"Average step time: {avg_step_time:.2f}ms")
    print(f"Calculated FPS: {calculated_fps:.1f}")
    print(f"IsaacGymEnvs baseline: 32,000 FPS")
    print(f"Performance gap: {32000/calculated_fps:.0f}x slower")
    print()
    
    # Operation breakdown
    print("Operation Breakdown:")
    print("-" * 50)
    
    total_accounted = 0
    for op_name, stats in summary.items():
        if op_name != 'total_step' and stats['count'] > 0:
            avg_time = stats['avg_ms']
            percentage = (avg_time / avg_step_time * 100) if avg_step_time > 0 else 0
            total_accounted += avg_time
            
            print(f"{op_name:20} | {avg_time:6.2f}ms | {percentage:5.1f}% | {stats['count']} calls")
    
    unaccounted = avg_step_time - total_accounted
    unaccounted_pct = (unaccounted / avg_step_time * 100) if avg_step_time > 0 else 0
    
    print("-" * 50)
    print(f"{'TOTAL ACCOUNTED':20} | {total_accounted:6.2f}ms | {total_accounted/avg_step_time*100:5.1f}%")
    print(f"{'UNACCOUNTED':20} | {unaccounted:6.2f}ms | {unaccounted_pct:5.1f}%")
    print(f"{'TOTAL STEP':20} | {avg_step_time:6.2f}ms | 100.0%")
    
    # Identify bottlenecks
    print(f"\n📊 Bottleneck Analysis:")
    sorted_ops = sorted([(name, stats['avg_ms']) for name, stats in summary.items() 
                        if name != 'total_step' and stats['count'] > 0], 
                       key=lambda x: x[1], reverse=True)
    
    for i, (op_name, avg_time) in enumerate(sorted_ops[:3]):
        percentage = (avg_time / avg_step_time * 100) if avg_step_time > 0 else 0
        print(f"   {i+1}. {op_name}: {avg_time:.2f}ms ({percentage:.1f}%)")
    
    # GPU utilization analysis
    print(f"\n🎯 Performance Analysis:")
    if calculated_fps < 1000:
        print("❌ Environment step frequency is critically low")
        print("   This is the primary bottleneck causing overall slowdown")
    
    if summary['simulation_step']['avg_ms'] > 3.0:
        print("⚠️  Simulation step timing is high")
        print("   May indicate GPU synchronization or physics overhead")
    
    if unaccounted_pct > 30:
        print("⚠️  High unaccounted time suggests framework overhead")
        print("   Queue coordination or async scheduling may be the issue")
    
    # Save detailed results
    results = {
        'timestamp': time.time(),
        'summary': summary,
        'calculated_fps': calculated_fps,
        'performance_gap': 32000/calculated_fps if calculated_fps > 0 else float('inf'),
        'unaccounted_time_pct': unaccounted_pct
    }
    
    results_file = Path(__file__).parent / "isaac_gym_step_profiling_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return results

async def compare_with_raw_isaac_gym():
    """Compare DNNE Isaac Gym performance with raw Isaac Gym operations"""
    print("\n\n🔬 Raw Isaac Gym Performance Comparison")
    print("=" * 60)
    
    import isaacgym
    import torch
    
    # Try to create a minimal Isaac Gym environment for comparison
    try:
        from isaacgymenvs.tasks.cartpole import Cartpole
        import yaml
        
        # Load IsaacGymEnvs config for fair comparison
        print("🔧 Creating raw IsaacGymEnvs environment...")
        
        # Minimal config for Cartpole
        config = {
            "name": "Cartpole",
            "physics_engine": "physx",
            "env": {
                "numEnvs": 512,
                "envSpacing": 4.0,
                "resetDist": 3.0,
                "maxEffort": 400.0,
                "clipObservations": 5.0,
                "clipActions": 1.0,
                "enableCameraSensors": False
            },
            "sim": {
                "dt": 0.0166,
                "substeps": 2,
                "up_axis": "z",
                "use_gpu_pipeline": True,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 0,
                    "solver_type": 1,
                    "use_gpu": True,
                    "num_position_iterations": 4,
                    "num_velocity_iterations": 0,
                    "contact_offset": 0.02,
                    "rest_offset": 0.001,
                    "bounce_threshold_velocity": 0.2,
                    "max_depenetration_velocity": 100.0,
                    "default_buffer_size_multiplier": 2.0,
                    "max_gpu_contact_pairs": 1048576,
                    "num_subscenes": 0,
                    "contact_collection": 0
                }
            },
            "task": {"randomize": False}
        }
        
        # Create the environment
        from omegaconf import DictConfig
        cfg = DictConfig(config)
        
        raw_env = Cartpole(
            cfg=cfg,
            rl_device="cuda",
            sim_device="cuda:0", 
            graphics_device_id=0,
            headless=True,
            virtual_screen_capture=False,
            force_render=False
        )
        
        print("✅ Raw IsaacGymEnvs environment created")
        print(f"   Number of environments: {raw_env.num_envs}")
        
        # Time raw environment steps
        num_steps = 200
        print(f"🚀 Timing {num_steps} raw Isaac Gym steps...")
        
        # Reset environment
        obs = raw_env.reset()
        
        # Warmup
        for _ in range(10):
            actions = torch.zeros(raw_env.num_envs, raw_env.num_actions, device="cuda")
            obs, rewards, dones, info = raw_env.step(actions)
        
        # Time the steps
        step_times = []
        torch.cuda.synchronize()
        
        for step in range(num_steps):
            # Create random actions
            actions = torch.randn(raw_env.num_envs, raw_env.num_actions, device="cuda")
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            obs, rewards, dones, info = raw_env.step(actions)
            
            torch.cuda.synchronize()
            step_time = (time.perf_counter() - start) * 1000
            step_times.append(step_time)
            
            if step % 50 == 0:
                print(f"  Raw step {step}: {step_time:.3f}ms")
        
        # Calculate raw performance
        avg_raw_step = statistics.mean(step_times)
        raw_fps = 1000 / avg_raw_step
        
        print(f"\n📊 Raw Isaac Gym Performance:")
        print(f"   Average step time: {avg_raw_step:.3f}ms")
        print(f"   Raw FPS: {raw_fps:.0f}")
        print(f"   Expected Isaac Gym baseline: 32,000 FPS")
        print(f"   Raw vs baseline: {32000/raw_fps:.1f}x slower")
        
        return {
            'raw_step_time_ms': avg_raw_step,
            'raw_fps': raw_fps,
            'raw_vs_baseline': 32000/raw_fps
        }
        
    except Exception as e:
        print(f"❌ Could not create raw Isaac Gym comparison: {e}")
        print("   This is expected if IsaacGymEnvs is not properly configured")
        return None

async def main():
    """Main profiler execution"""
    
    # Ensure conda environment
    import os
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("❌ Please activate DNNE_PY38 conda environment first:")
        print("   source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        return
    
    print("Starting Isaac Gym Step Node Performance Analysis...")
    print()
    
    # Warm up GPU first
    print("🔥 Warming up GPU...")
    ensure_gpu_ready(verbose=False)
    
    # Run Isaac Gym Step Node profiling
    dnne_results = await profile_isaac_gym_step_node()
    
    # Try to compare with raw Isaac Gym
    raw_results = await compare_with_raw_isaac_gym()
    
    # Final comparison
    if raw_results:
        print(f"\n\n🎯 FINAL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"DNNE Isaac Gym Step: {dnne_results['calculated_fps']:.1f} FPS")
        print(f"Raw Isaac Gym: {raw_results['raw_fps']:.0f} FPS") 
        print(f"IsaacGymEnvs baseline: 32,000 FPS")
        print()
        
        dnne_overhead = raw_results['raw_fps'] / dnne_results['calculated_fps']
        print(f"DNNE framework overhead: {dnne_overhead:.1f}x slower than raw Isaac Gym")
        print(f"Raw Isaac Gym gap: {raw_results['raw_vs_baseline']:.1f}x slower than baseline")
    
    print(f"\n💡 Next Investigation Targets:")
    print("   1. Async queue overhead in environment steps")
    print("   2. GPU synchronization patterns")
    print("   3. Memory allocation during simulation")
    print("   4. Node coordination latency")

if __name__ == "__main__":
    # Run the profiler
    asyncio.run(main())