#!/usr/bin/env python3
"""
Raw Isaac Gym Performance Test

This script tests Isaac Gym performance outside of the DNNE framework
to establish a baseline and quantify DNNE's framework overhead.
"""

import time
import statistics
import sys
from pathlib import Path

# Import Isaac Gym first to avoid import order issues
import isaacgym
from isaacgym import gymapi, gymtorch

# Now we can safely import torch
import torch

def create_minimal_cartpole_environment(num_envs=512):
    """Create a minimal Cartpole environment using raw Isaac Gym API"""
    print(f"🔧 Creating minimal Cartpole environment with {num_envs} environments...")
    
    # Create gym instance
    gym = gymapi.acquire_gym()
    
    # Configure simulation
    sim_params = gymapi.SimParams()
    sim_params.dt = 0.0166  # 60 FPS
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # Configure PhysX
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.001
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 100.0
    sim_params.physx.default_buffer_size_multiplier = 2.0
    sim_params.physx.max_gpu_contact_pairs = 1024 * 1024
    sim_params.physx.contact_collection = gymapi.CC_NEVER
    
    sim_params.use_gpu_pipeline = True
    
    # Create simulation
    device_id = 0
    sim = gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)
    
    if sim is None:
        raise RuntimeError("Failed to create simulation")
    
    # Create ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    
    # Load Cartpole asset
    asset_root = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/assets"
    asset_file = "urdf/cartpole.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    # Create environments
    spacing = 4.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    
    num_per_row = int(num_envs**0.5)
    
    envs = []
    actor_handles = []
    
    for i in range(num_envs):
        # Create environment
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)
        
        # Create cartpole actor
        pose = gymapi.Transform()
        pose.p.z = 2.0
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        actor_handle = gym.create_actor(env, cartpole_asset, pose, "cartpole", i, 1, 0)
        actor_handles.append(actor_handle)
        
        # Configure DOF properties
        dof_props = gym.get_actor_dof_properties(env, actor_handle)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0
        gym.set_actor_dof_properties(env, actor_handle, dof_props)
    
    # Prepare simulation
    gym.prepare_sim(sim)
    
    # Get DOF state tensor
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    dof_pos = dof_state.view(num_envs, 2, 2)[..., 0]  # positions
    dof_vel = dof_state.view(num_envs, 2, 2)[..., 1]  # velocities
    
    print("✅ Environment created successfully")
    print(f"   DOF state shape: {dof_state.shape}")
    
    return gym, sim, envs, actor_handles, dof_state, dof_pos, dof_vel

def compute_cartpole_reward(observations, reset_dist=3.0, max_episode_length=500, progress_buf=None):
    """Compute Cartpole rewards using the same logic as IsaacGymEnvs"""
    cart_pos = observations[:, 0]
    cart_vel = observations[:, 1] 
    pole_angle = observations[:, 2]
    pole_vel = observations[:, 3]
    
    # Reward calculation (same as IsaacGymEnvs)
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
    
    # Reset conditions
    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward), torch.zeros_like(reward))
    reset = torch.where(torch.abs(pole_angle) > 1.57, torch.ones_like(reward), reset)  # ~90 degrees
    
    # Apply reset penalty
    reward = torch.where(reset > 0, torch.ones_like(reward) * -2.0, reward)
    
    return reward, reset

def benchmark_raw_isaac_gym(num_steps=1000, num_envs=512):
    """Benchmark raw Isaac Gym performance"""
    print(f"⚡ Raw Isaac Gym Performance Benchmark")
    print("=" * 60)
    print(f"Testing {num_steps} steps with {num_envs} environments")
    print()
    
    # Create environment
    gym, sim, envs, actor_handles, dof_state, dof_pos, dof_vel = create_minimal_cartpole_environment(num_envs)
    
    # Initialize progress buffer for episode tracking
    progress_buf = torch.zeros(num_envs, dtype=torch.int32, device="cuda")
    
    print(f"🚀 Starting {num_steps} step benchmark...")
    
    # Warmup (let physics settle)
    for _ in range(50):
        actions = torch.zeros(num_envs, 1, device="cuda")
        actions_tensor = torch.zeros(num_envs * 2, device="cuda", dtype=torch.float)
        actions_tensor[::2] = actions.squeeze() * 400.0  # max effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        gym.set_dof_actuation_force_tensor(sim, forces)
        gym.simulate(sim)
        gym.fetch_results(sim, True)
    
    print("✅ Warmup complete, starting timed benchmark...")
    
    step_times = []
    total_start = time.perf_counter()
    
    for step in range(num_steps):
        step_start = time.perf_counter()
        
        # === ENVIRONMENT STEP ===
        
        # 1. Generate random actions
        actions = torch.randn(num_envs, 1, device="cuda") * 0.5  # Random actions
        
        # 2. Apply actions
        actions_tensor = torch.zeros(num_envs * 2, device="cuda", dtype=torch.float)
        actions_tensor[::2] = actions.squeeze() * 400.0  # max effort from cartpole config
        forces = gymtorch.unwrap_tensor(actions_tensor)
        gym.set_dof_actuation_force_tensor(sim, forces)
        
        # 3. Step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # 4. Get observations
        gym.refresh_dof_state_tensor(sim)
        
        # Build observations tensor [cart_pos, cart_vel, pole_angle, pole_vel]
        observations = torch.zeros(num_envs, 4, device="cuda")
        observations[:, 0] = dof_pos[:, 0]  # cart position
        observations[:, 1] = dof_vel[:, 0]  # cart velocity
        observations[:, 2] = dof_pos[:, 1]  # pole angle
        observations[:, 3] = dof_vel[:, 1]  # pole velocity
        
        # 5. Compute rewards
        rewards, resets = compute_cartpole_reward(observations)
        
        # 6. Handle resets (simplified)
        reset_indices = torch.nonzero(resets, as_tuple=False).squeeze(-1)
        if len(reset_indices) > 0:
            # Reset positions and velocities for environments that need reset
            new_positions = 0.2 * (torch.rand((len(reset_indices), 2), device="cuda") - 0.5)
            new_velocities = 0.5 * (torch.rand((len(reset_indices), 2), device="cuda") - 0.5)
            
            dof_pos[reset_indices] = new_positions
            dof_vel[reset_indices] = new_velocities
            
            # Apply to simulation
            reset_indices_int32 = reset_indices.to(dtype=torch.int32)
            gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state),
                                           gymtorch.unwrap_tensor(reset_indices_int32), 
                                           len(reset_indices_int32))
        
        # Update progress
        progress_buf += 1
        progress_buf[reset_indices] = 0
        
        step_time = (time.perf_counter() - step_start) * 1000
        step_times.append(step_time)
        
        # Progress indicator
        if step % 200 == 0:
            print(f"  Step {step}: {step_time:.3f}ms")
    
    total_time = time.perf_counter() - total_start
    
    # Calculate statistics
    avg_step_time = statistics.mean(step_times)
    min_step_time = min(step_times)
    max_step_time = max(step_times)
    std_step_time = statistics.stdev(step_times)
    
    raw_fps = 1000 / avg_step_time
    total_fps = num_steps / total_time
    
    print(f"\n📊 RAW ISAAC GYM PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Steps completed: {num_steps}")
    print(f"Environments: {num_envs}")
    print()
    
    print(f"Step timing statistics:")
    print(f"  Average: {avg_step_time:.3f}ms")
    print(f"  Minimum: {min_step_time:.3f}ms") 
    print(f"  Maximum: {max_step_time:.3f}ms")
    print(f"  Std dev: {std_step_time:.3f}ms")
    print()
    
    print(f"Performance metrics:")
    print(f"  Raw FPS (from avg step): {raw_fps:.0f}")
    print(f"  Total FPS (wall clock): {total_fps:.0f}")
    print(f"  IsaacGymEnvs baseline: 32,000 FPS")
    print(f"  Raw vs baseline: {32000/raw_fps:.1f}x slower")
    print()
    
    # Cleanup
    gym.destroy_sim(sim)
    
    return {
        'avg_step_time_ms': avg_step_time,
        'raw_fps': raw_fps,
        'total_fps': total_fps,
        'baseline_gap': 32000/raw_fps,
        'step_times': step_times
    }

def compare_with_dnne():
    """Compare raw Isaac Gym performance with DNNE results"""
    print(f"\n🔬 COMPARISON WITH DNNE")
    print("=" * 60)
    
    # Known DNNE performance from previous tests
    dnne_fps = 129  # From environment_step_timing.py
    dnne_step_time = 1000 / dnne_fps
    
    print(f"Known DNNE performance:")
    print(f"  DNNE FPS: {dnne_fps}")
    print(f"  DNNE step time: {dnne_step_time:.2f}ms")
    print()
    
    return dnne_fps, dnne_step_time

def main():
    """Main benchmark execution"""
    
    # Check environment
    import os
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("❌ Please activate DNNE_PY38 conda environment first:")
        print("   source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        return
    
    print("🚀 Raw Isaac Gym Performance Analysis")
    print("=" * 60)
    print("Goal: Establish baseline performance outside DNNE framework")
    print()
    
    # Run benchmark
    results = benchmark_raw_isaac_gym(num_steps=1000, num_envs=512)
    
    # Compare with DNNE
    dnne_fps, dnne_step_time = compare_with_dnne()
    
    # Calculate framework overhead
    framework_overhead = dnne_step_time / results['avg_step_time_ms']
    
    print(f"📊 FRAMEWORK OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"Raw Isaac Gym: {results['avg_step_time_ms']:.3f}ms per step ({results['raw_fps']:.0f} FPS)")
    print(f"DNNE system: {dnne_step_time:.2f}ms per step ({dnne_fps} FPS)")
    print(f"Framework overhead: {framework_overhead:.1f}x slower")
    print(f"Overhead per step: {dnne_step_time - results['avg_step_time_ms']:.2f}ms")
    print()
    
    print(f"Gap analysis:")
    print(f"  Raw Isaac Gym vs baseline: {results['baseline_gap']:.1f}x slower")
    print(f"  DNNE vs baseline: {32000/dnne_fps:.0f}x slower")
    print(f"  DNNE framework adds: {framework_overhead:.1f}x additional overhead")
    
    # Save results
    results_file = Path(__file__).parent / "raw_isaac_gym_results.txt"
    with open(results_file, 'w') as f:
        f.write("Raw Isaac Gym Performance Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Results: {results}\n")
        f.write(f"DNNE comparison: {dnne_fps} FPS\n")
        f.write(f"Framework overhead: {framework_overhead:.1f}x\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    main()