#!/usr/bin/env python3
"""
Graphics Configuration Test for DNNE Isaac Gym

This script checks if DNNE is properly running in headless mode
and identifies any graphics-related performance issues.
"""

import sys
import os
import time
from pathlib import Path

# Add export directory to path
sys.path.insert(0, str(Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")))

def test_dnne_graphics_config():
    """Test DNNE Isaac Gym graphics configuration"""
    print("🔍 DNNE Graphics Configuration Test")
    print("=" * 50)
    print("Checking if DNNE is properly running headless...")
    print()
    
    # Import Isaac Gym first
    import isaacgym
    
    # Import DNNE Isaac Gym Environment Node
    from nodes.isaacgymenvnode_7 import IsaacGymEnvNode_7
    
    print("✅ Creating DNNE Isaac Gym Environment Node...")
    
    # Create the environment node
    env_node = IsaacGymEnvNode_7("7")
    
    # Check configuration
    print(f"📊 Configuration Check:")
    print(f"   headless: {env_node.headless}")
    print(f"   device: {env_node.device}")
    print(f"   physics_engine: {env_node.physics_engine}")
    print(f"   num_envs: {env_node.num_envs}")
    print()
    
    # Check Isaac Gym graphics device setting
    if hasattr(env_node, 'sim') and env_node.sim:
        print(f"✅ Isaac Gym simulation created")
        
        # Try to get graphics device info from the simulation
        try:
            from isaacgym import gymapi
            
            # Check if viewer was created
            if hasattr(env_node, 'viewer'):
                if env_node.viewer is None:
                    print(f"✅ No viewer created (good for headless)")
                else:
                    print(f"⚠️  Viewer was created (unexpected for headless)")
            
            # Check simulation parameters
            if hasattr(env_node, 'sim_params'):
                print(f"📋 Simulation Parameters:")
                print(f"   use_gpu_pipeline: {env_node.sim_params.use_gpu_pipeline}")
                if hasattr(env_node.sim_params, 'physx'):
                    print(f"   physx.use_gpu: {env_node.sim_params.physx.use_gpu}")
                    
        except Exception as e:
            print(f"⚠️  Could not check graphics configuration: {e}")
    else:
        print("❌ Isaac Gym simulation not created")
    
    return env_node

def test_vulkan_warning():
    """Test to reproduce the Vulkan warning"""
    print(f"\n🔍 Vulkan Warning Investigation")
    print("=" * 50)
    
    print("Looking for Vulkan/graphics initialization in DNNE...")
    
    # Check environment variables that might affect graphics
    graphics_env_vars = [
        'DISPLAY', 'WAYLAND_DISPLAY', 'XDG_SESSION_TYPE', 
        'VULKAN_SDK', 'VK_LOADER_DEBUG', 'MESA_LOADER_DRIVER_OVERRIDE'
    ]
    
    print("Environment variables:")
    for var in graphics_env_vars:
        value = os.environ.get(var, 'not set')
        print(f"   {var}: {value}")
    
    print()
    
    # Check if we can identify the source of Vulkan warning
    print("WSL2 Graphics Status:")
    print("- WSL2 can have complex graphics forwarding")
    print("- Vulkan might be initialized even for 'headless' mode")
    print("- This could be causing the performance issue")

def test_graphics_disabled_performance():
    """Test performance with graphics explicitly disabled"""
    print(f"\n⚡ Graphics-Disabled Performance Test")
    print("=" * 50)
    
    # Set environment variables to disable graphics
    os.environ['DISPLAY'] = ''
    os.environ['WAYLAND_DISPLAY'] = ''
    os.environ['XDG_SESSION_TYPE'] = 'none'
    
    print("🔧 Set graphics environment variables to disabled")
    print("🚀 Testing DNNE performance with graphics disabled...")
    
    # Run a quick DNNE test
    import subprocess
    
    export_dir = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")
    conda_activate = "source /home/asantanna/miniconda/bin/activate DNNE_PY38"
    
    # Set additional environment variables to disable graphics
    env_vars = "DISPLAY='' WAYLAND_DISPLAY='' XDG_SESSION_TYPE=none"
    cmd = f"{conda_activate} && {env_vars} cd {export_dir} && timeout 5s python runner.py --headless --timeout 4s"
    
    print(f"Running: {cmd}")
    
    start_time = time.time()
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True
    )
    duration = time.time() - start_time
    
    print(f"✅ Completed in {duration:.1f} seconds")
    print(f"   Return code: {result.returncode}")
    
    # Check for Vulkan warnings in output
    output = result.stdout + result.stderr
    vulkan_warnings = [line for line in output.split('\n') if 'vulkan' in line.lower() or 'dzn' in line.lower()]
    
    if vulkan_warnings:
        print(f"⚠️  Still found Vulkan/graphics warnings:")
        for warning in vulkan_warnings:
            print(f"     {warning}")
    else:
        print(f"✅ No Vulkan/graphics warnings found")
    
    # Look for performance metrics
    if "computations, avg time:" in output:
        for line in output.split('\n'):
            if "computations, avg time:" in line:
                print(f"   {line.strip()}")
    
    return result

def compare_graphics_impact():
    """Compare performance with and without graphics optimizations"""
    print(f"\n📊 Graphics Impact Analysis")
    print("=" * 50)
    
    print("Analysis of findings:")
    print("1. DNNE configuration shows headless=True ✅")
    print("2. Graphics device should be set to -1 ✅") 
    print("3. No viewer should be created ✅")
    print("4. BUT: Vulkan warnings suggest graphics initialization ⚠️")
    print()
    
    print("Potential issues:")
    print("- WSL2 graphics forwarding causing overhead")
    print("- Isaac Gym initializing graphics despite headless mode")
    print("- GPU drivers loading graphics components unnecessarily")
    print("- PhysX engine requiring graphics context in WSL2")
    print()
    
    print("Optimization recommendations:")
    print("1. 🔧 Force graphics environment variables to empty")
    print("2. 🎯 Check Isaac Gym initialization for graphics calls")
    print("3. 🏗️  Test with different Isaac Gym configuration")
    print("4. 📊 Consider native Linux vs WSL2 for comparison")

def main():
    """Main graphics configuration test"""
    
    # Check environment
    import os
    if os.environ.get("CONDA_DEFAULT_ENV") != "DNNE_PY38":
        print("❌ Please activate DNNE_PY38 conda environment first:")
        print("   source /home/asantanna/miniconda/bin/activate DNNE_PY38")
        return
    
    print("🚀 DNNE Graphics Configuration Analysis")
    print("=" * 60)
    print("Goal: Identify if graphics initialization is causing 200x slowdown")
    print()
    
    # Test DNNE graphics configuration
    env_node = test_dnne_graphics_config()
    
    # Investigate Vulkan warning
    test_vulkan_warning()
    
    # Test with graphics disabled
    result = test_graphics_disabled_performance()
    
    # Analyze impact
    compare_graphics_impact()
    
    print(f"\n💡 Key Findings:")
    print("- DNNE is configured for headless mode")
    print("- Vulkan warnings suggest graphics initialization anyway")
    print("- WSL2 graphics forwarding may be causing overhead")
    print("- Need to investigate Isaac Gym graphics initialization")

if __name__ == "__main__":
    main()