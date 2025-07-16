#!/usr/bin/env python3
"""
Minimal test of DNNE Cartpole - just try to start it
"""

import sys
import os

# Set environment
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PPO_CYCLE_DEBUG"] = "1"

print("[TEST] Starting DNNE runner...")

# Add paths
sys.path.insert(0, "/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO")

# Try to import the runner
try:
    print("[TEST] Importing nodes...")
    
    # Import in correct order - Isaac Gym nodes first
    from nodes.isaacgymenvnode_7 import IsaacGymEnvNode_7
    print("[TEST] ✓ IsaacGymEnvNode imported")
    
    from nodes.isaacgymstepnode_9 import IsaacGymStepNode_9
    print("[TEST] ✓ IsaacGymStepNode imported")
    
    from nodes.ppoagentnode_3 import PPOAgentNode_3
    print("[TEST] ✓ PPOAgentNode imported")
    
    from nodes.ppotrainernode_6 import PPOTrainerNode_6
    print("[TEST] ✓ PPOTrainerNode imported")
    
    print("[TEST] All imports successful!")
    
    # Try to create nodes
    print("\n[TEST] Creating nodes...")
    
    env_node = IsaacGymEnvNode_7("7")
    print("[TEST] ✓ IsaacGymEnvNode created")
    
    # The env node might fail during Isaac Gym init
    # Let's see how far we get
    
except Exception as e:
    import traceback
    print(f"\n[TEST] ❌ Error: {e}")
    traceback.print_exc()
    
print("\n[TEST] Test complete")