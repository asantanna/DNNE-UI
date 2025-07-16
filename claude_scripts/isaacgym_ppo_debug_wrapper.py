#!/usr/bin/env python3
"""
IsaacGymEnvs PPO Debug Wrapper

Adds PPO_CYCLE_DEBUG logging to IsaacGymEnvs by monkey-patching the rl_games A2C agent.
This allows us to capture the same debug information from both systems for comparison.
"""

import os
import sys
import importlib
import torch

# Add IsaacGymEnvs to path
isaac_dir = '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs'
sys.path.insert(0, isaac_dir)

# Import IsaacGymEnvs training module
from isaacgymenvs import train

# Import rl_games to monkey-patch
import rl_games
from rl_games.algos_torch import a2c_continuous

# Store original play_steps method
original_play_steps = a2c_continuous.A2CAgent.play_steps

# Counter for steps
step_counter = 0

def patched_play_steps(self, n_steps):
    """Patched play_steps that adds PPO_CYCLE_DEBUG logging"""
    global step_counter
    
    # Call original method
    result = original_play_steps(self, n_steps)
    
    # Add PPO_CYCLE_DEBUG logging if enabled
    if os.environ.get("PPO_CYCLE_DEBUG"):
        # Extract values from the batch_dict
        batch_dict = result
        
        # Log first few steps
        if step_counter < 5:
            try:
                # Get the first action, value, and reward
                if 'actions' in batch_dict:
                    action = batch_dict['actions'][0, 0].item() if batch_dict['actions'].numel() > 0 else 0.0
                else:
                    action = 0.0
                    
                if 'values' in batch_dict:
                    value = batch_dict['values'][0].item() if batch_dict['values'].numel() > 0 else 0.0
                else:
                    value = 0.0
                    
                if 'rewards' in batch_dict:
                    reward = batch_dict['rewards'][0].item() if batch_dict['rewards'].numel() > 0 else 0.0
                else:
                    reward = 0.0
                
                print(f"[PPO_CYCLE] Step {step_counter + 1}: action={action:.4f}, value={value:.4f}, reward={reward:.4f}")
                step_counter += 1
                
            except Exception as e:
                print(f"[PPO_CYCLE] Error extracting values: {e}")
        
        # Check if we've completed a full PPO cycle (16 steps)
        if 'obses' in batch_dict and batch_dict['obses'].shape[0] >= 16:
            print(f"[PPO_CYCLE] Buffer full ({batch_dict['obses'].shape[0]} steps) - starting PPO update")
            
            # Log first 5 values from each tensor
            if 'rewards' in batch_dict:
                rewards = batch_dict['rewards'][:5].tolist() if batch_dict['rewards'].numel() >= 5 else batch_dict['rewards'].tolist()
                print(f"[PPO_CYCLE] First 5 rewards: {rewards}")
                
            if 'values' in batch_dict:
                values = batch_dict['values'][:5].tolist() if batch_dict['values'].numel() >= 5 else batch_dict['values'].tolist()
                print(f"[PPO_CYCLE] First 5 values: {values}")
                
            if 'actions' in batch_dict:
                actions = batch_dict['actions'][:5, 0].tolist() if batch_dict['actions'].shape[0] >= 5 else batch_dict['actions'][:, 0].tolist()
                print(f"[PPO_CYCLE] First 5 actions: {actions}")
                
            if 'dones' in batch_dict:
                dones = batch_dict['dones'][:5].tolist() if batch_dict['dones'].numel() >= 5 else batch_dict['dones'].tolist()
                print(f"[PPO_CYCLE] First 5 dones: {dones}")
            
            # Exit if PPO_STOP_AFTER_CYCLE is set
            if os.environ.get("PPO_STOP_AFTER_CYCLE"):
                print("[PPO_CYCLE] First PPO cycle complete - exiting now")
                sys.exit(0)
    
    return result

# Monkey-patch the play_steps method
a2c_continuous.A2CAgent.play_steps = patched_play_steps

# If fixed seed is set, configure it
if os.environ.get("FIXED_SEED"):
    seed = int(os.environ.get("FIXED_SEED"))
    print(f"🔍 Fixed seed mode: {seed}")
    
    # Set all random seeds
    import torch
    import numpy as np
    import random
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Now run the normal IsaacGymEnvs training
if __name__ == "__main__":
    print("🚀 Starting IsaacGymEnvs with PPO_CYCLE_DEBUG wrapper")
    train.main()