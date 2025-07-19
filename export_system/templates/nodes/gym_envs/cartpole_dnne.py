# cartpole_dnne.py
"""
CartpoleDNNE - DNNE adaptation of IsaacGymEnvs Cartpole
Inherits from IGE's Cartpole implementation and adds DNNE-specific features
"""

import numpy as np
import os
import torch
import sys

# Import DNNE_print from centralized location
from isaacgymenvs.utils.debug_utils import DNNE_print

# Import Isaac Gym first (before torch imports in parent class)
import isaacgym
from isaacgym import gymutil, gymtorch, gymapi

# Now import the parent class
from isaacgymenvs.tasks.cartpole import Cartpole, compute_cartpole_reward


class CartpoleDNNE(Cartpole):
    """
    DNNE-compatible Cartpole environment
    Inherits from IsaacGymEnvs Cartpole and adds async/DNNE features
    """
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture=False, force_render=False):
        """Initialize CartpoleDNNE with same interface as IGE"""
        
        # Add any DNNE-specific initialization here
        self.dnne_mode = True
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        import builtins
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        self.verbose = getattr(builtins, 'VERBOSE', False)
        
        # Call parent class initialization
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        from isaacgymenvs.utils.debug_utils import DNNE_print
        DNNE_print("B", "ENV_INIT", f"CartpoleDNNE initialized with {self.num_envs} environments")
        if self.ppo_cycle_debug:
            DNNE_print("B", "PPO_CYCLE", f"CartpoleDNNE initialized: num_envs={self.num_envs}, device={self.device}")
        
    def step_async(self, actions):
        """
        DNNE-compatible async step function
        Calls the standard step() but can be used in async context
        """
        # Don't log here - the parent class step() already logs PPO_CYCLE_DEBUG messages
        # Use the parent class step() which handles everything properly
        obs_dict, rewards, dones, infos = self.step(actions)
        
        # Extract observations from dict (VecTask returns {"obs": tensor})
        observations = obs_dict["obs"]
        
        # Verbose logging for outputs
        if self.verbose:
            step_num = getattr(self, '_step_count', 0)
            DNNE_print("D", "PPO_STEP", f"CartpoleDNNE step {step_num} - observations shape: {observations.shape}")
            DNNE_print("D", "PPO_STEP", f"Rewards: min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")
            DNNE_print("D", "PPO_STEP", f"Dones: {dones.sum().item()} environments done")
            
            # Log some observation details (cart pos, cart vel, pole angle, pole vel)
            if observations.shape[1] >= 4:
                cart_pos = observations[:, 0]
                pole_angle = observations[:, 2]
                DNNE_print("D", "PPO_STEP", f"Cart pos: min={cart_pos.min().item():.4f}, max={cart_pos.max().item():.4f}")
                DNNE_print("D", "PPO_STEP", f"Pole angle: min={pole_angle.min().item():.4f}, max={pole_angle.max().item():.4f}")
        
        return observations, rewards, dones, infos
    
    def reset(self):
        """Override reset - parent class handles debug logging"""
        # Call parent reset (which already logs the reset with caller info)
        obs_dict = super().reset()
        
        # Add any DNNE-specific reset logic here if needed
        
        return obs_dict
    
    def get_initial_observations(self):
        """Get initial observations after reset for DNNE"""
        # DNNE needs to trigger the initial reset that IGE does during its first step
        # Check if environments need to be reset (reset_buf initialized to 1)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.compute_observations()
        
        return self.obs_buf.clone()
    
    def set_custom_reward_fn(self, reward_fn):
        """Allow custom reward computation for DNNE flexibility"""
        self.custom_reward_fn = reward_fn
        
    def compute_reward(self):
        """Override to support custom reward functions"""
        if hasattr(self, 'custom_reward_fn') and self.custom_reward_fn is not None:
            # Use custom reward function
            self.rew_buf[:], self.reset_buf[:] = self.custom_reward_fn(
                self.obs_buf, self.reset_buf, self.progress_buf
            )
        else:
            # Use default Cartpole reward
            super().compute_reward()
    
    def get_env_state(self):
        """Get current environment state for DNNE state management"""
        return {
            "dof_pos": self.dof_pos.clone(),
            "dof_vel": self.dof_vel.clone(),
            "progress_buf": self.progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "obs_buf": self.obs_buf.clone()
        }
    
    def set_env_state(self, state):
        """Restore environment state for DNNE state management"""
        self.dof_pos[:] = state["dof_pos"]
        self.dof_vel[:] = state["dof_vel"]
        self.progress_buf[:] = state["progress_buf"]
        self.reset_buf[:] = state["reset_buf"]
        self.obs_buf[:] = state["obs_buf"]
        
        # Apply state to simulation
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )