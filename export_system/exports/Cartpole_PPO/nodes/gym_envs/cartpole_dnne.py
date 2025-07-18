# cartpole_dnne.py
"""
CartpoleDNNE - DNNE adaptation of IsaacGymEnvs Cartpole
Inherits from IGE's Cartpole implementation and adds DNNE-specific features
"""

import numpy as np
import os
import torch
import sys

# Add IsaacGymEnvs to path
sys.path.append("/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs")

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
        self.step_count = 0
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        
        # Call parent class initialization
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        print(f"[CartpoleDNNE] Initialized with {self.num_envs} environments")
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] CartpoleDNNE initialized: num_envs={self.num_envs}, device={self.device}")
        
    def step_async(self, actions):
        """
        DNNE-compatible async step function
        Calls the standard step() but can be used in async context
        """
        self.step_count += 1
        
        # PPO_CYCLE_DEBUG logging for actions
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] CartpoleDNNE step {self.step_count} - actions shape: {actions.shape}")
            print(f"[PPO_CYCLE_DEBUG] Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
        
        # Use the parent class step() which handles everything properly
        obs_dict, rewards, dones, infos = self.step(actions)
        
        # Extract observations from dict (VecTask returns {"obs": tensor})
        observations = obs_dict["obs"]
        
        # PPO_CYCLE_DEBUG logging for outputs
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] CartpoleDNNE step {self.step_count} - observations shape: {observations.shape}")
            print(f"[PPO_CYCLE_DEBUG] Rewards: min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")
            print(f"[PPO_CYCLE_DEBUG] Dones: {dones.sum().item()} environments done")
            
            # Log some observation details (cart pos, cart vel, pole angle, pole vel)
            if observations.shape[1] >= 4:
                cart_pos = observations[:, 0]
                pole_angle = observations[:, 2]
                print(f"[PPO_CYCLE_DEBUG] Cart pos: min={cart_pos.min().item():.4f}, max={cart_pos.max().item():.4f}")
                print(f"[PPO_CYCLE_DEBUG] Pole angle: min={pole_angle.min().item():.4f}, max={pole_angle.max().item():.4f}")
        
        return observations, rewards, dones, infos
    
    def get_initial_observations(self):
        """Get initial observations after reset for DNNE"""
        # Match IGE behavior: just return current obs_buf (zeros initially)
        # The actual reset happens on first step when post_physics_step sees reset_buf=1
        return self.obs_buf
    
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