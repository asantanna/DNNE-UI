# isaac_gym_base_nodes.py
"""
Base Isaac Gym nodes for DNNE
Generic environment setup and stepping nodes that work with any IsaacGymEnvs environment
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from inspect import cleandoc
from .base_node import LearningNodeBase
from .robotics_types import SimHandle

# Add paths for Isaac Gym
sys.path.append("/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym/python")
sys.path.append("/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs")

# Isaac Gym imports (with proper error handling)
try:
    # Important: Isaac Gym must be imported before torch
    import isaacgym
    from isaacgym import gymapi, gymutil, gymtorch
    from isaacgym.torch_utils import *
    ISAAC_GYM_AVAILABLE = True
except ImportError as e:
    # Isaac Gym not available - robotics nodes will be disabled
    ISAAC_GYM_AVAILABLE = False
    gymapi = None
    gymutil = None
    gymtorch = None


class IsaacGymEnvNode(LearningNodeBase):
    """
    Isaac Gym Environment Node
    Sets up and manages Isaac Gym environments using IsaacGymEnvs implementations
    """
    
    CATEGORY = "robotics"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_name": ("STRING", {
                    "default": "Cartpole",
                    "multiline": False,
                    "tooltip": "Isaac Gym environment name (e.g., Cartpole, Ant, Humanoid)"
                }),
                "num_envs": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Number of parallel environments"
                }),
                "headless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run in headless mode (no GUI)"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for simulation"
                }),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("ENV_HANDLE", "TENSOR")
    RETURN_NAMES = ("env_handle", "observations")
    FUNCTION = "setup_environment"
    DESCRIPTION = cleandoc(__doc__)
    
    def __init__(self):
        super().__init__()
        self.env = None
        self.viewer = None
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        
    def setup_environment(self, env_name: str, num_envs: int, headless: bool, device: str):
        """
        Set up Isaac Gym environment using IsaacGymEnvs
        """
        if not ISAAC_GYM_AVAILABLE:
            raise RuntimeError("Isaac Gym is not available. Please install Isaac Gym first.")
        
        # Import environment based on name
        if env_name.lower() == "cartpole":
            from gym_envs.cartpole_dnne import CartpoleDNNE
            env_class = CartpoleDNNE
        else:
            raise NotImplementedError(f"Environment {env_name} not yet implemented in DNNE")
        
        # Create config for environment
        cfg = {
            "env": {
                "numEnvs": num_envs,
                "envSpacing": 2.0,
                "resetDist": 2.0,
                "maxEffort": 10.0,
                "numObservations": 4,
                "numActions": 1,
            },
            "sim": {
                "dt": 1.0/60.0,
                "substeps": 2,
                "up_axis": "z",
                "use_gpu_pipeline": device == "cuda",
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "num_threads": 4,
                    "solver_type": 1,
                    "use_gpu": device == "cuda",
                    "num_position_iterations": 4,
                    "num_velocity_iterations": 1,
                    "contact_offset": 0.02,
                    "rest_offset": 0.001,
                    "bounce_threshold_velocity": 0.2,
                    "max_depenetration_velocity": 100.0,
                    "default_buffer_size_multiplier": 5.0,
                    "max_gpu_contact_pairs": 8388608,
                    "num_subscenes": 4,
                    "contact_collection": 0,
                },
            },
        }
        
        # Set up devices
        rl_device = device + ":0" if device == "cuda" else device
        sim_device = device + ":0" if device == "cuda" else device
        graphics_device_id = -1 if headless else 0
        
        # Create environment instance
        self.env = env_class(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=False,
            force_render=False
        )
        
        # Get initial observations
        initial_observations = self.env.get_initial_observations()
        
        # Create environment handle
        env_handle = {
            "environment": self.env,
            "gym": self.env.gym,
            "sim": self.env.sim,
            "viewer": self.env.viewer if hasattr(self.env, 'viewer') else None,
            "device": device,
            "num_envs": num_envs,
        }
        
        print(f"[IsaacGymEnvNode] Created {env_name} with {num_envs} environments")
        
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] IsaacGymEnvNode - Initial observations shape: {initial_observations.shape}")
            print(f"[PPO_CYCLE_DEBUG] Initial obs: min={initial_observations.min().item():.4f}, max={initial_observations.max().item():.4f}, mean={initial_observations.mean().item():.4f}")
        
        return (env_handle, initial_observations)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution on each run"""
        return float("inf")


class IsaacGymStepNode(LearningNodeBase):
    """
    Isaac Gym Step Node
    Executes simulation steps with dual-mode support for RL training synchronization
    """
    
    CATEGORY = "robotics"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "env_handle": ("ENV_HANDLE",),
                "actions": ("TENSOR",),  # Changed from ACTION to TENSOR for simplicity
            },
            "optional": {
                "trigger": ("SYNC",),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "DICT", "TENSOR")
    RETURN_NAMES = ("observations", "rewards", "done", "info", "next_observations")
    FUNCTION = "step_simulation"
    DESCRIPTION = cleandoc(__doc__)
    
    def __init__(self):
        super().__init__()
        self.cached_observations = None
        self.cached_rewards = None
        self.cached_done = None
        self.cached_info = None
        self.step_count = 0
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
    
    def step_simulation(self, env_handle: Dict, actions: torch.Tensor, trigger: Optional[Dict] = None):
        """Execute simulation step with dual-mode support"""
        
        # Extract environment from handle
        env = env_handle["environment"]
        num_envs = env_handle["num_envs"]
        
        # Handle trigger-based output mode
        if trigger is not None:
            # Return cached observations from previous step
            next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(num_envs, 4)
            
            if self.ppo_cycle_debug and self.cached_observations is not None:
                print(f"[PPO_CYCLE_DEBUG] IsaacGymStepNode TRIGGER MODE - Releasing cached observations")
                print(f"[PPO_CYCLE_DEBUG] Cached obs shape: {next_observations.shape}")
            
            return (
                torch.zeros(num_envs, 4),  # observations (dummy)
                torch.zeros(num_envs),     # rewards (dummy)  
                torch.zeros(num_envs, dtype=torch.bool),  # done (dummy)
                {},                        # info (dummy)
                next_observations,         # next_observations (cached)
            )
        
        # Normal execution mode: step environment
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] IsaacGymStepNode step {self.step_count + 1} - NORMAL MODE")
            print(f"[PPO_CYCLE_DEBUG] Input actions shape: {actions.shape}")
            print(f"[PPO_CYCLE_DEBUG] Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
        
        observations, rewards, done, info = env.step_async(actions)
        
        # Cache for later trigger-based output
        self.cached_observations = observations
        self.cached_rewards = rewards
        self.cached_done = done
        self.cached_info = info
        
        self.step_count += 1
        
        # PPO_CYCLE_DEBUG logging for outputs
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] IsaacGymStepNode - After step {self.step_count}:")
            print(f"[PPO_CYCLE_DEBUG] Observations cached: shape={observations.shape}")
            print(f"[PPO_CYCLE_DEBUG] Rewards: min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")
            print(f"[PPO_CYCLE_DEBUG] Done count: {done.sum().item()}")
        
        # Regular debug logging
        if self.step_count % 100 == 0:
            print(f"[IsaacGymStepNode] Step {self.step_count}: "
                  f"obs_shape={observations.shape}, "
                  f"reward_mean={rewards.mean().item():.3f}, "
                  f"done_count={done.sum().item()}")
        
        return (
            observations,              # Current observations
            rewards,                  # Current rewards
            done,                     # Current done flags
            info,                     # Current info
            torch.zeros(num_envs, 4), # next_observations (empty until triggered)
        )
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")