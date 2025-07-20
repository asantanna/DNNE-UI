from typing import Dict, Any
import isaacgym
import torch
import numpy as np
import os
from framework import QueueNode, SensorNode

# Template variables - replaced during export

# Import DNNE_print from centralized location
from isaacgymenvs.utils.debug_utils import DNNE_print

class IsaacGymEnvNode_7(QueueNode):
    """Isaac Gym environment node using new CartpoleDNNE approach"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs(["env_handle", "observations"])
        
        # Configuration
        self.env_name = "Cartpole"
        self.num_envs = 512
        self.headless = True
        self.device = "cuda"
        
        # Check for command line override of headless setting
        try:
            import builtins
            if hasattr(builtins, 'VISUAL_MODE') and builtins.VISUAL_MODE:
                self.headless = False
                self.logger.info("Visual mode enabled via command line")
            elif hasattr(builtins, 'HEADLESS_MODE') and builtins.HEADLESS_MODE:
                self.headless = True
                self.logger.info("Headless mode forced via command line")
        except:
            pass  # Use default from template
        
        # Environment instance
        self.env = None
        self.env_initialized = False
        self.initial_observations = None  # Cache initial observations
        
        # Enable PPO_CYCLE_DEBUG and verbose logging
        import os
        import builtins
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        self.verbose = getattr(builtins, 'VERBOSE', False)
        
        # Initialize environment
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment using CartpoleDNNE"""
        try:
            # Paths should already be configured in runner.py
            # Isaac Gym must be imported before torch
            import isaacgym
            
            # Import CartpoleDNNE from gym_envs subdirectory
            import os
            import sys
            nodes_dir = os.path.join(os.path.dirname(__file__))
            gym_envs_dir = os.path.join(nodes_dir, 'gym_envs')
            if gym_envs_dir not in sys.path:
                sys.path.insert(0, gym_envs_dir)
            
            from cartpole_dnne import CartpoleDNNE
            
            # Create config for environment (matching IsaacGymEnvs format)
            cfg = {
                "name": "Cartpole",
                "physics_engine": "physx",
                "env": {
                    "numEnvs": self.num_envs,
                    "envSpacing": 4.0,
                    "resetDist": 3.0,
                    "maxEffort": 10.0,
                    "numObservations": 4,
                    "numActions": 1,
                },
                "sim": {
                    "dt": 1.0/60.0,
                    "substeps": 2,
                    "up_axis": "z",
                    "use_gpu_pipeline": self.device == "cuda",
                    "gravity": [0.0, 0.0, -9.81],
                    "physx": {
                        "num_threads": 4,
                        "solver_type": 1,
                        "use_gpu": self.device == "cuda",
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
            rl_device = self.device + ":0" if self.device == "cuda" else self.device
            sim_device = self.device + ":0" if self.device == "cuda" else self.device
            graphics_device_id = -1 if self.headless else 0
            
            # Create environment instance
            self.env = CartpoleDNNE(
                cfg=cfg,
                rl_device=rl_device,
                sim_device=sim_device,
                graphics_device_id=graphics_device_id,
                headless=self.headless,
                virtual_screen_capture=False,
                force_render=False
            )
            
            # Call reset to match IGE initialization behavior
            _ = self.env.reset()
            
            self.env_initialized = True
            self.logger.info(f"CartpoleDNNE initialized with {self.num_envs} environments")
            
            if self.verbose:
                from isaacgymenvs.utils.debug_utils import DNNE_print
                DNNE_print("D", "ENV_INIT", "IsaacGymEnvNode - Initialized CartpoleDNNE")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}")
            raise RuntimeError(f"Environment initialization failed: {e}")
    
    async def compute(self) -> Dict[str, Any]:
        """Get initial observations and return environment handle"""
        if not self.env_initialized or self.env is None:
            raise RuntimeError("Environment not initialized")
        
        try:
            # Debug: Track how many times this is called
            if not hasattr(self, 'compute_count'):
                self.compute_count = 0
            self.compute_count += 1
            
            if self.verbose:
                from isaacgymenvs.utils.debug_utils import DNNE_print
                DNNE_print("D", "ENV_COMPUTE", f"IsaacGymEnvNode.compute() call #{self.compute_count}")
            
            # Get initial observations
            initial_observations = self.env.get_initial_observations()
            
            # Create environment handle
            env_handle = {
                "environment": self.env,
                "gym": self.env.gym,
                "sim": self.env.sim,
                "viewer": self.env.viewer if hasattr(self.env, 'viewer') else None,
                "device": self.device,
                "num_envs": self.num_envs,
            }
            
            if self.verbose:
                DNNE_print("D", "ENV_COMPUTE", f"IsaacGymEnvNode - Initial observations shape: {initial_observations.shape}")
                DNNE_print("D", "ENV_COMPUTE", f"Initial obs: min={initial_observations.min().item():.4f}, max={initial_observations.max().item():.4f}, mean={initial_observations.mean().item():.4f}")
            
            return {
                "env_handle": env_handle,
                "observations": initial_observations
            }
            
        except Exception as e:
            self.logger.error(f"Error in environment compute: {e}")
            raise
    
    def cleanup(self):
        """Clean up environment resources"""
        if self.env is not None:
            # VecTask handles cleanup automatically
            pass
        
        self.logger.info("Environment resources cleaned up")
