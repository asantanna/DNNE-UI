# Template variables - replaced during export
template_vars = {
    "NODE_ID": "isaac_gym_1",
    "CLASS_NAME": "IsaacGymEnvNode",
    "ENV_NAME": "Cartpole",
    "NUM_ENVS": 64,
    "ISAAC_GYM_PATH": "/home/asantanna/DNNE-LINUX-SUPPORT/isaacgym",
    "ISAAC_GYM_ENVS_PATH": "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs",
    "HEADLESS": True,
    "DEVICE": "cuda",
    "PHYSICS_ENGINE": "physx"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym environment node using clean class hierarchy"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=[])
        self.setup_outputs(["observations", "sim_handle"])
        
        # Configuration
        self.env_name = "{ENV_NAME}"
        self.num_envs = {NUM_ENVS}
        self.isaac_gym_path = "{ISAAC_GYM_PATH}"
        self.isaac_gym_envs_path = "{ISAAC_GYM_ENVS_PATH}"
        self.headless = {HEADLESS}
        self.device = "{DEVICE}"
        self.physics_engine = "{PHYSICS_ENGINE}"
        
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
        
        # Isaac Gym objects
        self.gym = None
        self.sim = None
        self.viewer = None
        self.sim_params = None
        self.device_id = 0 if self.device == "cuda" else -1
        
        # Environment instance (using clean class hierarchy)
        self.environment = None
        self.env_initialized = False
        
        # Initialize Isaac Gym
        self._initialize_isaac_gym()
    
    def _initialize_isaac_gym(self):
        """Initialize Isaac Gym simulation"""
        try:
            # Isaac Gym must be imported before torch
            import isaacgym
            from isaacgym import gymapi, gymutil, gymtorch
            import isaacgym.torch_utils as torch_utils
            
            # Validate paths
            import os
            if not os.path.exists(self.isaac_gym_path):
                raise ValueError(f"Isaac Gym path does not exist: {self.isaac_gym_path}")
            
            if not os.path.exists(self.isaac_gym_envs_path):
                raise ValueError(f"Isaac Gym Envs path does not exist: {self.isaac_gym_envs_path}")
            
            # Initialize Isaac Gym
            self.gym = gymapi.acquire_gym()
            
            # Configure simulation parameters
            self.sim_params = gymapi.SimParams()
            
            # Physics engine setup
            if self.physics_engine == "physx":
                self.sim_params.physx.solver_type = 1
                self.sim_params.physx.num_position_iterations = 4
                self.sim_params.physx.num_velocity_iterations = 1
                self.sim_params.physx.num_threads = 4
                self.sim_params.physx.use_gpu = (self.device == "cuda")
                self.sim_params.use_gpu_pipeline = (self.device == "cuda")
            else:  # flex
                self.sim_params.flex.solver_type = 5
                self.sim_params.flex.num_outer_iterations = 4
                self.sim_params.flex.num_inner_iterations = 10
            
            # General simulation parameters
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
            
            # Create simulation
            compute_device = self.device_id if self.device == "cuda" else 0
            graphics_device = 0 if not self.headless else -1
            
            self.sim = self.gym.create_sim(
                compute_device, graphics_device, 
                gymapi.SIM_PHYSX if self.physics_engine == "physx" else gymapi.SIM_FLEX,
                self.sim_params
            )
            
            if self.sim is None:
                raise RuntimeError("Failed to create Isaac Gym simulation")
            
            # Create ground plane
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.add_ground(self.sim, plane_params)
            
            # Create environment using clean class hierarchy
            self._create_environment()
            
            # Create viewer if not headless
            if not self.headless:
                self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                if self.viewer is None:
                    self.logger.warning("Failed to create viewer")
            
            # Prepare simulation
            self.gym.prepare_sim(self.sim)
            
            self.env_initialized = True
            self.logger.info(f"Isaac Gym initialized: {self.env_name} with {self.num_envs} environments")
            
        except ImportError as e:
            self.logger.error(f"Isaac Gym not available: {e}")
            raise RuntimeError("Isaac Gym is not installed or not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize Isaac Gym: {e}")
            raise
    
    def _create_environment(self):
        """Create environment instance using clean class hierarchy"""
        try:
            # Import environment factory
            import sys
            import os
            
            # Add templates directory to path to import environment classes
            template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
            if template_dir not in sys.path:
                sys.path.insert(0, template_dir)
            
            from environments import create_environment
            
            # Create environment instance
            self.environment = create_environment(
                env_name=self.env_name,
                gym=self.gym,
                sim=self.sim,
                sim_params=self.sim_params,
                num_envs=self.num_envs,
                device=self.device,
                logger=self.logger,
                isaac_gym_envs_path=self.isaac_gym_envs_path
            )
            
            # Create all environments
            self.environment.create_environments(spacing=4.0)
            
            self.logger.info(f"Created {self.environment.get_environment_name()} environment with {self.num_envs} instances")
            
        except Exception as e:
            self.logger.error(f"Failed to create environment: {e}")
            raise RuntimeError(f"Environment creation failed: {e}")
    
    async def compute(self, actions=None) -> Dict[str, Any]:
        """Execute environment step and return observations"""
        if not self.env_initialized or self.environment is None:
            raise RuntimeError("Isaac Gym environment not initialized")
        
        try:
            # Get initial observations (for environment setup)
            observations = self.environment.get_observations()
            
            # Update viewer if available
            if self.viewer is not None:
                self.environment.update_viewer(self.viewer)
            
            # Return observations and environment handle
            return {
                "observations": observations,
                "sim_handle": self  # Pass environment node as sim_handle
            }
            
        except Exception as e:
            self.logger.error(f"Error in environment compute: {e}")
            raise
    
    def step_environment(self, actions):
        """Step the environment with actions (called by IsaacGymStep node)"""
        if self.environment is None:
            raise RuntimeError("Environment not initialized")
        
        # Use environment's step_simulation method
        return self.environment.step_simulation(actions)
    
    def cleanup(self):
        """Clean up Isaac Gym resources"""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
        
        self.logger.info("Isaac Gym resources cleaned up")