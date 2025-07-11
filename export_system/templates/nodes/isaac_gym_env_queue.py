# Template variables - replaced during export
template_vars = {
    "NODE_ID": "isaac_gym_1",
    "CLASS_NAME": "IsaacGymEnvNode",
    "ENV_NAME": "Cartpole",
    "NUM_ENVS": 64,
    "ISAAC_GYM_PATH": "/home/asantanna/isaacgym",
    "ISAAC_GYM_ENVS_PATH": "/home/asantanna/IsaacGymEnvs",
    "HEADLESS": True,
    "DEVICE": "cuda",
    "PHYSICS_ENGINE": "physx"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym environment node for robotics simulation"""
    
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
        self.envs = []
        self.actors = []
        self.viewer = None
        self.sim_params = None
        self.device_id = 0 if self.device == "cuda" else -1
        
        # Environment state
        self.env_initialized = False
        self.step_count = 0
        self.episode_count = 0
        self.total_rewards = None
        
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
                raise ValueError(f"Isaac Gym path does not exist: {{self.isaac_gym_path}}")
            
            if not os.path.exists(self.isaac_gym_envs_path):
                raise ValueError(f"Isaac Gym Envs path does not exist: {{self.isaac_gym_envs_path}}")
            
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
            self.sim_params.dt = 1.0/60.0  # 60 Hz
            self.sim_params.substeps = 2
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
            
            # Create environments
            self._create_environments()
            
            # Create viewer if not headless
            if not self.headless:
                self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                if self.viewer is None:
                    self.logger.warning("Failed to create viewer")
            
            # Prepare simulation
            self.gym.prepare_sim(self.sim)
            
            # Initialize state tracking
            self.total_rewards = [0.0] * self.num_envs
            self.env_initialized = True
            
            self.logger.info(f"Isaac Gym initialized: {{self.env_name}} with {{self.num_envs}} environments")
            
        except ImportError as e:
            self.logger.error(f"Isaac Gym not available: {{e}}")
            raise RuntimeError("Isaac Gym is not installed or not available")
        except Exception as e:
            self.logger.error(f"Failed to initialize Isaac Gym: {{e}}")
            raise
    
    def _create_environments(self):
        """Create Isaac Gym environments"""
        import os
        from isaacgym import gymapi
        
        # Environment-specific asset loading
        if self.env_name.lower() == "cartpole":
            self._create_cartpole_environments()
        elif self.env_name.lower() == "ant":
            self._create_ant_environments()
        elif self.env_name.lower() == "humanoid":
            self._create_humanoid_environments()
        else:
            self._create_generic_environments()
    
    def _create_cartpole_environments(self):
        """Create Cartpole environments"""
        import os
        from isaacgym import gymapi
        
        # Load cartpole asset
        asset_root = os.path.join(self.isaac_gym_envs_path, "assets")
        asset_file = "urdf/cartpole.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        
        try:
            cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            self.logger.warning("Cartpole asset not found, creating simple fallback")
            cartpole_asset = self.gym.create_box(self.sim, 1.0, 1.0, 1.0, asset_options)
        
        # Create environments
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        envs_per_row = int(self.num_envs ** 0.5)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            
            # Add cartpole actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, cartpole_asset, pose, "cartpole", i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_ant_environments(self):
        """Create Ant environments"""
        import os
        from isaacgym import gymapi
        
        asset_root = os.path.join(self.isaac_gym_envs_path, "assets")
        asset_file = "mjcf/nv_ant.xml"
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 64.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        try:
            ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            raise RuntimeError(f"Failed to load Ant asset from {{asset_root}}/{{asset_file}}")
        
        # Create environments
        spacing = 4.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        envs_per_row = int(self.num_envs ** 0.5)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.75)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, ant_asset, pose, "ant", i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_humanoid_environments(self):
        """Create Humanoid environments"""
        import os
        from isaacgym import gymapi
        
        asset_root = os.path.join(self.isaac_gym_envs_path, "assets")
        asset_file = "mjcf/nv_humanoid.xml"
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 64.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        try:
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            raise RuntimeError(f"Failed to load Humanoid asset from {{asset_root}}/{{asset_file}}")
        
        # Create environments
        spacing = 5.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        envs_per_row = int(self.num_envs ** 0.5)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.34)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, humanoid_asset, pose, "humanoid", i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_generic_environments(self):
        """Create generic environments"""
        import os
        from isaacgym import gymapi
        
        asset_root = os.path.join(self.isaac_gym_envs_path, "assets")
        
        # Common asset locations
        possible_paths = [
            f"mjcf/{{self.env_name.lower()}}.xml",
            f"urdf/{{self.env_name.lower()}}.urdf",
            f"mjcf/nv_{{self.env_name.lower()}}.xml",
            f"urdf/nv_{{self.env_name.lower()}}.urdf",
        ]
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        asset = None
        for asset_file in possible_paths:
            try:
                asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                self.logger.info(f"Loaded asset: {{asset_file}}")
                break
            except:
                continue
        
        if asset is None:
            raise RuntimeError(f"Failed to load asset for environment: {{self.env_name}}")
        
        # Create environments
        spacing = 4.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        envs_per_row = int(self.num_envs ** 0.5)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, envs_per_row)
            
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, asset, pose, self.env_name.lower(), i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    async def compute(self, actions=None) -> Dict[str, Any]:
        """Execute one simulation step"""
        if not self.env_initialized:
            raise RuntimeError("Isaac Gym environment not initialized")
        
        # Apply actions if provided
        if actions is not None:
            self._apply_actions(actions)
        
        # Step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Update viewer if available
        if self.viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
        
        # Get current state
        robot_state = self._get_robot_state()
        sensor_data = self._get_sensor_data()
        rewards = self._compute_rewards()
        done = self._check_done()
        info = self._get_info()
        
        # Update counters
        self.step_count += 1
        
        # Update total rewards
        for i, reward in enumerate(rewards):
            self.total_rewards[i] += reward
        
        # Log progress
        if self.step_count % 1000 == 0:
            avg_reward = sum(self.total_rewards) / len(self.total_rewards)
            self.logger.info(f"Step {{self.step_count}}: avg reward = {{avg_reward:.4f}}")
        
        # Return initial observations for RL training
        observations = self._get_initial_observations()
        
        return {{
            "observations": observations,
            "sim_handle": self  # Pass the environment node itself as sim_handle
        }}
    
    def _apply_actions(self, actions):
        """Apply actions to simulation"""
        # This is a simplified implementation
        # In practice, you'd apply the actual joint commands based on the environment
        pass
    
    def _get_robot_state(self):
        """Get current robot state from simulation"""
        # Simplified implementation - return basic state info
        return {{
            "joint_positions": [0.0] * 8,  # Placeholder
            "joint_velocities": [0.0] * 8,
            "base_position": [0.0, 0.0, 1.0],
            "base_orientation": [0.0, 0.0, 0.0, 1.0],
            "timestamp": self.gym.get_sim_time(self.sim)
        }}
    
    def _get_sensor_data(self):
        """Get current sensor data"""
        # Simplified IMU data
        return {{
            "sensor_type": "imu",
            "linear_acceleration": [0.0, 0.0, 9.81],
            "angular_velocity": [0.0, 0.0, 0.0],
            "timestamp": self.gym.get_sim_time(self.sim)
        }}
    
    def _compute_rewards(self):
        """Compute rewards for each environment"""
        # Simplified reward computation
        # In practice, this would be environment-specific
        return [1.0] * self.num_envs  # Basic survival reward
    
    def _check_done(self):
        """Check if episodes are done"""
        # Simplified done check
        return [False] * self.num_envs
    
    def _get_info(self):
        """Get additional info"""
        return {{
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "average_reward": sum(self.total_rewards) / len(self.total_rewards),
            "sim_time": self.gym.get_sim_time(self.sim)
        }}
    
    def cleanup(self):
        """Clean up Isaac Gym resources"""
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
        
        self.logger.info("Isaac Gym resources cleaned up")
    
    def _get_initial_observations(self):
        """Get initial observations tensor in standard RL format"""
        import torch
        
        # Create initial observation tensor matching the observation space
        # This should match what the neural network expects
        
        # Example observation space for robotics:
        # - Joint positions (8 DOF)
        # - Joint velocities (8 DOF) 
        # - Base orientation (4 quaternion components)
        # - Additional state info (1 upright indicator)
        obs_dim = 21  # 8 + 8 + 4 + 1
        
        device_torch = torch.device(self.device if self.device == "cuda" and torch.cuda.is_available() else "cpu")
        observations = torch.zeros(self.num_envs, obs_dim, device=device_torch, dtype=torch.float32)
        
        # Initialize with reasonable default values
        try:
            # Joint positions - start at zero (neutral pose)
            observations[:, 0:8] = 0.0
            
            # Joint velocities - start at zero (stationary)
            observations[:, 8:16] = 0.0
            
            # Base orientation - identity quaternion (w=1, x=0, y=0, z=0)
            observations[:, 16] = 1.0  # w
            observations[:, 17:20] = 0.0  # x, y, z
            
            # Upright indicator - start upright
            observations[:, 20] = 1.0
            
            self.logger.info(f"Initialized observations tensor: shape={{observations.shape}}, device={{observations.device}}")
            
        except Exception as e:
            self.logger.error(f"Error initializing observations: {{e}}")
            # Return zeros if initialization fails
            observations = torch.zeros(self.num_envs, obs_dim, device=device_torch, dtype=torch.float32)
        
        return observations