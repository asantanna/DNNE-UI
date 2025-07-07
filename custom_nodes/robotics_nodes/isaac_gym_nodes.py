# isaac_gym_nodes.py
"""
Isaac Gym integration nodes for DNNE
Provides Isaac Gym environment setup and control for robotics simulation
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from .base_node import LearningNodeBase
from .robotics_types import RobotState, Action, SensorData, SimHandle, Context

# Isaac Gym imports (with proper error handling)
try:
    # Important: Isaac Gym must be imported before torch
    import isaacgym
    from isaacgym import gymapi, gymutil, gymtorch
    from isaacgym.torch_utils import *
    ISAAC_GYM_AVAILABLE = True
except ImportError as e:
    print(f"Isaac Gym not available: {e}")
    ISAAC_GYM_AVAILABLE = False
    # Mock objects for when Isaac Gym isn't available
    gymapi = None
    gymutil = None
    gymtorch = None

class IsaacGymEnvNode(LearningNodeBase):
    """
    Isaac Gym Environment Node
    Sets up and manages Isaac Gym environments for robotics simulation
    """
    
    CATEGORY = "robotics/simulation"
    
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
                "isaac_gym_path": ("STRING", {
                    "default": "/home/asantanna/isaacgym",
                    "multiline": False,
                    "tooltip": "Path to Isaac Gym installation"
                }),
                "isaac_gym_envs_path": ("STRING", {
                    "default": "/home/asantanna/IsaacGymEnvs",
                    "multiline": False,
                    "tooltip": "Path to Isaac Gym Envs installation"
                }),
                "headless": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run in headless mode (no GUI)"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device for simulation"
                }),
                "physics_engine": (["physx", "flex"], {
                    "default": "physx",
                    "tooltip": "Physics engine to use"
                }),
            },
            "optional": {
                "context": ("CONTEXT",),
            }
        }
    
    RETURN_TYPES = ("SIM_HANDLE", "TENSOR", "CONTEXT")
    RETURN_NAMES = ("sim_handle", "observations", "context")
    FUNCTION = "setup_environment"
    
    def __init__(self):
        super().__init__()
        self.gym = None
        self.sim = None
        self.envs = []
        self.actors = []
        self.viewer = None
        self.sim_params = None
        self.device_id = 0
        self.env_initialized = False
        
    def setup_environment(self, env_name: str, num_envs: int, isaac_gym_path: str, 
                         isaac_gym_envs_path: str, headless: bool, device: str, 
                         physics_engine: str, context: Optional[Context] = None):
        """
        Set up Isaac Gym environment
        """
        if not ISAAC_GYM_AVAILABLE:
            raise RuntimeError("Isaac Gym is not available. Please install Isaac Gym first.")
        
        # Validate paths
        if not os.path.exists(isaac_gym_path):
            raise ValueError(f"Isaac Gym path does not exist: {isaac_gym_path}")
        
        if not os.path.exists(isaac_gym_envs_path):
            raise ValueError(f"Isaac Gym Envs path does not exist: {isaac_gym_envs_path}")
        
        # Set device
        self.device_id = 0 if device == "cuda" else -1
        
        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()
        
        # Configure simulation parameters
        self.sim_params = gymapi.SimParams()
        
        # Physics engine setup
        if physics_engine == "physx":
            self.sim_params.physx.solver_type = 1
            self.sim_params.physx.num_position_iterations = 4
            self.sim_params.physx.num_velocity_iterations = 1
            self.sim_params.physx.num_threads = 4
            self.sim_params.physx.use_gpu = (device == "cuda")
            self.sim_params.use_gpu_pipeline = (device == "cuda")
        else:  # flex
            self.sim_params.flex.solver_type = 5
            self.sim_params.flex.num_outer_iterations = 4
            self.sim_params.flex.num_inner_iterations = 10
        
        # General simulation parameters
        self.sim_params.dt = 1.0/60.0  # 60 Hz
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # Create simulation
        compute_device = self.device_id if device == "cuda" else 0
        graphics_device = 0 if not headless else -1
        
        self.sim = self.gym.create_sim(
            compute_device, graphics_device, 
            gymapi.SIM_PHYSX if physics_engine == "physx" else gymapi.SIM_FLEX,
            self.sim_params
        )
        
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")
        
        # Create ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
        # Load environment-specific assets and create environments
        self._create_environments(env_name, num_envs, isaac_gym_envs_path)
        
        # Create viewer if not headless
        if not headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("Warning: Failed to create viewer")
        
        # Prepare simulation
        self.gym.prepare_sim(self.sim)
        
        # Create simulation handle
        sim_handle = SimHandle(
            sim=self.sim,
            envs=self.envs,
            viewer=self.viewer,
            dt=self.sim_params.dt,
            device=device,
            graphics_device=graphics_device
        )
        
        # Get initial observations as tensor
        initial_observations = self._get_initial_observations(num_envs, device)
        
        # Update context
        if context is None:
            context = Context()
        context.store("sim_handle", sim_handle)
        context.store("env_name", env_name)
        context.store("num_envs", num_envs)
        context.store("isaac_gym_initialized", True)
        
        self.env_initialized = True
        
        return (sim_handle, initial_observations, context)
    
    def _create_environments(self, env_name: str, num_envs: int, isaac_gym_envs_path: str):
        """Create the specified environments"""
        
        # Environment-specific asset loading
        if env_name.lower() == "cartpole":
            self._create_cartpole_environments(num_envs)
        elif env_name.lower() == "ant":
            self._create_ant_environments(num_envs, isaac_gym_envs_path)
        elif env_name.lower() == "humanoid":
            self._create_humanoid_environments(num_envs, isaac_gym_envs_path)
        else:
            # Generic environment creation
            self._create_generic_environments(env_name, num_envs, isaac_gym_envs_path)
    
    def _create_cartpole_environments(self, num_envs: int):
        """Create Cartpole environments"""
        # Load cartpole asset
        asset_root = os.path.join(os.path.dirname(__file__), "../../assets")
        asset_file = "cartpole.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        
        # Try to load asset (fallback to default if not found)
        try:
            cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            # Create a simple cartpole programmatically if asset not found
            print("Warning: Cartpole asset not found, creating simple box")
            cartpole_asset = self._create_simple_cartpole_asset()
        
        # Create environments
        spacing = 2.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(num_envs)))
            
            # Add cartpole actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, cartpole_asset, pose, "cartpole", i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_ant_environments(self, num_envs: int, isaac_gym_envs_path: str):
        """Create Ant environments"""
        # Load ant asset from IsaacGymEnvs
        asset_root = os.path.join(isaac_gym_envs_path, "assets")
        asset_file = "mjcf/nv_ant.xml"
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 64.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        try:
            ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            raise RuntimeError(f"Failed to load Ant asset from {asset_root}/{asset_file}")
        
        # Create environments
        spacing = 4.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(num_envs)))
            
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 0.75)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, ant_asset, pose, "ant", i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_humanoid_environments(self, num_envs: int, isaac_gym_envs_path: str):
        """Create Humanoid environments"""
        # Load humanoid asset
        asset_root = os.path.join(isaac_gym_envs_path, "assets")
        asset_file = "mjcf/nv_humanoid.xml"
        
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 64.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        try:
            humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        except:
            raise RuntimeError(f"Failed to load Humanoid asset from {asset_root}/{asset_file}")
        
        # Create environments
        spacing = 5.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(num_envs)))
            
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.34)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, humanoid_asset, pose, "humanoid", i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_generic_environments(self, env_name: str, num_envs: int, isaac_gym_envs_path: str):
        """Create generic environments"""
        # Try to load from IsaacGymEnvs assets
        asset_root = os.path.join(isaac_gym_envs_path, "assets")
        
        # Common asset locations
        possible_paths = [
            f"mjcf/{env_name.lower()}.xml",
            f"urdf/{env_name.lower()}.urdf",
            f"mjcf/nv_{env_name.lower()}.xml",
            f"urdf/nv_{env_name.lower()}.urdf",
        ]
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        asset = None
        for asset_file in possible_paths:
            try:
                asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
                break
            except:
                continue
        
        if asset is None:
            raise RuntimeError(f"Failed to load asset for environment: {env_name}")
        
        # Create environments
        spacing = 4.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        
        for i in range(num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(num_envs)))
            
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env, asset, pose, env_name.lower(), i, 1)
            
            self.envs.append(env)
            self.actors.append(actor_handle)
    
    def _create_simple_cartpole_asset(self):
        """Create a simple cartpole asset programmatically"""
        # This is a fallback if no asset file is found
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        
        # Create a simple box asset as fallback
        return self.gym.create_box(self.sim, 1.0, 1.0, 1.0, asset_options)
    
    def _get_initial_observations(self, num_envs: int, device: str) -> torch.Tensor:
        """Get initial observations tensor in standard RL format"""
        # Create initial observation tensor
        # This should match the observation space expected by the neural network
        
        # Example observation space for a typical robotics environment:
        # - Joint positions (8 DOF)
        # - Joint velocities (8 DOF) 
        # - Base orientation (4 quaternion components)
        # - Additional state info (1 upright indicator)
        obs_dim = 21  # 8 + 8 + 4 + 1
        
        device_torch = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        observations = torch.zeros(num_envs, obs_dim, device=device_torch, dtype=torch.float32)
        
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
            
        except Exception as e:
            print(f"Warning: Error initializing observations: {e}")
            # Return zeros if initialization fails
            observations = torch.zeros(num_envs, obs_dim, device=device_torch, dtype=torch.float32)
        
        return observations
    
    def _get_initial_state(self, num_envs: int) -> RobotState:
        """Get initial robot state from simulation"""
        # This is a simplified version - in practice, you'd query the actual DOF states
        
        # For now, return a generic state
        joint_positions = torch.zeros(num_envs, 8)  # Assuming 8 DOF
        joint_velocities = torch.zeros(num_envs, 8)
        base_position = torch.zeros(num_envs, 3)
        base_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0).repeat(num_envs, 1)
        
        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_position=base_position,
            base_orientation=base_orientation,
            timestamp=0.0,
            frame_id="base_link"
        )
    
    def _get_sensor_data(self, num_envs: int) -> SensorData:
        """Get initial sensor data"""
        # Create dummy sensor data
        sensor_data = SensorData(
            sensor_type="imu",
            data=torch.zeros(num_envs, 6),  # 3 linear accel + 3 angular vel
            linear_acceleration=torch.tensor([0.0, 0.0, 9.81]).unsqueeze(0).repeat(num_envs, 1),
            angular_velocity=torch.zeros(num_envs, 3),
            timestamp=0.0,
            frame_id="imu_link"
        )
        
        return sensor_data
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute to ensure fresh simulation state"""
        return float("inf")  # Always changed
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate Isaac Gym availability and paths"""
        if not ISAAC_GYM_AVAILABLE:
            return False
        
        isaac_gym_path = kwargs.get("isaac_gym_path", "")
        isaac_gym_envs_path = kwargs.get("isaac_gym_envs_path", "")
        
        if not os.path.exists(isaac_gym_path):
            return False
        
        if not os.path.exists(isaac_gym_envs_path):
            return False
        
        return True


class IsaacGymStepNode(LearningNodeBase):
    """
    Isaac Gym Step Node
    Executes a single simulation step with the given actions
    """
    
    CATEGORY = "robotics/simulation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sim_handle": ("SIM_HANDLE",),
                "actions": ("ACTION",),
            },
            "optional": {
                "context": ("CONTEXT",),
                "trigger": ("SYNC",),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "DICT", "TENSOR", "CONTEXT")
    RETURN_NAMES = ("observations", "rewards", "done", "info", "next_observations", "context")
    FUNCTION = "step_simulation"
    
    def __init__(self):
        super().__init__()
        self.cached_observations = None
        self.cached_rewards = None
        self.cached_done = None
        self.cached_info = None
        self.step_count = 0
    
    def step_simulation(self, sim_handle: SimHandle, actions: Action, context: Optional[Context] = None, trigger: Optional[Dict] = None):
        """Execute one simulation step with RL interface and state caching"""
        
        if not sim_handle.is_valid():
            raise RuntimeError("Invalid simulation handle")
        
        if not ISAAC_GYM_AVAILABLE:
            raise RuntimeError("Isaac Gym is not available")
        
        gym = gymapi.acquire_gym()
        num_envs = len(sim_handle.envs)
        
        # Handle trigger-based next_observations output
        if trigger is not None:
            # Output cached observations when triggered by TrainingStep
            next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(num_envs, 1)
            return (
                torch.zeros(num_envs, 1),  # observations (dummy)
                torch.zeros(num_envs),     # rewards (dummy)  
                torch.zeros(num_envs, dtype=torch.bool),  # done (dummy)
                {},                        # info (dummy)
                next_observations,         # next_observations (cached)
                context if context is not None else Context()
            )
        
        # Normal execution: apply actions and step simulation
        
        # Apply actions to simulation
        if actions.joint_commands is not None:
            self._apply_actions(gym, sim_handle, actions)
        
        # Step physics
        gym.simulate(sim_handle.sim)
        gym.fetch_results(sim_handle.sim, True)
        
        # Update viewer if available
        if sim_handle.viewer is not None:
            gym.step_graphics(sim_handle.sim)
            gym.draw_viewer(sim_handle.viewer, sim_handle.sim, True)
            gym.sync_frame_time(sim_handle.sim)
        
        # Get RL outputs in standard format
        observations = self._get_observations(sim_handle)
        rewards = self._compute_rewards(sim_handle)
        done = self._check_done(sim_handle)
        info = self._get_info(sim_handle)
        
        # Cache observations for later trigger-based output
        self.cached_observations = observations
        self.cached_rewards = rewards
        self.cached_done = done
        self.cached_info = info
        
        # Update context
        if context is None:
            context = Context()
        context.step_count += 1
        context.store("last_step_time", gym.get_sim_time(sim_handle.sim))
        self.step_count += 1
        
        return (
            observations,              # Current step observations
            rewards,                  # Current step rewards
            done,                     # Current step done flags
            info,                     # Current step info
            torch.zeros(num_envs, 1), # next_observations (empty until triggered)
            context
        )
    
    def _apply_actions(self, gym, sim_handle: SimHandle, actions: Action):
        """Apply actions to Isaac Gym simulation"""
        try:
            # Example: Apply joint commands if actions contain joint data
            if hasattr(actions, 'joint_commands') and actions.joint_commands is not None:
                # Apply joint commands to all environments
                for i, env in enumerate(sim_handle.envs):
                    if i < len(sim_handle.actors):
                        # In practice, you would set DOF targets here
                        # gym.set_dof_position_target_tensor(sim_handle.sim, actions.joint_commands)
                        pass
            
            # Example: Apply forces if actions contain force data  
            elif hasattr(actions, 'forces') and actions.forces is not None:
                # Apply forces to actors
                for i, env in enumerate(sim_handle.envs):
                    if i < len(sim_handle.actors):
                        # gym.apply_rigid_body_force_tensors(sim_handle.sim, actions.forces, None, gymapi.ENV_SPACE)
                        pass
                        
        except Exception as e:
            print(f"Warning: Failed to apply actions: {e}")
    
    def _get_observations(self, sim_handle: SimHandle) -> torch.Tensor:
        """Get observations tensor in standard RL format"""
        num_envs = len(sim_handle.envs)
        
        # In a real implementation, you would:
        # 1. Get DOF states from the simulation
        # 2. Get rigid body states
        # 3. Flatten into observation vector
        
        # For now, create a simplified observation tensor
        # Typical observations might include: joint positions, velocities, base pose, etc.
        obs_dim = 21  # 8 joint pos + 8 joint vel + 4 quaternion + 1 upright indicator
        device_torch = torch.device(self.device if hasattr(self, 'device') else "cpu")
        observations = torch.zeros(num_envs, obs_dim, device=device_torch, dtype=torch.float32)
        
        # Example observation construction (simplified)
        try:
            # Joint positions (8 DOF) - placeholder
            observations[:, 0:8] = 0.0
            
            # Joint velocities (8 DOF) - placeholder  
            observations[:, 8:16] = 0.0
            
            # Base orientation quaternion (4 values) - identity quaternion
            observations[:, 16] = 1.0  # w
            observations[:, 17:20] = 0.0  # x, y, z
            
            # Upright indicator (1 value) - placeholder
            observations[:, 20] = 1.0  # Assume upright
            
        except Exception as e:
            print(f"Warning: Error constructing observations: {e}")
            # Return zeros if construction fails
            observations = torch.zeros(num_envs, obs_dim, device=device_torch, dtype=torch.float32)
        
        return observations
    
    def _compute_rewards(self, sim_handle: SimHandle) -> torch.Tensor:
        """Compute rewards for each environment"""
        num_envs = len(sim_handle.envs)
        
        # In a real implementation, reward computation would be environment-specific
        # Examples:
        # - Cartpole: reward for keeping pole upright
        # - Ant: reward for forward movement
        # - Humanoid: reward for maintaining balance and forward progress
        
        try:
            # Simple reward computation (environment-specific)
            rewards = torch.ones(num_envs, dtype=torch.float32)  # Basic survival reward
            
            # Example: Add rewards based on observations
            # observations = self._get_observations(sim_handle)
            # upright_bonus = observations[:, 20] * 0.1  # Bonus for being upright
            # rewards += upright_bonus
            
        except Exception as e:
            print(f"Warning: Error computing rewards: {e}")
            rewards = torch.zeros(num_envs, dtype=torch.float32)
        
        return rewards
    
    def _check_done(self, sim_handle: SimHandle) -> torch.Tensor:
        """Check if episodes are done"""
        num_envs = len(sim_handle.envs)
        
        try:
            # In a real implementation, you would check environment-specific termination conditions
            # Examples:
            # - Cartpole: pole angle too large
            # - Ant: robot fell over  
            # - Humanoid: robot fell down
            # - Timeout: max episode length reached
            
            done = torch.zeros(num_envs, dtype=torch.bool)
            
            # Example termination checks
            # observations = self._get_observations(sim_handle)
            # upright = observations[:, 20]
            # done = upright < 0.1  # Episode ends if not upright
            
            # Check for timeout
            if self.step_count > 1000:  # Max episode length
                done[:] = True
                
        except Exception as e:
            print(f"Warning: Error checking done conditions: {e}")
            done = torch.zeros(num_envs, dtype=torch.bool)
        
        return done
    
    def _get_info(self, sim_handle: SimHandle) -> Dict[str, Any]:
        """Get additional information dictionary"""
        try:
            gym = gymapi.acquire_gym()
            info = {
                "step_count": self.step_count,
                "sim_time": gym.get_sim_time(sim_handle.sim),
                "num_envs": len(sim_handle.envs),
                "physics_dt": sim_handle.dt if hasattr(sim_handle, 'dt') else 0.0167,
                # Add environment-specific info as needed
            }
        except Exception as e:
            print(f"Warning: Error getting info: {e}")
            info = {
                "step_count": self.step_count,
                "sim_time": 0.0,
                "num_envs": 0,
                "physics_dt": 0.0167
            }
        
        return info
    
    def _get_robot_state(self, sim_handle: SimHandle) -> RobotState:
        """Get current robot state from simulation"""
        # This would query actual DOF states from Isaac Gym
        # Simplified version for now
        num_envs = len(sim_handle.envs)
        
        joint_positions = torch.zeros(num_envs, 8)
        joint_velocities = torch.zeros(num_envs, 8)
        base_position = torch.zeros(num_envs, 3)
        base_orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]).unsqueeze(0).repeat(num_envs, 1)
        
        return RobotState(
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            base_position=base_position,
            base_orientation=base_orientation,
            timestamp=float(gymapi.acquire_gym().get_sim_time(sim_handle.sim)),
            frame_id="base_link"
        )
    
    def _get_sensor_data(self, sim_handle: SimHandle) -> SensorData:
        """Get current sensor data"""
        num_envs = len(sim_handle.envs)
        
        return SensorData(
            sensor_type="imu",
            data=torch.zeros(num_envs, 6),
            linear_acceleration=torch.tensor([0.0, 0.0, 9.81]).unsqueeze(0).repeat(num_envs, 1),
            angular_velocity=torch.zeros(num_envs, 3),
            timestamp=float(gymapi.acquire_gym().get_sim_time(sim_handle.sim)),
            frame_id="imu_link"
        )


class ORNode(LearningNodeBase):
    """
    OR/ANY Node for RL Training Loop State Routing
    Outputs when ANY input becomes available - used for routing initial state vs ongoing state
    """
    
    CATEGORY = "robotics/utility"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "input_a": ("TENSOR",),
                "input_b": ("TENSOR",),
                "input_c": ("TENSOR",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output",)
    FUNCTION = "route_input"
    
    def __init__(self):
        super().__init__()
        self.last_input_source = None
        self.output_count = 0
    
    def route_input(self, input_a: Optional[torch.Tensor] = None, 
                   input_b: Optional[torch.Tensor] = None, 
                   input_c: Optional[torch.Tensor] = None):
        """Route the first available input to output"""
        
        # Check inputs in order of priority (A, B, C)
        if input_a is not None:
            self.last_input_source = "A"
            self.output_count += 1
            print(f"OR Node: Routing input A (shape: {input_a.shape}) - output #{self.output_count}")
            return (input_a,)
        
        elif input_b is not None:
            self.last_input_source = "B"
            self.output_count += 1
            print(f"OR Node: Routing input B (shape: {input_b.shape}) - output #{self.output_count}")
            return (input_b,)
        
        elif input_c is not None:
            self.last_input_source = "C"
            self.output_count += 1
            print(f"OR Node: Routing input C (shape: {input_c.shape}) - output #{self.output_count}")
            return (input_c,)
        
        else:
            # No inputs available - this shouldn't happen in normal operation
            raise RuntimeError("OR Node: No inputs available")
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")  # Always changed
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate that at least one input will be provided"""
        # This node should work with any tensor inputs
        return True


class CartpoleActionNode(LearningNodeBase):
    """
    Cartpole Action Node
    Converts neural network output to Isaac Gym ACTION format for Cartpole environment
    """
    
    CATEGORY = "robotics/cartpole"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "network_output": ("TENSOR",),
                "max_push_effort": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Maximum force that can be applied to cart"
                }),
            },
            "optional": {
                "context": ("CONTEXT",),
            }
        }
    
    RETURN_TYPES = ("ACTION", "CONTEXT")
    RETURN_NAMES = ("action", "context")
    FUNCTION = "convert_to_action"
    
    def __init__(self):
        super().__init__()
        
    def convert_to_action(self, network_output: torch.Tensor, max_push_effort: float, context: Optional[Context] = None):
        """
        Convert neural network output to Isaac Gym ACTION format for Cartpole
        
        Args:
            network_output: Raw network output tensor [1] or [1, 1] (single force value)
            max_push_effort: Maximum force scaling factor
            context: Optional context
            
        Returns:
            ACTION: Properly formatted action for IsaacGymStepNode
        """
        
        # Ensure network_output is properly shaped
        if network_output.dim() > 1:
            network_output = network_output.squeeze()
        
        if network_output.dim() == 0:
            network_output = network_output.unsqueeze(0)
            
        # Scale by max effort (same as IsaacGym Cartpole implementation)
        scaled_force = network_output[0] * max_push_effort
        
        # For Cartpole: 2 DOF (cart translation, pole rotation)
        # Only cart (DOF 0) is actuated, pole (DOF 1) is passive
        forces = torch.zeros(2, dtype=torch.float32, device=network_output.device)
        forces[0] = scaled_force  # Apply force to cart only
        
        # Create ACTION object
        action = Action(
            forces=forces,
            joint_commands=None,  # Not used for Cartpole
            torques=None          # Not used for Cartpole
        )
        
        # Update context
        if context is None:
            context = Context()
        context.store("last_action_force", scaled_force.item())
        context.store("max_push_effort", max_push_effort)
        
        return (action, context)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")  # Always changed
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs"""
        network_output = kwargs.get("network_output")
        if network_output is not None and not isinstance(network_output, torch.Tensor):
            return False
        return True


class CartpoleRewardNode(LearningNodeBase):
    """
    Cartpole Reward Node
    Computes Cartpole-specific rewards matching IsaacGym implementation
    """
    
    CATEGORY = "robotics/cartpole"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observations": ("TENSOR",),
                "reset_dist": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Cart position limit before episode reset"
                }),
                "invert_for_loss": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Return negative reward for training (loss = -reward)"
                }),
            },
            "optional": {
                "context": ("CONTEXT",),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "CONTEXT")
    RETURN_NAMES = ("reward_or_loss", "done", "info_dict", "context")
    FUNCTION = "compute_reward"
    
    def __init__(self):
        super().__init__()
        self.episode_steps = 0
        self.max_episode_length = 500  # Standard Cartpole episode length
        
    def compute_reward(self, observations: torch.Tensor, reset_dist: float, invert_for_loss: bool, 
                      context: Optional[Context] = None):
        """
        Compute Cartpole reward matching IsaacGym implementation
        
        Args:
            observations: Tensor [1, 4] containing [cart_pos, cart_vel, pole_angle, pole_vel]
            reset_dist: Maximum cart position before reset
            invert_for_loss: Whether to return negative reward for training
            context: Optional context
            
        Returns:
            reward_or_loss: Reward (or negative reward if invert_for_loss=True)
            done: Episode termination flag
            info_dict: Additional information as tensor (placeholder)
        """
        
        # Ensure observations is properly shaped [1, 4]
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
            
        # Extract state components
        cart_pos = observations[0, 0]    # Cart position
        cart_vel = observations[0, 1]    # Cart velocity  
        pole_angle = observations[0, 2]  # Pole angle
        pole_vel = observations[0, 3]    # Pole velocity
        
        # Compute reward (matching IsaacGym Cartpole implementation)
        # reward = 1.0 - pole_angle² - 0.01 * |cart_vel| - 0.005 * |pole_vel|
        reward = (1.0 - pole_angle * pole_angle 
                 - 0.01 * torch.abs(cart_vel) 
                 - 0.005 * torch.abs(pole_vel))
        
        # Check termination conditions
        done = False
        
        # Cart position out of bounds
        if torch.abs(cart_pos) > reset_dist:
            reward = torch.tensor(-2.0, dtype=torch.float32, device=observations.device)
            done = True
            
        # Pole angle too large (fell over)
        if torch.abs(pole_angle) > (np.pi / 2):
            reward = torch.tensor(-2.0, dtype=torch.float32, device=observations.device)
            done = True
            
        # Episode length exceeded
        self.episode_steps += 1
        if self.episode_steps >= self.max_episode_length:
            done = True
            self.episode_steps = 0  # Reset for next episode
            
        # Reset episode counter if done
        if done and self.episode_steps < self.max_episode_length:
            self.episode_steps = 0
            
        # Convert to loss if requested
        if invert_for_loss:
            output = -reward
        else:
            output = reward
            
        # Create done tensor
        done_tensor = torch.tensor([done], dtype=torch.bool, device=observations.device)
        
        # Create info tensor (placeholder for additional data)
        info_tensor = torch.tensor([self.episode_steps], dtype=torch.float32, device=observations.device)
        
        # Update context
        if context is None:
            context = Context()
        context.store("last_reward", reward.item())
        context.store("episode_steps", self.episode_steps)
        context.store("done", done)
        context.total_reward += reward.item()
        
        return (output, done_tensor, info_tensor, context)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute when inputs change"""
        return float("inf")  # Always changed
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs"""
        observations = kwargs.get("observations")
        if observations is not None and not isinstance(observations, torch.Tensor):
            return False
        return True