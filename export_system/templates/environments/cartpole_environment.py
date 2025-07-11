"""
Cartpole Environment Implementation

Based on IsaacGymEnvs Cartpole implementation with proper randomization,
force-based action application, and physics state extraction.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from .base_environment import IsaacGymEnvironment


class CartpoleEnvironment(IsaacGymEnvironment):
    """Cartpole environment implementation following IsaacGymEnvs patterns"""
    
    def __init__(self, gym, sim, sim_params, num_envs: int, device: str, logger, isaac_gym_envs_path: str):
        super().__init__(gym, sim, sim_params, num_envs, device, logger)
        
        self.isaac_gym_envs_path = isaac_gym_envs_path
        
        # Cartpole-specific parameters (from IsaacGymEnvs config)
        self.reset_dist = 3.0  # Cart position limit for episode termination
        self.max_push_effort = 400.0  # Maximum force magnitude (N)
        self.max_episode_length = 500  # Maximum steps per episode
        
        # DOF configuration
        self.num_dof = 2  # Cart position + pole angle
        
        # Cartpole asset handle
        self.cartpole_asset = None
        
    def get_environment_name(self) -> str:
        return "Cartpole"
    
    def get_simulation_params(self) -> Dict[str, float]:
        """Cartpole simulation parameters from IsaacGymEnvs"""
        return {
            "dt": 0.0166,  # 1/60 Hz (60 FPS)
            "substeps": 2
        }
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Cartpole action space: 1D force applied to cart"""
        return {
            "dimension": 1,
            "low": -1.0,  # Normalized action
            "high": 1.0,  # Normalized action  
            "scaling": self.max_push_effort,  # Applied as: action * max_push_effort
            "description": "Force applied to cart (normalized -1 to +1, scaled to ±400N)"
        }
    
    def get_observation_space_info(self) -> Dict[str, Any]:
        """Cartpole observation space: [cart_pos, cart_vel, pole_angle, pole_angular_vel]"""
        return {
            "dimension": 4,
            "description": [
                "Cart position (m)",
                "Cart velocity (m/s)", 
                "Pole angle (rad)",
                "Pole angular velocity (rad/s)"
            ]
        }
    
    def create_environments(self, spacing: float = 4.0, num_per_row: int = None) -> None:
        """Create Cartpole environments with IsaacGymEnvs-compatible setup"""
        from isaacgym import gymapi
        
        # Set simulation parameters
        self.set_simulation_parameters()
        
        # Load cartpole asset
        self._load_cartpole_asset()
        
        # Calculate grid layout
        if num_per_row is None:
            num_per_row = int(np.sqrt(self.num_envs))
        
        # Get environment bounds
        env_lower, env_upper = self.get_environment_bounds(spacing)
        
        # Create environments
        for i in range(self.num_envs):
            # Create environment instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            # Create cartpole actor
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, 2.0)  # Z-up configuration
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            actor_handle = self.gym.create_actor(env_ptr, self.cartpole_asset, pose, "cartpole", i, 1, 0)
            
            # Configure DOF properties (critical for proper physics)
            self._configure_dof_properties(env_ptr, actor_handle)
            
            self.envs.append(env_ptr)
            self.actors.append(actor_handle)
        
        # Initialize state tensors after all environments are created
        self.initialize_state_tensors()
        
        # Reset all environments with randomized initial states
        all_env_ids = torch.arange(self.num_envs, device=self.torch_device)
        self.reset_environments(all_env_ids)
        
        self.logger.info(f"Created {self.num_envs} Cartpole environments with randomized initial states")
    
    def _load_cartpole_asset(self) -> None:
        """Load the cartpole URDF asset"""
        from isaacgym import gymapi
        
        # Asset paths (matching IsaacGymEnvs)
        asset_root = os.path.join(self.isaac_gym_envs_path, "assets")
        asset_file = "urdf/cartpole.urdf"
        
        # Asset options (matching IsaacGymEnvs)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Slider rail is fixed
        
        try:
            self.cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            self.num_dof = self.gym.get_asset_dof_count(self.cartpole_asset)
            self.logger.info(f"Loaded Cartpole asset: {asset_file}, DOF: {self.num_dof}")
        except Exception as e:
            self.logger.error(f"Failed to load Cartpole asset from {asset_root}/{asset_file}: {e}")
            # Create fallback box asset
            self.cartpole_asset = self.gym.create_box(self.sim, 1.0, 1.0, 1.0, asset_options)
            self.num_dof = 2  # Assume 2 DOF for fallback
            self.logger.warning("Using fallback box asset for Cartpole")
    
    def _configure_dof_properties(self, env_ptr, actor_handle) -> None:
        """Configure DOF properties following IsaacGymEnvs pattern"""
        from isaacgym import gymapi
        
        # Get and configure DOF properties
        dof_props = self.gym.get_actor_dof_properties(env_ptr, actor_handle)
        
        # DOF 0: Cart (controlled by force)
        dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
        
        # DOF 1: Pole (free-swinging, no actuation)
        dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
        
        # Zero stiffness and damping for realistic physics
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0
        
        # Apply properties
        self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)
    
    def reset_environments(self, env_ids: torch.Tensor) -> None:
        """Reset specified environments with randomized initial states"""
        if len(env_ids) == 0:
            return
        
        # Randomized initial positions (matching IsaacGymEnvs)
        # Cart position: -0.1 to +0.1 meters
        # Pole angle: -0.1 to +0.1 radians  
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.torch_device) - 0.5)
        
        # Randomized initial velocities
        # Cart velocity: -0.25 to +0.25 m/s
        # Pole angular velocity: -0.25 to +0.25 rad/s
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.torch_device) - 0.5)
        
        # Apply randomized states
        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]
        
        # Update simulation with new states
        from isaacgym import gymtorch
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # Move tensors to CPU for Isaac Gym indexed update
        dof_state_cpu = self.dof_state.cpu()
        env_ids_cpu = env_ids_int32.cpu()
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(dof_state_cpu),
            gymtorch.unwrap_tensor(env_ids_cpu),
            len(env_ids_cpu)
        )
        
        # Reset tracking buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        self.logger.debug(f"Reset {len(env_ids)} environments with randomized states")
    
    def apply_actions(self, actions) -> None:
        """Apply force-based actions to cart (following IsaacGymEnvs pattern)"""
        import torch
        
        # Handle different action formats
        if isinstance(actions, dict):
            # Dictionary format from CartpoleActionNode
            if "forces" in actions and actions["forces"] is not None:
                # Use pre-computed forces from action node
                cart_force = actions["forces"][0]  # Get cart force (DOF 0)
            else:
                # No valid forces, use zero
                cart_force = torch.tensor(0.0, device=self.torch_device)
        elif isinstance(actions, torch.Tensor):
            # Tensor format - convert to force
            cart_force = actions.to(self.torch_device).squeeze() * self.max_push_effort
            if cart_force.dim() > 0:
                cart_force = cart_force[0]  # Take first element if multiple
        else:
            # Unknown format, use zero force
            cart_force = torch.tensor(0.0, device=self.torch_device)
            self.logger.warning(f"Unknown action format: {type(actions)}, using zero force")
        
        # Create force tensor for all DOFs across all environments
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.torch_device, dtype=torch.float)
        
        # Apply the same cart force to all environments using strided indexing
        actions_tensor[::self.num_dof] = cart_force  # Every num_dof-th element (cart DOF)
        
        # Apply forces to simulation
        from isaacgym import gymtorch
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)
    
    def get_observations(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get current observations from physics simulation"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.torch_device)
        
        # CRITICAL: Refresh DOF state tensor before reading
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # Create observation tensor: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        obs_dim = 4
        observations = torch.zeros((len(env_ids), obs_dim), device=self.torch_device, dtype=torch.float32)
        
        # Extract observations (matching IsaacGymEnvs pattern)
        observations[:, 0] = self.dof_pos[env_ids, 0].squeeze()  # Cart position
        observations[:, 1] = self.dof_vel[env_ids, 0].squeeze()  # Cart velocity
        observations[:, 2] = self.dof_pos[env_ids, 1].squeeze()  # Pole angle  
        observations[:, 3] = self.dof_vel[env_ids, 1].squeeze()  # Pole angular velocity
        
        return observations
    
    def compute_rewards(self) -> torch.Tensor:
        """Compute rewards following IsaacGymEnvs Cartpole reward function"""
        # Get current observations
        observations = self.get_observations()
        
        cart_pos = observations[:, 0]
        cart_vel = observations[:, 1]
        pole_angle = observations[:, 2]
        pole_vel = observations[:, 3]
        
        # Reward function (matching IsaacGymEnvs)
        # Reward for staying upright with minimal movement
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        
        # Penalty for failure conditions
        reward = torch.where(torch.abs(cart_pos) > self.reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        
        return reward
    
    def check_termination(self) -> torch.Tensor:
        """Check termination conditions for episode resets"""
        # Get current observations  
        observations = self.get_observations()
        
        cart_pos = observations[:, 0]
        pole_angle = observations[:, 2]
        
        # Termination conditions (matching IsaacGymEnvs)
        reset = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        
        # Cart moved too far
        reset = torch.where(torch.abs(cart_pos) > self.reset_dist, torch.ones_like(reset), reset)
        
        # Pole fell over
        reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset), reset)
        
        # Episode timeout
        reset = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(reset), reset)
        
        return reset.long()  # Convert to long tensor for compatibility