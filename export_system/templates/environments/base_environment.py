"""
Base Isaac Gym Environment Class

Provides abstract interface for Isaac Gym environments with common functionality.
All environment-specific implementations should inherit from this base class.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging


class IsaacGymEnvironment(ABC):
    """Base class for all Isaac Gym environment implementations"""
    
    def __init__(self, gym, sim, sim_params, num_envs: int, device: str, logger: logging.Logger):
        """
        Initialize base environment
        
        Args:
            gym: Isaac Gym instance
            sim: Isaac Gym simulation handle
            sim_params: Simulation parameters
            num_envs: Number of parallel environments
            device: Device for tensor operations ("cuda" or "cpu")
            logger: Logger instance
        """
        self.gym = gym
        self.sim = sim
        self.sim_params = sim_params
        self.num_envs = num_envs
        self.device = device
        self.logger = logger
        
        # Environment containers
        self.envs = []
        self.actors = []
        
        # State tracking
        self.step_count = 0
        self.episode_count = 0
        self.progress_buf = None
        self.reset_buf = None
        
        # DOF state tensors (to be initialized by subclasses)
        self.dof_state = None
        self.dof_pos = None
        self.dof_vel = None
        self.num_dof = 0
        
        # Device for torch tensors
        self.torch_device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def get_environment_name(self) -> str:
        """Return the name of this environment type"""
        pass
    
    @abstractmethod
    def get_simulation_params(self) -> Dict[str, float]:
        """Return environment-specific simulation parameters (dt, substeps)"""
        pass
    
    @abstractmethod
    def get_action_space_info(self) -> Dict[str, Any]:
        """Return action space information (dimension, limits, scaling)"""
        pass
    
    @abstractmethod
    def get_observation_space_info(self) -> Dict[str, Any]:
        """Return observation space information (dimension, limits)"""
        pass
    
    @abstractmethod
    def create_environments(self, spacing: float, num_per_row: int) -> None:
        """
        Create all environment instances with assets and actors
        
        Args:
            spacing: Distance between environments
            num_per_row: Number of environments per row in grid layout
        """
        pass
    
    @abstractmethod
    def reset_environments(self, env_ids: torch.Tensor) -> None:
        """
        Reset specified environments to random initial states
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        pass
    
    @abstractmethod
    def apply_actions(self, actions: torch.Tensor) -> None:
        """
        Apply actions to the simulation
        
        Args:
            actions: Action tensor with shape (num_envs, action_dim)
        """
        pass
    
    @abstractmethod
    def get_observations(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get current observations from simulation
        
        Args:
            env_ids: Optional tensor of environment indices. If None, get all.
            
        Returns:
            Observation tensor with shape (num_envs, obs_dim) or (len(env_ids), obs_dim)
        """
        pass
    
    @abstractmethod
    def compute_rewards(self) -> torch.Tensor:
        """
        Compute rewards for all environments
        
        Returns:
            Reward tensor with shape (num_envs,)
        """
        pass
    
    @abstractmethod
    def check_termination(self) -> torch.Tensor:
        """
        Check termination conditions for all environments
        
        Returns:
            Boolean tensor with shape (num_envs,) indicating which environments should reset
        """
        pass
    
    def initialize_state_tensors(self) -> None:
        """Initialize DOF state tensors (call after create_environments)"""
        if self.num_dof == 0:
            raise RuntimeError("num_dof must be set before initializing state tensors")
            
        # Get DOF state tensor from Isaac Gym
        from isaacgym import gymtorch
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        # Create views for positions and velocities
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        
        # Initialize tracking buffers
        self.progress_buf = torch.zeros(self.num_envs, device=self.torch_device, dtype=torch.long)
        self.reset_buf = torch.zeros(self.num_envs, device=self.torch_device, dtype=torch.long)
        
        self.logger.info(f"Initialized state tensors: {self.num_envs} envs, {self.num_dof} DOF")
    
    def step_simulation(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute one complete simulation step
        
        Args:
            actions: Action tensor with shape (num_envs, action_dim)
            
        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Apply actions before physics step
        self.apply_actions(actions)
        
        # Step physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Update progress
        self.progress_buf += 1
        self.step_count += 1
        
        # Check for resets and handle them
        self.reset_buf = self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_environments(env_ids)
        
        # Get current state
        observations = self.get_observations()
        rewards = self.compute_rewards()
        done = self.reset_buf.clone()
        
        # Create info dictionary
        info = {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "num_resets": len(env_ids),
            "sim_time": self.gym.get_sim_time(self.sim)
        }
        
        return observations, rewards, done, info
    
    def update_viewer(self, viewer) -> None:
        """Update viewer if available"""
        if viewer is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
    
    def get_environment_bounds(self, spacing: float) -> Tuple[Any, Any]:
        """Get environment bounds for creation"""
        from isaacgym import gymapi
        
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        return env_lower, env_upper
    
    def set_simulation_parameters(self) -> None:
        """Set environment-specific simulation parameters"""
        params = self.get_simulation_params()
        
        if "dt" in params:
            self.sim_params.dt = params["dt"]
            
        if "substeps" in params:
            self.sim_params.substeps = params["substeps"]
            
        self.logger.info(f"{self.get_environment_name()} simulation: dt={self.sim_params.dt}, substeps={self.sim_params.substeps}")