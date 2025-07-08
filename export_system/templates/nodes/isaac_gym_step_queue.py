# Template variables - replaced during export
template_vars = {
    "NODE_ID": "isaac_step_1",
    "CLASS_NAME": "IsaacGymStepNode"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Isaac Gym step node for advancing simulation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["sim_handle", "actions", "trigger"])
        self.setup_outputs(["observations", "rewards", "done", "info", "next_observations"])
        
        # Step tracking
        self.step_count = 0
        
        # State caching for RL training loop
        self.cached_observations = None
        self.cached_rewards = None
        self.cached_done = None
        self.cached_info = None
        
    async def compute(self, sim_handle, actions, trigger) -> Dict[str, Any]:
        """Execute one simulation step with RL interface and state caching"""
        try:
            # Import Isaac Gym at runtime
            import isaacgym
            from isaacgym import gymapi, gymutil, gymtorch
            import torch
            
            gym = gymapi.acquire_gym()
            
            # Validate simulation handle
            if not hasattr(sim_handle, 'sim') or sim_handle.sim is None:
                raise RuntimeError("Invalid simulation handle")
            
            num_envs = len(sim_handle.envs) if hasattr(sim_handle, 'envs') else 1
            
            # Handle trigger-based next_observations output
            if trigger is not None:
                # Output cached observations when triggered by TrainingStep
                next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(num_envs, 21)
                self.logger.info(f"Trigger received, outputting cached observations: shape={{next_observations.shape}}")
                return {{
                    "observations": torch.zeros(num_envs, 21),  # observations (dummy)
                    "rewards": torch.zeros(num_envs),           # rewards (dummy)  
                    "done": torch.zeros(num_envs, dtype=torch.bool),  # done (dummy)
                    "info": {{}},                               # info (dummy)
                    "next_observations": next_observations      # next_observations (cached)
                }}
            
            # Normal execution: apply actions and step simulation
            
            # Apply actions to simulation
            if actions is not None:
                self._apply_actions(gym, sim_handle, actions)
            
            # Step physics
            gym.simulate(sim_handle.sim)
            gym.fetch_results(sim_handle.sim, True)
            
            # Update viewer if available
            if hasattr(sim_handle, 'viewer') and sim_handle.viewer is not None:
                gym.step_graphics(sim_handle.sim)
                gym.draw_viewer(sim_handle.viewer, sim_handle.sim, True)
                gym.sync_frame_time(sim_handle.sim)
            
            # Get RL outputs in standard format
            observations = self._get_observations(gym, sim_handle)
            rewards = self._compute_rewards(gym, sim_handle)
            done = self._check_done(gym, sim_handle)
            info = self._get_info(gym, sim_handle)
            
            # Cache observations for later trigger-based output
            self.cached_observations = observations
            self.cached_rewards = rewards
            self.cached_done = done
            self.cached_info = info
            
            # Update step counter
            self.step_count += 1
            
            # Log progress periodically
            if self.step_count % 1000 == 0:
                avg_reward = torch.mean(rewards).item() if len(rewards) > 0 else 0.0
                self.logger.info(f"Step {{self.step_count}}: avg reward = {{avg_reward:.4f}}")
            
            return {{
                "observations": observations,              # Current step observations
                "rewards": rewards,                       # Current step rewards
                "done": done,                            # Current step done flags
                "info": info,                            # Current step info
                "next_observations": torch.zeros(num_envs, 21)  # next_observations (empty until triggered)
            }}
            
        except ImportError:
            self.logger.error("Isaac Gym not available")
            raise RuntimeError("Isaac Gym is not installed or not available")
        except Exception as e:
            self.logger.error(f"Error in simulation step: {{e}}")
            raise
    
    def _apply_actions(self, gym, sim_handle, actions):
        """Apply actions to simulation"""
        # This is a simplified implementation
        # In a real implementation, you would:
        # 1. Parse the actions based on the environment type
        # 2. Apply joint commands, forces, or torques
        # 3. Set DOF states or apply forces to actors
        
        try:
            # Example: Apply joint commands if actions contain joint data
            if hasattr(actions, 'joint_commands') and actions.joint_commands is not None:
                # Apply joint commands to all environments
                for i, env in enumerate(sim_handle.envs):
                    if i < len(sim_handle.actors):
                        # In practice, you would set DOF targets here
                        # gym.set_dof_target_position(env, sim_handle.actors[i], actions.joint_commands[i])
                        pass
            
            # Example: Apply forces if actions contain force data
            elif hasattr(actions, 'forces') and actions.forces is not None:
                # Apply forces to actors
                for i, env in enumerate(sim_handle.envs):
                    if i < len(sim_handle.actors):
                        # gym.apply_rigid_body_force_tensors(sim_handle.sim, actions.forces[i], None, gymapi.ENV_SPACE)
                        pass
            
        except Exception as e:
            self.logger.warning(f"Failed to apply actions: {{e}}")
    
    def _get_observations(self, gym, sim_handle):
        """Get observations tensor in standard RL format"""
        import torch
        
        try:
            num_envs = len(sim_handle.envs) if hasattr(sim_handle, 'envs') else 1
            
            # In a real implementation, you would:
            # 1. Get DOF states from the simulation
            # 2. Get rigid body states
            # 3. Flatten into observation vector
            
            # Create observation tensor matching the observation space
            obs_dim = 21  # 8 joint pos + 8 joint vel + 4 quaternion + 1 upright indicator
            device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            observations = torch.zeros(num_envs, obs_dim, device=device_torch, dtype=torch.float32)
            
            # Example observation construction (simplified)
            # Joint positions (8 DOF) - placeholder
            observations[:, 0:8] = 0.0
            
            # Joint velocities (8 DOF) - placeholder  
            observations[:, 8:16] = 0.0
            
            # Base orientation quaternion (4 values) - identity quaternion
            observations[:, 16] = 1.0  # w
            observations[:, 17:20] = 0.0  # x, y, z
            
            # Upright indicator (1 value) - placeholder
            observations[:, 20] = 1.0  # Assume upright
            
            return observations
            
        except Exception as e:
            self.logger.error(f"Error getting observations: {{e}}")
            # Return zeros if construction fails
            num_envs = 1
            obs_dim = 21
            device_torch = torch.device("cpu")
            return torch.zeros(num_envs, obs_dim, device=device_torch, dtype=torch.float32)
    
    def _compute_rewards(self, gym, sim_handle):
        """Compute rewards for each environment"""
        import torch
        
        try:
            num_envs = len(sim_handle.envs) if hasattr(sim_handle, 'envs') else 1
            
            # In a real implementation, reward computation would be environment-specific
            # For example:
            # - Cartpole: reward for keeping pole upright
            # - Ant: reward for forward movement
            # - Humanoid: reward for maintaining balance and forward progress
            
            # Simplified implementation - basic survival reward
            device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rewards = torch.ones(num_envs, dtype=torch.float32, device=device_torch)
            
            # Example: Add rewards based on observations
            # observations = self._get_observations(gym, sim_handle)
            # upright_bonus = observations[:, 20] * 0.1  # Bonus for being upright
            # rewards += upright_bonus
            
            return rewards
            
        except Exception as e:
            self.logger.error(f"Error computing rewards: {{e}}")
            device_torch = torch.device("cpu")
            return torch.zeros(1, dtype=torch.float32, device=device_torch)
    
    def _check_done(self, gym, sim_handle):
        """Check if episodes are done"""
        import torch
        
        try:
            num_envs = len(sim_handle.envs) if hasattr(sim_handle, 'envs') else 1
            
            # In a real implementation, you would check environment-specific termination conditions
            # For example:
            # - Cartpole: pole angle too large
            # - Ant: robot fell over
            # - Humanoid: robot fell down
            # - Timeout: max episode length reached
            
            device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            done = torch.zeros(num_envs, dtype=torch.bool, device=device_torch)
            
            # Example termination checks
            # observations = self._get_observations(gym, sim_handle)
            # upright = observations[:, 20]
            # done = upright < 0.1  # Episode ends if not upright
            
            # Check for timeout
            if self.step_count > 1000:  # Max episode length
                done[:] = True
                
            return done
            
        except Exception as e:
            self.logger.error(f"Error checking done: {{e}}")
            device_torch = torch.device("cpu")
            return torch.zeros(1, dtype=torch.bool, device=device_torch)
    
    def _get_info(self, gym, sim_handle):
        """Get additional information"""
        try:
            return {{
                "step_count": self.step_count,
                "sim_time": gym.get_sim_time(sim_handle.sim),
                "num_envs": len(sim_handle.envs),
                "physics_dt": sim_handle.dt if hasattr(sim_handle, 'dt') else 0.0167
            }}
            
        except Exception as e:
            self.logger.error(f"Error getting info: {{e}}")
            return {{
                "step_count": self.step_count,
                "sim_time": 0.0,
                "num_envs": 0,
                "physics_dt": 0.0167
            }}