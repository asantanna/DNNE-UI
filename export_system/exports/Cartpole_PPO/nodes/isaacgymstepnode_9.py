"""Node implementation for IsaacGymStepNode (ID: 9)"""
from typing import Dict, Any
import torch
import numpy as np
# Isaac Gym imports are handled at runtime in the template
from framework.base import QueueNode, SensorNode

# Template variables - replaced during export

class IsaacGymStepNode_9(QueueNode):
    """Isaac Gym step node with dual-mode execution for RL synchronization"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        # Setup all inputs including optional ones
        self.setup_inputs(required=["env_handle", "actions", "trigger"])
        self.setup_outputs(["observations", "rewards", "done", "info", "next_observations"])
        
        # State caching for RL synchronization
        self.cached_observations = None
        self.cached_rewards = None
        self.cached_done = None
        self.cached_info = None
        self.step_count = 0
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
    
    async def compute(self, env_handle, actions, trigger) -> Dict[str, Any]:
        """Execute simulation step with dual-mode support"""
        
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] IsaacGymStepNode.compute() called!")
            print(f"[PPO_CYCLE_DEBUG] - env_handle: {type(env_handle)}")
            print(f"[PPO_CYCLE_DEBUG] - actions: {type(actions)}, shape={actions.shape if hasattr(actions, 'shape') else 'N/A'}")
            print(f"[PPO_CYCLE_DEBUG] - trigger: {trigger}")
        
        # Extract environment from handle
        env = env_handle["environment"]
        num_envs = env_handle["num_envs"]
        
        # Handle trigger-based output mode
        # CRITICAL FIX: Only use trigger mode for actual training_complete signals
        # "collecting" signals should run normal mode
        if trigger is not None and isinstance(trigger, dict) and trigger.get('signal_type') != 'collecting':
            # Return cached observations from previous step
            next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(num_envs, 4)
            
            if self.ppo_cycle_debug and self.cached_observations is not None:
                print(f"[PPO_CYCLE_DEBUG] IsaacGymStepNode TRIGGER MODE - Releasing cached observations")
                print(f"[PPO_CYCLE_DEBUG] Cached obs shape: {next_observations.shape}")
            
            return {
                "observations": torch.zeros(num_envs, 4),  # dummy
                "rewards": torch.zeros(num_envs),          # dummy  
                "done": torch.zeros(num_envs, dtype=torch.bool),  # dummy
                "info": {},                                # dummy
                "next_observations": next_observations,    # cached
            }
        
        # Normal execution mode: step environment
        if self.ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] IsaacGymStepNode step {self.step_count + 1} - NORMAL MODE")
            print(f"[PPO_CYCLE_DEBUG] Input actions shape: {actions.shape}")
            print(f"[PPO_CYCLE_DEBUG] Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
        
        # Use CartpoleDNNE's step_async method
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
            self.logger.info(f"Step {self.step_count}: "
                           f"obs_shape={observations.shape}, "
                           f"reward_mean={rewards.mean().item():.3f}, "
                           f"done_count={done.sum().item()}")
        
        return {
            "observations": observations,              # Current observations
            "rewards": rewards,                       # Current rewards
            "done": done,                            # Current done flags
            "info": info,                            # Current info
            "next_observations": torch.zeros(num_envs, 4), # empty until triggered
        }
