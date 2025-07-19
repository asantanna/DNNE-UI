from typing import Dict, Any
import torch
import numpy as np
# Isaac Gym imports are handled at runtime in the template
from framework import QueueNode, SensorNode

# Template variables - replaced during export

# Import DNNE_print from centralized location
from isaacgymenvs.utils.debug_utils import DNNE_print

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
        import builtins
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        self.verbose = getattr(builtins, 'VERBOSE', False)
    
    async def compute(self, env_handle, actions, trigger) -> Dict[str, Any]:
        """Execute simulation step with dual-mode support"""
        
        if self.verbose:
            from isaacgymenvs.utils.debug_utils import DNNE_print
            DNNE_print("D", "STEP_COMPUTE", "IsaacGymStepNode.compute() called!")
            DNNE_print("D", "STEP_COMPUTE", f"- env_handle: {type(env_handle)}")
            DNNE_print("D", "STEP_COMPUTE", f"- actions: {type(actions)}, shape={actions.shape if hasattr(actions, 'shape') else 'N/A'}")
            DNNE_print("D", "STEP_COMPUTE", f"- trigger: {trigger}")
        
        # Extract environment from handle
        env = env_handle["environment"]
        num_envs = env_handle["num_envs"]
        
        # Handle trigger-based output mode
        # CRITICAL FIX: Only use trigger mode for actual training_complete signals
        # "collecting" signals should run normal mode
        if trigger is not None and isinstance(trigger, dict) and trigger.get('signal_type') != 'collecting':
            # Return cached observations from previous step
            next_observations = self.cached_observations if self.cached_observations is not None else torch.zeros(num_envs, 4)
            
            if self.verbose and self.cached_observations is not None:
                DNNE_print("D", "STEP_TRIGGER", "IsaacGymStepNode TRIGGER MODE - Releasing cached observations")
                DNNE_print("D", "STEP_TRIGGER", f"Cached obs shape: {next_observations.shape}")
            
            return {
                "observations": torch.zeros(num_envs, 4),  # dummy
                "rewards": torch.zeros(num_envs),          # dummy  
                "done": torch.zeros(num_envs, dtype=torch.bool),  # dummy
                "info": {},                                # dummy
                "next_observations": next_observations,    # cached
            }
        
        # Normal execution mode: step environment
        if self.verbose:
            DNNE_print("D", "STEP_NORMAL", f"IsaacGymStepNode step {self.step_count + 1} - NORMAL MODE")
            DNNE_print("D", "STEP_NORMAL", f"Input actions shape: {actions.shape}")
            DNNE_print("D", "STEP_NORMAL", f"Actions: min={actions.min().item():.4f}, max={actions.max().item():.4f}, mean={actions.mean().item():.4f}")
        
        # Use CartpoleDNNE's step_async method
        observations, rewards, done, info = env.step_async(actions)
        
        # Render if viewer is enabled (for visual mode)
        if hasattr(env, 'viewer') and env.viewer is not None:
            env.render()
        
        # Cache for later trigger-based output
        self.cached_observations = observations
        self.cached_rewards = rewards
        self.cached_done = done
        self.cached_info = info
        
        self.step_count += 1
        
        # Verbose logging for outputs
        if self.verbose:
            DNNE_print("D", "STEP_RESULT", f"IsaacGymStepNode - After step {self.step_count}:")
            DNNE_print("D", "STEP_RESULT", f"Observations cached: shape={observations.shape}")
            DNNE_print("D", "STEP_RESULT", f"Rewards: min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")
            DNNE_print("D", "STEP_RESULT", f"Done count: {done.sum().item()}")
        
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
