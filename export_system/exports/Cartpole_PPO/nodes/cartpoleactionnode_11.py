from typing import Dict, Any
import torch
from typing import Dict, Any, Optional
from framework import QueueNode, SensorNode

# Template variables - replaced during export

def DNNE_print(message):
    """Print with [DNNE_DEBUG] prefix for easy grep filtering"""
    print(f"[DNNE_DEBUG] {message}")

class CartpoleActionNode_11(QueueNode):
    """Cartpole Action Node - Convert network output to Isaac Gym ACTION format"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["policy"])
        self.setup_outputs(["action"])
        
        # Configuration
        self.max_push_effort = 10
        
        self.logger.info(f"CartpoleActionNode {node_id} initialized with max_push_effort={self.max_push_effort}")
        
    async def compute(self, policy) -> Dict[str, Any]:
        """
        Convert PPO policy output to Isaac Gym ACTION format for Cartpole
        
        Args:
            policy: PolicyOutput dictionary containing action tensor
            
        Returns:
            action: ACTION object with forces for Isaac Gym
        """
        
        import torch
        import os
        import builtins
        
        ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        verbose = getattr(builtins, 'VERBOSE', False)
        
        try:
            # Extract action tensor from PolicyOutput dictionary
            action_tensor = policy["action"]
            
            if verbose:
                print(f"CartpoleActionNode.compute() called!")
                print(f"Input action_tensor shape: {action_tensor.shape}")
            
            # CRITICAL FIX: Return action tensor directly for Isaac Gym
            # The VecTask expects a simple tensor of shape [num_envs, num_actions]
            # NOT a dictionary with forces/torques
            
            if verbose:
                print(f"Returning action_tensor directly: shape={action_tensor.shape}")
                print(f"Action values: min={action_tensor.min().item():.4f}, max={action_tensor.max().item():.4f}")
            
            return {
                "action": action_tensor  # Return raw action tensor, scaling happens in pre_physics_step
            }
            
        except Exception as e:
            self.logger.error(f"Error in CartpoleActionNode {self.node_id}: {e}")
            # Return safe default - zeros for all environments
            import torch
            # Assume 512 environments if we can't determine from policy
            num_envs = 512
            default_action = torch.zeros(num_envs, 1, dtype=torch.float32)
            return {
                "action": default_action
            }
