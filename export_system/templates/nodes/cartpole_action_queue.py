# Template variables - replaced during export
template_vars = {
    "NODE_ID": "cartpole_action_1",
    "CLASS_NAME": "CartpoleActionNode",
    "MAX_PUSH_EFFORT": 10
}

# Import DNNE_print from centralized location
from isaacgymenvs.utils.debug_utils import DNNE_print

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Cartpole Action Node - Convert network output to Isaac Gym ACTION format"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["policy"])
        self.setup_outputs(["action"])
        
        # Configuration
        self.max_push_effort = {MAX_PUSH_EFFORT}
        
        self.logger.info(f"CartpoleActionNode {{node_id}} initialized with max_push_effort={{self.max_push_effort}}")
        
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
                from isaacgymenvs.utils.debug_utils import DNNE_print
                DNNE_print("D", "PPO_ACTION", "CartpoleActionNode.compute() called!")
                DNNE_print("D", "PPO_ACTION", f"Input action_tensor shape: {{action_tensor.shape}}")
            
            # CRITICAL FIX: Return action tensor directly for Isaac Gym
            # The VecTask expects a simple tensor of shape [num_envs, num_actions]
            # NOT a dictionary with forces/torques
            
            if verbose:
                DNNE_print("D", "PPO_ACTION", f"Returning action_tensor directly: shape={{action_tensor.shape}}")
                DNNE_print("D", "PPO_ACTION", f"Action values: min={{action_tensor.min().item():.4f}}, max={{action_tensor.max().item():.4f}}")
            
            return {{
                "action": action_tensor  # Return raw action tensor, scaling happens in pre_physics_step
            }}
            
        except Exception as e:
            self.logger.error(f"Error in CartpoleActionNode {{self.node_id}}: {{e}}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Re-raise the exception to trigger immediate exit
            raise
