# Template variables - replaced during export
template_vars = {
    "NODE_ID": "cartpole_action_1",
    "CLASS_NAME": "CartpoleActionNode",
    "MAX_PUSH_EFFORT": 10.0
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Cartpole Action Node - Convert network output to Isaac Gym ACTION format"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["policy"])
        self.setup_outputs(["action"])
        
        # Configuration
        self.max_push_effort = {MAX_PUSH_EFFORT}
        
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
        
        try:
            # Extract action tensor from PolicyOutput dictionary
            action_tensor = policy["action"]
            
            # Ensure action_tensor is properly shaped
            if action_tensor.dim() > 1:
                action_tensor = action_tensor.squeeze()
            
            if action_tensor.dim() == 0:
                action_tensor = action_tensor.unsqueeze(0)
                
            # Scale by max effort (same as IsaacGym Cartpole implementation)
            scaled_force = action_tensor[0] * self.max_push_effort
            
            # For Cartpole: 2 DOF (cart translation, pole rotation)
            # Only cart (DOF 0) is actuated, pole (DOF 1) is passive
            forces = torch.zeros(2, dtype=torch.float32, device=action_tensor.device)
            forces[0] = scaled_force  # Apply force to cart only
            
            # Create ACTION object (simplified for export)
            action = {
                "forces": forces,
                "joint_commands": None,  # Not used for Cartpole
                "torques": None          # Not used for Cartpole
            }
            
            self.logger.debug(f"Generated action force: {scaled_force.item():.3f}")
            
            return {
                "action": action
            }
            
        except Exception as e:
            self.logger.error(f"Error in CartpoleActionNode {self.node_id}: {e}")
            # Return safe default
            default_action = {
                "forces": torch.zeros(2, dtype=torch.float32),
                "joint_commands": None,
                "torques": None
            }
            return {
                "action": default_action
            }