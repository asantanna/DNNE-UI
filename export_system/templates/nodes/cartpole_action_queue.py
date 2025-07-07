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
        self.setup_inputs(required=["network_output"])
        self.setup_outputs(["action", "context"])
        
        # Configuration
        self.max_push_effort = {MAX_PUSH_EFFORT}
        
        self.logger.info(f"CartpoleActionNode {node_id} initialized with max_push_effort={self.max_push_effort}")
        
    async def compute(self, network_output, context=None) -> Dict[str, Any]:
        """
        Convert neural network output to Isaac Gym ACTION format for Cartpole
        
        Args:
            network_output: Raw network output tensor [1] or [1, 1] (single force value)
            context: Optional context dictionary
            
        Returns:
            action: ACTION object with forces for Isaac Gym
            context: Updated context
        """
        
        import torch
        
        try:
            # Ensure network_output is properly shaped
            if network_output.dim() > 1:
                network_output = network_output.squeeze()
            
            if network_output.dim() == 0:
                network_output = network_output.unsqueeze(0)
                
            # Scale by max effort (same as IsaacGym Cartpole implementation)
            scaled_force = network_output[0] * self.max_push_effort
            
            # For Cartpole: 2 DOF (cart translation, pole rotation)
            # Only cart (DOF 0) is actuated, pole (DOF 1) is passive
            forces = torch.zeros(2, dtype=torch.float32, device=network_output.device)
            forces[0] = scaled_force  # Apply force to cart only
            
            # Create ACTION object (simplified for export)
            action = {
                "forces": forces,
                "joint_commands": None,  # Not used for Cartpole
                "torques": None          # Not used for Cartpole
            }
            
            # Update context
            if context is None:
                context = {}
            context["last_action_force"] = scaled_force.item()
            context["max_push_effort"] = self.max_push_effort
            
            self.logger.debug(f"Generated action force: {scaled_force.item():.3f}")
            
            return {
                "action": action,
                "context": context
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
                "action": default_action,
                "context": context if context is not None else {}
            }