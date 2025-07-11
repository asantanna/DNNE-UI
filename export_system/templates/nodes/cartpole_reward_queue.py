# Template variables - replaced during export
template_vars = {
    "NODE_ID": "cartpole_reward_1",
    "CLASS_NAME": "CartpoleRewardNode",
    "RESET_DIST": 2.0,
    "INVERT_FOR_LOSS": True
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """Cartpole Reward Node - Compute Cartpole-specific rewards matching IsaacGym implementation"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["observations"])
        self.setup_outputs(["reward_or_loss", "done", "info_dict"])
        
        # Configuration
        self.reset_dist = {RESET_DIST}
        self.invert_for_loss = {INVERT_FOR_LOSS}
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_length = 500  # Standard Cartpole episode length
        
        self.logger.info(f"CartpoleRewardNode {node_id} initialized with reset_dist={self.reset_dist}, invert_for_loss={self.invert_for_loss}")
        
    async def compute(self, observations) -> Dict[str, Any]:
        """
        Compute Cartpole reward matching IsaacGym implementation
        
        Args:
            observations: Tensor [1, 4] containing [cart_pos, cart_vel, pole_angle, pole_vel]
            
        Returns:
            reward_or_loss: Reward (or negative reward if invert_for_loss=True)
            done: Episode termination flag
            info_dict: Additional information as tensor
        """
        
        import torch
        import numpy as np
        
        try:
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
            if torch.abs(cart_pos) > self.reset_dist:
                reward = torch.tensor(-2.0, dtype=torch.float32, device=observations.device)
                done = True
                self.logger.debug(f"Episode ended: cart position {cart_pos.item():.3f} > {self.reset_dist}")
                
            # Pole angle too large (fell over)
            if torch.abs(pole_angle) > (np.pi / 2):
                reward = torch.tensor(-2.0, dtype=torch.float32, device=observations.device)
                done = True
                self.logger.debug(f"Episode ended: pole angle {pole_angle.item():.3f} > {np.pi/2:.3f}")
                
            # Episode length exceeded
            self.episode_steps += 1
            if self.episode_steps >= self.max_episode_length:
                done = True
                self.episode_steps = 0  # Reset for next episode
                self.logger.debug(f"Episode ended: max length {self.max_episode_length} reached")
                
            # Reset episode counter if done
            if done and self.episode_steps < self.max_episode_length:
                self.episode_steps = 0
                
            # Convert to loss if requested
            if self.invert_for_loss:
                output = -reward
            else:
                output = reward
                
            # Create done tensor
            done_tensor = torch.tensor([done], dtype=torch.bool, device=observations.device)
            
            # Create info tensor (episode steps)
            info_tensor = torch.tensor([self.episode_steps], dtype=torch.float32, device=observations.device)
            
            self.logger.debug(f"Reward: {reward.item():.3f}, Output: {output.item():.3f}, Done: {done}, Steps: {self.episode_steps}")
            
            return {
                "reward_or_loss": output,
                "done": done_tensor,
                "info_dict": info_tensor
            }
            
        except Exception as e:
            self.logger.error(f"Error in CartpoleRewardNode {self.node_id}: {e}")
            
            # Return safe defaults
            safe_output = torch.tensor(-1.0 if self.invert_for_loss else 1.0, 
                                     dtype=torch.float32, device=observations.device)
            safe_done = torch.tensor([False], dtype=torch.bool, device=observations.device)
            safe_info = torch.tensor([self.episode_steps], dtype=torch.float32, device=observations.device)
            
            return {
                "reward_or_loss": safe_output,
                "done": safe_done,
                "info_dict": safe_info
            }