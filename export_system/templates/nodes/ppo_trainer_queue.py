# Template variables - replaced during export
template_vars = {
    "NODE_ID": "ppo_trainer_1",
    "CLASS_NAME": "PPOTrainerNode",
    "HORIZON_LENGTH": 16,
    "NUM_EPOCHS": 4,
    "MINIBATCH_SIZE": 32,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_PARAM": 0.2,
    "VALUE_COEF": 0.5,
    "ENTROPY_COEF": 0.01,
    "LEARNING_RATE": 3e-4,
    "MAX_GRAD_NORM": 0.5,
    "CHECKPOINT_ENABLED": False,
    "CHECKPOINT_TRIGGER_TYPE": "epoch",
    "CHECKPOINT_TRIGGER_VALUE": "10"
}

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """PPO Trainer Node - Complete PPO Training Algorithm"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["state", "policy_output", "reward", "done", "model"])
        self.setup_outputs(["loss", "training_complete"])
        
        # Configuration from template
        self.horizon_length = {HORIZON_LENGTH}
        self.num_epochs = {NUM_EPOCHS}
        self.minibatch_size = {MINIBATCH_SIZE}
        self.gamma = {GAMMA}
        self.gae_lambda = {GAE_LAMBDA}
        self.clip_param = {CLIP_PARAM}
        self.value_coef = {VALUE_COEF}
        self.entropy_coef = {ENTROPY_COEF}
        self.learning_rate = {LEARNING_RATE}
        self.max_grad_norm = {MAX_GRAD_NORM}
        
        # Training state
        self.reset_buffer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = None
        self.step_count = 0
        
        # Checkpoint configuration
        self.checkpoint_enabled = {CHECKPOINT_ENABLED}
        self.checkpoint_trigger_type = "{CHECKPOINT_TRIGGER_TYPE}"
        self.checkpoint_trigger_value = "{CHECKPOINT_TRIGGER_VALUE}"
        self.checkpoint_manager = None
        self.last_loss = None
        
        # Initialize checkpoint manager if enabled
        if self.checkpoint_enabled:
            from run_utils import CheckpointManager, validate_checkpoint_config
            
            # Validate checkpoint configuration
            checkpoint_config = {{
                'enabled': self.checkpoint_enabled,
                'trigger_type': self.checkpoint_trigger_type,
                'trigger_value': self.checkpoint_trigger_value
            }}
            
            try:
                validate_checkpoint_config(checkpoint_config)
                # Get checkpoint directory from command line args (set by runner.py)
                try:
                    import builtins
                    save_checkpoint_dir = getattr(builtins, 'SAVE_CHECKPOINT_DIR', None)
                except:
                    save_checkpoint_dir = None
                    
                self.checkpoint_manager = CheckpointManager(
                    node_id=node_id,
                    checkpoint_dir=save_checkpoint_dir
                )
                self.logger.info(f"Checkpoint manager initialized: {self.checkpoint_trigger_type} trigger")
            except ValueError as e:
                self.logger.error(f"Checkpoint configuration error: {e}")
                self.checkpoint_enabled = False
        
        self.logger.info(f"PPOTrainerNode {node_id} initialized with horizon={self.horizon_length}, epochs={self.num_epochs}")
        
    def reset_buffer(self):
        """Reset the trajectory buffer"""
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_values = []
        self.buffer_log_probs = []
        self.buffer_dones = []
        self.buffer_full = False
        
    def compute_gae_advantages(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) advantages
        
        Args:
            rewards: [horizon_length] reward tensor
            values: [horizon_length] value estimates  
            dones: [horizon_length] episode termination flags
            
        Returns:
            advantages: [horizon_length] GAE advantages
            returns: [horizon_length] discounted returns
        """
        horizon_length = len(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute advantages using GAE recursion
        lastgaelam = 0
        for t in reversed(range(horizon_length)):
            if t == horizon_length - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = 0  # Assume episode ends
            else:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = values[t + 1]
                
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
        
    def ppo_update(self, states, actions, old_log_probs, advantages, returns, model):
        """
        Perform PPO update on collected batch
        
        Args:
            states, actions, old_log_probs, advantages, returns: Trajectory data
            model: PyTorch model to update
            
        Returns:
            average_loss: Average loss over all updates
        """
        
        import torch.optim as optim
        import torch.nn as nn
        import torch.distributions as dist
        import numpy as np
        
        # Setup optimizer if needed
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
        model.train()
        total_losses = []
        batch_size = len(states)
        
        # Multiple epochs over the data
        for epoch in range(self.num_epochs):
            # Create minibatches
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = min(start + self.minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass through model
                features = model['shared'](mb_states)
                
                # Get current values
                current_values = model['value'](features).squeeze(-1)
                
                # Get current policy
                if 'policy_log_std' in model:
                    # Continuous action space
                    action_mean = model['policy_mean'](features)
                    action_std = torch.exp(model['policy_log_std']['log_std'])
                    
                    policy_dist = dist.Normal(action_mean, action_std)
                    current_log_probs = policy_dist.log_prob(mb_actions).sum(dim=-1)
                    entropy = policy_dist.entropy().sum(dim=-1)
                    
                else:
                    # Discrete action space  
                    action_logits = model['policy_mean'](features)
                    policy_dist = dist.Categorical(logits=action_logits)
                    current_log_probs = policy_dist.log_prob(mb_actions.squeeze(-1))
                    entropy = policy_dist.entropy()
                
                # Compute PPO loss
                ratio = torch.exp(current_log_probs - mb_old_log_probs)
                
                # Actor loss (PPO clipped objective)
                surr1 = mb_advantages * ratio
                surr2 = mb_advantages * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (MSE)
                critic_loss = nn.MSELoss()(current_values, mb_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                total_losses.append(total_loss.item())
                
        return torch.tensor(np.mean(total_losses), device=self.device)
        
    async def compute(self, state, policy_output, reward, done, model) -> Dict[str, Any]:
        """
        PPO training step - collect trajectory and train when buffer is full
        
        Args:
            state: Current state tensor
            policy_output: PolicyOutput dictionary from PPOAgent
            reward: Current reward tensor
            done: Episode termination flag
            model: PyTorch model for training
            
        Returns:
            loss: Training loss (0 if still collecting)
            training_complete: Sync signal when training is done
        """
        
        import torch
        import torch.nn as nn
        import torch.distributions as dist
        import numpy as np
        import os
        
        try:
            # Ensure tensors are on correct device
            state = state.to(self.device)
            reward = reward.to(self.device) 
            done = done.to(self.device)
            
            # Extract values from policy_output dictionary
            action = policy_output["action"].to(self.device)
            value = policy_output["value"].to(self.device)
            log_prob = policy_output["log_prob"].to(self.device)
            
            # Add to buffer
            self.buffer_states.append(state.clone())
            self.buffer_actions.append(action.clone())
            self.buffer_rewards.append(reward.clone())
            self.buffer_values.append(value.clone())
            self.buffer_log_probs.append(log_prob.clone())
            self.buffer_dones.append(done.clone())
            
            # Check if buffer is full
            if len(self.buffer_states) >= self.horizon_length:
                # Convert buffer to tensors
                states = torch.stack(self.buffer_states)
                actions = torch.stack(self.buffer_actions)
                rewards = torch.stack(self.buffer_rewards)
                values = torch.stack(self.buffer_values)
                log_probs = torch.stack(self.buffer_log_probs)
                dones = torch.stack(self.buffer_dones)
                
                # Compute advantages and returns using GAE
                advantages, returns = self.compute_gae_advantages(rewards, values, dones)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Perform PPO training
                total_loss = self.ppo_update(states, actions, log_probs, advantages, returns, model)
                
                # Update step count
                self.step_count += 1
                
                # Handle checkpointing
                if self.checkpoint_enabled and self.checkpoint_manager:
                    current_loss = total_loss.item()
                    self.last_loss = current_loss
                    
                    # Check if we should checkpoint
                    should_checkpoint = False
                    if self.checkpoint_trigger_type == "epoch":
                        should_checkpoint = self.checkpoint_manager.should_checkpoint(
                            "epoch", self.checkpoint_trigger_value, current_epoch=self.step_count
                        )
                    elif self.checkpoint_trigger_type == "time":
                        should_checkpoint = self.checkpoint_manager.should_checkpoint(
                            "time", self.checkpoint_trigger_value
                        )
                    elif self.checkpoint_trigger_type == "best_metric":
                        # Use loss as metric (lower is better)
                        should_checkpoint = self.checkpoint_manager.should_checkpoint(
                            "best_metric", "min", current_metric=current_loss
                        )
                    
                    if should_checkpoint:
                        # Prepare metadata with training state and hyperparameters
                        metadata = {{
                            'trigger_type': self.checkpoint_trigger_type,
                            'trigger_value': self.checkpoint_trigger_value,
                            'training_step': self.step_count,
                            'loss': current_loss,
                            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                            'hyperparameters': {{
                                'horizon_length': self.horizon_length,
                                'num_epochs': self.num_epochs,
                                'minibatch_size': self.minibatch_size,
                                'gamma': self.gamma,
                                'gae_lambda': self.gae_lambda,
                                'clip_param': self.clip_param,
                                'value_coef': self.value_coef,
                                'entropy_coef': self.entropy_coef,
                                'learning_rate': self.learning_rate,
                                'max_grad_norm': self.max_grad_norm
                            }}
                        }}
                        
                        # Save checkpoint (only model weights + metadata)
                        self.checkpoint_manager.save_checkpoint(
                            model.state_dict(), metadata=metadata
                        )
                
                # Reset buffer
                self.reset_buffer()
                
                # Create completion signal
                training_complete = {{
                    "signal_type": "training_complete",
                    "step": self.step_count,
                    "loss": total_loss.item(),
                    "source_node": f"ppo_trainer_{self.node_id}"
                }}
                
                self.logger.info(f"PPO training step {self.step_count} complete, loss: {total_loss.item():.4f}")
                
                return {{
                    "loss": total_loss,
                    "training_complete": training_complete
                }}
            
            else:
                # Still collecting, return dummy outputs
                dummy_loss = torch.tensor(0.0, device=self.device)
                dummy_signal = {
                    "signal_type": "collecting", 
                    "buffer_size": len(self.buffer_states),
                    "horizon_length": self.horizon_length,
                    "source_node": f"ppo_trainer_{self.node_id}"
                }
                
                return {
                    "loss": dummy_loss,
                    "training_complete": dummy_signal
                }
                
        except Exception as e:
            self.logger.error(f"Error in PPOTrainerNode {self.node_id}: {e}")
            
            # Return safe defaults
            safe_loss = torch.tensor(-1.0, device=self.device)
            safe_signal = {
                "signal_type": "error",
                "error": str(e),
                "source_node": f"ppo_trainer_{self.node_id}"
            }
            
            return {
                "loss": safe_loss,
                "training_complete": safe_signal
            }