# Template variables - replaced during export
template_vars = {
    "NODE_ID": "ppo_trainer_1",
    "CLASS_NAME": "PPOTrainerNode",
    "MAX_EPOCHS": 100,  # Maximum training epochs (when to stop)
    "HORIZON_LENGTH": 16,
    "MINI_EPOCHS_NUM": 8,  # rl_games naming: mini_epochs_num instead of ppo_epochs
    "MINIBATCH_SIZE": 8192,
    "GAMMA": 0.99,
    "TAU": 0.95,  # rl_games naming: tau instead of gae_lambda
    "E_CLIP": 0.2,  # rl_games naming: e_clip instead of clip_param
    "CRITIC_COEF": 4.0,  # rl_games naming: critic_coef instead of value_coef
    "ENTROPY_COEF": 0.0,
    "LEARNING_RATE": 3e-4,
    "GRAD_NORM": 1.0,  # rl_games naming: grad_norm instead of max_grad_norm
    "CLIP_VALUE": True,
    "BOUNDS_LOSS_COEF": 0.0001,
    "BOUND_LOSS_TYPE": "bound",
    "CHECKPOINT_ENABLED": False,
    "CHECKPOINT_TRIGGER_TYPE": "epoch",
    "CHECKPOINT_TRIGGER_VALUE": "10"
}

"""Node implementation for PPOTrainerNode using rl_games components"""
import time
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from framework.base import QueueNode, SensorNode

# Import rl_games PPO components
import sys
import os
template_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(template_dir)
from rlgames_ppo_components import RLGamesPPOComponents

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """PPO Trainer Node using rl_games components - maintains DNNE async coordination"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["state", "policy_output", "reward", "done", "model"])
        self.setup_outputs(["loss", "training_complete"])
        
        # Configuration from template - using rl_games parameter names
        import builtins
        if hasattr(builtins, 'EPOCHS_OVERRIDE') and builtins.EPOCHS_OVERRIDE is not None:
            self.max_epochs = builtins.EPOCHS_OVERRIDE
            self.logger.info(f"Using epochs override: {{self.max_epochs}} (instead of workflow value: {MAX_EPOCHS})")
        else:
            self.max_epochs = {MAX_EPOCHS}
            
        # rl_games compatible configuration
        rlgames_config = {{
            'horizon_length': {HORIZON_LENGTH},
            'mini_epochs_num': {MINI_EPOCHS_NUM},
            'minibatch_size': {MINIBATCH_SIZE},
            'gamma': {GAMMA},
            'tau': {TAU},
            'e_clip': {E_CLIP},
            'critic_coef': {CRITIC_COEF},
            'entropy_coef': {ENTROPY_COEF},
            'learning_rate': {LEARNING_RATE},
            'grad_norm': {GRAD_NORM},
            'clip_value': {CLIP_VALUE},
            'bounds_loss_coef': {BOUNDS_LOSS_COEF},
            'bound_loss_type': "{BOUND_LOSS_TYPE}"
        }}
        
        # Initialize rl_games PPO components
        self.ppo_components = RLGamesPPOComponents(rlgames_config)
        
        # Maintain DNNE parameter access (for backward compatibility)
        self.horizon_length = {HORIZON_LENGTH}
        self.mini_epochs_num = {MINI_EPOCHS_NUM}
        self.minibatch_size = {MINIBATCH_SIZE}
        self.gamma = {GAMMA}
        self.tau = {TAU}
        self.e_clip = {E_CLIP}
        self.critic_coef = {CRITIC_COEF}
        self.entropy_coef = {ENTROPY_COEF}
        self.learning_rate = {LEARNING_RATE}
        self.grad_norm = {GRAD_NORM}
        
        # Training state
        self.reset_buffer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = None
        self.step_count = 0
        self.current_epoch = 0
        self.training_complete = False
        
        # Check if we're in inference mode
        import builtins
        self.inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        
        # Checkpoint configuration
        self.checkpoint_enabled = {CHECKPOINT_ENABLED}
        self.checkpoint_trigger_type = "{CHECKPOINT_TRIGGER_TYPE}"
        self.checkpoint_trigger_value = "{CHECKPOINT_TRIGGER_VALUE}"
        self.checkpoint_save_on_exit = True
        self.checkpoint_manager = None
        self.last_loss = None
        
        # Initialize checkpoint manager if enabled
        if self.checkpoint_enabled:
            from run_utils import CheckpointManager, validate_checkpoint_config
            
            checkpoint_config = {{
                'enabled': self.checkpoint_enabled,
                'trigger_type': self.checkpoint_trigger_type,
                'trigger_value': self.checkpoint_trigger_value
            }}
            
            try:
                validate_checkpoint_config(checkpoint_config)
                import builtins
                save_checkpoint_dir = getattr(builtins, 'SAVE_CHECKPOINT_DIR', None)
                    
                self.checkpoint_manager = CheckpointManager(
                    node_id=node_id,
                    checkpoint_dir=save_checkpoint_dir
                )
                self.logger.info(f"Checkpoint manager initialized: {{self.checkpoint_trigger_type}} trigger")
            except ValueError as e:
                self.logger.error(f"Checkpoint configuration error: {{e}}")
                self.checkpoint_enabled = False
        
        self.logger.info(f"PPOTrainerNode {{node_id}} initialized with rl_games components - max_epochs={{self.max_epochs}}, horizon={{self.horizon_length}}, mini_epochs={{self.mini_epochs_num}}")
        
    def reset_buffer(self):
        """Reset the trajectory buffer"""
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_values = []
        self.buffer_log_probs = []
        self.buffer_dones = []
        self.buffer_action_means = []  # Store mu for rl_games
        self.buffer_action_stds = []   # Store sigma for rl_games
        self.buffer_full = False
        
    def prepare_rlgames_input_dict(self, states, actions, rewards, values, log_probs, dones, action_means, action_stds):
        """
        Convert DNNE buffer data to rl_games input_dict format
        
        Args:
            states, actions, rewards, values, log_probs, dones: DNNE trajectory data
            action_means, action_stds: Policy parameters for rl_games
            
        Returns:
            input_dict: rl_games compatible data dictionary
        """
        # Compute GAE advantages using rl_games method
        advantages = self.ppo_components.discount_values(rewards, values, dones)
        
        # Compute returns
        returns = advantages + values
        
        # Create rl_games input dictionary
        input_dict = {{
            'old_values': values.detach(),
            'old_logp_actions': log_probs.detach(),
            'advantages': advantages.detach(),
            'returns': returns.detach(),
            'actions': actions.detach(),
            'obs': states.detach(),
            'mu': action_means.detach(),
            'sigma': action_stds.detach(),
            'dones': dones.detach()
        }}
        
        return input_dict
    
    def rlgames_ppo_update(self, states, actions, rewards, values, log_probs, dones, action_means, action_stds, model):
        """
        Perform PPO update using rl_games components
        Replaces custom ppo_update() method with rl_games implementation
        
        Args:
            states, actions, rewards, values, log_probs, dones: Trajectory data
            action_means, action_stds: Policy parameters
            model: PyTorch model to update
            
        Returns:
            average_loss: Average loss over all updates
        """
        
        # Skip training in inference mode
        if self.inference_mode:
            return torch.zeros(1, device=self.device)
        
        # Setup optimizer if needed
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
        model.train()
        total_losses = []
        batch_size = len(states)
        
        # Prepare rl_games input dictionary
        input_dict = self.prepare_rlgames_input_dict(
            states, actions, rewards, values, log_probs, dones, action_means, action_stds
        )
        
        # Multiple mini-epochs over the data (rl_games pattern)
        for mini_epoch in range(self.mini_epochs_num):
            # Create minibatches
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = min(start + self.minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Create minibatch input_dict
                mb_input_dict = {{}}
                for key, value in input_dict.items():
                    mb_input_dict[key] = value[mb_indices]
                
                # Use rl_games PPO components for loss computation
                train_result, loss = self.ppo_components.train_actor_critic(mb_input_dict, model)
                
                # Backpropagation (DNNE maintains control over optimization)
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping using rl_games parameter
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm)
                
                self.optimizer.step()
                
                total_losses.append(loss.item())
                
        return torch.tensor(np.mean(total_losses), device=self.device)
    
    async def run(self):
        """Override run to send initial training_complete trigger"""
        self.running = True
        self.logger.info(f"Starting PPOTrainer node {{self.node_id}} with rl_games components")
        
        # CRITICAL: Send initial training_complete trigger to break circular dependency
        await self.send_output("training_complete", {{"trigger": True, "step": 0}})
        self.logger.info("Sent initial training_complete trigger to break circular dependency")
        
        # Now proceed with normal QueueNode execution
        await super().run()
        
    async def compute(self, state, policy_output, reward, done, model) -> Dict[str, Any]:
        """
        PPO training step using rl_games components - maintains DNNE async coordination
        
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
        
        # In inference mode, just pass through signals without training
        if self.inference_mode:
            return {{
                "loss": torch.zeros(1, device=self.device),
                "training_complete": {{"signal": "complete", "timestamp": time.time()}}
            }}
        
        # If training is complete, stop processing immediately
        if self.training_complete:
            from framework.base import TrainingCompleteException
            raise TrainingCompleteException(
                self.node_id, 
                f"PPO training complete after {{self.current_epoch}}/{{self.max_epochs}} epochs"
            )
        
        try:
            # Ensure tensors are on correct device
            state = state.to(self.device)
            reward = reward.to(self.device) 
            done = done.to(self.device)
            
            # Extract values from policy_output dictionary
            action = policy_output["action"].to(self.device)
            value = policy_output["value"].to(self.device)
            log_prob = policy_output["log_prob"].to(self.device)
            
            # Extract action parameters for rl_games (if available)
            action_mean = policy_output.get("action_mean", torch.zeros_like(action))
            action_std = policy_output.get("action_std", torch.ones_like(action))
            
            # Add to buffer (detach to avoid gradient conflicts)
            self.buffer_states.append(state.detach().clone())
            self.buffer_actions.append(action.detach().clone())
            self.buffer_rewards.append(reward.detach().clone())
            self.buffer_values.append(value.detach().clone())
            self.buffer_log_probs.append(log_prob.detach().clone())
            self.buffer_dones.append(done.detach().clone())
            self.buffer_action_means.append(action_mean.detach().clone())
            self.buffer_action_stds.append(action_std.detach().clone())
            
            # Check if buffer is full (DNNE async coordination maintained)
            if len(self.buffer_states) >= self.horizon_length:
                # Convert buffer to tensors
                states = torch.stack(self.buffer_states)
                actions = torch.stack(self.buffer_actions)
                rewards = torch.stack(self.buffer_rewards)
                values = torch.stack(self.buffer_values)
                log_probs = torch.stack(self.buffer_log_probs)
                dones = torch.stack(self.buffer_dones)
                action_means = torch.stack(self.buffer_action_means)
                action_stds = torch.stack(self.buffer_action_stds)
                
                # Perform PPO training using rl_games components
                total_loss = self.rlgames_ppo_update(
                    states, actions, rewards, values, log_probs, dones, 
                    action_means, action_stds, model
                )
                
                # Update step count and epoch count
                self.step_count += 1
                self.current_epoch += 1
                
                # Check if we've reached max epochs
                if self.current_epoch >= self.max_epochs:
                    self.training_complete = True
                    self.logger.info(f"🎯 PPO Trainer reached max_epochs ({{self.max_epochs}}) - signaling completion")
                
                # Handle checkpointing (unchanged from original)
                if self.checkpoint_enabled and self.checkpoint_manager:
                    current_loss = total_loss.item()
                    self.last_loss = current_loss
                    
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
                        should_checkpoint = self.checkpoint_manager.should_checkpoint(
                            "best_metric", "min", current_metric=current_loss
                        )
                    
                    if should_checkpoint:
                        metadata = {{
                            'trigger_type': self.checkpoint_trigger_type,
                            'trigger_value': self.checkpoint_trigger_value,
                            'training_step': self.step_count,
                            'loss': current_loss,
                            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                            'hyperparameters': {{
                                'max_epochs': self.max_epochs,
                                'horizon_length': self.horizon_length,
                                'mini_epochs_num': self.mini_epochs_num,
                                'minibatch_size': self.minibatch_size,
                                'gamma': self.gamma,
                                'tau': self.tau,
                                'e_clip': self.e_clip,
                                'critic_coef': self.critic_coef,
                                'entropy_coef': self.entropy_coef,
                                'learning_rate': self.learning_rate,
                                'grad_norm': self.grad_norm
                            }}
                        }}
                        
                        self.checkpoint_manager.save_checkpoint(
                            model.state_dict(), metadata=metadata
                        )
                
                # Reset buffer (DNNE async coordination maintained)
                self.reset_buffer()
                
                # Create completion signal
                training_complete = {{
                    "signal_type": "training_complete",
                    "step": self.step_count,
                    "loss": total_loss.item(),
                    "source_node": f"ppo_trainer_{{self.node_id}}"
                }}
                
                self.logger.info(f"PPO training step {{self.step_count}} complete (rl_games), loss: {{total_loss.item():.4f}}")
                
                return {{
                    "loss": total_loss,
                    "training_complete": training_complete
                }}
            
            else:
                # Still collecting, return dummy outputs (DNNE async coordination maintained)
                dummy_loss = torch.tensor(0.0, device=self.device)
                dummy_signal = {{
                    "signal_type": "collecting", 
                    "buffer_size": len(self.buffer_states),
                    "horizon_length": self.horizon_length,
                    "source_node": f"ppo_trainer_{{self.node_id}}"
                }}
                
                return {{
                    "loss": dummy_loss,
                    "training_complete": dummy_signal
                }}
                
        except Exception as e:
            self.logger.error(f"Error in PPOTrainerNode {{self.node_id}}: {{e}}")
            
            # Return safe defaults
            safe_loss = torch.tensor(-1.0, device=self.device)
            safe_signal = {{
                "signal_type": "error",
                "error": str(e),
                "source_node": f"ppo_trainer_{{self.node_id}}"
            }}
            
            return {{
                "loss": safe_loss,
                "training_complete": safe_signal
            }}
    
    async def save_checkpoint_on_exit(self, exit_reason: str) -> bool:
        """Save checkpoint on exit if enabled (unchanged from original)"""
        if not self.checkpoint_enabled or not self.checkpoint_save_on_exit or not self.checkpoint_manager:
            return False
            
        try:
            import time
            
            metadata = {{
                'exit_type': 'on_exit',
                'exit_reason': exit_reason,
                'timestamp': time.time(),
                'training_step': self.step_count,
                'max_epochs': self.max_epochs,
                'current_epoch': self.current_epoch,
                'horizon_length': self.horizon_length,
                'mini_epochs_num': self.mini_epochs_num,
                'minibatch_size': self.minibatch_size,
                'hyperparameters': {{
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'e_clip': self.e_clip,
                    'critic_coef': self.critic_coef,
                    'entropy_coef': self.entropy_coef,
                    'learning_rate': self.learning_rate,
                    'grad_norm': self.grad_norm
                }},
                'last_loss': self.last_loss,
                'rlgames_integration': True
            }}
            
            success = self.checkpoint_manager.save_checkpoint(
                {{}}, metadata=metadata
            )
            
            if success:
                self.logger.info(f"💾 Exit checkpoint saved for rl_games PPOTrainer node {{self.node_id}}")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to save exit checkpoint for rl_games PPOTrainer node {{self.node_id}}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving exit checkpoint: {{e}}")
            return False