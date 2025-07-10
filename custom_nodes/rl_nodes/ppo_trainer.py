# ppo_trainer.py
"""
PPO Trainer Node - Complete PPO Training Algorithm
Handles trajectory collection, advantage computation, and policy optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from inspect import cleandoc
from .rl_types import PPOBatch, PolicyOutput, compute_gae_advantages

class PPOTrainerNode:
    """
    PPO Trainer Node
    Collects PPO trajectories, computes advantages using GAE and performs PPO updates.
    For checkpoint debugging: check console logs or exported code for actual node ID.
    """
    
    def __init__(self):
        self.reset_buffer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = None
        self.step_count = 0
        self.checkpoint_manager = None
        self.last_loss = None  # For best metric checkpointing
        
    def reset_buffer(self):
        """Reset the trajectory buffer"""
        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_values = []
        self.buffer_log_probs = []
        self.buffer_dones = []
        self.buffer_full = False
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": ("TENSOR", {"tooltip": "Current environment state tensor from simulation"}),
                "policy_output": ("POLICY_OUTPUT", {"tooltip": "Policy output containing action, value, and log probability"}),
                "reward": ("TENSOR", {"tooltip": "Reward signal from environment (scalar per environment)"}),
                "done": ("TENSOR", {"tooltip": "Episode termination flags (boolean per environment)"}),
                "model": ("MODEL", {"tooltip": "Neural network model to train (actor-critic architecture)"}),
                "horizon_length": ("INT", {"default": 16, "min": 4, "max": 256, "tooltip": "Number of steps to collect before training (trajectory length)"}),
                "num_epochs": ("INT", {"default": 4, "min": 1, "max": 20, "tooltip": "Number of training epochs per collected trajectory"}),
                "minibatch_size": ("INT", {"default": 32, "min": 4, "max": 512, "tooltip": "Batch size for SGD updates during training"}),
                "gamma": ("FLOAT", {"default": 0.99, "min": 0.9, "max": 1.0, "step": 0.01, "tooltip": "Discount factor for future rewards (0.99 = long-term focus)"}),
                "gae_lambda": ("FLOAT", {"default": 0.95, "min": 0.9, "max": 1.0, "step": 0.01, "tooltip": "GAE lambda parameter for advantage estimation (higher = less bias)"}),
                "clip_param": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.5, "step": 0.01, "tooltip": "PPO clipping parameter (prevents large policy updates)"}),
                "value_coef": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1, "tooltip": "Coefficient for value function loss in total loss"}),
                "entropy_coef": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001, "tooltip": "Coefficient for entropy bonus (encourages exploration)"}),
                "learning_rate": ("FLOAT", {"default": 3e-4, "min": 1e-6, "max": 1e-1, "step": 1e-6, "tooltip": "Learning rate for the optimizer (3e-4 is PPO standard)"}),
                "max_grad_norm": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Maximum gradient norm for clipping (prevents exploding gradients)"}),
                # Checkpoint parameters
                "checkpoint_enabled": ("BOOLEAN", {"default": False, "tooltip": "Enable automatic checkpoint saving during training. Checkpoints saved to 'node_<ID>' subdirectories."}),
                "checkpoint_trigger_type": (["epoch", "time", "best_metric"], {"default": "epoch", "tooltip": "When to save checkpoints: every N training steps, time intervals, or metric improvements"}),
                "checkpoint_trigger_value": ("STRING", {"default": "10", "tooltip": "Trigger value: number (epochs), time format (1h30m), or 'min'/'max' (metrics)"}),
            },
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
        
    RETURN_TYPES = ("TENSOR", "SYNC")
    RETURN_NAMES = ("loss", "training_complete")
    
    FUNCTION = "train_step"
    CATEGORY = "rl"
    DESCRIPTION = cleandoc(__doc__)
    
    def train_step(self, state, policy_output, reward, done, model, 
                   horizon_length=16, num_epochs=4, minibatch_size=32,
                   gamma=0.99, gae_lambda=0.95, clip_param=0.2,
                   value_coef=0.5, entropy_coef=0.01, learning_rate=3e-4,
                   max_grad_norm=0.5, checkpoint_enabled=False,
                   checkpoint_trigger_type="epoch", checkpoint_trigger_value="10",
                   unique_id=None):
        """
        PPO training step - collect trajectory and train when buffer is full
        
        Args:
            state: Current state tensor
            policy_output: PolicyOutput from PPOAgent
            reward: Current reward tensor
            done: Episode termination flag
            model: PyTorch model for training
            ... (other parameters are hyperparameters)
            
        Returns:
            loss: Training loss (0 if still collecting)
            training_complete: Sync signal when training is done
        """
        
        # Log actual node ID for checkpoint debugging
        actual_node_id = unique_id or f"ppo_trainer_{id(self)}"
        if checkpoint_enabled:
            print(f"🔍 PPO Trainer Node ID: {actual_node_id} (look for 'node_{actual_node_id}' in checkpoint directories)")
        
        # Initialize checkpoint manager if needed
        if checkpoint_enabled and self.checkpoint_manager is None:
            # Import here to avoid circular imports in export
            from ..export_system.templates.base.run_utils import CheckpointManager, validate_checkpoint_config
            
            # Validate checkpoint configuration
            checkpoint_config = {
                'enabled': checkpoint_enabled,
                'trigger_type': checkpoint_trigger_type,
                'trigger_value': checkpoint_trigger_value
            }
            try:
                validate_checkpoint_config(checkpoint_config)
            except ValueError as e:
                print(f"⚠️ Checkpoint configuration error: {e}")
                checkpoint_enabled = False
            
            if checkpoint_enabled:
                # Get checkpoint directory from command line args (set by runner.py)
                try:
                    import builtins
                    save_checkpoint_dir = getattr(builtins, 'SAVE_CHECKPOINT_DIR', None)
                except:
                    save_checkpoint_dir = None
                    
                self.checkpoint_manager = CheckpointManager(
                    node_id=actual_node_id,
                    checkpoint_dir=save_checkpoint_dir
                )
        
        # Ensure tensors are on correct device
        state = state.to(self.device)
        reward = reward.to(self.device) 
        done = done.to(self.device)
        policy_output = policy_output.to_device(self.device)
        
        # Add to buffer
        self.buffer_states.append(state.clone())
        self.buffer_actions.append(policy_output.action.clone())
        self.buffer_rewards.append(reward.clone())
        self.buffer_values.append(policy_output.value.clone())
        self.buffer_log_probs.append(policy_output.log_prob.clone())
        self.buffer_dones.append(done.clone())
        
        # Check if buffer is full
        if len(self.buffer_states) >= horizon_length:
            # Convert buffer to tensors
            states = torch.stack(self.buffer_states)
            actions = torch.stack(self.buffer_actions)
            rewards = torch.stack(self.buffer_rewards)
            values = torch.stack(self.buffer_values)
            log_probs = torch.stack(self.buffer_log_probs)
            dones = torch.stack(self.buffer_dones)
            
            # Compute advantages and returns using GAE
            advantages, returns = compute_gae_advantages(
                rewards, values, dones, gamma, gae_lambda
            )
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Create PPO batch
            batch = PPOBatch(states, actions, rewards, values, log_probs, 
                           dones, advantages, returns)
            
            # Perform PPO training
            total_loss = self.ppo_update(batch, model, num_epochs, minibatch_size,
                                       clip_param, value_coef, entropy_coef,
                                       learning_rate, max_grad_norm)
            
            # Update step count
            self.step_count += 1
            
            # Handle checkpointing
            if checkpoint_enabled and self.checkpoint_manager:
                current_loss = total_loss.item()
                self.last_loss = current_loss
                
                # Check if we should checkpoint
                should_checkpoint = False
                if checkpoint_trigger_type == "epoch":
                    should_checkpoint = self.checkpoint_manager.should_checkpoint(
                        "epoch", checkpoint_trigger_value, current_epoch=self.step_count
                    )
                elif checkpoint_trigger_type == "time":
                    should_checkpoint = self.checkpoint_manager.should_checkpoint(
                        "time", checkpoint_trigger_value
                    )
                elif checkpoint_trigger_type == "best_metric":
                    # Use loss as metric (lower is better)
                    should_checkpoint = self.checkpoint_manager.should_checkpoint(
                        "best_metric", "min", current_metric=current_loss
                    )
                
                if should_checkpoint:
                    # Prepare metadata with training state and hyperparameters
                    metadata = {
                        'trigger_type': checkpoint_trigger_type,
                        'trigger_value': checkpoint_trigger_value,
                        'training_step': self.step_count,
                        'loss': current_loss,
                        'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                        'hyperparameters': {
                            'horizon_length': horizon_length,
                            'num_epochs': num_epochs,
                            'minibatch_size': minibatch_size,
                            'gamma': gamma,
                            'gae_lambda': gae_lambda,
                            'clip_param': clip_param,
                            'value_coef': value_coef,
                            'entropy_coef': entropy_coef,
                            'learning_rate': learning_rate,
                            'max_grad_norm': max_grad_norm
                        }
                    }
                    
                    # Save checkpoint (only model weights + metadata)
                    self.checkpoint_manager.save_checkpoint(
                        model.state_dict(), metadata=metadata
                    )
            
            # Reset buffer
            self.reset_buffer()
            
            # Create completion signal
            training_complete = {
                "signal_type": "training_complete",
                "step": self.step_count,
                "loss": total_loss.item(),
                "source_node": "ppo_trainer"
            }
            
            return (total_loss, training_complete)
        
        else:
            # Still collecting, return dummy outputs
            dummy_loss = torch.tensor(0.0, device=self.device)
            dummy_signal = {"signal_type": "collecting", "source_node": "ppo_trainer"}
            return (dummy_loss, dummy_signal)
            
    def ppo_update(self, batch, model, num_epochs, minibatch_size, clip_param,
                   value_coef, entropy_coef, learning_rate, max_grad_norm):
        """
        Perform PPO update on collected batch
        
        Args:
            batch: PPOBatch with trajectory data
            model: PyTorch model to update
            ... (hyperparameters)
            
        Returns:
            average_loss: Average loss over all updates
        """
        
        # Setup optimizer if needed
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
        model.train()
        total_losses = []
        
        # Multiple epochs over the data
        for epoch in range(num_epochs):
            # Create minibatches
            batch_size = batch.length
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Get minibatch data
                mb_states = batch.states[mb_indices]
                mb_actions = batch.actions[mb_indices]
                mb_old_log_probs = batch.log_probs[mb_indices]
                mb_advantages = batch.advantages[mb_indices]
                mb_returns = batch.returns[mb_indices]
                
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
                surr2 = mb_advantages * torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (MSE)
                critic_loss = nn.MSELoss()(current_values, mb_returns)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
                
                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                self.optimizer.step()
                
                total_losses.append(total_loss.item())
                
        return torch.tensor(np.mean(total_losses), device=self.device)
    
    def load_checkpoint(self, model, load_checkpoint_dir=None):
        """
        Load checkpoint and restore training state
        
        Args:
            model: PyTorch model to load state into
            load_checkpoint_dir: Override load directory (from command line)
            
        Returns:
            bool: True if checkpoint loaded successfully
        """
        if not self.checkpoint_manager:
            print("⚠️ No checkpoint manager initialized")
            return False
        
        # Load checkpoint from command line directory or default
        checkpoint_data = self.checkpoint_manager.load_checkpoint(load_checkpoint_dir)
        if not checkpoint_data:
            print("⚠️ No checkpoint found to load")
            return False
        
        try:
            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Load optimizer state from metadata
            metadata = checkpoint_data['metadata']
            if self.optimizer and 'optimizer_state' in metadata:
                optimizer_state = metadata['optimizer_state']
                if optimizer_state:
                    self.optimizer.load_state_dict(optimizer_state)
            
            # Restore training state from metadata
            self.step_count = metadata.get('training_step', 0)
            self.last_loss = metadata.get('loss', None)
            
            print(f"✅ Checkpoint loaded successfully - resuming from step {self.step_count}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            return False

# Register the node
NODE_CLASS_MAPPINGS = {"PPOTrainerNode": PPOTrainerNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PPOTrainerNode": "PPO Trainer"}