import time
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from framework import QueueNode, SensorNode

# Template variables - replaced during export

"""Node implementation for PPOTrainerNode using rl_games components"""
import time
import os
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import numpy as np
from framework import QueueNode, SensorNode

# Import PPO components from rl_games_dnne
import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT')
from rl_games_dnne.dnne_exports import PPOComponents, RunningMeanStd

class PPOTrainerNode_6(QueueNode):
    """PPO Trainer Node using rl_games components - maintains DNNE async coordination"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["state", "policy_output", "reward", "done", "model"])
        self.setup_outputs(["loss", "training_complete"])
        
        # Configuration from template - using rl_games parameter names
        import builtins
        if hasattr(builtins, 'EPOCHS_OVERRIDE') and builtins.EPOCHS_OVERRIDE is not None:
            self.max_epochs = builtins.EPOCHS_OVERRIDE
            self.logger.info(f"Using epochs override: {self.max_epochs} (instead of workflow value: 100)")
        else:
            self.max_epochs = 100
            
        # rl_games compatible configuration
        rlgames_config = {
            'horizon_length': 16,
            'mini_epochs_num': 8,
            'minibatch_size': 8192,
            'gamma': 0.99,
            'tau': 0.95,
            'e_clip': 0.2,
            'critic_coef': 4,
            'entropy_coef': 0,
            'learning_rate': 0.0003,
            'grad_norm': 1,
            'clip_value': True,
            'bounds_loss_coef': 0.0001,
            'bound_loss_type': "bound"
        }
        
        # Initialize PPO components
        self.ppo_components = PPOComponents(rlgames_config)
        
        # Maintain DNNE parameter access (for backward compatibility)
        self.horizon_length = 16
        self.mini_epochs_num = 8
        self.minibatch_size = 8192
        self.gamma = 0.99
        self.tau = 0.95
        self.e_clip = 0.2
        self.critic_coef = 4
        self.entropy_coef = 0
        self.learning_rate = 0.0003
        self.grad_norm = 1
        
        # Training state
        self.reset_buffer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = None
        self.step_count = 0
        self.current_epoch = 0
        self.training_complete = False
        
        # PPO cycle tracking
        self.ppo_cycles_completed = 0
        self.stop_after_cycle = None
        ppo_stop_env = os.environ.get('PPO_STOP_AFTER_CYCLE')
        if ppo_stop_env:
            try:
                self.stop_after_cycle = int(ppo_stop_env)
                self.logger.info(f"PPO_STOP_AFTER_CYCLE set to {self.stop_after_cycle}")
            except ValueError:
                self.logger.warning(f"Invalid PPO_STOP_AFTER_CYCLE value: {ppo_stop_env}")
        
        # Value function normalization (matching IsaacGymEnvs)
        self.value_rms = None  # Will be initialized on first use
        
        # Check if we're in inference mode
        import builtins
        self.inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        self.fixed_seed_debug = getattr(builtins, 'FIXED_SEED', None) is not None
        
        # Checkpoint configuration
        self.checkpoint_enabled = True
        self.checkpoint_trigger_type = "epoch"
        self.checkpoint_trigger_value = "5"
        self.checkpoint_save_on_exit = True
        self.checkpoint_manager = None
        self.last_loss = None
        
        # Initialize checkpoint manager if enabled
        if self.checkpoint_enabled:
            from framework.checkpoint import CheckpointManager, validate_checkpoint_config
            
            checkpoint_config = {
                'enabled': self.checkpoint_enabled,
                'trigger_type': self.checkpoint_trigger_type,
                'trigger_value': self.checkpoint_trigger_value
            }
            
            try:
                validate_checkpoint_config(checkpoint_config)
                import builtins
                save_checkpoint_dir = getattr(builtins, 'SAVE_CHECKPOINT_DIR', None)
                    
                self.checkpoint_manager = CheckpointManager(
                    node_id=node_id,
                    checkpoint_dir=save_checkpoint_dir
                )
                self.logger.info(f"Checkpoint manager initialized: {self.checkpoint_trigger_type} trigger")
            except ValueError as e:
                self.logger.error(f"Checkpoint configuration error: {e}")
                self.checkpoint_enabled = False
        
        self.logger.info(f"PPOTrainerNode {node_id} initialized with rl_games components - max_epochs={self.max_epochs}, horizon={self.horizon_length}, mini_epochs={self.mini_epochs_num}")
        
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
        
    def prepare_rlgames_input_dict(self, states, actions, rewards, values, log_probs, dones, action_means, action_stds, 
                                   last_values, last_dones, horizon_length, num_envs):
        """
        Convert DNNE buffer data to rl_games input_dict format
        
        Args:
            states, actions, rewards, values, log_probs, dones: DNNE trajectory data (already flattened)
            action_means, action_stds: Policy parameters for rl_games
            last_values: Value of the state after the last action (bootstrap value)
            last_dones: Done flags for the state after the last action
            horizon_length: Number of steps in trajectory
            num_envs: Number of parallel environments
            
        Returns:
            input_dict: rl_games compatible data dictionary
        """
        # Reshape flattened tensors back to [horizon_length, num_envs] for GAE computation
        rewards_2d = rewards.view(horizon_length, num_envs)
        values_2d = values.view(horizon_length, num_envs)
        dones_2d = dones.view(horizon_length, num_envs)
        
        # Append bootstrap values and dones for GAE computation
        # rl_games expects values and dones to have shape [horizon_length + 1, num_envs]
        values_with_bootstrap = torch.cat([values_2d, last_values.unsqueeze(0)], dim=0)
        dones_with_bootstrap = torch.cat([dones_2d, last_dones.unsqueeze(0)], dim=0)
        
        # Compute GAE advantages using rl_games method
        advantages = self.ppo_components.discount_values(rewards_2d, values_with_bootstrap, dones_with_bootstrap)
        
        # Flatten advantages back to match the flattened format
        advantages = advantages.transpose(0, 1).reshape(-1)
        
        # Compute returns (only for the trajectory, not including bootstrap)
        returns = advantages + values
        
        if self.fixed_seed_debug:
            self.logger.info(f"[PPO Trainer Debug] Computing GAE with gamma={self.gamma}, tau={self.tau}")
            self.logger.info(f"[PPO Trainer Debug] Raw rewards: {rewards[:5].tolist()}")
            self.logger.info(f"[PPO Trainer Debug] Raw values: {values[:5].tolist()}")
            self.logger.info(f"[PPO Trainer Debug] Computed advantages: {advantages[:5].tolist()}")
            self.logger.info(f"[PPO Trainer Debug] Computed returns: {returns[:5].tolist()}")
        
        # Initialize value normalization on first use (matching IsaacGymEnvs)
        if self.value_rms is None and not self.inference_mode:
            self.value_rms = RunningMeanStd(shape=(1,), device=self.device)
            self.logger.info("Initialized value function normalization")
        
        # Update value normalization statistics with returns (only in training)
        if self.value_rms is not None and not self.inference_mode:
            # Flatten returns for statistics update
            flat_returns = returns.view(-1, 1)
            self.value_rms.update(flat_returns)
            
            # Normalize returns for value function targets
            normalized_returns = self.value_rms.normalize(flat_returns).view_as(returns)
        else:
            # In inference mode or before initialization, use raw returns
            normalized_returns = returns
        
        # Create rl_games input dictionary
        input_dict = {
            'old_values': values.detach(),
            'old_logp_actions': log_probs.detach(),
            'advantages': advantages.detach(),
            'returns': normalized_returns.detach(),  # Use normalized returns as value targets
            'actions': actions.detach(),
            'obs': states.detach(),
            'mu': action_means.detach(),
            'sigma': action_stds.detach(),
            'dones': dones.detach()
        }
        
        # PPO_BATCH debug logging to match IGE
        import os
        if os.environ.get('PPO_CYCLE_DEBUG', '0') == '1':
            # Note: IGE shows shapes before flattening, but we show after
            print(f"[DNNE_DEBUG] PPO_BATCH: Advantages shape: {advantages.shape}, mean: {advantages.mean().item():.4f}, std: {advantages.std().item():.4f}")
            print(f"[DNNE_DEBUG] PPO_BATCH: Returns shape: {returns.shape}, mean: {returns.mean().item():.4f}, std: {returns.std().item():.4f}")
            print(f"[DNNE_DEBUG] PPO_BATCH: Values shape: {values.shape}, mean: {values.mean().item():.4f}, std: {values.std().item():.4f}")
            print(f"[DNNE_DEBUG] PPO_BATCH: First 5 advantages: {advantages.flatten()[:5].tolist()}")
            print(f"[DNNE_DEBUG] PPO_BATCH: First 5 returns: {returns.flatten()[:5].tolist()}")
        
        return input_dict
    
    def rlgames_ppo_update(self, states, actions, rewards, values, log_probs, dones, action_means, action_stds, model, last_state, last_done):
        """
        Perform PPO update using rl_games components
        Replaces custom ppo_update() method with rl_games implementation
        
        Args:
            states, actions, rewards, values, log_probs, dones: Trajectory data
            action_means, action_stds: Policy parameters
            model: PyTorch model to update
            last_state: The state after the last action (for bootstrap value)
            last_done: Done flag after the last action
            
        Returns:
            average_loss: Average loss over all updates
        """
        # print(f"[DEBUG] rlgames_ppo_update STARTED - current_epoch={self.current_epoch}, max_epochs={self.max_epochs}")
        
        # Skip training in inference mode
        if self.inference_mode:
            return torch.zeros(1, device=self.device)
        
        # Debug shapes
        import os
        ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        if ppo_cycle_debug:
            print(f"[PPO_CYCLE_DEBUG] rlgames_ppo_update input shapes:")
            print(f"  states: {states.shape}")
            print(f"  actions: {actions.shape}")
            print(f"  rewards: {rewards.shape}")
            print(f"  values: {values.shape}")
            print(f"  log_probs: {log_probs.shape}")
            print(f"  dones: {dones.shape}")
        
        # Setup optimizer if needed
        if self.optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            
        model.train()
        total_losses = []
        batch_size = len(states)
        
        # Get bootstrap value from the model for the last state
        with torch.no_grad():
            # Get shared features for last state
            last_features = model['shared'](last_state)
            # Get value prediction
            last_values = model['value'](last_features).squeeze(-1)
        
        # Calculate dimensions for reshape
        num_envs = last_state.shape[0]  # Number of environments
        horizon_length = batch_size // num_envs  # Number of timesteps
        
        # Prepare rl_games input dictionary
        input_dict = self.prepare_rlgames_input_dict(
            states, actions, rewards, values, log_probs, dones, action_means, action_stds, 
            last_values, last_done, horizon_length, num_envs
        )
        
        if self.fixed_seed_debug:
            self.logger.info(f"[PPO Trainer Debug] Starting {self.mini_epochs_num} mini-epochs")
            self.logger.info(f"[PPO Trainer Debug] Advantages: {input_dict['advantages'][:5].tolist()}")
            self.logger.info(f"[PPO Trainer Debug] Returns: {input_dict['returns'][:5].tolist()}")
        
        # Multiple mini-epochs over the data (rl_games pattern)
        for mini_epoch in range(self.mini_epochs_num):
            # print(f"[DEBUG] Starting mini-epoch {mini_epoch + 1}/{self.mini_epochs_num}")
            # Create minibatches
            indices = torch.randperm(batch_size)
            
            if ppo_cycle_debug and mini_epoch == 0:
                print(f"[PPO_CYCLE_DEBUG] Mini-epoch {mini_epoch}: batch_size={batch_size}, minibatch_size={self.minibatch_size}")
                print(f"[PPO_CYCLE_DEBUG] Number of minibatches: {(batch_size + self.minibatch_size - 1) // self.minibatch_size}")
            
            minibatch_count = 0
            for start in range(0, batch_size, self.minibatch_size):
                end = min(start + self.minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                # Create minibatch input_dict
                mb_input_dict = {}
                for key, value in input_dict.items():
                    try:
                        mb_input_dict[key] = value[mb_indices]
                    except IndexError as e:
                        self.logger.error(f"IndexError in minibatch creation:")
                        self.logger.error(f"  Key: {key}")
                        self.logger.error(f"  Value shape: {value.shape}")
                        self.logger.error(f"  mb_indices max: {mb_indices.max().item()}")
                        self.logger.error(f"  batch_size: {batch_size}")
                        raise e
                
                minibatch_count += 1
                # if minibatch_count == 1 or minibatch_count % 10 == 0:
                #     print(f"[DEBUG] Mini-epoch {mini_epoch + 1}, Minibatch {minibatch_count}")
                
                # Use rl_games PPO components for loss computation
                # print(f"[DEBUG] Calling train_actor_critic...")
                train_result, loss = self.ppo_components.train_actor_critic(mb_input_dict, model)
                # print(f"[DEBUG] train_actor_critic returned, loss={loss.item():.4f}")
                
                # Backpropagation (DNNE maintains control over optimization)
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping using rl_games parameter
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_norm)
                
                self.optimizer.step()
                
                total_losses.append(loss.item())
            
            # print(f"[DEBUG] Completed mini-epoch {mini_epoch + 1}/{self.mini_epochs_num}")
                
        result = torch.tensor(np.mean(total_losses), device=self.device)
        # print(f"[DEBUG] rlgames_ppo_update COMPLETED - returning loss={result.item()}")
        return result
    
    async def run(self):
        """Override run to send initial training_complete trigger"""
        self.running = True
        self.logger.info(f"Starting PPOTrainer node {self.node_id} with rl_games components")
        
        # CRITICAL: Send initial training_complete trigger to break circular dependency
        await self.send_output("training_complete", {"trigger": True, "step": 0})
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
            return {
                "loss": torch.zeros(1, device=self.device),
                "training_complete": {"signal": "complete", "timestamp": time.time()}
            }
        
        # If training is complete, stop processing immediately
        if self.training_complete:
            # print(f"[DEBUG] PPOTrainerNode.compute() - training_complete is True, raising TrainingCompleteException")
            # print(f"[DEBUG] Current epoch: {self.current_epoch}, max_epochs: {self.max_epochs}")
            from framework import TrainingCompleteException
            stop_reason = []
            if self.current_epoch >= self.max_epochs:
                stop_reason.append(f"{self.current_epoch}/{self.max_epochs} epochs")
            if self.stop_after_cycle and self.ppo_cycles_completed >= self.stop_after_cycle:
                stop_reason.append(f"{self.ppo_cycles_completed}/{self.stop_after_cycle} PPO cycles")
            
            raise TrainingCompleteException(
                self.node_id, 
                f"PPO training complete after {' and '.join(stop_reason)}"
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
            # Use normalized observations if available (critical for correct training)
            normalized_obs = policy_output.get("normalized_observations", state)
            self.buffer_states.append(normalized_obs.detach().clone())
            self.buffer_actions.append(action.detach().clone())
            
            # PPO_CYCLE_DEBUG logging to match IGE
            import os
            ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
            if ppo_cycle_debug and len(self.buffer_states) < 5:
                step_num = len(self.buffer_states)
                # Get first environment's values for logging
                first_action = action[0].item() if action.dim() > 0 else action.item()
                first_value = value[0].item() if value.dim() > 0 else value.item()
                first_reward = reward[0].item() if reward.dim() > 0 else reward.item()
                print(f"[DNNE_DEBUG] PPO_CYCLE: Step {step_num}: action={first_action:.4f}, value={first_value:.4f}, reward={first_reward:.4f}")
            
            if self.fixed_seed_debug and self.step_count == 0:
                self.logger.info(f"[PPO Trainer Debug] First state shape: {state.shape}")
                self.logger.info(f"[PPO Trainer Debug] First state (first 5): {state[0][:5].tolist()}")
                self.logger.info(f"[PPO Trainer Debug] First normalized state (first 5): {normalized_obs[0][:5].tolist()}")
                self.logger.info(f"[PPO Trainer Debug] Action: {policy_output['action'][0].tolist() if policy_output['action'].dim() > 1 else policy_output['action'].tolist()}")
                self.logger.info(f"[PPO Trainer Debug] Value: {policy_output['value'][0].item() if policy_output['value'].numel() > 1 else policy_output['value'].item()}")
                self.logger.info(f"[PPO Trainer Debug] Log prob: {policy_output['log_prob'][0].item() if policy_output['log_prob'].numel() > 1 else policy_output['log_prob'].item()}")
                self.logger.info(f"[PPO Trainer Debug] Reward: {reward[0].item() if reward.numel() > 1 else reward.item()}")
                self.logger.info(f"[PPO Trainer Debug] Done: {done[0].item() if done.numel() > 1 else done.item()}")
            self.buffer_rewards.append(reward.detach().clone())
            self.buffer_values.append(value.detach().clone())
            self.buffer_log_probs.append(log_prob.detach().clone())
            self.buffer_dones.append(done.detach().clone())
            self.buffer_action_means.append(action_mean.detach().clone())
            self.buffer_action_stds.append(action_std.detach().clone())
            
            # Add debug to track buffer growth
            import os
            ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
            if ppo_cycle_debug:
                print(f"[PPO_CYCLE_DEBUG] Buffer size: {len(self.buffer_states)}, horizon: {self.horizon_length}")
            
            # Check if buffer is full (DNNE async coordination maintained)
            # CRITICAL FIX: Each buffer entry contains data for ALL environments
            # So we need horizon_length entries, not horizon_length * num_envs
            if len(self.buffer_states) >= self.horizon_length:
                if ppo_cycle_debug:
                    print(f"[PPO_CYCLE_DEBUG] Buffer full! Length: {len(self.buffer_states)}")
                if self.fixed_seed_debug:
                    self.logger.info(f"[PPO Trainer Debug] Starting PPO update with {len(self.buffer_states)} steps")
                    self.logger.info(f"[PPO Trainer Debug] Buffer state shape: {self.buffer_states[0].shape}")
                    
                # Convert buffer to tensors
                # CRITICAL: Only use exactly horizon_length items to avoid index errors
                # Stack creates [horizon_length, num_envs, ...] tensors
                states = torch.stack(self.buffer_states[:self.horizon_length])
                actions = torch.stack(self.buffer_actions[:self.horizon_length])
                rewards = torch.stack(self.buffer_rewards[:self.horizon_length])
                values = torch.stack(self.buffer_values[:self.horizon_length])
                log_probs = torch.stack(self.buffer_log_probs[:self.horizon_length])
                dones = torch.stack(self.buffer_dones[:self.horizon_length])
                action_means = torch.stack(self.buffer_action_means[:self.horizon_length])
                action_stds = torch.stack(self.buffer_action_stds[:self.horizon_length])
                
                # CRITICAL FIX: Reshape from [horizon_length, num_envs, ...] to [horizon_length * num_envs, ...]
                # This matches what rl_games expects for minibatch creation
                # Enable PPO_CYCLE_DEBUG logging if set
                import os
                ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
                
                if ppo_cycle_debug:
                    print(f"[PPO_CYCLE_DEBUG] Stacked shapes:")
                    print(f"  states: {states.shape}")
                    print(f"  actions: {actions.shape}")
                    print(f"  rewards: {rewards.shape}")
                    print(f"  values: {values.shape}")
                    print(f"  log_probs: {log_probs.shape}")
                    print(f"  dones: {dones.shape}")
                    print(f"  action_means: {action_means.shape}")
                    print(f"  action_stds: {action_stds.shape}")
                
                # Handle both possible shapes: [steps, features] or [steps, num_envs, features]
                if states.dim() == 2:
                    # Already flattened, probably single environment
                    batch_size = states.shape[0]
                    num_envs = 1
                else:
                    # Use actual number of steps collected, not horizon_length
                    num_steps = states.shape[0]
                    num_envs = states.shape[1]
                    batch_size = num_steps * num_envs
                
                # Apply swap_and_flatten01 pattern from rl_games
                # This transposes [horizon, envs, ...] to [envs, horizon, ...] then flattens to [envs*horizon, ...]
                def swap_and_flatten(tensor):
                    if tensor.dim() == 2:
                        # Already [horizon*envs, features]
                        return tensor
                    elif tensor.dim() == 3:
                        # [horizon, envs, features] -> [envs, horizon, features] -> [envs*horizon, features]
                        return tensor.transpose(0, 1).reshape(batch_size, -1)
                    elif tensor.dim() == 1:
                        # Special case for scalars that were incorrectly shaped
                        # This shouldn't happen but let's handle it
                        return tensor.unsqueeze(-1).expand(batch_size, 1).squeeze(-1)
                    else:
                        # For higher dims, just flatten after transpose
                        return tensor.transpose(0, 1).reshape(batch_size, *tensor.shape[2:])
                
                states = swap_and_flatten(states)
                actions = swap_and_flatten(actions)
                # For scalar tensors (rewards, values, etc), we need to flatten to 1D
                rewards = rewards.transpose(0, 1).reshape(-1)
                values = values.transpose(0, 1).reshape(-1)
                log_probs = log_probs.transpose(0, 1).reshape(-1)
                dones = dones.transpose(0, 1).reshape(-1)
                action_means = swap_and_flatten(action_means)
                action_stds = swap_and_flatten(action_stds)
                
                if ppo_cycle_debug:
                    print(f"[PPO_CYCLE_DEBUG] After swap_and_flatten:")
                    print(f"  states: {states.shape}")
                    print(f"  actions: {actions.shape}")
                    print(f"  rewards: {rewards.shape}")
                    print(f"  values: {values.shape}")
                    print(f"  log_probs: {log_probs.shape}")
                    print(f"  batch_size: {batch_size}")
                
                if self.fixed_seed_debug:
                    self.logger.info(f"[PPO Trainer Debug] Rewards: {rewards.tolist()}")
                    self.logger.info(f"[PPO Trainer Debug] Values: {values[:5].tolist()}")
                    self.logger.info(f"[PPO Trainer Debug] Dones: {dones.sum().item()} episodes completed")
                
                # Perform PPO training using rl_games components
                try:
                    if ppo_cycle_debug:
                        print(f"[PPO_CYCLE_DEBUG] Calling rlgames_ppo_update...")
                    # Pass the current state and done flag as bootstrap values
                    total_loss = self.rlgames_ppo_update(
                        states, actions, rewards, values, log_probs, dones, 
                        action_means, action_stds, model, state, done
                    )
                    if ppo_cycle_debug:
                        print(f"[PPO_CYCLE_DEBUG] rlgames_ppo_update completed! Loss: {total_loss.item()}")
                except Exception as e:
                    self.logger.error(f"Error in rlgames_ppo_update: {e}")
                    raise
                
                # Update step count and epoch count
                self.step_count += 1
                self.current_epoch += 1
                
                # Update PPO cycle count
                self.ppo_cycles_completed += 1
                if ppo_cycle_debug or self.stop_after_cycle:
                    self.logger.info(f"PPO cycle {self.ppo_cycles_completed} completed")
                
                # Check if we've reached max epochs
                if self.current_epoch >= self.max_epochs:
                    self.training_complete = True
                    # print(f"[DEBUG] PPOTrainerNode - Setting training_complete=True")
                    # print(f"[DEBUG] current_epoch={self.current_epoch}, max_epochs={self.max_epochs}")
                    self.logger.info(f"🎯 PPO Trainer reached max_epochs ({self.max_epochs}) - signaling completion")
                
                # Check if we've reached PPO cycle limit
                if self.stop_after_cycle and self.ppo_cycles_completed >= self.stop_after_cycle:
                    self.training_complete = True
                    self.logger.info(f"🎯 PPO Trainer reached PPO cycle limit ({self.stop_after_cycle}) - signaling completion")
                
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
                        metadata = {
                            'trigger_type': self.checkpoint_trigger_type,
                            'trigger_value': self.checkpoint_trigger_value,
                            'training_step': self.step_count,
                            'loss': current_loss,
                            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                            'hyperparameters': {
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
                            }
                        }
                        
                        self.checkpoint_manager.save_checkpoint(
                            model.state_dict(), metadata=metadata
                        )
                
                # Reset buffer (DNNE async coordination maintained)
                self.reset_buffer()
                
                # Create completion signal
                training_complete = {
                    "signal_type": "training_complete",
                    "step": self.step_count,
                    "loss": total_loss.item(),
                    "source_node": f"ppo_trainer_{self.node_id}"
                }
                
                self.logger.info(f"PPO training step {self.step_count} complete (rl_games), loss: {total_loss.item():.4f}")
                
                return {
                    "loss": total_loss,
                    "training_complete": training_complete
                }
            
            else:
                # Still collecting, return dummy outputs (DNNE async coordination maintained)
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
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # CRITICAL: Reset buffer even on error to prevent infinite growth
            self.reset_buffer()
            
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
    
    async def save_checkpoint_on_exit(self, exit_reason: str) -> bool:
        """Save checkpoint on exit if enabled (unchanged from original)"""
        if not self.checkpoint_enabled or not self.checkpoint_save_on_exit or not self.checkpoint_manager:
            return False
            
        try:
            import time
            
            metadata = {
                'exit_type': 'on_exit',
                'exit_reason': exit_reason,
                'timestamp': time.time(),
                'training_step': self.step_count,
                'max_epochs': self.max_epochs,
                'current_epoch': self.current_epoch,
                'horizon_length': self.horizon_length,
                'mini_epochs_num': self.mini_epochs_num,
                'minibatch_size': self.minibatch_size,
                'hyperparameters': {
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'e_clip': self.e_clip,
                    'critic_coef': self.critic_coef,
                    'entropy_coef': self.entropy_coef,
                    'learning_rate': self.learning_rate,
                    'grad_norm': self.grad_norm
                },
                'last_loss': self.last_loss,
                'rlgames_integration': True
            }
            
            success = self.checkpoint_manager.save_checkpoint(
                {}, metadata=metadata
            )
            
            if success:
                self.logger.info(f"💾 Exit checkpoint saved for rl_games PPOTrainer node {self.node_id}")
                return True
            else:
                self.logger.warning(f"⚠️ Failed to save exit checkpoint for rl_games PPOTrainer node {self.node_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving exit checkpoint: {e}")
            return False
