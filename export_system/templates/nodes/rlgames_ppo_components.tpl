"""
rl_games PPO Components - Surgical Extraction for DNNE
Extracted from rl_games to provide identical PPO implementation while maintaining DNNE's async architecture.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple


class RLGamesPPOComponents:
    """
    Surgical extraction of rl_games PPO components for DNNE integration.
    Provides identical PPO implementation while maintaining async queue coordination.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with rl_games-compatible configuration
        
        Args:
            config: Configuration dict with rl_games parameter names
        """
        # Map DNNE parameters to rl_games naming conventions
        self.e_clip = config.get('e_clip', config.get('clip_param', 0.2))
        self.critic_coef = config.get('critic_coef', config.get('value_coef', 4.0))
        self.entropy_coef = config.get('entropy_coef', 0.0)
        self.tau = config.get('tau', config.get('gae_lambda', 0.95))
        self.gamma = config.get('gamma', 0.99)
        self.grad_norm = config.get('grad_norm', config.get('max_grad_norm', 1.0))
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.mini_epochs_num = config.get('mini_epochs_num', config.get('ppo_epochs', 8))
        self.minibatch_size = config.get('minibatch_size', 8192)
        self.horizon_length = config.get('horizon_length', 16)
        self.clip_value = config.get('clip_value', True)
        self.bounds_loss_coef = config.get('bounds_loss_coef', 0.0001)
        self.bound_loss_type = config.get('bound_loss_type', 'bound')
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.last_lr = self.learning_rate
        self.ppo = True  # Always PPO mode
        
    def discount_values(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute GAE advantages using rl_games implementation
        Extracted from rl_games.common.a2c_common.discount_values()
        
        Args:
            rewards: [horizon_length, num_envs] reward tensor
            values: [horizon_length, num_envs] value estimates  
            dones: [horizon_length, num_envs] episode termination flags
            
        Returns:
            advantages: [horizon_length, num_envs] GAE advantages
        """
        horizon_length, num_envs = rewards.shape
        lastgaelam = torch.zeros(num_envs, device=rewards.device)
        mb_advs = torch.zeros_like(rewards)
        
        for t in reversed(range(horizon_length)):
            if t == horizon_length - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = torch.zeros_like(values[t])  # Assume episode ends
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            
            # Compute temporal difference
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            
            # Update GAE advantage
            lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
            mb_advs[t] = lastgaelam
        
        return mb_advs
    
    def actor_loss(self, old_action_neglog_probs: torch.Tensor, 
                   action_neglog_probs: torch.Tensor, 
                   advantage: torch.Tensor) -> torch.Tensor:
        """
        PPO clipped actor loss - extracted from rl_games.common.common_losses.actor_loss()
        
        Args:
            old_action_neglog_probs: Previous negative log probabilities
            action_neglog_probs: Current negative log probabilities  
            advantage: GAE advantages
            
        Returns:
            actor_loss: PPO clipped actor loss
        """
        ratio = torch.exp(old_action_neglog_probs - action_neglog_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
        a_loss = torch.max(-surr1, -surr2)
        return a_loss
    
    def critic_loss(self, value_preds_batch: torch.Tensor, 
                    values: torch.Tensor, 
                    return_batch: torch.Tensor) -> torch.Tensor:
        """
        PPO value function loss with clipping - extracted from rl_games.common.common_losses.critic_loss()
        
        Args:
            value_preds_batch: Previous value predictions
            values: Current value predictions
            return_batch: Target returns
            
        Returns:
            critic_loss: Value function loss
        """
        if self.clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-self.e_clip, self.e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2
        return c_loss
    
    def bound_loss(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Action bound regularization loss - extracted from rl_games a2c_continuous.bound_loss()
        
        Args:
            mu: Action means
            
        Returns:
            bound_loss: Regularization loss for action bounds
        """
        if self.bounds_loss_coef is not None and self.bounds_loss_coef > 0:
            if self.bound_loss_type == 'regularisation':
                b_loss = (mu * mu).sum(axis=-1)
            elif self.bound_loss_type == 'bound':
                soft_bound = 1.1
                mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
                mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
            else:
                b_loss = torch.zeros_like(mu[..., 0])
        else:
            b_loss = torch.zeros_like(mu[..., 0])
        return b_loss
    
    def policy_kl(self, mu: torch.Tensor, sigma: torch.Tensor, 
                  old_mu: torch.Tensor, old_sigma: torch.Tensor) -> torch.Tensor:
        """
        KL divergence between old and new policies for continuous actions
        Extracted from rl_games.algos_torch.torch_ext.policy_kl()
        
        Args:
            mu, sigma: Current policy parameters
            old_mu, old_sigma: Previous policy parameters
            
        Returns:
            kl_divergence: KL divergence between policies
        """
        # KL divergence for Gaussian distributions
        # KL(N(mu1, sigma1) || N(mu2, sigma2)) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/(2*sigma2^2) - 1/2
        kl = torch.log(old_sigma / sigma) + (sigma**2 + (mu - old_mu)**2) / (2 * old_sigma**2) - 0.5
        return kl.sum(dim=-1)
    
    def calc_gradients(self, input_dict: Dict[str, torch.Tensor], 
                       model: Dict[str, nn.Module]) -> Tuple[torch.Tensor, ...]:
        """
        Core PPO gradient calculation - adapted from rl_games a2c_continuous.calc_gradients()
        
        Args:
            input_dict: Dictionary containing training data
            model: Dictionary of PyTorch model components
            
        Returns:
            train_result: Tuple of (a_loss, c_loss, entropy, kl_dist, lr, lr_mul, mu, sigma, b_loss)
        """
        # Extract input data
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        
        # Ensure tensors are on correct device
        obs_batch = obs_batch.to(self.device)
        actions_batch = actions_batch.to(self.device)
        advantage = advantage.to(self.device)
        
        # Normalize advantages
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # Forward pass through model
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs': obs_batch,
        }
        
        # Get model outputs - handle DNNE ModuleDict format
        with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for simplicity
            if isinstance(model, nn.ModuleDict) and 'shared' in model:
                # DNNE ModuleDict format - the standard DNNE PPO model
                features = model['shared'](obs_batch)
                values = model['value'](features).squeeze(-1)
                
                if 'policy_log_std' in model:
                    # Continuous action space
                    mu = model['policy_mean'](features)
                    sigma = torch.exp(model['policy_log_std']['log_std']).expand_as(mu)
                    
                    # Create policy distribution
                    policy_dist = torch.distributions.Normal(mu, sigma)
                    action_log_probs = policy_dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = policy_dist.entropy().sum(dim=-1)
                else:
                    # Discrete action space
                    action_logits = model['policy_mean'](features)
                    policy_dist = torch.distributions.Categorical(logits=action_logits)
                    action_log_probs = policy_dist.log_prob(actions_batch.squeeze(-1))
                    entropy = policy_dist.entropy()
                    mu = action_logits  # For consistency
                    sigma = torch.ones_like(mu)
            elif hasattr(model, 'actor') and hasattr(model, 'critic'):
                # Alternative DNNE PPO model format with .actor/.critic attributes
                actor_output = model.actor(obs_batch)
                values = model.critic(obs_batch).squeeze(-1)
                
                # Handle action distribution based on action space
                if hasattr(model.actor, 'log_std'):
                    # Continuous action space
                    mu = actor_output
                    log_std = model.actor.log_std.expand_as(mu)
                    sigma = torch.exp(log_std)
                    
                    # Create policy distribution
                    policy_dist = torch.distributions.Normal(mu, sigma)
                    action_log_probs = policy_dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = policy_dist.entropy().sum(dim=-1)
                else:
                    # Discrete action space
                    action_logits = actor_output
                    policy_dist = torch.distributions.Categorical(logits=action_logits)
                    action_log_probs = policy_dist.log_prob(actions_batch.squeeze(-1))
                    entropy = policy_dist.entropy()
                    mu = action_logits  # For consistency
                    sigma = torch.ones_like(mu)
            elif isinstance(model, dict) and 'shared' in model:
                # Pure dict format (rl_games style)
                features = model['shared'](obs_batch)
                values = model['value'](features).squeeze(-1)
                
                if 'policy_log_std' in model:
                    # Continuous action space
                    mu = model['policy_mean'](features)
                    sigma = torch.exp(model['policy_log_std']['log_std'])
                    
                    # Create policy distribution
                    policy_dist = torch.distributions.Normal(mu, sigma)
                    action_log_probs = policy_dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = policy_dist.entropy().sum(dim=-1)
                else:
                    # Discrete action space
                    action_logits = model['policy_mean'](features)
                    policy_dist = torch.distributions.Categorical(logits=action_logits)
                    action_log_probs = policy_dist.log_prob(actions_batch.squeeze(-1))
                    entropy = policy_dist.entropy()
                    mu = action_logits  # For consistency
                    sigma = torch.ones_like(mu)
            else:
                raise ValueError(f"Model format not recognized - got type: {type(model)}, keys: {getattr(model, 'keys', lambda: 'N/A')()}")
        
        # Convert log_probs to negative log_probs (rl_games convention)
        action_neglog_probs = -action_log_probs
        old_action_neglog_probs = -old_action_log_probs_batch
        
        # Compute losses
        a_loss = self.actor_loss(old_action_neglog_probs, action_neglog_probs, advantage)
        c_loss = self.critic_loss(value_preds_batch, values, return_batch)
        b_loss = self.bound_loss(mu)
        
        # Total loss
        loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
        
        # Compute KL divergence
        with torch.no_grad():
            kl_dist = self.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch)
            kl_dist = kl_dist.mean()
        
        lr_mul = 1.0
        
        # Return rl_games compatible result tuple
        train_result = (
            a_loss.mean(),      # actor loss
            c_loss.mean(),      # critic loss  
            entropy.mean(),     # entropy
            kl_dist,            # kl divergence
            self.last_lr,       # learning rate
            lr_mul,             # lr multiplier
            mu.detach(),        # action means
            sigma.detach(),     # action stds
            b_loss.mean()       # bound loss
        )
        
        return train_result, loss.mean()
    
    def train_actor_critic(self, input_dict: Dict[str, torch.Tensor], 
                          model: Dict[str, nn.Module]) -> Tuple[torch.Tensor, ...]:
        """
        PPO training step - wrapper around calc_gradients
        Maintains rl_games interface compatibility
        
        Args:
            input_dict: Training data dictionary
            model: PyTorch model components
            
        Returns:
            train_result: Training metrics tuple
        """
        train_result, loss = self.calc_gradients(input_dict, model)
        return train_result, loss