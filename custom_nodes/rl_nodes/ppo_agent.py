# ppo_agent.py
"""
PPO Agent Node - Actor-Critic Network for PPO Algorithm
Combines policy and value networks in a single node for efficiency
"""

import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from inspect import cleandoc
from .rl_types import PolicyOutput

class PPOAgentNode:
    """
    PPO Agent Node - Actor-Critic Network
    
    Combines actor (policy) and critic (value) networks in a single node.
    Outputs actions, values, and log probabilities efficiently.
    """
    
    def __init__(self):
        pass
    
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observations": ("TENSOR", {"tooltip": "Environment observations tensor"}),
                "hidden_sizes": ("STRING", {"default": "64,64", "tooltip": "Comma-separated hidden layer sizes (e.g., '64,64' for two 64-unit layers)"}),
                "activation": (["relu", "tanh", "elu"], {"default": "elu", "tooltip": "Activation function for hidden layers"}),
                "action_space": (["continuous", "discrete"], {"default": "continuous", "tooltip": "Type of action space: continuous (real values) or discrete (integers)"}),
                "action_dim": ("INT", {"default": 1, "min": 1, "max": 10, "tooltip": "Dimension of action space (number of actions)"}),
                "learning_rate": ("FLOAT", {"default": 3e-4, "min": 1e-6, "max": 1e-1, "step": 1e-6, "tooltip": "Learning rate for the optimizer (3e-4 is standard for PPO)"}),
            },
            "optional": {
                "deterministic": ("BOOLEAN", {"default": False, "tooltip": "Use deterministic actions (no exploration noise)"}),
                "init_log_std": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1, "tooltip": "Initial log standard deviation for continuous action noise"}),
            }
        }
    
    RETURN_TYPES = ("POLICY_OUTPUT", "MODEL")
    RETURN_NAMES = ("policy_output", "model")
    
    FUNCTION = "forward"
    CATEGORY = "rl"
    DESCRIPTION = cleandoc(__doc__)
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def build_model(self, obs_dim, hidden_sizes_str, activation, action_space, action_dim, init_log_std):
        """Build the actor-critic network"""
        
        # Parse hidden sizes
        hidden_sizes = [int(x.strip()) for x in hidden_sizes_str.split(",")]
        
        # Get activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU
            
        # Build shared layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                act_fn()
            ])
            prev_size = hidden_size
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Build policy head
        if action_space == "continuous":
            self.policy_mean = nn.Linear(prev_size, action_dim)
            self.policy_log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)
        else:  # discrete
            self.policy_logits = nn.Linear(prev_size, action_dim)
            
        # Build value head
        self.value_head = nn.Linear(prev_size, 1)
        
        # Store configuration
        self.action_space = action_space
        self.action_dim = action_dim
        
        # Create the complete model
        self.model = nn.ModuleDict({
            'shared': self.shared_layers,
            'policy_mean': self.policy_mean if action_space == "continuous" else self.policy_logits,
            'value': self.value_head
        })
        
        if action_space == "continuous":
            self.model['policy_log_std'] = nn.ParameterDict({'log_std': self.policy_log_std})
            
        self.model.to(self.device)
        
        return self.model
        
    def forward(self, observations, hidden_sizes="64,64", activation="elu", 
                action_space="continuous", action_dim=1, learning_rate=3e-4,
                deterministic=False, init_log_std=0.0):
        """
        Forward pass through actor-critic network
        
        Args:
            node_id: Node ID widget (informational only)
            observations: Input state tensor [batch_size, obs_dim]
            hidden_sizes: Comma-separated hidden layer sizes
            activation: Activation function for hidden layers
            action_space: Type of action space (continuous/discrete)
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            deterministic: Use deterministic actions (no exploration)
            init_log_std: Initial log standard deviation for continuous actions
            
        Returns:
            policy_output: PolicyOutput containing action, value, log_prob
            model: PyTorch model for optimizer connection
        """
        
        # Ensure observations is on correct device
        if isinstance(observations, torch.Tensor):
            observations = observations.to(self.device)
        else:
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
            
        # Handle batch dimension
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
            
        batch_size, obs_dim = observations.shape
        
        # Build model if needed
        if self.model is None:
            self.build_model(obs_dim, hidden_sizes, activation, action_space, action_dim, init_log_std)
            
        # Forward pass through shared layers
        features = self.shared_layers(observations)
        
        # Compute value
        value = self.value_head(features)
        if single_sample:
            value = value.squeeze(0)  # Remove batch dimension for single sample
            
        # Compute policy output
        if action_space == "continuous":
            # Continuous action space - Gaussian policy
            action_mean = self.policy_mean(features)
            action_std = torch.exp(self.policy_log_std)
            
            # Create distribution
            policy_dist = dist.Normal(action_mean, action_std)
            
            # Sample action
            if deterministic:
                action = action_mean
            else:
                action = policy_dist.sample()
                
            # Compute log probability
            log_prob = policy_dist.log_prob(action).sum(dim=-1)
            
            # Store action parameters for later use
            action_params = torch.cat([action_mean, action_std.expand_as(action_mean)], dim=-1)
            
        else:
            # Discrete action space - Categorical policy
            action_logits = self.policy_logits(features)
            
            # Create distribution
            policy_dist = dist.Categorical(logits=action_logits)
            
            # Sample action
            if deterministic:
                action = torch.argmax(action_logits, dim=-1, keepdim=True)
            else:
                action = policy_dist.sample().unsqueeze(-1)
                
            # Compute log probability
            log_prob = policy_dist.log_prob(action.squeeze(-1))
            
            # Store action parameters
            action_params = torch.softmax(action_logits, dim=-1)
            
        # Remove batch dimension for single samples
        if single_sample:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            action_params = action_params.squeeze(0)
            
        # Create PolicyOutput
        policy_output = PolicyOutput(
            action=action,
            value=value,
            log_prob=log_prob,
            action_params=action_params
        )
        
        return (policy_output, self.model)

# Register the node
NODE_CLASS_MAPPINGS = {"PPOAgentNode": PPOAgentNode}
NODE_DISPLAY_NAME_MAPPINGS = {"PPOAgentNode": "PPO Agent (Actor-Critic)"}