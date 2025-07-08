# rl_types.py
"""
Custom types for DNNE RL nodes
Defines specialized data types for reinforcement learning workflows
"""

import torch

# Define RL-specific types
RL_TYPES = {
    "PPO_BATCH": "PPO_BATCH",  # Complete trajectory batch for PPO training
    "POLICY_OUTPUT": "POLICY_OUTPUT",  # Combined actions, values, log_probs
}

def register_rl_types():
    """Register RL types with the DNNE type system"""
    try:
        # Import the main DNNE types system
        import nodes
        
        # Add our RL types to the global type registry
        for type_name, type_value in RL_TYPES.items():
            if hasattr(nodes, 'TYPES') and hasattr(nodes.TYPES, 'append'):
                if type_value not in nodes.TYPES:
                    nodes.TYPES.append(type_value)
            
        print(f"Registered {len(RL_TYPES)} RL types")
        
    except Exception as e:
        print(f"Warning: Could not register RL types: {e}")

class PPOBatch:
    """
    Container for PPO training batch data
    Contains complete trajectory information for PPO algorithm
    """
    def __init__(self, states, actions, rewards, values, log_probs, dones, advantages, returns):
        self.states = states          # [horizon_length, state_dim]
        self.actions = actions        # [horizon_length, action_dim] 
        self.rewards = rewards        # [horizon_length]
        self.values = values          # [horizon_length]
        self.log_probs = log_probs    # [horizon_length]
        self.dones = dones            # [horizon_length]
        self.advantages = advantages  # [horizon_length] - computed using GAE
        self.returns = returns        # [horizon_length] - advantages + values
        
        self.length = len(states)
        
    def to_device(self, device):
        """Move all tensors to specified device"""
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.log_probs = self.log_probs.to(device)
        self.dones = self.dones.to(device)
        self.advantages = self.advantages.to(device)
        self.returns = self.returns.to(device)
        return self

class PolicyOutput:
    """
    Container for policy network outputs
    Bundles action, value, and log_prob to minimize async queue operations
    """
    def __init__(self, action, value, log_prob, action_params=None):
        self.action = action          # Sampled action
        self.value = value            # Value estimate
        self.log_prob = log_prob      # Log probability of action
        self.action_params = action_params  # Raw policy parameters (mean, std for continuous)
        
    def to_device(self, device):
        """Move all tensors to specified device"""
        self.action = self.action.to(device)
        self.value = self.value.to(device)
        self.log_prob = self.log_prob.to(device)
        if self.action_params is not None:
            self.action_params = self.action_params.to(device)
        return self

def compute_gae_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) advantages
    
    Args:
        rewards: [horizon_length] reward tensor
        values: [horizon_length] value estimates  
        dones: [horizon_length] episode termination flags
        gamma: discount factor
        gae_lambda: GAE lambda parameter
        
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
            
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    
    # Returns = advantages + values
    returns = advantages + values
    
    return advantages, returns