# Template variables - replaced during export
template_vars = {
    "NODE_ID": "ppo_agent_1", 
    "CLASS_NAME": "PPOAgentNode",
    "HIDDEN_SIZES": "32,32",
    "ACTIVATION": "elu",
    "ACTION_SPACE": "continuous",
    "ACTION_DIM": 1,
    "LEARNING_RATE": 0.0003,
    "DETERMINISTIC": False,
    "INIT_LOG_STD": 0
}

# Import RunningMeanStd from rl_games_dnne
import sys
sys.path.append('/home/asantanna/DNNE-LINUX-SUPPORT')
from rl_games_dnne.dnne_exports import RunningMeanStd

# Debug print function for consistent logging
def DNNE_print(message):
    """Print with [DNNE_DEBUG] prefix for easy grep filtering"""
    print(f"[DNNE_DEBUG] {{message}}")

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """PPO Agent Node - Actor-Critic Network for PPO Algorithm"""
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.setup_inputs(required=["observations"])
        self.setup_outputs(["policy_output", "model"])
        
        # Configuration from template
        self.hidden_sizes = "{HIDDEN_SIZES}"
        self.activation = "{ACTIVATION}"
        self.action_space = "{ACTION_SPACE}"
        self.action_dim = {ACTION_DIM}
        self.learning_rate = {LEARNING_RATE}
        self.deterministic = {DETERMINISTIC}
        self.init_log_std = {INIT_LOG_STD}
        
        # Model state
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Observation normalization
        self.obs_rms = None  # Will be initialized on first observation
        
        # Check if we're in inference mode
        import builtins
        self.inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        self.fixed_seed_debug = getattr(builtins, 'FIXED_SEED', None) is not None
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        
        # Initialize step counter
        self.step_count = 0
        
        self.logger.info(f"PPOAgentNode {{node_id}} initialized with action_space={{self.action_space}}, action_dim={{self.action_dim}}")
        if self.fixed_seed_debug:
            self.logger.info("🔍 Fixed seed debug mode enabled - will log all computation values")
        
    def build_model(self, obs_dim):
        """Build the actor-critic network"""
        import torch.nn as nn
        import torch.distributions as dist
        
        # Parse hidden sizes
        hidden_sizes = [int(x.strip()) for x in self.hidden_sizes.split(",")]
        
        # Get activation function
        if self.activation == "relu":
            act_fn = nn.ReLU
        elif self.activation == "tanh":
            act_fn = nn.Tanh
        elif self.activation == "elu":
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
        if self.action_space == "continuous":
            self.policy_mean = nn.Linear(prev_size, self.action_dim)
            self.policy_log_std = nn.Parameter(torch.ones(self.action_dim) * self.init_log_std)
        else:  # discrete
            self.policy_logits = nn.Linear(prev_size, self.action_dim)
            
        # Build value head
        self.value_head = nn.Linear(prev_size, 1)
        
        # Create the complete model
        self.model = nn.ModuleDict({
            'shared': self.shared_layers,
            'policy_mean': self.policy_mean if self.action_space == "continuous" else self.policy_logits,
            'value': self.value_head
        })
        
        if self.action_space == "continuous":
            self.model['policy_log_std'] = nn.ParameterDict({'log_std': self.policy_log_std})
            
        self.model.to(self.device)
        
        # Set to eval mode if in inference
        if self.inference_mode:
            self.model.eval()
            self.logger.info("PPO model set to evaluation mode for inference")
            
        return self.model
        
    async def compute(self, observations) -> Dict[str, Any]:
        """
        Forward pass through actor-critic network
        
        Args:
            observations: Input state tensor [batch_size, obs_dim]
            
        Returns:
            policy_output: PolicyOutput containing action, value, log_prob
            model: PyTorch model for optimizer connection
        """
        
        import torch
        import torch.nn as nn
        import torch.distributions as dist
        
        try:
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
                if self.ppo_cycle_debug and self.step_count == 0:
                    DNNE_print("=== NETWORK INITIALIZATION ===\n")
                self.build_model(obs_dim)
                if self.fixed_seed_debug:
                    # Log initial model weights
                    self.logger.info("[PPO Agent Debug] Initial model weights:")
                    for name, param in self.model.named_parameters():
                        if param.numel() < 10:
                            self.logger.info(f"  {{name}}: {{param.data.tolist()}}")
                        else:
                            self.logger.info(f"  {{name}}: shape={{param.shape}}, first 5={{param.data.flatten()[:5].tolist()}}")
                
            # Initialize observation normalization on first call
            if self.obs_rms is None:
                self.obs_rms = RunningMeanStd(obs_dim, device=self.device)
                self.logger.info(f"Initialized observation normalization for {{obs_dim}} features")
                
            # Update statistics only in training mode
            if not self.inference_mode:
                self.obs_rms.update(observations)
                
            # Normalize observations
            normalized_obs = self.obs_rms.normalize(observations)
            
            # Debug logging for fixed seed mode
            if self.fixed_seed_debug:
                self.logger.info(f"[PPO Agent Debug] Raw observations (first 5): {{observations[0][:5].tolist()}}")
                self.logger.info(f"[PPO Agent Debug] Obs mean: {{self.obs_rms.mean[:5].tolist()}}")
                self.logger.info(f"[PPO Agent Debug] Obs var: {{self.obs_rms.var[:5].tolist()}}")
                self.logger.info(f"[PPO Agent Debug] Normalized obs (first 5): {{normalized_obs[0][:5].tolist()}}")
                
            # Forward pass through shared layers
            features = self.shared_layers(normalized_obs)
            
            # Compute value
            value = self.value_head(features)
            value = value.squeeze(1)  # Remove second dimension: [batch_size, 1] -> [batch_size]
            
            if self.fixed_seed_debug:
                self.logger.info(f"[PPO Agent Debug] Features shape: {{features.shape}}")
                self.logger.info(f"[PPO Agent Debug] Features (first 5): {{features[0][:5].tolist()}}")
                self.logger.info(f"[PPO Agent Debug] Value output: {{value[0].item()}}")
                
            # Compute policy output
            if self.action_space == "continuous":
                # Continuous action space - Gaussian policy
                action_mean = self.policy_mean(features)
                # CRITICAL FIX: Expand action_std to match batch dimension
                action_std = torch.exp(self.policy_log_std).expand_as(action_mean)
                
                # Create distribution
                policy_dist = dist.Normal(action_mean, action_std)
                
                # Sample action
                if self.deterministic:
                    action = action_mean
                else:
                    action = policy_dist.sample()
                    
                # Compute log probability
                log_prob = policy_dist.log_prob(action).sum(dim=-1)
                
                # Increment step counter
                self.step_count += 1
                
                # Store action parameters separately for PPO trainer
                # action_params = torch.cat([action_mean, action_std.expand_as(action_mean)], dim=-1)  # Old format
                
                if self.fixed_seed_debug:
                    self.logger.info(f"[PPO Agent Debug] Action mean: {{action_mean[0].tolist()}}")
                    self.logger.info(f"[PPO Agent Debug] Action std: {{action_std.tolist()}}")
                    self.logger.info(f"[PPO Agent Debug] Sampled action: {{action[0].tolist()}}")
                    self.logger.info(f"[PPO Agent Debug] Log prob: {{log_prob[0].item()}}")
                
            else:
                # Discrete action space - Categorical policy
                action_logits = self.policy_logits(features)
                
                # Create distribution
                policy_dist = dist.Categorical(logits=action_logits)
                
                # Sample action
                if self.deterministic:
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
                if self.action_space == "continuous":
                    action_mean = action_mean.squeeze(0)
                    action_std = action_std.squeeze(0)
                else:
                    action_params = action_params.squeeze(0)
                
            # Create PolicyOutput-like dictionary
            policy_output = {
                "action": action,
                "value": value,
                "log_prob": log_prob,
                "normalized_observations": normalized_obs  # Include normalized observations for training
            }
            
            # Add action parameters based on action space
            if self.action_space == "continuous":
                policy_output["action_mean"] = action_mean
                policy_output["action_std"] = action_std
            else:
                policy_output["action_params"] = action_params
            
            # Debug logging removed - was causing 95% performance overhead due to tensor string formatting
            # If you need to debug, use: self.logger.debug(f"Forward pass complete, batch_size={{batch_size}}")
            
            return {{
                "policy_output": policy_output,
                "model": self.model
            }}
            
        except Exception as e:
            self.logger.error(f"Error in PPOAgentNode {{self.node_id}}: {{e}}")
            
            # Return safe defaults
            safe_action = torch.zeros(self.action_dim, device=self.device)
            safe_value = torch.tensor(0.0, device=self.device)
            safe_log_prob = torch.tensor(0.0, device=self.device)
            
            safe_policy_output = {
                "action": safe_action,
                "value": safe_value,
                "log_prob": safe_log_prob,
                "action_params": safe_action
            }
            
            return {
                "policy_output": safe_policy_output,
                "model": self.model if self.model is not None else nn.Module()
            }
