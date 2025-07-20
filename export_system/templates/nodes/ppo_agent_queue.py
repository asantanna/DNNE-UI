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

# Import RunningMeanStd from rl_games_dnne (paths configured in runner.py)
from rl_games_dnne.dnne_exports import RunningMeanStd

# Import DNNE_print from centralized location
from isaacgymenvs.utils.debug_utils import DNNE_print

# Import rl_games components for A2CBuilder
from rl_games_dnne.algos_torch.network_builder import A2CBuilder
from rl_games_dnne.algos_torch.models import ModelA2CContinuousLogStd

class {CLASS_NAME}_{NODE_ID}(QueueNode):
    """PPO Agent Node using A2CBuilder - Actor-Critic Network for PPO Algorithm"""
    
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
        self.network = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Observation normalization
        self.obs_rms = None  # Will be initialized on first observation
        
        # Check if we're in inference mode
        import builtins
        self.inference_mode = getattr(builtins, 'INFERENCE_MODE', False)
        self.fixed_seed_debug = getattr(builtins, 'FIXED_SEED', None) is not None
        self.verbose = getattr(builtins, 'VERBOSE', False)
        
        # Enable PPO_CYCLE_DEBUG logging if set
        import os
        self.ppo_cycle_debug = os.environ.get('PPO_CYCLE_DEBUG', '0') == '1'
        
        # Initialize step counter
        self.step_count = 0
        
        self.logger.info(f"PPOAgentNode {{node_id}} initialized with A2CBuilder, action_space={{self.action_space}}, action_dim={{self.action_dim}}")
        if self.fixed_seed_debug:
            self.logger.info("🔍 Fixed seed debug mode enabled - will log all computation values")
        
    def build_model(self, obs_dim):
        """Build the actor-critic network using A2CBuilder"""
        import torch.nn as nn
        import torch.distributions as dist
        
        # Debug: Check current torch seed state
        if self.ppo_cycle_debug:
            # Get current RNG state to see if seed was set
            from isaacgymenvs.utils.debug_utils import DNNE_print
            DNNE_print("D", "PPO_INITIAL", f"Building model with torch seed state hash: {hash(torch.get_rng_state().cpu().numpy().tobytes())}")
        
        # Parse hidden sizes
        hidden_sizes = [int(x.strip()) for x in self.hidden_sizes.split(",")]
        
        # Create network configuration for A2CBuilder
        network_config = {
            'name': 'actor_critic',
            'separate': False,
            'space': {
                'continuous': {
                    'mu_activation': 'None',  # Must be string 'None', not Python None
                    'sigma_activation': 'None',  # Must be string 'None', not Python None
                    'mu_init': {'name': 'default'},  # Must be dict
                    'sigma_init': {'name': 'const_initializer', 'val': self.init_log_std},
                    'fixed_sigma': False
                }
            },
            'mlp': {
                'units': hidden_sizes,
                'activation': self.activation,
                'initializer': {'name': 'default'},  # Must be dict, not string
                'd2rl': False,
                'layer_norm': False
            },
            'value_activation': 'None',  # Must be string 'None', not Python None
            'normalization': None
        }
        
        # Create A2CBuilder instance
        network_builder = A2CBuilder()
        network_builder.load(network_config)
        
        # Build the network with required parameters
        self.network = network_builder.build('a2c', actions_num=self.action_dim, input_shape=(obs_dim,))
        
        # Create model wrapper using ModelA2CContinuousLogStd
        model_wrapper = ModelA2CContinuousLogStd(network_builder)
        
        # Build model configuration
        model_config = {
            'input_shape': (obs_dim,),
            'actions_num': self.action_dim,
            'num_seqs': 1,  # Not using RNN
            'value_size': 1,
            'normalize_value': False,  # We handle value normalization in trainer
            'normalize_input': False,  # We handle observation normalization here
            'num_agents': 1,
            'horizon_length': 16,  # Default, not used in forward pass
            'use_diagnostics': True,
            'discrete': False  # Continuous action space
        }
        
        # Build and get the model
        self.model = model_wrapper.build(model_config)
        self.model.to(self.device)
        
        # Log network details after building model (like IGE)
        if self.ppo_cycle_debug and self.step_count == 0:
            # Log observation normalization info
            DNNE_print("D", "PPO_INITIAL", "Obs normalization - mean: [0.0, 0.0, 0.0, 0.0]")
            DNNE_print("D", "PPO_INITIAL", "Obs normalization - var: [1.0, 1.0, 1.0, 1.0]") 
            DNNE_print("D", "PPO_INITIAL", "Obs normalization - count: 1.0")
            
            # Get actor network from A2CBuilder
            actor_mlp = self.model.a2c_network.actor_mlp
            if hasattr(actor_mlp, '0'):  # First layer in Sequential
                first_layer = actor_mlp[0]
                if hasattr(first_layer, 'weight'):
                    DNNE_print("D", "PPO_INITIAL", f"Actor first layer weights: {first_layer.weight[0][:4].tolist()}")
                    DNNE_print("D", "PPO_INITIAL", f"Actor first layer bias: {first_layer.bias[:4].tolist()}")
            
            # Log policy head weights
            if hasattr(self.model.a2c_network, 'mu'):
                mu_layer = self.model.a2c_network.mu
                DNNE_print("D", "PPO_INITIAL", f"Mu layer weights: {mu_layer.weight[0][:4].tolist()}")
                DNNE_print("D", "PPO_INITIAL", f"Mu layer bias: {mu_layer.bias.tolist()}")
        
        # Set to eval mode if in inference
        if self.inference_mode:
            self.model.eval()
            self.logger.info("PPO model set to evaluation mode for inference")
            
        return self.model
        
    async def compute(self, observations) -> Dict[str, Any]:
        """
        Forward pass through actor-critic network using A2CBuilder model
        
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
                    from isaacgymenvs.utils.debug_utils import DNNE_print
                    DNNE_print("D", "PPO_CYCLE", "=== PPO TRAINING CYCLE 1 START ===")
                    # Log initial observation details like IGE
                    first_obs = observations[0]
                    DNNE_print("D", "PPO_INITIAL", f"First observation: {first_obs[:4].tolist() if len(first_obs) >= 4 else first_obs.tolist()}")
                    DNNE_print("D", "PPO_INITIAL", f"Observation shape: {observations.shape}")
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
            
            # Prepare input dict for A2CBuilder model
            # During forward pass (collection phase), always use is_train=False
            # The trainer will use is_train=True when computing losses
            input_dict = {
                'obs': normalized_obs,
                'is_train': False,  # Always False during collection phase
                'prev_actions': None
            }
            
            # Forward pass through model
            result = self.model(input_dict)
            
            # Extract outputs from A2CBuilder model result
            action_mean = result['mus']
            action_std = result['sigmas']
            value = result['values'].squeeze(-1)  # Remove last dimension
            
            # Create distribution and sample action
            if self.action_space == "continuous":
                # Create normal distribution
                distr = torch.distributions.Normal(action_mean, action_std)
                
                # Sample action
                if self.deterministic:
                    action = action_mean
                else:
                    action = distr.sample()
                
                # Compute log probability
                log_prob = distr.log_prob(action).sum(dim=-1)
                
                # Increment step counter
                self.step_count += 1
                
                if self.fixed_seed_debug:
                    self.logger.info(f"[PPO Agent Debug] Action mean: {{action_mean[0].tolist()}}")
                    self.logger.info(f"[PPO Agent Debug] Action std: {{action_std[0].tolist()}}")
                    self.logger.info(f"[PPO Agent Debug] Sampled action: {{action[0].tolist()}}")
                    self.logger.info(f"[PPO Agent Debug] Log prob: {{log_prob[0].item()}}")
            else:
                # For discrete action spaces (not implemented in this template)
                raise NotImplementedError("Discrete action space not yet implemented with A2CBuilder")
                
            # Remove batch dimension for single samples
            if single_sample:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                action_mean = action_mean.squeeze(0)
                action_std = action_std.squeeze(0)
                value = value.squeeze(0)
                
            # Create PolicyOutput-like dictionary
            policy_output = {
                "action": action,
                "value": value,
                "log_prob": log_prob,
                "normalized_observations": normalized_obs,  # Include normalized observations for training
                "action_mean": action_mean,
                "action_std": action_std
            }
            
            # Return both policy output and model (for optimizer connection)
            return {{
                "policy_output": policy_output,
                "model": self.model
            }}
            
        except Exception as e:
            self.logger.error(f"Error in PPOAgentNode {{self.node_id}}: {{e}}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return safe defaults
            safe_action = torch.zeros(self.action_dim, device=self.device)
            safe_value = torch.tensor(0.0, device=self.device)
            safe_log_prob = torch.tensor(0.0, device=self.device)
            
            safe_policy_output = {
                "action": safe_action,
                "value": safe_value,
                "log_prob": safe_log_prob,
                "action_mean": safe_action,
                "action_std": torch.ones(self.action_dim, device=self.device)
            }
            
            return {
                "policy_output": safe_policy_output,
                "model": self.model if self.model is not None else nn.Module()
            }