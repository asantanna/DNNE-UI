# rl_nodes.py
"""
Export handlers for RL (Reinforcement Learning) nodes
"""

from typing import Dict, List
from ..graph_exporter import ExportableNode

class PPOAgentExporter(ExportableNode):
    """Exporter for PPOAgentNode"""
    
    @classmethod
    def get_template_name(cls) -> str:
        return "nodes/ppo_agent_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, connections: Dict, node_registry=None, all_nodes=None, all_links=None) -> Dict:
        """Prepare template variables for PPOAgentNode"""
        
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'hidden_sizes', 'widget_index': 0, 'default': '64,64'},
            {'name': 'activation', 'widget_index': 1, 'default': 'elu'},
            {'name': 'action_space', 'widget_index': 2, 'default': 'continuous'},
            {'name': 'action_dim', 'widget_index': 3, 'default': 1},
            {'name': 'learning_rate', 'widget_index': 4, 'default': 3e-4},
            {'name': 'deterministic', 'widget_index': 5, 'default': False},
            {'name': 'init_log_std', 'widget_index': 6, 'default': 0.0}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Get parameter values
        hidden_sizes = params['hidden_sizes']
        activation = params['activation']
        action_space = params['action_space']
        action_dim = params['action_dim']
        learning_rate = params['learning_rate']
        deterministic = params['deterministic']
        init_log_std = params['init_log_std']
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "PPOAgentNode",
            "HIDDEN_SIZES": hidden_sizes,
            "ACTIVATION": activation,
            "ACTION_SPACE": action_space,
            "ACTION_DIM": action_dim,
            "LEARNING_RATE": learning_rate,
            "DETERMINISTIC": deterministic,
            "INIT_LOG_STD": init_log_std
        }
    
    @classmethod
    def get_imports(cls) -> List[str]:
        return [
            "import torch",
            "import torch.nn as nn",
            "import torch.distributions as dist",
            "import numpy as np"
        ]
    
    @classmethod
    def get_input_names(cls) -> List[str]:
        return ["observations"]
    
    @classmethod
    def get_output_names(cls) -> List[str]:
        return ["policy_output", "model"]

class PPOTrainerExporter(ExportableNode):
    """Exporter for PPOTrainerNode"""
    
    @classmethod
    def get_template_name(cls) -> str:
        return "nodes/ppo_trainer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, connections: Dict, node_registry=None, all_nodes=None, all_links=None) -> Dict:
        """Prepare template variables for PPOTrainerNode"""
        
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'horizon_length', 'widget_index': 0, 'default': 16},
            {'name': 'num_epochs', 'widget_index': 1, 'default': 4},
            {'name': 'minibatch_size', 'widget_index': 2, 'default': 32},
            {'name': 'gamma', 'widget_index': 3, 'default': 0.99},
            {'name': 'gae_lambda', 'widget_index': 4, 'default': 0.95},
            {'name': 'clip_param', 'widget_index': 5, 'default': 0.2},
            {'name': 'value_coef', 'widget_index': 6, 'default': 0.5},
            {'name': 'entropy_coef', 'widget_index': 7, 'default': 0.01},
            {'name': 'learning_rate', 'widget_index': 8, 'default': 0.0003},
            {'name': 'max_grad_norm', 'widget_index': 9, 'default': 0.5},
            {'name': 'checkpoint_enabled', 'widget_index': 10, 'default': True},
            {'name': 'checkpoint_trigger_type', 'widget_index': 11, 'default': 'time'},
            {'name': 'checkpoint_trigger_value', 'widget_index': 12, 'default': '5m'}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Extract parameter values
        horizon_length = params['horizon_length']
        num_epochs = params['num_epochs']
        minibatch_size = params['minibatch_size']
        gamma = params['gamma']
        gae_lambda = params['gae_lambda']
        clip_param = params['clip_param']
        value_coef = params['value_coef']
        entropy_coef = params['entropy_coef']
        learning_rate = params['learning_rate']
        max_grad_norm = params['max_grad_norm']
        checkpoint_enabled = params['checkpoint_enabled']
        checkpoint_trigger_type = params['checkpoint_trigger_type']
        checkpoint_trigger_value = params['checkpoint_trigger_value']
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "PPOTrainerNode",
            "HORIZON_LENGTH": horizon_length,
            "NUM_EPOCHS": num_epochs,
            "MINIBATCH_SIZE": minibatch_size,
            "GAMMA": gamma,
            "GAE_LAMBDA": gae_lambda,
            "CLIP_PARAM": clip_param,
            "VALUE_COEF": value_coef,
            "ENTROPY_COEF": entropy_coef,
            "LEARNING_RATE": learning_rate,
            "MAX_GRAD_NORM": max_grad_norm,
            "CHECKPOINT_ENABLED": checkpoint_enabled,
            "CHECKPOINT_TRIGGER_TYPE": checkpoint_trigger_type,
            "CHECKPOINT_TRIGGER_VALUE": checkpoint_trigger_value
        }
    
    @classmethod
    def get_imports(cls) -> List[str]:
        return [
            "import torch",
            "import torch.nn as nn",
            "import torch.optim as optim",
            "import torch.distributions as dist",
            "import numpy as np"
        ]
    
    @classmethod
    def get_input_names(cls) -> List[str]:
        return ["state", "policy_output", "reward", "done", "model"]
    
    @classmethod
    def get_output_names(cls) -> List[str]:
        return ["loss", "training_complete"]

# Registration function
def register_rl_exporters(exporter):
    """Register all RL node exporters"""
    exporter.register_node("PPOAgentNode", PPOAgentExporter)
    exporter.register_node("PPOTrainerNode", PPOTrainerExporter)

# Node type mapping for export system  
RL_NODE_EXPORTERS = {
    "PPOAgentNode": PPOAgentExporter,
    "PPOTrainerNode": PPOTrainerExporter,
}