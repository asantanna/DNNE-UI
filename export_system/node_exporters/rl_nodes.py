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
        
        # Extract widget values with defaults (ComfyUI workflow format uses widgets_values list)
        widget_values = node_data.get("widgets_values", ["64,64", "elu", "continuous", 1, 3e-4, False, 0.0])
        
        # Get widget values
        hidden_sizes = widget_values[0] if len(widget_values) > 0 else "64,64"
        activation = widget_values[1] if len(widget_values) > 1 else "elu"
        action_space = widget_values[2] if len(widget_values) > 2 else "continuous"
        action_dim = widget_values[3] if len(widget_values) > 3 else 1
        learning_rate = widget_values[4] if len(widget_values) > 4 else 3e-4
        deterministic = widget_values[5] if len(widget_values) > 5 else False
        init_log_std = widget_values[6] if len(widget_values) > 6 else 0.0
        
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
        
        # Extract widget values with defaults (ComfyUI workflow format uses widgets_values list)
        widget_values = node_data.get("widgets_values", [16, 4, 32, 0.99, 0.95, 0.2, 0.5, 0.01, 3e-4, 0.5])
        
        # Get widget values
        horizon_length = widget_values[0] if len(widget_values) > 0 else 16
        num_epochs = widget_values[1] if len(widget_values) > 1 else 4
        minibatch_size = widget_values[2] if len(widget_values) > 2 else 32
        gamma = widget_values[3] if len(widget_values) > 3 else 0.99
        gae_lambda = widget_values[4] if len(widget_values) > 4 else 0.95
        clip_param = widget_values[5] if len(widget_values) > 5 else 0.2
        value_coef = widget_values[6] if len(widget_values) > 6 else 0.5
        entropy_coef = widget_values[7] if len(widget_values) > 7 else 0.01
        learning_rate = widget_values[8] if len(widget_values) > 8 else 3e-4
        max_grad_norm = widget_values[9] if len(widget_values) > 9 else 0.5
        
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
            "MAX_GRAD_NORM": max_grad_norm
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