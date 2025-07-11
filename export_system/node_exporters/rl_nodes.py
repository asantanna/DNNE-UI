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
        
        # Use standard inputs pattern like all other exporters
        inputs = node_data.get("inputs", {})
        
        # Validate that inputs exists
        if not inputs:
            raise ValueError(f"PPOTrainerNode {node_id}: missing inputs in node_data: {node_data}")
        
        # Required input keys for PPOTrainerNode
        required_keys = ["horizon_length", "num_epochs", "minibatch_size", "gamma", "gae_lambda", 
                        "clip_param", "value_coef", "entropy_coef", "learning_rate", "max_grad_norm",
                        "checkpoint_enabled", "checkpoint_trigger_type", "checkpoint_trigger_value"]
        missing_keys = [key for key in required_keys if key not in inputs]
        if missing_keys:
            raise ValueError(f"PPOTrainerNode {node_id}: missing required inputs {missing_keys} in node_data: {node_data}")
        
        # Get values from inputs (no fallbacks)
        horizon_length = inputs["horizon_length"]
        num_epochs = inputs["num_epochs"]
        minibatch_size = inputs["minibatch_size"]
        gamma = inputs["gamma"]
        gae_lambda = inputs["gae_lambda"]
        clip_param = inputs["clip_param"]
        value_coef = inputs["value_coef"]
        entropy_coef = inputs["entropy_coef"]
        learning_rate = inputs["learning_rate"]
        max_grad_norm = inputs["max_grad_norm"]
        checkpoint_enabled = inputs["checkpoint_enabled"]
        checkpoint_trigger_type = inputs["checkpoint_trigger_type"]
        checkpoint_trigger_value = inputs["checkpoint_trigger_value"]
        
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