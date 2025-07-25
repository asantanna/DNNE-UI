# rl_nodes.py
"""
Export handlers for RL (Reinforcement Learning) nodes
"""

from typing import Dict, List
from ..graph_exporter import ExportableNode

# Old PPO exporters - replaced with new virtual node system
class PPOAgentExporter_OLD(ExportableNode):
    """Exporter for PPOAgentNode"""
    
    @classmethod
    def get_template_name(cls) -> str:
        return "nodes/ppo_agent_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, connections: Dict, node_registry=None, all_nodes=None, all_links=None) -> Dict:
        """Prepare template variables for PPOAgentNode"""
        
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'hidden_sizes', 'widget_index': 0, 'default': '32,32'},  # Updated to match IsaacGymEnvs
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

class PPOTrainerExporter_OLD(ExportableNode):
    """Exporter for PPOTrainerNode"""
    
    @classmethod
    def get_template_name(cls) -> str:
        return "nodes/ppo_trainer_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id: str, node_data: Dict, connections: Dict, node_registry=None, all_nodes=None, all_links=None) -> Dict:
        """Prepare template variables for PPOTrainerNode"""
        
        # Use universal parameter reader for consistent data access
        # New widget order: max_epochs first, then horizon_length, ppo_epochs, etc.
        param_specs = [
            {'name': 'max_epochs', 'widget_index': 0, 'default': 100},  # Max training epochs
            {'name': 'horizon_length', 'widget_index': 1, 'default': 16},
            {'name': 'ppo_epochs', 'widget_index': 2, 'default': 8},  # PPO mini-epochs per trajectory
            {'name': 'minibatch_size', 'widget_index': 3, 'default': 8192},  # Updated for batch efficiency
            {'name': 'gamma', 'widget_index': 4, 'default': 0.99},
            {'name': 'gae_lambda', 'widget_index': 5, 'default': 0.95},
            {'name': 'clip_param', 'widget_index': 6, 'default': 0.2},
            {'name': 'value_coef', 'widget_index': 7, 'default': 4.0},  # Updated to match IsaacGymEnvs
            {'name': 'entropy_coef', 'widget_index': 8, 'default': 0.0},  # Updated to match IsaacGymEnvs
            {'name': 'learning_rate', 'widget_index': 9, 'default': 0.0003},
            {'name': 'max_grad_norm', 'widget_index': 10, 'default': 1.0},  # Updated to match IsaacGymEnvs
            {'name': 'checkpoint_enabled', 'widget_index': 11, 'default': True},
            {'name': 'checkpoint_trigger_type', 'widget_index': 12, 'default': 'time'},
            {'name': 'checkpoint_trigger_value', 'widget_index': 13, 'default': '5m'}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Extract parameter values
        max_epochs = params['max_epochs']
        horizon_length = params['horizon_length']
        ppo_epochs = params['ppo_epochs']
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
            "MAX_EPOCHS": max_epochs,
            "HORIZON_LENGTH": horizon_length,
            # rl_games compatible parameter names
            "MINI_EPOCHS_NUM": ppo_epochs,  # rl_games naming
            "MINIBATCH_SIZE": minibatch_size,
            "GAMMA": gamma,
            "TAU": gae_lambda,  # rl_games naming: tau instead of gae_lambda
            "E_CLIP": clip_param,  # rl_games naming: e_clip instead of clip_param
            "CRITIC_COEF": value_coef,  # rl_games naming: critic_coef instead of value_coef
            "ENTROPY_COEF": entropy_coef,
            "LEARNING_RATE": learning_rate,
            "GRAD_NORM": max_grad_norm,  # rl_games naming: grad_norm instead of max_grad_norm
            "CLIP_VALUE": True,  # rl_games parameter
            "BOUNDS_LOSS_COEF": 0.0001,  # rl_games parameter
            "BOUND_LOSS_TYPE": "bound",  # rl_games parameter
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
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """Return list of dependency files that need to be copied to export"""
        return ["rlgames_ppo_components.py"]

# New PPO exporters with virtual node support

class PPOConfigExporter(ExportableNode):
    """Exporter for PPO configuration virtual node"""
    
    @classmethod
    def is_virtual(cls):
        """PPOConfig is a virtual node - only provides configuration"""
        return True
    
    @classmethod
    def get_template_name(cls):
        # Virtual nodes don't need templates
        return None
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Virtual nodes don't generate code
        return {}
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["config"]
    
    @classmethod
    def get_input_names(cls):
        return []


class PPOAgentExporter(ExportableNode):
    """Exporter for PPO Agent node - the main RL training node"""
    
    @classmethod
    def is_virtual(cls):
        """PPOAgent is NOT virtual - it generates the actual training code"""
        return False
    
    @classmethod
    def get_template_name(cls):
        return "nodes/ppo_agent_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Get node parameters
        param_specs = [
            {'name': 'network_mlp_layers', 'widget_index': 0, 'default': "[256, 128, 64]"},
            {'name': 'network_activation', 'widget_index': 1, 'default': "elu"},
            {'name': 'separate_value_network', 'widget_index': 2, 'default': False},
            {'name': 'checkpoint_interval', 'widget_index': 3, 'default': 100},
            {'name': 'keep_checkpoints', 'widget_index': 4, 'default': 5},
            {'name': 'load_checkpoint', 'widget_index': 5, 'default': ""},
            {'name': 'log_interval', 'widget_index': 6, 'default': 10},
            {'name': 'save_interval', 'widget_index': 7, 'default': 1000},
            {'name': 'experiment_name', 'widget_index': 8, 'default': "PPO_DNNE"},
            {'name': 'mixed_precision', 'widget_index': 9, 'default': False},
            {'name': 'multi_gpu', 'widget_index': 10, 'default': False},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Extract configuration from connected virtual nodes
        env_config = cls._extract_env_config(node_id, all_nodes, all_links)
        ppo_config = cls._extract_ppo_config(node_id, all_nodes, all_links)
        
        # Merge all configuration
        template_vars = {
            "NODE_ID": node_id,
            "CLASS_NAME": "PPOAgentNode",
            "NETWORK_MLP_LAYERS": params['network_mlp_layers'],
            "NETWORK_ACTIVATION": params['network_activation'],
            "SEPARATE_VALUE_NETWORK": params['separate_value_network'],
            "CHECKPOINT_INTERVAL": params['checkpoint_interval'],
            "KEEP_CHECKPOINTS": params['keep_checkpoints'],
            "LOAD_CHECKPOINT": params['load_checkpoint'],
            "LOG_INTERVAL": params['log_interval'],
            "SAVE_INTERVAL": params['save_interval'],
            "EXPERIMENT_NAME": params['experiment_name'],
            "MIXED_PRECISION": params['mixed_precision'],
            "MULTI_GPU": params['multi_gpu'],
        }
        
        # Add environment configuration
        if env_config:
            template_vars.update({
                "ENV_TASK": env_config.get('task', 'Cartpole'),
                "ENV_NUM_ENVS": env_config.get('num_envs', 64),
                "ENV_SEED": env_config.get('seed', 42),
                "ENV_HEADLESS": env_config.get('headless', True),
                "ENV_GRAPHICS_DEVICE": env_config.get('graphics_device_id', 0),
                "ENV_SIM_DEVICE": env_config.get('sim_device', 'cuda:0'),
                "ENV_PHYSICS_ENGINE": env_config.get('physics_engine', 'physx'),
                "ENV_MULTI_GPU": env_config.get('multi_gpu', False),
                "ENV_ENABLE_CAMERAS": env_config.get('enable_cameras', False),
                "ENV_FORCE_RENDER": env_config.get('force_render', False),
                "ENV_USE_GPU_PIPELINE": env_config.get('use_gpu_pipeline', True),
                "ENV_NUM_THREADS": env_config.get('num_threads', 0),
                "ENV_SOLVER_TYPE": env_config.get('solver_type', 1),
                "ENV_NUM_SUBSCENES": env_config.get('num_subscenes', 0),
                "ISAAC_GYM_ENVS_PATH": env_config.get('isaac_gym_envs_path', '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs'),
            })
        
        # Add PPO configuration
        if ppo_config:
            template_vars.update({
                "PPO_MINIBATCH_SIZE": ppo_config.get('minibatch_size', 64),
                "PPO_HORIZON_LENGTH": ppo_config.get('horizon_length', 16),
                "PPO_LEARNING_RATE": ppo_config.get('learning_rate', 0.0003),
                "PPO_SCHEDULE_TYPE": ppo_config.get('schedule_type', 'adaptive'),
                "PPO_GAMMA": ppo_config.get('gamma', 0.99),
                "PPO_TAU": ppo_config.get('tau', 0.95),
                "PPO_E_CLIP": ppo_config.get('e_clip', 0.2),
                "PPO_CLIP_VALUE": ppo_config.get('clip_value', True),
                "PPO_MINI_EPOCHS": ppo_config.get('mini_epochs', 5),
                "PPO_CRITIC_COEF": ppo_config.get('critic_coef', 4),
                "PPO_ENTROPY_COEF": ppo_config.get('entropy_coef', 0.0),
                "PPO_BOUNDS_LOSS_COEF": ppo_config.get('bounds_loss_coef', 0.0),
                "PPO_MAX_AGENT_STEPS": ppo_config.get('max_agent_steps', 1e8),
                "PPO_NORMALIZE_ADVANTAGE": ppo_config.get('normalize_advantage', True),
                "PPO_NORMALIZE_INPUT": ppo_config.get('normalize_input', True),
                "PPO_VALUE_BOOTSTRAP": ppo_config.get('value_bootstrap', True),
                "PPO_CLIP_ACTIONS": ppo_config.get('clip_actions', False),
            })
        
        return template_vars
    
    @classmethod
    def get_imports(cls):
        # These imports are already in the template, so return empty list
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["metrics"]
    
    @classmethod  
    def get_input_names(cls):
        # PPOAgent doesn't have inputs since it consolidates virtual nodes
        return []
    
    @classmethod
    def _extract_env_config(cls, ppo_node_id, all_nodes, all_links):
        """Extract environment configuration from connected IsaacGymEnvs virtual node"""
        if not all_links or not all_nodes:
            return None
            
        # Find the env input connection (slot 0)
        env_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == ppo_node_id and to_slot == 0:  # env input
                    env_node_id = str(link[1])
                    break
        
        if not env_node_id:
            return None
            
        # Find the node data
        env_node_data = None
        for node in all_nodes:
            if str(node["id"]) == env_node_id:
                env_node_data = node
                break
                
        if not env_node_data:
            return None
            
        # Check if it's an IsaacGymEnvs node
        node_type = env_node_data.get("class_type") or env_node_data.get("type")
        if node_type != "IsaacGymEnvs":
            return None
            
        # Extract widget values
        widget_values = env_node_data.get("widgets_values", [])
        
        # Map widget values to config (based on IsaacGymEnvs node definition)
        # Widget values from workflow: ["Cartpole",64,42,"randomize",true,0,"cuda:0","physx",false,false,false,true,0,1,0]
        # The "randomize" value at index 3 seems to be an extra widget not in the node definition
        # Shifting indices by 1 after that point
        return {
            'task': widget_values[0] if len(widget_values) > 0 else 'Cartpole',
            'num_envs': widget_values[1] if len(widget_values) > 1 else 64,
            'seed': widget_values[2] if len(widget_values) > 2 else 42,
            'headless': widget_values[4] if len(widget_values) > 4 else True,  # Skip index 3 ("randomize")
            'graphics_device_id': widget_values[5] if len(widget_values) > 5 else 0,
            'sim_device': widget_values[6] if len(widget_values) > 6 else 'cuda:0',
            'physics_engine': widget_values[7] if len(widget_values) > 7 else 'physx',
            'multi_gpu': widget_values[8] if len(widget_values) > 8 else False,
            'enable_cameras': widget_values[9] if len(widget_values) > 9 else False,
            'force_render': widget_values[10] if len(widget_values) > 10 else False,
            'use_gpu_pipeline': widget_values[11] if len(widget_values) > 11 else True,
            'num_threads': widget_values[12] if len(widget_values) > 12 else 0,
            'solver_type': widget_values[13] if len(widget_values) > 13 else 1,
            'num_subscenes': widget_values[14] if len(widget_values) > 14 else 0,
            'isaac_gym_envs_path': '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs',  # Always use default path
        }
    
    @classmethod
    def _extract_ppo_config(cls, ppo_node_id, all_nodes, all_links):
        """Extract PPO configuration from connected PPOConfig virtual node"""
        if not all_links or not all_nodes:
            return None
            
        # Find the config input connection (slot 1)
        config_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == ppo_node_id and to_slot == 1:  # config input
                    config_node_id = str(link[1])
                    break
        
        if not config_node_id:
            return None
            
        # Find the node data
        config_node_data = None
        for node in all_nodes:
            if str(node["id"]) == config_node_id:
                config_node_data = node
                break
                
        if not config_node_data:
            return None
            
        # Check if it's a PPOConfig node
        node_type = config_node_data.get("class_type") or config_node_data.get("type")
        if node_type != "PPOConfig":
            return None
            
        # Extract widget values
        widget_values = config_node_data.get("widgets_values", [])
        
        # Map widget values to config (based on PPOConfig node definition)
        # Widget order from PPOConfig INPUT_TYPES:
        # 0: learning_rate, 1: num_epochs, 2: num_minibatches, 3: clip_param,
        # 4: value_loss_coef, 5: entropy_coef, 6: gamma, 7: gae_lambda,
        # 8: max_grad_norm, 9: horizon_length, 10: max_iterations, 11: lr_schedule,
        # 12: lr_schedule_kl_threshold, 13: use_clipped_value_loss, 14: normalize_advantage,
        # 15: normalize_input, 16: normalize_value, 17: reward_shaper_scale,
        # 18: e_clip, 19: truncate_grads
        
        return {
            'learning_rate': widget_values[0] if len(widget_values) > 0 else 0.0003,
            'mini_epochs': widget_values[1] if len(widget_values) > 1 else 4,
            'minibatch_size': widget_values[2] if len(widget_values) > 2 else 8,
            'e_clip': widget_values[3] if len(widget_values) > 3 else 0.2,
            'critic_coef': widget_values[4] if len(widget_values) > 4 else 0.5,
            'entropy_coef': widget_values[5] if len(widget_values) > 5 else 0.01,
            'gamma': widget_values[6] if len(widget_values) > 6 else 0.99,
            'tau': widget_values[7] if len(widget_values) > 7 else 0.95,
            'grad_norm': widget_values[8] if len(widget_values) > 8 else 0.5,
            'horizon_length': widget_values[9] if len(widget_values) > 9 else 16,
            'max_agent_steps': widget_values[10] * 1000000 if len(widget_values) > 10 else 10000000000,  # Convert to steps
            'schedule_type': widget_values[11] if len(widget_values) > 11 else 'constant',
            'lr_schedule_kl_threshold': widget_values[12] if len(widget_values) > 12 else 0.008,
            'clip_value': widget_values[13] if len(widget_values) > 13 else True,
            'normalize_advantage': widget_values[14] if len(widget_values) > 14 else True,
            'normalize_input': widget_values[15] if len(widget_values) > 15 else True,
            'value_bootstrap': widget_values[16] if len(widget_values) > 16 else True,
            'bounds_loss_coef': 0.0,  # Not directly in widgets, set to 0
            'clip_actions': False,  # Not in PPOConfig widgets
        }


# Registration function
def register_rl_exporters(exporter):
    """Register all RL node exporters"""
    # Old exporters - commented out
    # exporter.register_node("PPOAgentNode", PPOAgentExporter_OLD)
    # exporter.register_node("PPOTrainerNode", PPOTrainerExporter_OLD)
    
    # New exporters with virtual node support
    exporter.register_node("PPOConfig", PPOConfigExporter)
    exporter.register_node("PPOAgent", PPOAgentExporter)

# Node type mapping for export system  
RL_NODE_EXPORTERS = {
    # Old exporters - commented out
    # "PPOAgentNode": PPOAgentExporter_OLD,
    # "PPOTrainerNode": PPOTrainerExporter_OLD,
    
    # New exporters
    "PPOConfig": PPOConfigExporter,
    "PPOAgent": PPOAgentExporter,
}