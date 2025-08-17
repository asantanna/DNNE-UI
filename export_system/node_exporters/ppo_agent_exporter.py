#!/usr/bin/env python3
"""
Exporter for PPOAgent node using queue-based template
"""

import sys
import os

# Add parent directory to path to import dnne_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dnne_config import get_isaac_gym_envs_path

from ..graph_exporter import ExportableNode
from ..utils import export_utils

class PPOAgentExporter(ExportableNode):
    """Exporter for PPO Agent node - the main RL training node"""
    # PPOAgent is NOT virtual - it generates the actual training code
    # Virtual status is handled by @dnne_node decorator
    
    @classmethod
    def get_template_name(cls):
        return "nodes/ppo_agent_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Get node parameters from the UI widgets
        # The PPOAgent node has these widgets in order:
        # max_iterations, checkpoint_interval, eval_interval, eval_episodes, log_interval, save_path, resume_from
        param_specs = [
            {'name': 'max_iterations', 'widget_index': 0},
            {'name': 'checkpoint_interval', 'widget_index': 1},
            {'name': 'eval_interval', 'widget_index': 2},
            {'name': 'eval_episodes', 'widget_index': 3},
            {'name': 'log_interval', 'widget_index': 4},
            {'name': 'save_path', 'widget_index': 5},
            {'name': 'resume_from', 'widget_index': 6, 'default': ""},  # OK to be empty
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present (resume_from can be empty)
        required_params = ['max_iterations', 'checkpoint_interval', 'eval_interval', 
                          'eval_episodes', 'log_interval', 'save_path']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"PPOAgent node {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all training control parameters."
            )
        
        # Extract configuration from connected virtual nodes using proper query methods
        env_config = cls._get_env_config_via_query(node_id, all_nodes, all_links)
        ppo_config = cls._get_ppo_config_via_query(node_id, all_nodes, all_links)
        balancing_config = cls._get_balancing_config_via_query(node_id, all_nodes, all_links)
        
        # Load task-specific configuration from YAML if we have a task
        task_ppo_config = {}
        if env_config and 'task' in env_config:
            task_name = env_config['task']
            # Load PPO config YAML directly if it exists
            try:
                from pathlib import Path
                import yaml
                isaacgym_envs_path = get_isaac_gym_envs_path()
                ppo_cfg_path = isaacgym_envs_path / 'isaacgymenvs' / 'cfg' / 'train' / f'{task_name}PPO.yaml'
                
                if ppo_cfg_path.exists():
                    with open(ppo_cfg_path, 'r') as f:
                        ppo_yaml = yaml.safe_load(f)
                    # Extract network config from PPO YAML if available
                    if ppo_yaml and 'params' in ppo_yaml and 'network' in ppo_yaml['params']:
                        net_config = ppo_yaml['params']['network']
                        task_ppo_config['network_mlp_layers'] = net_config.get('mlp', {}).get('units', [256, 128, 64])
                        task_ppo_config['network_activation'] = net_config.get('mlp', {}).get('activation', 'elu')
                        task_ppo_config['separate_value_network'] = net_config.get('separate', False)
                    # Extract other PPO params
                    if ppo_yaml and 'params' in ppo_yaml and 'config' in ppo_yaml['params']:
                        config = ppo_yaml['params']['config']
                        task_ppo_config['mixed_precision'] = config.get('mixed_precision', False)
                        task_ppo_config['multi_gpu'] = config.get('multi_gpu', False)
            except Exception as e:
                # If we can't load PPO config, just use defaults
                pass
        
        # Get network configuration from task config with sensible defaults
        network_mlp_layers = task_ppo_config.get('network_mlp_layers', [256, 128, 64])
        network_activation = task_ppo_config.get('network_activation', 'elu')
        separate_value_network = task_ppo_config.get('separate_value_network', False)
        mixed_precision = task_ppo_config.get('mixed_precision', False)
        multi_gpu = task_ppo_config.get('multi_gpu', False)
        
        # Get other parameters from task config or use defaults
        keep_checkpoints = task_ppo_config.get('keep_checkpoints', 5)
        save_interval = task_ppo_config.get('save_interval', 1000)
        experiment_name = task_ppo_config.get('experiment_name', f"{env_config.get('task', 'PPO')}_PPO")
        
        # Merge all configuration
        template_vars = {
            "NODE_ID": node_id,
            "CLASS_NAME": "PPOAgentNode",
            # Network configuration from task YAML
            "NETWORK_MLP_LAYERS": network_mlp_layers,
            "NETWORK_ACTIVATION": network_activation,
            "SEPARATE_VALUE_NETWORK": separate_value_network,
            # Training control from UI widgets
            "CHECKPOINT_INTERVAL": params['checkpoint_interval'],
            "LOG_INTERVAL": params['log_interval'],
            "SAVE_PATH": params['save_path'],
            "RESUME_FROM": params['resume_from'],
            "MAX_ITERATIONS": params['max_iterations'],
            "EVAL_INTERVAL": params['eval_interval'],
            "EVAL_EPISODES": params['eval_episodes'],
            # Additional parameters from task config or defaults
            "KEEP_CHECKPOINTS": keep_checkpoints,
            "SAVE_INTERVAL": save_interval,
            "EXPERIMENT_NAME": experiment_name,
            "MIXED_PRECISION": mixed_precision,
            "MULTI_GPU": multi_gpu,
            # For compatibility - some templates might still use this
            "LOAD_CHECKPOINT": params.get('resume_from', ''),
        }
        
        # Add environment configuration
        if env_config:
            # Fail-fast: all required env config values must be present
            required_env_keys = [
                'task', 'num_envs', 'seed', 'seed_control', 'headless',
                'graphics_device_id', 'sim_device', 'physics_engine', 'multi_gpu',
                'enable_cameras', 'force_render', 'use_gpu_pipeline', 'num_threads',
                'solver_type', 'num_subscenes', 'isaac_gym_envs_path'
            ]
            missing_keys = [key for key in required_env_keys if key not in env_config]
            if missing_keys:
                raise ValueError(
                    f"IsaacGymEnvs config missing required keys: {missing_keys}. "
                    f"This indicates a mismatch between the exporter and visual node."
                )
            
            template_vars.update({
                "ENV_TASK": env_config['task'],
                "ENV_NUM_ENVS": env_config['num_envs'],
                "ENV_SEED": env_config['seed'],
                "ENV_SEED_CONTROL": env_config['seed_control'],
                "ENV_HEADLESS": env_config['headless'],
                "ENV_GRAPHICS_DEVICE": env_config['graphics_device_id'],
                "ENV_SIM_DEVICE": env_config['sim_device'],
                "ENV_PHYSICS_ENGINE": env_config['physics_engine'],
                "ENV_MULTI_GPU": env_config['multi_gpu'],
                "ENV_ENABLE_CAMERAS": env_config['enable_cameras'],
                "ENV_FORCE_RENDER": env_config['force_render'],
                "ENV_USE_GPU_PIPELINE": env_config['use_gpu_pipeline'],
                "ENV_NUM_THREADS": env_config['num_threads'],
                "ENV_SOLVER_TYPE": env_config['solver_type'],
                "ENV_NUM_SUBSCENES": env_config['num_subscenes'],
                "ISAAC_GYM_ENVS_PATH": env_config['isaac_gym_envs_path'],
            })
        
        # Add PPO configuration
        if ppo_config:
            # Fail-fast: all required PPO config values must be present
            required_ppo_keys = [
                'minibatch_size', 'horizon_length', 'learning_rate', 'schedule_type',
                'gamma', 'tau', 'e_clip', 'clip_value', 'mini_epochs', 'critic_coef',
                'entropy_coef', 'bounds_loss_coef', 'max_epochs', 'normalize_advantage',
                'normalize_input', 'normalize_value', 'grad_norm', 'lr_schedule_kl_threshold'
            ]
            missing_keys = [key for key in required_ppo_keys if key not in ppo_config]
            if missing_keys:
                raise ValueError(
                    f"PPOConfig missing required keys: {missing_keys}. "
                    f"This indicates a mismatch between the exporter and visual node."
                )
            
            template_vars.update({
                "PPO_MINIBATCH_SIZE": ppo_config['minibatch_size'],
                "PPO_HORIZON_LENGTH": ppo_config['horizon_length'],
                "PPO_LEARNING_RATE": ppo_config['learning_rate'],
                "PPO_SCHEDULE_TYPE": ppo_config['schedule_type'],
                "PPO_GAMMA": ppo_config['gamma'],
                "PPO_TAU": ppo_config['tau'],
                "PPO_E_CLIP": ppo_config['e_clip'],
                "PPO_CLIP_VALUE": ppo_config['clip_value'],
                "PPO_MINI_EPOCHS": ppo_config['mini_epochs'],
                "PPO_CRITIC_COEF": ppo_config['critic_coef'],
                "PPO_ENTROPY_COEF": ppo_config['entropy_coef'],
                "PPO_BOUNDS_LOSS_COEF": ppo_config['bounds_loss_coef'],
                "PPO_MAX_EPOCHS": ppo_config['max_epochs'],
                "PPO_NORMALIZE_ADVANTAGE": ppo_config['normalize_advantage'],
                "PPO_NORMALIZE_INPUT": ppo_config['normalize_input'],
                "PPO_NORMALIZE_VALUE": ppo_config['normalize_value'],
            })
            
            # Only add these if they exist in the PPO config (from YAML)
            if 'value_bootstrap' in ppo_config:
                template_vars["PPO_VALUE_BOOTSTRAP"] = ppo_config['value_bootstrap']
            if 'clip_actions' in ppo_config:
                template_vars["PPO_CLIP_ACTIONS"] = ppo_config['clip_actions']
        
        # Add balancing configuration if present
        if balancing_config:
            # Extract values from the config we built - use 0 only for truly optional fields
            freq = balancing_config.get('frequency', {})
            throughput = balancing_config.get('throughput', {})
            scheduling = balancing_config.get('scheduling', {})
            latency = balancing_config.get('latency', {})
            
            template_vars.update({
                "HAS_BALANCING_CONFIG": True,
                "BALANCING_MIN_HZ": freq.get('min_hz', 0),  # 0 means no min constraint
                "BALANCING_MAX_HZ": freq.get('max_hz', 0),  # 0 means no max constraint
                "BALANCING_TARGET_HZ": freq.get('target_hz', 0),  # 0 means no target
                "BALANCING_TARGET_PERCENTAGE": throughput.get('target_percentage', 0),  # 0 means no target
                "BALANCING_PRIORITY": scheduling.get('priority', 0),  # 0 is valid default priority
                "BALANCING_GUARANTEED": scheduling.get('guaranteed', False),  # False is valid default
                "BALANCING_MAX_LATENCY_MS": latency.get('max_latency_ms', 0),  # 0 means no constraint
            })
        else:
            template_vars["HAS_BALANCING_CONFIG"] = False
        
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
        # PPOAgent has inputs for virtual config nodes
        return ["env_config", "ppo_config", "balancing_config"]
    
    @classmethod
    def _get_ppo_config_via_query(cls, ppo_node_id, all_nodes, all_links):
        """Get PPO configuration from connected PPOConfig virtual node using query method.
        
        This method respects widget encapsulation by calling the PPOConfig exporter's
        query method instead of directly accessing its widgets.
        """
        # Export context should already be set by GraphExporter
        if not all_links or not all_nodes:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Cannot get PPO config - missing nodes or links data"
            )
            
        # Find the config input connection (slot 1)
        config_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == ppo_node_id and to_slot == 1:  # config input
                    config_node_id = str(link[1])
                    break
        
        if not config_node_id:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: No PPO configuration connected to ppo_config input. "
                f"Please connect a PPOConfig node."
            )
            
        # Find the node data
        config_node_data = export_utils.get_node_by_id(config_node_id)
        if not config_node_data:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Connected PPO config node {config_node_id} not found in workflow"
            )
            
        # Check if it's a PPOConfig node
        node_type = config_node_data.get("class_type") or config_node_data.get("type")
        if node_type != "PPOConfig":
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Expected PPOConfig node connected to ppo_config input, "
                f"but got {node_type} node instead"
            )
            
        # Get the PPOConfig exporter and call its query method
        ppo_config_exporter = export_utils.get_node_exporter("PPOConfig")
        if not ppo_config_exporter or not hasattr(ppo_config_exporter, 'get_ppo_config'):
            raise ValueError(
                f"PPOConfig exporter missing get_ppo_config() query method. "
                f"This indicates an incomplete virtual node implementation."
            )
        
        # Call the query method to get configuration
        return ppo_config_exporter.get_ppo_config(config_node_id, config_node_data)
    
    @classmethod
    def _get_env_config_via_query(cls, ppo_node_id, all_nodes, all_links):
        """Get environment configuration from connected IsaacGymEnvs virtual node using query method.
        
        This method respects widget encapsulation by calling the IsaacGymEnvs exporter's
        query method instead of directly accessing its widgets.
        """
        # Export context should already be set by GraphExporter
        if not all_links or not all_nodes:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Cannot get environment config - missing nodes or links data"
            )
            
        # Find the env input connection (slot 0)
        env_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == ppo_node_id and to_slot == 0:  # env input
                    env_node_id = str(link[1])
                    break
        
        if not env_node_id:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: No environment configuration connected to env_config input. "
                f"Please connect an IsaacGymEnvs node."
            )
            
        # Find the node data
        env_node_data = export_utils.get_node_by_id(env_node_id)
        if not env_node_data:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Connected environment node {env_node_id} not found in workflow"
            )
            
        # Check if it's an IsaacGymEnvs node
        node_type = env_node_data.get("class_type") or env_node_data.get("type")
        if node_type != "IsaacGymEnvs":
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Expected IsaacGymEnvs node connected to env_config input, "
                f"but got {node_type} node instead"
            )
            
        # Get the IsaacGymEnvs exporter and call its query method
        env_exporter = export_utils.get_node_exporter("IsaacGymEnvs")
        if not env_exporter or not hasattr(env_exporter, 'get_env_config'):
            raise ValueError(
                f"IsaacGymEnvs exporter missing get_env_config() query method. "
                f"This indicates an incomplete virtual node implementation."
            )
        
        # Call the query method to get configuration
        return env_exporter.get_env_config(env_node_id, env_node_data)
    
    @classmethod
    def _get_balancing_config_via_query(cls, ppo_node_id, all_nodes, all_links):
        """Get balancing configuration from connected BalancerConfig virtual node using query method.
        
        This method respects widget encapsulation by calling the BalancerConfig exporter's
        query method instead of directly accessing its widgets.
        """
        # Export context should already be set by GraphExporter
        if not all_links or not all_nodes:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Cannot get balancing config - missing nodes or links data"
            )
            
        # Find balancing_config input connection (slot 2)
        balancing_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == ppo_node_id and to_slot == 2:  # balancing_config input
                    balancing_node_id = str(link[1])
                    break
        
        if not balancing_node_id:
            return None  # This is optional, so returning None is OK
            
        # Find the node data
        balancing_node_data = export_utils.get_node_by_id(balancing_node_id)
        if not balancing_node_data:
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Connected balancing config node {balancing_node_id} not found in workflow"
            )
            
        # Check if it's a BalancerConfig node
        node_type = balancing_node_data.get("class_type") or balancing_node_data.get("type")
        if node_type != "BalancerConfig":
            raise RuntimeError(
                f"PPOAgent node {ppo_node_id}: Expected BalancerConfig node connected to balancing_config input, "
                f"but got {node_type} node instead"
            )
            
        # Get the BalancerConfig exporter and call its query method
        balancing_exporter = export_utils.get_node_exporter("BalancerConfig")
        if not balancing_exporter or not hasattr(balancing_exporter, 'get_balancing_config'):
            raise ValueError(
                f"BalancerConfig exporter missing get_balancing_config() query method. "
                f"This indicates an incomplete virtual node implementation."
            )
        
        # Call the query method to get configuration
        return balancing_exporter.get_balancing_config(balancing_node_id, balancing_node_data)
    
    @classmethod
    def _get_node_registry(cls):
        """Get the node registry for exporter lookups.
        
        This helper method builds a registry of node exporters.
        In production, this would be passed from GraphExporter.
        """
        # Import the exporters we need
        from .ppo_config_exporter import PPOConfigExporter
        from .isaac_gym_envs_exporter import IsaacGymEnvsExporter
        from .balancer_config_exporter import BalancerConfigExporter
        
        return {
            'PPOConfig': PPOConfigExporter,
            'IsaacGymEnvs': IsaacGymEnvsExporter,
            'BalancerConfig': BalancerConfigExporter,
        }