#!/usr/bin/env python3
"""
Exporter for PPOAgent node using queue-based template
"""

from ..graph_exporter import ExportableNode
from custom_nodes.utils.isaac_gym_config_loader import IsaacGymEnvConfigLoader

class PPOAgentExporter(ExportableNode):
    """Exporter for PPO Agent node - the main RL training node"""
    
    @classmethod
    def is_virtual(cls):
        """PPOAgent is NOT virtual - it generates the actual training code"""
        return False
    
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
        
        # Extract configuration from connected virtual nodes
        env_config = cls._extract_env_config(node_id, all_nodes, all_links)
        ppo_config = cls._extract_ppo_config(node_id, all_nodes, all_links)
        balancing_config = cls._extract_balancing_config(node_id, all_nodes, all_links)
        
        # Load task-specific configuration from YAML if we have a task
        task_ppo_config = {}
        if env_config and 'task' in env_config:
            task_name = env_config['task']
            loader = IsaacGymEnvConfigLoader()
            task_config = loader.get_task_config(task_name)
            if task_config and 'ppo_agent' in task_config:
                task_ppo_config = task_config['ppo_agent']
        
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
            
        # Use parameter specs to extract values from either inputs dict or widgets_values
        param_specs = [
            {'name': 'task', 'widget_index': 0},
            {'name': 'num_envs', 'widget_index': 1},
            {'name': 'seed', 'widget_index': 2},
            {'name': 'seed_control', 'widget_index': 3},
            {'name': 'headless', 'widget_index': 4},
            {'name': 'graphics_device_id', 'widget_index': 5},
            {'name': 'sim_device', 'widget_index': 6},
            {'name': 'physics_engine', 'widget_index': 7},
            {'name': 'multi_gpu', 'widget_index': 8},
            {'name': 'enable_cameras', 'widget_index': 9},
            {'name': 'force_render', 'widget_index': 10},
            {'name': 'use_gpu_pipeline', 'widget_index': 11},
            {'name': 'num_threads', 'widget_index': 12},
            {'name': 'solver_type', 'widget_index': 13},
            {'name': 'num_subscenes', 'widget_index': 14},
        ]
        
        # Get parameters using the helper that checks both inputs and widgets_values
        params = cls.get_node_parameters_batch(env_node_data, param_specs)
        
        # Debug logging
        import logging
        logging.info(f"[DNNE Export] IsaacGymEnvs params: {params}")
        
        # Validate all required parameters are present
        missing_params = [spec['name'] for spec in param_specs if params.get(spec['name']) is None]
        if missing_params:
            raise ValueError(
                f"IsaacGymEnvs node {env_node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Return config with all parameters
        params['isaac_gym_envs_path'] = '/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs'  # Always use default path
        return params
    
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
            
        # Use parameter specs matching PPOConfig INPUT_TYPES order
        param_specs = [
            {'name': 'learning_rate', 'widget_index': 0},
            {'name': 'num_epochs', 'widget_index': 1},  # maps to mini_epochs
            {'name': 'minibatch_size', 'widget_index': 2},
            {'name': 'clip_param', 'widget_index': 3},  # maps to e_clip
            {'name': 'value_loss_coef', 'widget_index': 4},  # maps to critic_coef
            {'name': 'entropy_coef', 'widget_index': 5},
            {'name': 'gamma', 'widget_index': 6},
            {'name': 'gae_lambda', 'widget_index': 7},  # maps to tau
            {'name': 'max_grad_norm', 'widget_index': 8},  # maps to grad_norm
            {'name': 'horizon_length', 'widget_index': 9},
            {'name': 'max_iterations', 'widget_index': 10},  # maps to max_epochs
            {'name': 'lr_schedule', 'widget_index': 11},  # maps to schedule_type
            {'name': 'lr_schedule_kl_threshold', 'widget_index': 12},
            {'name': 'use_clipped_value_loss', 'widget_index': 13},  # maps to clip_value
            {'name': 'normalize_advantage', 'widget_index': 14},
            {'name': 'normalize_input', 'widget_index': 15},
            {'name': 'normalize_value', 'widget_index': 16},
            {'name': 'reward_shaper_scale', 'widget_index': 17},  # not used but in node
            {'name': 'e_clip', 'widget_index': 18},  # duplicate param
            {'name': 'truncate_grads', 'widget_index': 19},  # not used
            {'name': 'bounds_loss_coef', 'widget_index': 20},
        ]
        
        # Get parameters using the helper that checks both inputs and widgets_values
        raw_params = cls.get_node_parameters_batch(config_node_data, param_specs)
        
        # Validate required parameters are present
        required_params = [
            'learning_rate', 'num_epochs', 'minibatch_size', 'clip_param',
            'value_loss_coef', 'entropy_coef', 'gamma', 'gae_lambda',
            'max_grad_norm', 'horizon_length', 'max_iterations', 'lr_schedule',
            'lr_schedule_kl_threshold', 'use_clipped_value_loss', 'normalize_advantage',
            'normalize_input', 'normalize_value', 'bounds_loss_coef'
        ]
        missing_params = [p for p in required_params if raw_params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"PPOConfig node {config_node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Map to the expected output format
        return {
            'learning_rate': raw_params['learning_rate'],
            'mini_epochs': raw_params['num_epochs'],
            'minibatch_size': raw_params['minibatch_size'],
            'e_clip': raw_params['clip_param'],
            'critic_coef': raw_params['value_loss_coef'],
            'entropy_coef': raw_params['entropy_coef'],
            'gamma': raw_params['gamma'],
            'tau': raw_params['gae_lambda'],
            'grad_norm': raw_params['max_grad_norm'],
            'horizon_length': raw_params['horizon_length'],
            'max_epochs': raw_params['max_iterations'],
            'schedule_type': raw_params['lr_schedule'],
            'lr_schedule_kl_threshold': raw_params['lr_schedule_kl_threshold'],
            'clip_value': raw_params['use_clipped_value_loss'],
            'normalize_advantage': raw_params['normalize_advantage'],
            'normalize_input': raw_params['normalize_input'],
            'normalize_value': raw_params['normalize_value'],
            'bounds_loss_coef': raw_params['bounds_loss_coef'],
        }
    
    @classmethod
    def _extract_balancing_config(cls, ppo_node_id, all_nodes, all_links):
        """Extract balancing configuration from connected BalancingConfig virtual node"""
        if not all_links or not all_nodes:
            return None
            
        # Find balancing_config input connection (in optional inputs)
        # PPO Agent has env (slot 0), config (slot 1), and balancing_config (slot 2)
        balancing_node_id = None
        for link in all_links:
            if len(link) >= 5:
                to_node, to_slot = str(link[3]), link[4]
                if to_node == ppo_node_id and to_slot == 2:  # balancing_config input
                    balancing_node_id = str(link[1])
                    break
        
        if not balancing_node_id:
            return None
            
        # Find the node data
        balancing_node_data = None
        for node in all_nodes:
            if str(node["id"]) == balancing_node_id:
                balancing_node_data = node
                break
                
        if not balancing_node_data:
            return None
            
        # Check if it's a BalancingConfig node
        node_type = balancing_node_data.get("class_type") or balancing_node_data.get("type")
        if node_type != "BalancingConfig":
            return None
            
        # Use parameter specs matching BalancingConfig INPUT_TYPES order
        param_specs = [
            {'name': 'enabled', 'widget_index': 0},
            {'name': 'min_hz', 'widget_index': 1},
            {'name': 'max_hz', 'widget_index': 2},
            {'name': 'target_hz', 'widget_index': 3},
            {'name': 'target_percentage', 'widget_index': 4},
            {'name': 'priority', 'widget_index': 5},
            {'name': 'guaranteed', 'widget_index': 6},
            {'name': 'max_latency_ms', 'widget_index': 7},
        ]
        
        # Get parameters using the helper that checks both inputs and widgets_values
        params = cls.get_node_parameters_batch(balancing_node_data, param_specs)
        
        # Check if config is enabled
        enabled = params.get('enabled')
        if enabled is None:
            raise ValueError(
                f"BalancingConfig node {balancing_node_id} missing 'enabled' parameter. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        if not enabled:
            return {"enabled": False, "type": "balancing_config"}
            
        # Build configuration structure matching what BalancingConfig.create_config() returns
        config = {
            'enabled': True,
            'frequency': {},
            'throughput': {},
            'scheduling': {},
            'latency': {},
        }
        
        # Add scheduling settings (always include if present)
        if params.get('priority') is not None:
            config['scheduling']['priority'] = params['priority']
        if params.get('guaranteed') is not None:
            config['scheduling']['guaranteed'] = params['guaranteed']
        
        # Add frequency settings if specified (>= 0 means care, -1 means don't care)
        if params.get('min_hz') is not None and params['min_hz'] >= 0:
            config['frequency']['min_hz'] = params['min_hz']
        if params.get('max_hz') is not None and params['max_hz'] >= 0:
            config['frequency']['max_hz'] = params['max_hz']
        if params.get('target_hz') is not None and params['target_hz'] >= 0:
            config['frequency']['target_hz'] = params['target_hz']
            
        # Add throughput settings if specified
        if params.get('target_percentage') is not None and params['target_percentage'] >= 0:
            config['throughput']['target_percentage'] = params['target_percentage']
            
        # Add latency settings if specified
        if params.get('max_latency_ms') is not None and params['max_latency_ms'] >= 0:
            config['latency']['max_latency_ms'] = params['max_latency_ms']
        
        return config