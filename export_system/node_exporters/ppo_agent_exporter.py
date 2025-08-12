#!/usr/bin/env python3
"""
Exporter for PPOAgent node using queue-based template
"""

from ..graph_exporter import ExportableNode

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
        balancing_config = cls._extract_balancing_config(node_id, all_nodes, all_links)
        
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
            template_vars.update({
                "HAS_BALANCING_CONFIG": True,
                "BALANCING_MIN_HZ": balancing_config.get('frequency', {}).get('min_hz', 0),
                "BALANCING_MAX_HZ": balancing_config.get('frequency', {}).get('max_hz', 0),
                "BALANCING_TARGET_HZ": balancing_config.get('frequency', {}).get('target_hz', 0),
                "BALANCING_TARGET_PERCENTAGE": balancing_config.get('throughput', {}).get('target_percentage', 0),
                "BALANCING_PRIORITY": balancing_config.get('scheduling', {}).get('priority', 0),
                "BALANCING_GUARANTEED": balancing_config.get('scheduling', {}).get('guaranteed', False),
                "BALANCING_MAX_LATENCY_MS": balancing_config.get('latency', {}).get('max_latency_ms', 0),
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
            
        # Extract widget values
        widget_values = env_node_data.get("widgets_values", [])
        
        # Debug logging
        import logging
        logging.info(f"[DNNE Export] IsaacGymEnvs widget_values: {widget_values}")
        
        # Map widget values to config (based on IsaacGymEnvs node definition)
        # Fail-fast: ensure we have all required widget values
        required_widget_count = 15  # 9 required + 5 optional + 1 for seed_control
        if len(widget_values) < required_widget_count:
            raise ValueError(
                f"IsaacGymEnvs node {env_node_id} has {len(widget_values)} widget values, "
                f"expected at least {required_widget_count}. "
                f"This may indicate a mismatch between the visual node and workflow."
            )
        
        # Extract values with explicit indexing - no defaults!
        return {
            'task': widget_values[0],
            'num_envs': widget_values[1],
            'seed': widget_values[2],
            'seed_control': widget_values[3],
            'headless': widget_values[4],
            'graphics_device_id': widget_values[5],
            'sim_device': widget_values[6],
            'physics_engine': widget_values[7],
            'multi_gpu': widget_values[8],
            'enable_cameras': widget_values[9],
            'force_render': widget_values[10],
            'use_gpu_pipeline': widget_values[11],
            'num_threads': widget_values[12],
            'solver_type': widget_values[13],
            'num_subscenes': widget_values[14],
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
        # 0: learning_rate, 1: num_epochs, 2: minibatch_size, 3: clip_param,
        # 4: value_loss_coef, 5: entropy_coef, 6: gamma, 7: gae_lambda,
        # 8: max_grad_norm, 9: horizon_length, 10: max_iterations, 11: lr_schedule,
        # 12: lr_schedule_kl_threshold, 13: use_clipped_value_loss, 14: normalize_advantage,
        # 15: normalize_input, 16: normalize_value, 17: reward_shaper_scale,
        # 18: e_clip, 19: truncate_grads, 20: bounds_loss_coef
        
        # Fail-fast: ensure we have all required widget values
        required_widget_count = 21  # 9 required + 12 optional parameters
        if len(widget_values) < required_widget_count:
            raise ValueError(
                f"PPOConfig node {config_node_id} has {len(widget_values)} widget values, "
                f"expected at least {required_widget_count}. "
                f"This may indicate a mismatch between the visual node and workflow."
            )
        
        # Direct mapping from widget values - no defaults!
        return {
            'learning_rate': widget_values[0],
            'mini_epochs': widget_values[1],
            'minibatch_size': widget_values[2],
            'e_clip': widget_values[3],
            'critic_coef': widget_values[4],
            'entropy_coef': widget_values[5],
            'gamma': widget_values[6],
            'tau': widget_values[7],
            'grad_norm': widget_values[8],
            'horizon_length': widget_values[9],
            'max_epochs': widget_values[10],
            'schedule_type': widget_values[11],
            'lr_schedule_kl_threshold': widget_values[12],
            'clip_value': widget_values[13],
            'normalize_advantage': widget_values[14],
            'normalize_input': widget_values[15],
            'normalize_value': widget_values[16],
            'bounds_loss_coef': widget_values[20],
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
            
        # Extract widget values
        widget_values = balancing_node_data.get("widgets_values", [])
        
        # Map widget values to config (based on BalancingConfig node definition)
        # Widget order: enabled, min_hz, max_hz, target_hz, target_percentage, priority, 
        #               guaranteed, max_latency_ms
        
        # Check if config is enabled (first widget)
        enabled = widget_values[0] if len(widget_values) > 0 else True
        if not enabled:
            return {"enabled": False, "type": "balancing_config"}
            
        # Build configuration structure matching what BalancingConfig.create_config() returns
        config = {
            'enabled': True,
            'frequency': {},
            'throughput': {},
            'scheduling': {
                'priority': widget_values[5] if len(widget_values) > 5 else 0,
                'guaranteed': widget_values[6] if len(widget_values) > 6 else False,
            },
            'latency': {},
        }
        
        # Add frequency settings if specified (>= 0 means care, -1 means don't care)
        if len(widget_values) > 1 and widget_values[1] >= 0:
            config['frequency']['min_hz'] = widget_values[1]
        if len(widget_values) > 2 and widget_values[2] >= 0:
            config['frequency']['max_hz'] = widget_values[2]
        if len(widget_values) > 3 and widget_values[3] >= 0:
            config['frequency']['target_hz'] = widget_values[3]
            
        # Add throughput settings if specified
        if len(widget_values) > 4 and widget_values[4] >= 0:
            config['throughput']['target_percentage'] = widget_values[4]
            
        # Add latency settings if specified
        if len(widget_values) > 7 and widget_values[7] >= 0:
            config['latency']['max_latency_ms'] = widget_values[7]
        
        return config