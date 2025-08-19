"""
Isaac Gym Environment Configuration Loader for DNNE.
Provides environment-specific configurations for the 3-node PPO setup.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import dnne_config

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading fails"""
    pass


class IsaacGymEnvConfigLoader:
    """
    Load and parse IsaacGymEnvs configurations for DNNE nodes.
    
    This loader extracts environment-specific parameters from IGE's YAML files
    and provides them in a format suitable for DNNE's 3-node PPO workflow:
    - IsaacGymEnvs node (environment configuration)
    - PPOConfig node (algorithm hyperparameters)  
    - PPOAgent node (network architecture and training settings)
    """
    
    # Singleton instance
    _instance = None
    _configs_cache = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, isaac_gym_envs_path: str = None):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return
            
        # Get path from dnne_config
        if isaac_gym_envs_path is None:
            isaac_gym_envs_path = str(dnne_config.get_isaac_gym_envs_path())
            if not isaac_gym_envs_path:
                raise ConfigurationError("isaac_gym_envs path not found in dnne_config")
        
        self.base_path = Path(isaac_gym_envs_path)
        self.cfg_path = self.base_path / "isaacgymenvs" / "cfg"
        self.task_cfg_path = self.cfg_path / "task"
        self.train_cfg_path = self.cfg_path / "train"
        self._initialized = True
        
        # Load global config once
        self._global_config = self._load_global_config()
    
    def _load_global_config(self) -> Dict[str, Any]:
        """Load the global config.yaml file."""
        global_config_path = self.cfg_path / "config.yaml"
        if global_config_path.exists():
            try:
                with open(global_config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading global config: {e}")
        return {}
        
    def get_available_tasks(self) -> List[str]:
        """Get list of available tasks that have PPO configurations."""
        if self._configs_cache is None:
            self._load_all_configs()
        return list(self._configs_cache.keys())
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for a specific task.
        
        Args:
            task_name: Name of the IsaacGymEnvs task (e.g., "Cartpole")
            
        Returns:
            Dictionary with configurations for all 3 DNNE nodes
        """
        if self._configs_cache is None:
            self._load_all_configs()
            
        config = self._configs_cache.get(task_name)
        if config is None:
            raise NotImplementedError(f"Task '{task_name}' not found. Available tasks: {list(self._configs_cache.keys())}")
        return config
    
    def get_node_defaults(self, task_name: str, node_type: str) -> Dict[str, Any]:
        """
        Get default values for a specific node type.
        
        Args:
            task_name: Name of the task
            node_type: One of "isaac_gym_env", "ppo_config", or "ppo_agent"
            
        Returns:
            Dictionary of default values for the node's widgets
        """
        config = self.get_task_config(task_name)
        # Node type must exist in config
        if node_type not in config:
            raise ValueError(f"Task '{task_name}' has no configuration for node type '{node_type}'")
        return config[node_type]
    
    def _load_all_configs(self):
        """Load all configurations and cache them."""
        self._configs_cache = {}
        
        if not self.task_cfg_path.exists():
            logger.error(f"Task config path not found: {self.task_cfg_path}")
            return
            
        # Find all tasks (with or without PPO configs)
        for task_file in sorted(self.task_cfg_path.glob("*.yaml")):
            if task_file.is_file():
                task_name = task_file.stem
                ppo_file = self.train_cfg_path / f"{task_name}PPO.yaml"
                
                # Load config whether or not PPO file exists
                config = self._load_environment_config(task_name, task_file, ppo_file)
                if config:
                    self._configs_cache[task_name] = config
    
    def _load_environment_config(self, task_name: str, task_file: Path, ppo_file: Path) -> Optional[Dict[str, Any]]:
        """Load configuration for a specific environment."""
        try:
            # Load task config
            with open(task_file, 'r') as f:
                task_config = yaml.safe_load(f)
                
            # Check if this config uses Hydra defaults (inheritance)
            # DNNE doesn't support Hydra inheritance, so skip these configs
            if 'defaults' in task_config:
                logger.debug(f"Skipping {task_name}: Uses Hydra defaults (inheritance not supported by DNNE)")
                return None
                
            # Load PPO config if it exists
            ppo_config = None
            if ppo_file.exists():
                with open(ppo_file, 'r') as f:
                    ppo_config = yaml.safe_load(f)
                    
                # Skip PPO configs that use Hydra defaults too
                if ppo_config and 'defaults' in ppo_config:
                    logger.info(f"Skipping {task_name}: PPO config uses Hydra defaults (inheritance not supported)")
                    return None
                
            # Extract configurations for each node
            result = {
                "task_name": task_name,
                "isaac_gym_env": self._extract_env_node_config(task_config, ppo_config),
            }
            
            # Only add PPO configs if PPO file exists
            if ppo_config:
                result["ppo_config"] = self._extract_ppo_config_node(ppo_config)
                result["ppo_agent"] = self._extract_ppo_agent_node(ppo_config, task_name)
                
            return result
            
        except Exception as e:
            logger.error(f"Error loading config for {task_name}: {e}")
            raise ConfigurationError(f"Failed to load config for {task_name}") from e
    
    def _resolve_value(self, value, default):
        """Resolve OmegaConf interpolations to concrete values."""
        if isinstance(value, str) and value.startswith('${'):
            # Handle resolve_default pattern: ${resolve_default:default_value,${...path}}
            if value.startswith('${resolve_default:'):
                # Extract the default value
                content = value[18:-1]  # Remove '${resolve_default:' and '}'
                parts = content.split(',', 1)
                if parts:
                    try:
                        return int(parts[0])
                    except ValueError:
                        try:
                            return float(parts[0])
                        except ValueError:
                            return parts[0]
            
            # Handle equality check: ${eq:${...pipeline},"gpu"}
            elif value.startswith('${eq:'):
                # For GPU pipeline, default to True
                return True
            
            # Handle simple references - return sensible defaults
            elif 'num_threads' in value:
                return 0
            elif 'solver_type' in value:
                return 1
            elif 'num_subscenes' in value:
                return 0
            elif 'multi_gpu' in value:
                return False
            elif 'numEnvs' in value:
                return 512
            
            # If we can't resolve it, return the default or original value
            return default if default is not None else value
        
        return value
    
    def get_task_dt(self, task_name: str) -> float:
        """
        Get the simulation timestep (dt) for a specific task.
        Fail-fast: raises NotImplementedError if dt is not found.
        
        Args:
            task_name: Name of the IsaacGymEnvs task
            
        Returns:
            The dt value from the task's sim configuration
        """
        config = self.get_task_config(task_name)
        
        # Look for dt in the cached task config
        if 'isaac_gym_env' in config and 'sim_dt' in config['isaac_gym_env']:
            return config['isaac_gym_env']['sim_dt']
            
        # If not cached, load from file directly
        task_file = self.task_cfg_path / f"{task_name}.yaml"
        if not task_file.exists():
            raise NotImplementedError(f"Task file not found for '{task_name}'")
            
        with open(task_file, 'r') as f:
            task_config = yaml.safe_load(f)
            
        # sim and dt are required fields
        if 'sim' not in task_config:
            raise NotImplementedError(f"Task '{task_name}' does not have 'sim' section in its YAML configuration")
        if 'dt' not in task_config['sim']:
            raise NotImplementedError(f"Task '{task_name}' does not specify sim.dt in its YAML configuration")
        dt = task_config['sim']['dt']
            
        return float(dt)
    
    def get_task_subtasks(self, task_name: str) -> List[str]:
        """
        Get available subtasks for a DNNE environment.
        
        Args:
            task_name: Name of the IsaacGymEnvs task
            
        Returns:
            List of available subtask names, or empty list if not a DNNE environment
        """
        # Load task configuration
        task_file = self.task_cfg_path / f"{task_name}.yaml"
        if not task_file.exists():
            raise NotImplementedError(f"Task file not found for '{task_name}'")
            
        with open(task_file, 'r') as f:
            task_config = yaml.safe_load(f)
            
        # Check if this is a DNNE environment
        # Default to False for non-DNNE environments (valid per comment)
        if not task_config.get('is_dnne_environment', False):
            return []
            
        # For now, return hardcoded subtasks for FrankaDNNE
        # In future, this could be read from YAML or discovered dynamically
        if task_name == "FrankaDNNE":
            return ["random_target"]
            
        return []
    
    def is_dnne_environment(self, task_name: str) -> bool:
        """
        Check if a task is a DNNE-specific environment.
        
        Args:
            task_name: Name of the IsaacGymEnvs task
            
        Returns:
            True if this is a DNNE environment, False otherwise
        """
        task_file = self.task_cfg_path / f"{task_name}.yaml"
        if not task_file.exists():
            return False
            
        try:
            with open(task_file, 'r') as f:
                task_config = yaml.safe_load(f)
            # Default to False for non-DNNE environments (valid per comment)
            return task_config.get('is_dnne_environment', False)
        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error checking if {task_name} is DNNE environment: {e}")
            return False
    
    def _extract_env_node_config(self, task_config: Dict, ppo_config: Optional[Dict]) -> Dict[str, Any]:
        """Extract configuration for IsaacGymEnvs node."""
        # env and sim sections may not exist - set to None for later validation
        env_cfg = task_config.get('env', None)
        sim_cfg = task_config.get('sim', None)
        
        # Resolve num_envs first
        if env_cfg is None or 'numEnvs' not in env_cfg:
            raise ValueError("Task config missing required 'env.numEnvs' field")
        num_envs_raw = env_cfg['numEnvs']
        num_envs = self._resolve_value(num_envs_raw, None)
        
        # Get num_actors from PPO config if available
        # PPO config is optional - workflow may be incomplete
        if ppo_config and 'params' in ppo_config and 'config' in ppo_config['params']:
            ppo_params = ppo_config['params']['config']
            if 'num_actors' not in ppo_params:
                raise ValueError("PPO config missing required 'num_actors' field")
            num_actors = ppo_params['num_actors']
        else:
            # PPO config not available - skip num_actors
            num_actors = None
        
        # If num_actors references numEnvs, use the resolved value
        if isinstance(num_actors, str) and 'numEnvs' in num_actors:
            num_actors = num_envs
        else:
            num_actors = self._resolve_value(num_actors, num_envs)
        
        # Get physics config - may not exist
        physx_cfg = sim_cfg.get('physx', {}) if sim_cfg else {}
        
        # Get global config values
        global_cfg = self._global_config
        
        # Skip nullAction extraction here - we don't have schema keys yet
        # For DNNE environments, nullAction is in nested_schemas which requires selection
        # For non-DNNE environments, it's optional in env:
        # This will be handled at export time when schema is known
        
        # Extract dt value - required if sim_cfg exists
        if sim_cfg is None or 'dt' not in sim_cfg:
            raise ValueError("Task config missing required 'sim.dt' field")
        dt = sim_cfg['dt']
        
        # All global config values are required
        if 'seed' not in global_cfg:
            raise ValueError("Global config missing required 'seed' field")
        if 'headless' not in global_cfg:
            raise ValueError("Global config missing required 'headless' field")
        if 'graphics_device_id' not in global_cfg:
            raise ValueError("Global config missing required 'graphics_device_id' field")
        if 'sim_device' not in global_cfg:
            raise ValueError("Global config missing required 'sim_device' field")
        if 'multi_gpu' not in global_cfg:
            raise ValueError("Global config missing required 'multi_gpu' field")
        if 'force_render' not in global_cfg:
            raise ValueError("Global config missing required 'force_render' field")
        
        # Physics engine - check task config first, then global
        if 'physics_engine' in task_config:
            physics_engine = self._resolve_value(task_config['physics_engine'], None)
        elif 'physics_engine' in global_cfg:
            physics_engine = self._resolve_value(global_cfg['physics_engine'], None)
        else:
            raise ValueError("Neither task nor global config specifies 'physics_engine'")
        
        # Enable cameras - optional field
        enable_cameras = env_cfg.get('enableCameraSensors') if env_cfg else None
        
        # GPU pipeline - check sim first, then global pipeline setting
        if sim_cfg and 'use_gpu_pipeline' in sim_cfg:
            use_gpu_pipeline = self._resolve_value(sim_cfg['use_gpu_pipeline'], None)
        elif 'pipeline' in global_cfg:
            use_gpu_pipeline = global_cfg['pipeline'] == 'gpu'
        else:
            raise ValueError("Neither sim nor global config specifies GPU pipeline setting")
        
        # PhysX settings - check physx_cfg first, then global
        if 'num_threads' in physx_cfg:
            num_threads = self._resolve_value(physx_cfg['num_threads'], None)
        elif 'num_threads' in global_cfg:
            num_threads = global_cfg['num_threads']
        else:
            raise ValueError("Neither physx nor global config specifies 'num_threads'")
            
        if 'solver_type' in physx_cfg:
            solver_type = self._resolve_value(physx_cfg['solver_type'], None)
        elif 'solver_type' in global_cfg:
            solver_type = global_cfg['solver_type']
        else:
            raise ValueError("Neither physx nor global config specifies 'solver_type'")
            
        if 'num_subscenes' in physx_cfg:
            num_subscenes = self._resolve_value(physx_cfg['num_subscenes'], None)
        elif 'num_subscenes' in global_cfg:
            num_subscenes = global_cfg['num_subscenes']
        else:
            raise ValueError("Neither physx nor global config specifies 'num_subscenes'")
        
        config = {
            "num_envs": num_envs,
            "seed": global_cfg['seed'],
            "seed_control": "fixed",  # Use the actual widget name
            "headless": global_cfg['headless'],
            "graphics_device_id": global_cfg['graphics_device_id'],
            "sim_device": global_cfg['sim_device'],
            "physics_engine": physics_engine,
            "multi_gpu": global_cfg['multi_gpu'],
            "force_render": global_cfg['force_render'],  # From global config!
            "use_gpu_pipeline": use_gpu_pipeline,
            "num_threads": num_threads,
            "solver_type": solver_type,
            "num_subscenes": num_subscenes,
            # null_action excluded - requires schema selection for DNNE envs
        }
        
        # Add optional fields only if they exist
        if enable_cameras is not None:
            config["enable_cameras"] = enable_cameras
        
        if dt is not None:
            config["sim_dt"] = float(dt)
            
        return config
    
    def _extract_ppo_config_node(self, ppo_config: Dict) -> Dict[str, Any]:
        """Extract configuration for PPOConfig node."""
        # Validate PPO config structure
        if 'params' not in ppo_config:
            raise ValueError("PPO config missing required 'params' section")
        params = ppo_config['params']
        
        if 'config' not in params:
            raise ValueError("PPO params missing required 'config' section")
        algo_config = params['config']
        
        # All algorithm parameters are required
        if 'minibatch_size' not in algo_config:
            raise ValueError("PPO config missing required 'minibatch_size'")
        if 'horizon_length' not in algo_config:
            raise ValueError("PPO config missing required 'horizon_length'")
        
        minibatch_size = algo_config['minibatch_size']
        horizon_length = algo_config['horizon_length']
        
        # Some configs use mini_epochs, some use num_epochs - check both
        if 'mini_epochs' in algo_config:
            mini_epochs = algo_config['mini_epochs']
        elif 'num_epochs' in algo_config:
            mini_epochs = algo_config['num_epochs']
        else:
            raise ValueError("PPO config missing required epochs parameter (mini_epochs or num_epochs)")
        
        # DNNE now uses minibatch_size directly, matching the YAML configuration
        
        # Validate all required PPO parameters
        required_params = [
            'learning_rate', 'e_clip', 'critic_coef', 'entropy_coef',
            'gamma', 'tau', 'lr_schedule', 'kl_threshold',
            'grad_norm', 'truncate_grads', 'normalize_advantage',
            'normalize_input', 'normalize_value', 'clip_value',
            'max_epochs', 'bounds_loss_coef'
        ]
        
        for param in required_params:
            if param not in algo_config:
                raise ValueError(f"PPO config missing required parameter: '{param}'")
        
        # Handle reward shaper separately as it's nested
        if 'reward_shaper' in algo_config and 'scale_value' in algo_config['reward_shaper']:
            reward_shaper_scale = algo_config['reward_shaper']['scale_value']
        else:
            raise ValueError("PPO config missing required 'reward_shaper.scale_value'")
        
        config_dict = {
            # Core PPO parameters
            "learning_rate": float(algo_config['learning_rate']),
            "num_epochs": mini_epochs,  # DNNE calls it num_epochs
            "minibatch_size": minibatch_size,  # Direct from YAML
            "horizon_length": horizon_length,
            
            # PPO specific
            "clip_param": algo_config['e_clip'],
            "value_loss_coef": algo_config['critic_coef'],
            "entropy_coef": algo_config['entropy_coef'],
            "gamma": algo_config['gamma'],
            "gae_lambda": algo_config['tau'],
            
            # Learning rate schedule
            "lr_schedule": algo_config['lr_schedule'],
            "lr_schedule_kl_threshold": algo_config['kl_threshold'],
            
            # Gradient clipping
            "max_grad_norm": algo_config['grad_norm'],
            "truncate_grads": algo_config['truncate_grads'],
            
            # Normalization
            "normalize_advantage": algo_config['normalize_advantage'],
            "normalize_input": algo_config['normalize_input'],
            "normalize_value": algo_config['normalize_value'],
            
            # Value function
            "use_clipped_value_loss": algo_config['clip_value'],
            
            # Training duration
            "max_iterations": self._resolve_value(algo_config['max_epochs'], None),
            
            # Reward shaping
            "reward_shaper_scale": reward_shaper_scale,
            
            # Additional e_clip parameter that might be in UI
            "e_clip": algo_config['e_clip'],
            
            # Bounds loss coefficient
            "bounds_loss_coef": algo_config['bounds_loss_coef'],
        }
        
        # Only add these parameters if they exist in the YAML
        if 'value_bootstrap' in algo_config:
            config_dict['value_bootstrap'] = algo_config['value_bootstrap']
        if 'clip_actions' in algo_config:
            config_dict['clip_actions'] = algo_config['clip_actions']
            
        return config_dict
    
    def _extract_ppo_agent_node(self, ppo_config: Dict, task_name: str = "PPO") -> Dict[str, Any]:
        """Extract configuration for PPOAgent node."""
        # Validate PPO config structure for agent
        if 'params' not in ppo_config:
            raise ValueError("PPO config missing required 'params' section")
        params = ppo_config['params']
        
        if 'network' not in params:
            raise ValueError("PPO params missing required 'network' section")
        network_cfg = params['network']
        
        if 'mlp' not in network_cfg:
            raise ValueError("PPO network config missing required 'mlp' section")
        mlp_cfg = network_cfg['mlp']
        
        if 'config' not in params:
            raise ValueError("PPO params missing required 'config' section")
        algo_config = params['config']
        
        # All network parameters are required
        if 'units' not in mlp_cfg:
            raise ValueError("PPO mlp config missing required 'units' parameter")
        if 'activation' not in mlp_cfg:
            raise ValueError("PPO mlp config missing required 'activation' parameter")
        if 'separate' not in network_cfg:
            raise ValueError("PPO network config missing required 'separate' parameter")
        
        # Convert units list to string format for UI
        units = mlp_cfg['units']
        units_str = str(units).replace("'", '"')  # Convert to JSON-like format
        
        # Validate required training settings
        if 'mixed_precision' not in algo_config:
            raise ValueError("PPO config missing required 'mixed_precision' parameter")
        if 'multi_gpu' not in algo_config:
            raise ValueError("PPO config missing required 'multi_gpu' parameter")
        if 'save_frequency' not in algo_config:
            raise ValueError("PPO config missing required 'save_frequency' parameter")
        
        return {
            # Network architecture
            "network_mlp_layers": units_str,
            "network_activation": mlp_cfg['activation'],
            "separate_value_network": network_cfg['separate'],
            
            # Training settings
            "mixed_precision": algo_config['mixed_precision'],
            "multi_gpu": self._resolve_value(algo_config['multi_gpu'], None),
            
            # Checkpointing
            "checkpoint_interval": algo_config['save_frequency'],
            "keep_checkpoints": 5,  # Default value - TODO: should this be in config?
            
            # Logging
            "log_interval": 10,  # Default value - TODO: should this be in config?
            "save_interval": algo_config['save_frequency'],
            
            # Experiment name
            "experiment_name": f"{task_name}_PPO",
            
            # Load checkpoint (empty by default)
            "load_checkpoint": "",
        }
    
    def _get_num_envs_from_config(self, algo_config: Dict) -> int:
        """Extract num_envs from config, handling references."""
        num_actors = algo_config.get('num_actors', '${....task.env.numEnvs}')
        
        # If it's a reference, use a default
        if isinstance(num_actors, str):
            return 512  # Default
        return int(num_actors)
    
    @classmethod
    def get_instance(cls) -> 'IsaacGymEnvConfigLoader':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Convenience function for node implementations
def get_task_defaults(task_name: str, node_type: str) -> Dict[str, Any]:
    """
    Get default configuration values for a specific task and node type.
    
    Args:
        task_name: Name of the IsaacGymEnvs task (e.g., "Cartpole")
        node_type: One of "isaac_gym_env", "ppo_config", or "ppo_agent"
        
    Returns:
        Dictionary of default values for the node's widgets
    """
    loader = IsaacGymEnvConfigLoader.get_instance()
    return loader.get_node_defaults(task_name, node_type)