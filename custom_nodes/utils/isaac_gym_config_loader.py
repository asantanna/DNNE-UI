"""
Isaac Gym Environment Configuration Loader for DNNE.
Provides environment-specific configurations for the 3-node PPO setup.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


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
            
        # Determine the correct path based on platform
        if isaac_gym_envs_path is None:
            import platform
            if platform.system() == "Windows":
                # Running on Windows - use WSL path
                isaac_gym_envs_path = r"\\wsl.localhost\Ubuntu\home\asantanna\DNNE-LINUX-SUPPORT\IsaacGymEnvs"
            else:
                # Running on Linux/WSL
                isaac_gym_envs_path = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"
        
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
            
        return self._configs_cache.get(task_name, {})
    
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
        node_key = f"{node_type}_node"
        return config.get(node_key, {})
    
    def _load_all_configs(self):
        """Load all configurations and cache them."""
        self._configs_cache = {}
        
        if not self.task_cfg_path.exists():
            logger.error(f"Task config path not found: {self.task_cfg_path}")
            return
            
        # Find all tasks with PPO configs
        for task_file in sorted(self.task_cfg_path.glob("*.yaml")):
            if task_file.is_file():
                task_name = task_file.stem
                ppo_file = self.train_cfg_path / f"{task_name}PPO.yaml"
                
                if ppo_file.exists():
                    config = self._load_environment_config(task_name, task_file, ppo_file)
                    if config:
                        self._configs_cache[task_name] = config
    
    def _load_environment_config(self, task_name: str, task_file: Path, ppo_file: Path) -> Optional[Dict[str, Any]]:
        """Load configuration for a specific environment."""
        try:
            # Load task config
            with open(task_file, 'r') as f:
                task_config = yaml.safe_load(f)
                
            # Load PPO config
            with open(ppo_file, 'r') as f:
                ppo_config = yaml.safe_load(f)
                
            # Extract configurations for each node
            return {
                "task_name": task_name,
                "isaac_gym_env_node": self._extract_env_node_config(task_config, ppo_config),
                "ppo_config_node": self._extract_ppo_config_node(ppo_config),
                "ppo_agent_node": self._extract_ppo_agent_node(ppo_config, task_name)
            }
            
        except Exception as e:
            logger.error(f"Error loading config for {task_name}: {e}")
            return None
    
    def _resolve_value(self, value, default=None):
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
    
    def _extract_env_node_config(self, task_config: Dict, ppo_config: Dict) -> Dict[str, Any]:
        """Extract configuration for IsaacGymEnvs node."""
        env_cfg = task_config.get('env', {})
        sim_cfg = task_config.get('sim', {})
        
        # Get num_envs from PPO config if available
        ppo_params = ppo_config.get('params', {}).get('config', {})
        num_actors = ppo_params.get('num_actors', '${....task.env.numEnvs}')
        
        # Resolve num_envs
        num_envs_raw = env_cfg.get('numEnvs', 512)
        num_envs = self._resolve_value(num_envs_raw, 512)
        
        # If num_actors references numEnvs, use the resolved value
        if isinstance(num_actors, str) and 'numEnvs' in num_actors:
            num_actors = num_envs
        else:
            num_actors = self._resolve_value(num_actors, num_envs)
        
        # Get physics config with proper resolution
        physx_cfg = sim_cfg.get('physx', {})
        
        # Get global config values
        global_cfg = self._global_config
        
        return {
            "num_envs": num_envs,
            "seed": global_cfg.get('seed', 42),
            "control_after_generate": "fixed",  # Use the actual widget name
            "headless": global_cfg.get('headless', False),
            "graphics_device_id": global_cfg.get('graphics_device_id', 0),
            "sim_device": global_cfg.get('sim_device', 'cuda:0'),
            "physics_engine": self._resolve_value(
                task_config.get('physics_engine', global_cfg.get('physics_engine', 'physx')), 
                'physx'
            ),
            "multi_gpu": global_cfg.get('multi_gpu', False),
            "enable_cameras": env_cfg.get('enableCameraSensors', False),
            "force_render": global_cfg.get('force_render', True),  # From global config!
            "use_gpu_pipeline": self._resolve_value(
                sim_cfg.get('use_gpu_pipeline', global_cfg.get('pipeline', 'gpu') == 'gpu'), 
                True
            ),
            "num_threads": self._resolve_value(
                physx_cfg.get('num_threads', global_cfg.get('num_threads', 4)), 
                4
            ),
            "solver_type": self._resolve_value(
                physx_cfg.get('solver_type', global_cfg.get('solver_type', 1)), 
                1
            ),
            "num_subscenes": self._resolve_value(
                physx_cfg.get('num_subscenes', global_cfg.get('num_subscenes', 4)), 
                4
            ),
        }
    
    def _extract_ppo_config_node(self, ppo_config: Dict) -> Dict[str, Any]:
        """Extract configuration for PPOConfig node."""
        params = ppo_config.get('params', {})
        algo_config = params.get('config', {})
        
        # Critical: Get the actual minibatch_size, not num_minibatches!
        minibatch_size = algo_config.get('minibatch_size', 32768)
        horizon_length = algo_config.get('horizon_length', 16)
        
        # Some configs use mini_epochs, some use num_epochs
        mini_epochs = algo_config.get('mini_epochs', algo_config.get('num_epochs', 8))
        
        # DNNE now uses minibatch_size directly, matching the YAML configuration
        
        config_dict = {
            # Core PPO parameters
            "learning_rate": float(algo_config.get('learning_rate', 3e-4)),
            "num_epochs": mini_epochs,  # DNNE calls it num_epochs
            "minibatch_size": minibatch_size,  # Direct from YAML
            "horizon_length": horizon_length,
            
            # PPO specific
            "clip_param": algo_config.get('e_clip', 0.2),
            "value_loss_coef": algo_config.get('critic_coef', 2.0),
            "entropy_coef": algo_config.get('entropy_coef', 0.0),
            "gamma": algo_config.get('gamma', 0.99),
            "gae_lambda": algo_config.get('tau', 0.95),
            
            # Learning rate schedule
            "lr_schedule": algo_config.get('lr_schedule', 'adaptive'),
            "lr_schedule_kl_threshold": algo_config.get('kl_threshold', 0.008),
            
            # Gradient clipping
            "max_grad_norm": algo_config.get('grad_norm', 1.0),
            "truncate_grads": algo_config.get('truncate_grads', True),
            
            # Normalization
            "normalize_advantage": algo_config.get('normalize_advantage', True),
            "normalize_input": algo_config.get('normalize_input', True),
            "normalize_value": algo_config.get('normalize_value', True),
            
            # Value function
            "use_clipped_value_loss": algo_config.get('clip_value', True),
            
            # Training duration
            "max_iterations": self._resolve_value(algo_config.get('max_epochs', 1000), 1000),
            
            # Reward shaping
            "reward_shaper_scale": algo_config.get('reward_shaper', {}).get('scale_value', 1.0),
            
            # Additional e_clip parameter that might be in UI
            "e_clip": algo_config.get('e_clip', 0.2),
            
            # Bounds loss coefficient
            "bounds_loss_coef": algo_config.get('bounds_loss_coef', 0.0001),
        }
        
        # Only add these parameters if they exist in the YAML
        if 'value_bootstrap' in algo_config:
            config_dict['value_bootstrap'] = algo_config['value_bootstrap']
        if 'clip_actions' in algo_config:
            config_dict['clip_actions'] = algo_config['clip_actions']
            
        return config_dict
    
    def _extract_ppo_agent_node(self, ppo_config: Dict, task_name: str = "PPO") -> Dict[str, Any]:
        """Extract configuration for PPOAgent node."""
        params = ppo_config.get('params', {})
        network_cfg = params.get('network', {})
        mlp_cfg = network_cfg.get('mlp', {})
        algo_config = params.get('config', {})
        
        # Convert units list to string format for UI
        units = mlp_cfg.get('units', [256, 256])
        units_str = str(units).replace("'", '"')  # Convert to JSON-like format
        
        return {
            # Network architecture
            "network_mlp_layers": units_str,
            "network_activation": mlp_cfg.get('activation', 'elu'),
            "separate_value_network": network_cfg.get('separate', False),
            
            # Training settings
            "mixed_precision": algo_config.get('mixed_precision', False),
            "multi_gpu": self._resolve_value(algo_config.get('multi_gpu', False), False),
            
            # Checkpointing
            "checkpoint_interval": algo_config.get('save_frequency', 100),
            "keep_checkpoints": 5,  # Default value
            
            # Logging
            "log_interval": 10,  # Default value
            "save_interval": algo_config.get('save_frequency', 1000),
            
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