#!/usr/bin/env python3
"""
Complete configuration loader for IsaacGymEnvs environments.
Extracts all necessary parameters for DNNE's 3-node PPO setup.
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class IsaacGymEnvConfigLoader:
    """Load and parse IsaacGymEnvs configurations for DNNE."""
    
    def __init__(self, isaac_gym_envs_path: str = "/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs"):
        self.base_path = Path(isaac_gym_envs_path)
        self.cfg_path = self.base_path / "isaacgymenvs" / "cfg"
        self.task_cfg_path = self.cfg_path / "task"
        self.train_cfg_path = self.cfg_path / "train"
        
    def get_available_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tasks with their PPO compatibility."""
        tasks = {}
        
        if not self.task_cfg_path.exists():
            print(f"Error: Task config path not found: {self.task_cfg_path}")
            return tasks
            
        for task_file in sorted(self.task_cfg_path.glob("*.yaml")):
            if task_file.is_file():
                task_name = task_file.stem
                ppo_file = self.train_cfg_path / f"{task_name}PPO.yaml"
                
                tasks[task_name] = {
                    "task_config": str(task_file),
                    "has_ppo": ppo_file.exists(),
                    "ppo_config": str(ppo_file) if ppo_file.exists() else None
                }
                
        return tasks
    
    def load_task_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Load task-specific configuration."""
        task_file = self.task_cfg_path / f"{task_name}.yaml"
        
        if not task_file.exists():
            print(f"Error: Task config not found: {task_file}")
            return None
            
        try:
            with open(task_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading task config: {e}")
            return None
    
    def load_ppo_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Load PPO training configuration."""
        ppo_file = self.train_cfg_path / f"{task_name}PPO.yaml"
        
        if not ppo_file.exists():
            print(f"Error: PPO config not found: {ppo_file}")
            return None
            
        try:
            with open(ppo_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading PPO config: {e}")
            return None
    
    def get_environment_config(self, task_name: str) -> Dict[str, Any]:
        """
        Get complete configuration for all 3 DNNE nodes.
        Returns a dictionary with sections for each node.
        """
        # Load both configs
        task_config = self.load_task_config(task_name)
        ppo_config = self.load_ppo_config(task_name)
        
        if not task_config or not ppo_config:
            return {}
        
        # Extract values for each node
        result = {
            "task_name": task_name,
            "isaac_gym_env_node": self._extract_env_node_config(task_config, ppo_config),
            "ppo_config_node": self._extract_ppo_config_node(ppo_config),
            "ppo_agent_node": self._extract_ppo_agent_node(ppo_config, task_name)
        }
        
        return result
    
    def _extract_env_node_config(self, task_config: Dict, ppo_config: Dict) -> Dict[str, Any]:
        """Extract configuration for IsaacGymEnvs node."""
        env_cfg = task_config.get('env', {})
        sim_cfg = task_config.get('sim', {})
        
        # Get num_envs from PPO config if available
        ppo_params = ppo_config.get('params', {}).get('config', {})
        num_actors = ppo_params.get('num_actors', '${....task.env.numEnvs}')
        
        # If num_actors references numEnvs, get it from task config
        if isinstance(num_actors, str) and 'numEnvs' in num_actors:
            num_envs = env_cfg.get('numEnvs', 512)
        else:
            num_envs = num_actors
        
        return {
            "num_envs": num_envs,
            "episode_length": env_cfg.get('episodeLength', 500),
            "enable_debug_vis": env_cfg.get('enableDebugVis', False),
            "aggregate_mode": env_cfg.get('aggregateMode', 3),
            "control_frequency_inv": env_cfg.get('controlFrequencyInv', 1),
            "clip_observations": env_cfg.get('clipObservations', 100.0),
            "clip_actions": env_cfg.get('clipActions', 100.0),
            
            # Sim parameters
            "dt": sim_cfg.get('dt', 0.0166),  # 60 Hz default
            "substeps": sim_cfg.get('substeps', 1),
            "gravity": sim_cfg.get('gravity', [0.0, 0.0, -9.81]),
            "physics_engine": sim_cfg.get('physx', {}).get('solver_type', 1),
            "use_gpu_pipeline": sim_cfg.get('use_gpu_pipeline', True),
            
            # Domain randomization
            "enable_domain_randomization": env_cfg.get('enableCameraSensors', False),
        }
    
    def _extract_ppo_config_node(self, ppo_config: Dict) -> Dict[str, Any]:
        """Extract configuration for PPOConfig node."""
        params = ppo_config.get('params', {})
        algo_config = params.get('config', {})
        
        # Calculate minibatch_size correctly
        minibatch_size = algo_config.get('minibatch_size', 32768)
        horizon_length = algo_config.get('horizon_length', 16)
        
        # Some configs use mini_epochs, some use num_epochs
        mini_epochs = algo_config.get('mini_epochs', algo_config.get('num_epochs', 8))
        
        return {
            # Core PPO parameters
            "learning_rate": algo_config.get('learning_rate', 3e-4),
            "mini_epochs": mini_epochs,
            "minibatch_size": minibatch_size,  # This is the critical value!
            "horizon_length": horizon_length,
            
            # PPO specific
            "clip_param": algo_config.get('e_clip', 0.2),
            "value_loss_coef": algo_config.get('critic_coef', 2.0),
            "entropy_coef": algo_config.get('entropy_coef', 0.0),
            "gamma": algo_config.get('gamma', 0.99),
            "gae_lambda": algo_config.get('tau', 0.95),
            
            # Learning rate schedule
            "lr_schedule": algo_config.get('lr_schedule', 'adaptive'),
            "kl_threshold": algo_config.get('kl_threshold', 0.008),
            
            # Gradient clipping
            "max_grad_norm": algo_config.get('grad_norm', 1.0),
            "truncate_grads": algo_config.get('truncate_grads', True),
            
            # Normalization
            "normalize_advantage": algo_config.get('normalize_advantage', True),
            "normalize_input": algo_config.get('normalize_input', True),
            "normalize_value": algo_config.get('normalize_value', True),
            
            # Value function
            "use_clipped_value_loss": algo_config.get('clip_value', True),
            "value_bootstrap": algo_config.get('value_bootstrap', True),
            
            # Bounds loss (for continuous control)
            "bounds_loss_coef": algo_config.get('bounds_loss_coef', 0.0),
            
            # Training duration
            "max_iterations": algo_config.get('max_epochs', 1000),
            
            # Reward shaping
            "reward_shaper_scale": algo_config.get('reward_shaper', {}).get('scale_value', 1.0),
        }
    
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
            
            # Initialization
            "network_initializer": mlp_cfg.get('initializer', {}).get('name', 'default'),
            
            # Training settings
            "mixed_precision": algo_config.get('mixed_precision', False),
            "multi_gpu": algo_config.get('multi_gpu', False),
            
            # Checkpointing
            "checkpoint_interval": algo_config.get('save_frequency', 100),
            "keep_checkpoints": 5,  # Usually not specified in IGE configs
            
            # Logging
            "log_interval": 10,  # Usually not specified
            "save_interval": algo_config.get('save_frequency', 1000),
            
            # Experiment name (can be customized by user)
            "experiment_name": algo_config.get('name', f'{task_name}_PPO'),
            
            # Additional settings
            "score_to_win": algo_config.get('score_to_win', float('inf')),
            "max_agent_steps": algo_config.get('max_agent_steps', 1e10),
        }
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get configurations for all available tasks."""
        tasks = self.get_available_tasks()
        configs = {}
        
        for task_name, task_info in tasks.items():
            if task_info['has_ppo']:
                config = self.get_environment_config(task_name)
                if config:
                    configs[task_name] = config
                    
        return configs
    
    def save_configs_to_file(self, output_file: Path):
        """Save all configurations to a JSON file."""
        configs = self.get_all_configs()
        
        with open(output_file, 'w') as f:
            json.dump(configs, f, indent=2)
            
        print(f"Saved {len(configs)} environment configurations to {output_file}")

def main():
    """Test the configuration loader."""
    loader = IsaacGymEnvConfigLoader()
    
    # Test with specific environment
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        task_name = "Cartpole"
    
    print(f"\n{'='*60}")
    print(f"Configuration for {task_name}")
    print('='*60)
    
    config = loader.get_environment_config(task_name)
    
    if not config:
        print(f"Failed to load configuration for {task_name}")
        return
    
    # Print configuration for each node
    print("\n1. IsaacGymEnvs Node Configuration:")
    print("-" * 40)
    for key, value in config['isaac_gym_env_node'].items():
        print(f"  {key}: {value}")
    
    print("\n2. PPOConfig Node Configuration:")
    print("-" * 40)
    for key, value in config['ppo_config_node'].items():
        print(f"  {key}: {value}")
    
    print("\n3. PPOAgent Node Configuration:")
    print("-" * 40)
    for key, value in config['ppo_agent_node'].items():
        print(f"  {key}: {value}")
    
    # Save all configs
    output_file = Path(__file__).parent / "isaac_gym_all_env_configs.json"
    loader.save_configs_to_file(output_file)
    
    # Also save just this environment's config
    env_output = Path(__file__).parent / f"{task_name}_dnne_config.json"
    with open(env_output, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {env_output}")

if __name__ == "__main__":
    main()