#!/usr/bin/env python3
"""
Load PPO configuration for a specific IsaacGymEnvs task.
This can be used by DNNE to get the correct configuration values.
"""

import yaml
import json
import sys
from pathlib import Path

def load_ppo_config(task_name):
    """Load PPO configuration for a specific task."""
    
    # Path to PPO config
    config_path = Path(f"/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg/train/{task_name}PPO.yaml")
    
    if not config_path.exists():
        print(f"Error: PPO config not found for task '{task_name}' at {config_path}")
        return None
    
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract relevant PPO parameters
    params = config.get('params', {})
    algo_config = params.get('config', {})
    
    ppo_config = {
        'minibatch_size': algo_config.get('minibatch_size', 8192),
        'horizon_length': algo_config.get('horizon_length', 16),
        'learning_rate': algo_config.get('learning_rate', 3e-4),
        'schedule_type': algo_config.get('lr_schedule', 'constant'),
        'gamma': algo_config.get('gamma', 0.99),
        'tau': algo_config.get('tau', 0.95),
        'e_clip': algo_config.get('e_clip', 0.2),
        'clip_value': algo_config.get('clip_value', True),
        'mini_epochs': algo_config.get('mini_epochs', 8),
        'critic_coef': algo_config.get('critic_coef', 4),
        'entropy_coef': algo_config.get('entropy_coef', 0.0),
        'bounds_loss_coef': algo_config.get('bounds_loss_coef', 0.0001),
        'max_epochs': algo_config.get('max_epochs', 100),
        'normalize_advantage': algo_config.get('normalize_advantage', True),
        'normalize_input': algo_config.get('normalize_input', True),
        'normalize_value': algo_config.get('normalize_value', True),
        'num_actors': algo_config.get('num_actors', '${....task.env.numEnvs}'),
        'kl_threshold': algo_config.get('kl_threshold', 0.008),
    }
    
    # Also extract network architecture
    network_config = params.get('network', {})
    mlp_config = network_config.get('mlp', {})
    
    ppo_config['network'] = {
        'units': mlp_config.get('units', [256, 256]),
        'activation': mlp_config.get('activation', 'elu'),
        'separate': network_config.get('separate', False),
    }
    
    return ppo_config

def main():
    """Main function to test loading configs."""
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        task_name = "Cartpole"
    
    print(f"Loading PPO config for task: {task_name}")
    config = load_ppo_config(task_name)
    
    if config:
        print(f"\nPPO Configuration for {task_name}:")
        for key, value in config.items():
            if key != 'network':
                print(f"  {key}: {value}")
        
        print(f"\nNetwork Configuration:")
        for key, value in config['network'].items():
            print(f"  {key}: {value}")
        
        # Save as JSON
        output_file = Path(__file__).parent / f"{task_name}_ppo_config.json"
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to: {output_file}")

if __name__ == "__main__":
    main()