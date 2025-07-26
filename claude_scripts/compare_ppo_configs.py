#!/usr/bin/env python3
"""
Compare exported PPO configuration with YAML configuration
"""

import yaml
from pathlib import Path
import re

def load_yaml_config():
    """Load PPO configuration from YAML file."""
    yaml_path = Path("/home/asantanna/DNNE-LINUX-SUPPORT/IsaacGymEnvs/isaacgymenvs/cfg/train/CartpolePPO.yaml")
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    params = config.get('params', {})
    algo_config = params.get('config', {})
    network_config = params.get('network', {})
    mlp_config = network_config.get('mlp', {})
    
    return {
        'minibatch_size': algo_config.get('minibatch_size', 8192),
        'horizon_length': algo_config.get('horizon_length', 16),
        'learning_rate': algo_config.get('learning_rate', 3e-4),
        'schedule_type': algo_config.get('lr_schedule', 'adaptive'),
        'gamma': algo_config.get('gamma', 0.99),
        'tau': algo_config.get('tau', 0.95),
        'e_clip': algo_config.get('e_clip', 0.2),
        'clip_value': algo_config.get('clip_value', True),
        'mini_epochs': algo_config.get('mini_epochs', 8),
        'critic_coef': algo_config.get('critic_coef', 4),
        'entropy_coef': algo_config.get('entropy_coef', 0.0),
        'bounds_loss_coef': algo_config.get('bounds_loss_coef', 0.0001),
        'normalize_advantage': algo_config.get('normalize_advantage', True),
        'normalize_input': algo_config.get('normalize_input', True),
        'normalize_value': algo_config.get('normalize_value', True),
        'network_units': mlp_config.get('units', [256, 256]),
        'network_activation': mlp_config.get('activation', 'elu'),
        'network_separate': network_config.get('separate', False),
    }

def load_exported_config():
    """Load PPO configuration from exported runner.py file."""
    runner_path = Path("/mnt/e/ALS-Projects/DNNE/DNNE-UI/export_system/exports/Cartpole_PPO/nodes/ppoagentnode_11.py")
    with open(runner_path, 'r') as f:
        content = f.read()
    
    # Extract self.ppo_config dictionary
    ppo_match = re.search(r'self\.ppo_config = \{([^}]+)\}', content, re.DOTALL)
    if not ppo_match:
        print("ERROR: Could not find self.ppo_config in exported file")
        return {}
    
    ppo_config_str = ppo_match.group(1)
    
    # Extract values
    exported = {}
    
    # Parse each line
    for line in ppo_config_str.split('\n'):
        line = line.strip()
        if ':' in line:
            # Clean up the line
            line = line.rstrip(',')
            key, value = line.split(':', 1)
            key = key.strip().strip("'\"")
            value = value.strip()
            
            # Convert value to appropriate type
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif value.startswith("'") or value.startswith('"'):
                value = value.strip("'\"")
            else:
                try:
                    # Try to convert to number
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass
            
            exported[key] = value
    
    # Also extract network configuration
    network_match = re.search(r'self\.network_mlp_layers = \[([^\]]+)\]', content)
    if network_match:
        units_str = network_match.group(1)
        exported['network_units'] = [int(x.strip()) for x in units_str.split(',')]
    
    activation_match = re.search(r'self\.network_activation = ["\']([^"\']+)["\']', content)
    if activation_match:
        exported['network_activation'] = activation_match.group(1)
    
    separate_match = re.search(r'self\.separate_value_network = (True|False)', content)
    if separate_match:
        exported['network_separate'] = separate_match.group(1) == 'True'
    
    return exported

def compare_configs():
    """Compare YAML and exported configurations."""
    yaml_config = load_yaml_config()
    exported_config = load_exported_config()
    
    print("Configuration Comparison: YAML vs Exported")
    print("=" * 60)
    
    # Compare each field
    all_keys = set(yaml_config.keys()) | set(exported_config.keys())
    
    differences = []
    matches = []
    missing_in_export = []
    extra_in_export = []
    
    for key in sorted(all_keys):
        yaml_value = yaml_config.get(key, "NOT_IN_YAML")
        export_value = exported_config.get(key, "NOT_IN_EXPORT")
        
        if yaml_value == "NOT_IN_YAML":
            extra_in_export.append((key, export_value))
        elif export_value == "NOT_IN_EXPORT":
            missing_in_export.append((key, yaml_value))
        elif yaml_value != export_value:
            # Special handling for float comparison
            if isinstance(yaml_value, float) and isinstance(export_value, float):
                if abs(yaml_value - export_value) < 1e-10:
                    matches.append((key, yaml_value, export_value))
                else:
                    differences.append((key, yaml_value, export_value))
            else:
                differences.append((key, yaml_value, export_value))
        else:
            matches.append((key, yaml_value, export_value))
    
    # Print results
    if matches:
        print("\n✅ Matching Values:")
        for key, yaml_val, export_val in matches:
            print(f"  {key}: {yaml_val}")
    
    if differences:
        print("\n❌ Different Values:")
        for key, yaml_val, export_val in differences:
            print(f"  {key}:")
            print(f"    YAML:     {yaml_val}")
            print(f"    Exported: {export_val}")
    
    if missing_in_export:
        print("\n⚠️ Missing in Export:")
        for key, yaml_val in missing_in_export:
            print(f"  {key}: {yaml_val}")
    
    if extra_in_export:
        print("\n📌 Extra in Export (not in YAML):")
        for key, export_val in extra_in_export:
            print(f"  {key}: {export_val}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Matches: {len(matches)}")
    print(f"  Differences: {len(differences)}")
    print(f"  Missing in export: {len(missing_in_export)}")
    print(f"  Extra in export: {len(extra_in_export)}")

if __name__ == "__main__":
    compare_configs()