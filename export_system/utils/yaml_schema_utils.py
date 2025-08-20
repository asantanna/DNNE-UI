"""
YAML Schema Navigation Utilities for DNNE Export System

Provides utilities for navigating hierarchical YAML schemas used by IsaacGymEnvs,
especially for DNNE environments with nested configuration levels.
"""

from typing import Dict, Any, Optional, List, Tuple


def navigate_schema(yaml_config: Dict[str, Any], path: List[str]) -> Optional[Dict[str, Any]]:
    """
    Navigate through a nested YAML schema following a path.
    
    Args:
        yaml_config: The root YAML configuration dictionary
        path: List of keys to navigate (e.g., ['env', 'dnne', 'nested_schemas'])
        
    Returns:
        The value at the path, or None if path doesn't exist
    """
    current = yaml_config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def get_dnne_schema_levels(yaml_config: Dict[str, Any]) -> List[str]:
    """
    Get the schema levels defined in a DNNE configuration.
    
    Args:
        yaml_config: Task YAML configuration
        
    Returns:
        List of schema level names (e.g., ['subtask', 'controlType'])
    """
    dnne_config = navigate_schema(yaml_config, ['env', 'dnne'])
    
    if dnne_config and 'schema_levels' in dnne_config:
        return dnne_config['schema_levels']
    return []


def get_nested_schema_value(
    yaml_config: Dict[str, Any], 
    level_values: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Navigate to a specific schema based on level selections.
    
    Args:
        yaml_config: Task YAML configuration
        level_values: Dictionary mapping level names to selected values
                     (e.g., {'subtask': 'random_target', 'controlType': 'joint_tor'})
        
    Returns:
        The schema dictionary at the specified path, or None if not found
    """
    # Find DNNE config - only look under env
    dnne_config = navigate_schema(yaml_config, ['env', 'dnne'])
    
    if not dnne_config or 'nested_schemas' not in dnne_config:
        return None
    
    # Get schema levels in order
    schema_levels = dnne_config.get('schema_levels', [])
    if not schema_levels:
        return None
    
    # Navigate through nested schemas
    current_schema = dnne_config['nested_schemas']
    
    for level in schema_levels:
        if level not in level_values:
            # Level not specified, can't continue navigation
            break
            
        value = level_values[level]
        if not isinstance(current_schema, dict) or value not in current_schema:
            # Path doesn't exist
            return None
            
        current_schema = current_schema[value]
    
    return current_schema if isinstance(current_schema, dict) else None


def extract_observation_action_sizes(
    yaml_config: Dict[str, Any],
    level_values: Dict[str, str]
) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract numObservations and numActions from a DNNE schema.
    
    Args:
        yaml_config: Task YAML configuration
        level_values: Dictionary mapping level names to selected values
        
    Returns:
        Tuple of (numObservations, numActions), either can be None if not found
    """
    schema = get_nested_schema_value(yaml_config, level_values)
    
    if schema:
        num_obs = schema.get('numObservations')
        num_acts = schema.get('numActions')
        return num_obs, num_acts
    
    # Fallback: check root env config for non-DNNE tasks
    env_config = navigate_schema(yaml_config, ['env'])
    if env_config:
        num_obs = env_config.get('numObservations')
        num_acts = env_config.get('numActions')
        return num_obs, num_acts
    
    return None, None


def get_null_action(
    yaml_config: Dict[str, Any],
    level_values: Dict[str, str]
) -> Optional[List[float]]:
    """
    Extract nullAction from a DNNE schema.
    
    Args:
        yaml_config: Task YAML configuration
        level_values: Dictionary mapping level names to selected values
        
    Returns:
        List of null action values, or None if not found
    """
    schema = get_nested_schema_value(yaml_config, level_values)
    
    if schema and 'nullAction' in schema:
        return schema['nullAction']
    
    # Fallback: check root env config
    env_config = navigate_schema(yaml_config, ['env'])
    if env_config and 'nullAction' in env_config:
        return env_config['nullAction']
    
    return None


def get_schema_defaults(yaml_config: Dict[str, Any]) -> Dict[str, str]:
    """
    Get default values for each schema level.
    
    Args:
        yaml_config: Task YAML configuration
        
    Returns:
        Dictionary mapping level names to their default values
    """
    dnne_config = navigate_schema(yaml_config, ['env', 'dnne'])
    
    if not dnne_config:
        return {}
    
    defaults = {}
    schema_levels = dnne_config.get('schema_levels', [])
    
    for level in schema_levels:
        default_key = f'default_{level}'
        if default_key in dnne_config:
            defaults[level] = dnne_config[default_key]
    
    return defaults


def has_dnne_schema(yaml_config: Dict[str, Any]) -> bool:
    """
    Check if a YAML configuration has DNNE schema support.
    
    Args:
        yaml_config: Task YAML configuration
        
    Returns:
        True if DNNE schemas are present, False otherwise
    """
    dnne_config = navigate_schema(yaml_config, ['env', 'dnne'])
    
    return dnne_config is not None and 'nested_schemas' in dnne_config