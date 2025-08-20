"""
Override Parser Module for DNNE Runner

Handles parsing of --override command line arguments to allow runtime
configuration of node parameters without re-exporting from the UI.

Syntax: --override node_id:param=value,node_id:param=value
        --override subsystem:param=value (applies to all nodes in subsystem)
"""

import re
from typing import Dict, List, Tuple, Any, Union


def parse_override_args(override_string: str, subsystem_to_nodes: Dict[str, List[str]] = None) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Parse --override string into node configurations
    
    Args:
        override_string: Command line override string
        subsystem_to_nodes: Optional mapping of subsystem names to node IDs
        
    Returns:
        Tuple of (configs dict, errors list)
        configs: {node_id: {param: value}}
        errors: List of error messages
        
    Examples:
        56:checkpoint_enabled=True
        56:checkpoint_trigger_type="At End"
        42:learning_rate=0.001,56:checkpoint_enabled=True
        training:learning_rate=0.001 (applies to all training nodes)
    """
    configs = {}
    errors = []
    
    if not override_string:
        return configs, errors
    
    # Split by comma, but respect quoted strings
    # This regex splits by comma but not within quotes
    parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', override_string)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Parse single override
        result = parse_single_override(part, subsystem_to_nodes)
        if isinstance(result, str):  # Error message
            errors.append(result)
        else:
            targets, param, value = result
            # targets can be a list of node IDs (from subsystem expansion) or single node ID
            if isinstance(targets, list):
                for node_id in targets:
                    if node_id not in configs:
                        configs[node_id] = {}
                    configs[node_id][param] = value
            else:
                node_id = targets
                if node_id not in configs:
                    configs[node_id] = {}
                configs[node_id][param] = value
            
    return configs, errors


def parse_single_override(override_str: str, subsystem_to_nodes: Dict[str, List[str]] = None) -> Union[Tuple[Union[str, List[str]], str, Any], str]:
    """Parse a single override expression
    
    Args:
        override_str: Single override like "56:checkpoint_enabled=True" or "training:learning_rate=0.001"
        subsystem_to_nodes: Optional mapping of subsystem names to node IDs
        
    Returns:
        Either (target, param, value) tuple or error message string
        target can be a single node_id string or list of node_ids (from subsystem expansion)
    """
    # Match pattern: target:param=value (allowing spaces around colon)
    # target can be a node_id (digits) or subsystem name (word characters)
    match = re.match(r'^([\w]+)\s*:\s*([^=]+)=(.*)$', override_str.strip())
    if not match:
        return f"Invalid override format: '{override_str}' (expected: node_id:param=value or subsystem:param=value)"
        
    target = match.group(1)
    param = match.group(2).strip()
    value_str = match.group(3).strip()
    
    # Parse the value
    value = parse_value(value_str)
    
    # Check if target is a subsystem name
    if subsystem_to_nodes and target in subsystem_to_nodes:
        # Expand subsystem to all its node IDs
        node_ids = subsystem_to_nodes[target]
        if not node_ids:
            return f"Subsystem '{target}' has no nodes in this workflow"
        return node_ids, param, value
    elif target.isdigit():
        # It's a node ID
        return target, param, value
    else:
        # Unknown target - could be a node ID that's not purely numeric or an unknown subsystem
        # Treat it as a node ID for backward compatibility
        return target, param, value


def parse_value(value_str: str) -> Any:
    """Parse a value string into appropriate Python type
    
    Args:
        value_str: String representation of value
        
    Returns:
        Parsed value (bool, int, float, or string)
        
    Priority:
    1. Quoted strings (preserves spaces and special chars)
    2. Boolean literals (True/False, case insensitive)
    3. Numeric values (int or float)
    4. Unquoted strings (fallback)
    """
    # Check for quoted string
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]  # Remove quotes
        
    # Check for boolean (case insensitive)
    if value_str.lower() == 'true':
        return True
    elif value_str.lower() == 'false':
        return False
        
    # Try numeric conversion
    try:
        # Try int first (no decimal point)
        if '.' not in value_str:
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        # Keep as string if not numeric
        return value_str


def format_override_example() -> str:
    """Return formatted example usage for help text"""
    return """Examples:
  Enable checkpoint for node 56:
    --override 56:checkpoint_enabled=True
    
  Set multiple parameters:
    --override 56:checkpoint_enabled=True,56:checkpoint_trigger_type=end
    
  Apply to all training nodes:
    --override training:learning_rate=0.001
    
  Apply to all RL nodes:
    --override rl:gamma=0.99,rl:clip_epsilon=0.2
    
  Mix node IDs and subsystems:
    --override training:learning_rate=0.001,56:checkpoint_enabled=True,64:max_epochs=50
    
  String values with spaces must be quoted:
    --override 56:checkpoint_trigger_type="At End"
    
  Available subsystems:
    training, data, network, rl, robotics, control, util, telemetry, monitoring
"""


if __name__ == "__main__":
    # Simple test cases
    test_cases = [
        '56:checkpoint_enabled=True',
        '56:checkpoint_trigger_type=end',
        '42:learning_rate=0.001,56:checkpoint_enabled=True',
        '38:batch_size=128,39:max_iterations=5000',
        'invalid:format',
        '56:empty=',
        '56:quoted_string="hello world"',
    ]
    
    print("Testing override parser:")
    for test in test_cases:
        print(f"\nInput: {test}")
        configs, errors = parse_override_args(test)
        if configs:
            print(f"  Configs: {configs}")
        if errors:
            print(f"  Errors: {errors}")