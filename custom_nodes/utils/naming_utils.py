"""
Shared utilities for converting between naming conventions
"""

import re


def to_snake_case(name: str) -> str:
    """
    Convert PascalCase or camelCase to snake_case.
    
    Examples:
        MNISTDatasetNode -> mnist_dataset_node
        LinearLayer -> linear_layer
        PPOAgent -> ppo_agent
        ORNode -> or_node
    """
    # Handle acronyms and consecutive capitals
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def to_pascal_case(name: str) -> str:
    """
    Convert snake_case to PascalCase.
    
    Examples:
        mnist_dataset_node -> MnistDatasetNode
        linear_layer -> LinearLayer
        ppo_agent -> PpoAgent
        or_node -> OrNode
    """
    components = name.split('_')
    return ''.join(x.capitalize() for x in components)


def node_class_to_exporter_filename(class_name: str) -> str:
    """
    Convert a node class name to its exporter filename.
    
    Examples:
        MNISTDatasetNode -> mnist_dataset_exporter.py
        LinearLayerNode -> linear_layer_exporter.py
        PPOAgent -> ppo_agent_exporter.py
    """
    # Remove 'Node' suffix if present
    if class_name.endswith('Node'):
        class_name = class_name[:-4]
    
    # Convert to snake_case and add _exporter.py
    base_name = to_snake_case(class_name)
    return f"{base_name}_exporter.py"


def node_class_to_exporter_class(class_name: str) -> str:
    """
    Convert a node class name to its exporter class name.
    
    Examples:
        MNISTDatasetNode -> MNISTDatasetExporter
        LinearLayerNode -> LinearLayerExporter
        PPOAgent -> PPOAgentExporter
    """
    # Remove 'Node' suffix if present
    if class_name.endswith('Node'):
        class_name = class_name[:-4]
    
    # Add Exporter suffix
    return f"{class_name}Exporter"


def visnode_filename_to_node_class(filename: str) -> str:
    """
    Convert a visnode filename to expected node class name.
    
    Examples:
        mnist_dataset_visnode.py -> MNISTDatasetNode
        linear_layer_visnode.py -> LinearLayerNode
        ppo_agent_visnode.py -> PPOAgentNode
    """
    # Remove _visnode.py suffix
    if filename.endswith('_visnode.py'):
        base_name = filename[:-11]
    else:
        base_name = filename
    
    # Special handling for acronyms
    special_cases = {
        'mnist_dataset': 'MNISTDataset',
        'cifar10_dataset': 'CIFAR10Dataset',
        'sgd_optimizer': 'SGDOptimizer',
        'ppo_agent': 'PPOAgent',
        'ppo_config': 'PPOConfig',
        'or': 'OR',
    }
    
    if base_name in special_cases:
        return special_cases[base_name] + 'Node'
    
    # Default conversion
    return to_pascal_case(base_name) + 'Node'