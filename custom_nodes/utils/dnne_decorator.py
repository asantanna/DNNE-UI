"""
DNNE node decorator for automatic registration and metadata management
"""

from typing import Dict, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Global registry for decorated nodes
_node_registry: Dict[str, Dict[str, Any]] = {}


def dnne_node(*, is_virtual: bool):
    """
    Decorator for DNNE nodes that handles registration and metadata.
    
    Args:
        is_virtual: Required keyword-only parameter. True for configuration-only nodes
                   that don't generate runtime code, False for nodes that do.
    
    Usage:
        @dnne_node(is_virtual=False)
        class MNISTDatasetNode(RoboticsNodeBase):
            ...
    """
    def decorator(cls: Type) -> Type:
        # Store metadata about the node
        node_name = cls.__name__
        _node_registry[node_name] = {
            'class': cls,
            'is_virtual': is_virtual,
            'module': cls.__module__,
        }
        
        # Add a class attribute for easy access (but don't override if it exists)
        if not hasattr(cls, '_dnne_metadata'):
            cls._dnne_metadata = {
                'is_virtual': is_virtual,
            }
        
        logger.debug(f"Registered node {node_name} (virtual={is_virtual})")
        
        return cls
    
    return decorator


def get_node_registry() -> Dict[str, Dict[str, Any]]:
    """Get the complete node registry."""
    return _node_registry.copy()


def get_node_metadata(node_class_or_name) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a node by class or name.
    
    Args:
        node_class_or_name: Either the node class or its string name
        
    Returns:
        Dictionary with node metadata or None if not found
    """
    if isinstance(node_class_or_name, str):
        return _node_registry.get(node_class_or_name)
    
    # If it's a class, find it by checking the registry
    for name, metadata in _node_registry.items():
        if metadata['class'] is node_class_or_name:
            return metadata
    
    # Check if the class has metadata attached directly
    if hasattr(node_class_or_name, '_dnne_metadata'):
        return node_class_or_name._dnne_metadata
    
    return None


def is_virtual_node(node_class_or_name) -> bool:
    """
    Check if a node is virtual (configuration-only).
    
    Args:
        node_class_or_name: Either the node class or its string name
        
    Returns:
        True if the node is virtual, False otherwise
    """
    metadata = get_node_metadata(node_class_or_name)
    if metadata:
        return metadata.get('is_virtual', False)
    return False


def get_all_node_classes() -> Dict[str, Type]:
    """
    Get all registered node classes.
    
    Returns:
        Dictionary mapping node names to their classes
    """
    return {name: metadata['class'] for name, metadata in _node_registry.items()}


def clear_registry():
    """Clear the node registry. Mainly useful for testing."""
    global _node_registry
    _node_registry = {}