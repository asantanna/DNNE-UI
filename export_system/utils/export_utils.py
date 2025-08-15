#!/usr/bin/env python3
"""
Utility functions for the export system.
Provides context-aware helpers for node traversal and exporter access.
"""

import logging
from typing import Optional, Dict, Any, List

# Global reference to current export context
_current_context = None

def set_export_context(context: Dict[str, Any]):
    """Set the current export context (called by GraphExporter)"""
    global _current_context
    _current_context = context

def clear_export_context():
    """Clear the current export context"""
    global _current_context
    _current_context = None

def get_export_context() -> Dict[str, Any]:
    """Get the current export context"""
    if _current_context is None:
        raise RuntimeError("No export context available. This function must be called during export.")
    return _current_context

def get_node_by_id(node_id: str) -> Optional[Dict]:
    """Get node data by ID from current export context"""
    context = get_export_context()
    nodes = context.get('nodes', [])
    
    for node in nodes:
        if str(node.get('id')) == str(node_id):
            return node
    return None

def get_node_exporter(node_type: str):
    """Get exporter class for a node type"""
    context = get_export_context()
    node_registry = context.get('node_registry', {})
    return node_registry.get(node_type)

def follow_node_connection(node_id: str, output_name: str) -> Optional[str]:
    """
    Follow a connection from a node's output to find the connected node.
    
    Args:
        node_id: ID of the source node
        output_name: Name of the output connector (e.g., "layers", "output")
    
    Returns:
        ID of the connected node, or None if no connection found
    """
    context = get_export_context()
    links = context.get('links', [])
    
    # Get the node to find its output slot index
    node = get_node_by_id(node_id)
    if not node:
        return None
    
    # Get the exporter to find output names
    node_type = node.get('class_type') or node.get('type')
    exporter = get_node_exporter(node_type)
    if not exporter:
        return None
    
    # Get output slot index for the named output
    try:
        output_names = exporter.get_output_names()
        if output_name not in output_names:
            return None
        output_slot = output_names.index(output_name)
    except (AttributeError, ValueError):
        return None
    
    # Find the link from this node's output slot
    for link in links:
        # Link format: [link_id, from_node, from_slot, to_node, to_slot]
        if len(link) >= 5:
            from_node = str(link[1])
            from_slot = link[2]
            to_node = str(link[3])
            
            if from_node == str(node_id) and from_slot == output_slot:
                return to_node
    
    return None

def get_connected_input(node_id: str, input_name: str) -> Optional[Dict]:
    """
    Get information about what's connected to a node's input.
    
    Args:
        node_id: ID of the target node
        input_name: Name of the input connector
    
    Returns:
        Dict with 'from_node' and 'from_slot', or None if no connection
    """
    context = get_export_context()
    links = context.get('links', [])
    
    # Get the node to find its input slot index
    node = get_node_by_id(node_id)
    if not node:
        return None
    
    # Get the exporter to find input names
    node_type = node.get('class_type') or node.get('type')
    exporter = get_node_exporter(node_type)
    if not exporter:
        return None
    
    # Get input slot index for the named input
    try:
        input_names = exporter.get_input_names()
        if input_name not in input_names:
            return None
        input_slot = input_names.index(input_name)
    except (AttributeError, ValueError):
        return None
    
    # Find the link to this node's input slot
    for link in links:
        # Link format: [link_id, from_node, from_slot, to_node, to_slot]
        if len(link) >= 5:
            to_node = str(link[3])
            to_slot = link[4]
            
            if to_node == str(node_id) and to_slot == input_slot:
                return {
                    'from_node': str(link[1]),
                    'from_slot': link[2]
                }
    
    return None