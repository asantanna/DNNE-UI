#!/usr/bin/env python3
"""
Exporter for ConcatNode using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_UTIL

class ConcatExporter(ExportableNode):
    """Exporter for the Concat tensor concatenation node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/concat_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget values for mode and pad_mode
        widgets = node_data.get("widgets_values", [])
        
        # Default values if not provided
        mode = "wait for all"
        pad_mode = "pad with zeros"
        
        # Extract from widgets array based on expected order
        if len(widgets) >= 1:
            mode = widgets[0]
        if len(widgets) >= 2:
            pad_mode = widgets[1]
        
        # CRITICAL: Per tensor dimension standards in CLAUDE.md:
        # Concat MUST operate on dim=1 (features), never dim=0 (batch)
        # This overrides any UI configuration until UI is updated
        concat_dim = 1  # ALWAYS feature dimension
        
        # Determine which inputs are actually connected
        connected_inputs = []
        for input_name in ["input_a", "input_b", "input_c", "input_d"]:
            if "inputs" in connections and input_name in connections["inputs"]:
                # Handle both single connection (dict) and multiple connections (list)
                input_connections = connections["inputs"][input_name]
                if isinstance(input_connections, list):
                    # Multiple connections - check if not empty
                    if len(input_connections) > 0:
                        connected_inputs.append(input_name)
                else:
                    # Legacy single connection format
                    connected_inputs.append(input_name)
            
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "ConcatNode",
            "MODE": mode,
            "PAD_MODE": pad_mode,
            "CONCAT_DIM": concat_dim,  # Add dimension parameter
            "CONNECTED_INPUTS": connected_inputs  # Pass list of connected inputs
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "from typing import Dict, Any, Optional, List",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input_a", "input_b", "input_c", "input_d"]
    
    @classmethod
    def get_required_input_names(cls):
        """Override to make all inputs optional.
        
        Concat node can work with any subset of connected inputs.
        Custom validation ensures at least 2 inputs are connected.
        """
        return []  # All inputs are optional
    
    @classmethod
    def validate_required_connections(cls, node_id: str, connections: dict):
        """Custom validation for Concat node.
        
        Ensures at least 2 inputs are connected for meaningful concatenation.
        """
        # Count connected inputs
        connected_count = 0
        connected_names = []
        
        if "inputs" in connections:
            for input_name in cls.get_input_names():
                if input_name in connections["inputs"]:
                    # Handle both single connection (dict) and multiple connections (list)
                    input_connections = connections["inputs"][input_name]
                    if isinstance(input_connections, list):
                        # Multiple connections - check if not empty
                        if len(input_connections) > 0:
                            connected_count += 1
                            connected_names.append(input_name)
                    else:
                        # Legacy single connection format
                        connected_count += 1
                        connected_names.append(input_name)
        
        # Require at least 2 inputs for meaningful concatenation
        if connected_count < 2:
            raise ValueError(
                f"Concat node {node_id} requires at least 2 connected inputs for concatenation. "
                f"Currently connected: {connected_names if connected_names else 'none'}"
            )
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Return initial schema for concatenated output"""
        return {
            "outputs": {
                "output": {
                    "type": "tensor", 
                    "dtype": "float32",
                    "flattened_size": None  # Will be resolved based on inputs
                }
            }
        }
    
    @classmethod
    def get_output_schema(cls, node_data, connections=None, node_registry=None, 
                         all_nodes=None, all_links=None):
        """Get output schema, resolving concatenated size from inputs"""
        # Get base schema
        schema = cls.get_initial_output_schema(node_data)
        
        # Try to calculate flattened size from connected inputs
        if connections and "inputs" in connections:
            total_size = 0
            has_all_sizes = True
            
            # Check each connected input
            for input_name in ["input_a", "input_b", "input_c", "input_d"]:
                if input_name in connections["inputs"]:
                    input_info = connections["inputs"][input_name]
                    
                    # Handle multi-connection inputs (input_info is a list)
                    if isinstance(input_info, list):
                        # For concat, we typically only care about the first connection
                        # since all connections should have the same schema
                        if len(input_info) > 0:
                            input_info = input_info[0]
                        else:
                            continue
                    
                    # Try to get schema from connected node
                    if node_registry and all_nodes and all_links:
                        source_node_id = input_info["from_node"]
                        source_output_slot = input_info["from_slot"]
                        
                        # Find source node
                        for node in all_nodes:
                            if str(node["id"]) == source_node_id:
                                source_node_type = node.get("class_type") or node.get("type")
                                if source_node_type in node_registry:
                                    source_exporter = node_registry[source_node_type]
                                    # Create temporary exporter to get connections
                                    from ..graph_exporter import GraphExporter
                                    temp_exporter = GraphExporter()
                                    temp_exporter.node_registry = node_registry
                                    source_connections = temp_exporter._get_node_connections(
                                        source_node_id, all_links, all_nodes
                                    )
                                    
                                    # Get output schema from source
                                    source_schema = source_exporter.get_output_schema_by_connector(
                                        source_output_slot, node, source_connections,
                                        node_registry, all_nodes, all_links
                                    )
                                    
                                    if source_schema and "flattened_size" in source_schema:
                                        if source_schema["flattened_size"] is not None:
                                            total_size += source_schema["flattened_size"]
                                        else:
                                            has_all_sizes = False
                                            break
                                    else:
                                        has_all_sizes = False
                                        break
                                break
            
            # Update schema if we have all sizes
            if has_all_sizes and total_size > 0:
                schema["outputs"]["output"]["flattened_size"] = total_size
        
        return schema

    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_UTIL