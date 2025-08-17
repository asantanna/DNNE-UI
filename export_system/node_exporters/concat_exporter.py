#!/usr/bin/env python3
"""
Exporter for ConcatNode using queue-based template
"""

from ..graph_exporter import ExportableNode

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
            
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "ConcatNode",
            "MODE": mode,
            "PAD_MODE": pad_mode
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