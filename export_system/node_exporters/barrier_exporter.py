#!/usr/bin/env python3
"""
Exporter for BarrierNode using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_CONTROL


class BarrierExporter(ExportableNode):
    """Exporter for the Barrier synchronization node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/barrier_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract hold_mode widget value
        widget_values = node_data.get("widget_values", [])
        hold_mode = widget_values[0] if widget_values else "FIFO"
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BarrierNode",
            "HOLD_MODE": f'"{hold_mode}"'
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "from collections import deque",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input", "release"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_CONTROL
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """
        Barrier is a passthrough node - it outputs whatever it receives as input.
        The actual schema will be resolved from the input connection.
        """
        return {
            "outputs": {
                "output": {
                    "type": None,  # Will be resolved from input
                    "passthrough": True
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Resolve schema from input since this is a passthrough node"""
        if key == "type" and parent_schema.get("passthrough"):
            # Get the schema from our input
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if "input" in input_schema and input_schema["input"]:
                # Copy the entire input schema to output
                parent_schema.update(input_schema["input"])
                return parent_schema.get("type")
        
        return None