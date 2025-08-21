#!/usr/bin/env python3
"""
Exporter for Eat_NNode using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_CONTROL


class Eat_NExporter(ExportableNode):
    """Exporter for the Eat_N synchronization node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/eat_n_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget values
        widget_values = node_data.get("widget_values", [])
        
        # Parse widgets based on their order in INPUT_TYPES
        # Widget 0: num_to_eat (INT)
        # Widget 1: trigger_mode (ENUM)
        num_to_eat = widget_values[0] if len(widget_values) > 0 else 1
        trigger_mode = widget_values[1] if len(widget_values) > 1 else "every_eat"
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "Eat_NNode",
            "NUM_TO_EAT": str(num_to_eat),
            "TRIGGER_MODE": f'"{trigger_mode}"'
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_output_names(cls):
        return ["output", "trigger"]
    
    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_CONTROL
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """
        Eat_N is a passthrough node after consuming N inputs.
        The output schema matches the input, and trigger is a control signal.
        """
        return {
            "outputs": {
                "output": {
                    "type": None,  # Will be resolved from input
                    "passthrough": True
                },
                "trigger": {
                    "type": "trigger",
                    "shape": []  # Scalar trigger signal
                }
            }
        }
    
    @classmethod
    def _resolve_schema_value(cls, key, parent_schema, node_data, connections, 
                            node_registry, all_nodes, all_links):
        """Resolve schema from input since output is a passthrough"""
        if key == "type" and parent_schema.get("passthrough"):
            # Get the schema from our input
            input_schema = cls.get_input_schema(node_data, connections, 
                                              node_registry, all_nodes, all_links)
            
            if "input" in input_schema and input_schema["input"]:
                # Copy the entire input schema to output
                parent_schema.update(input_schema["input"])
                return parent_schema.get("type")
        
        return None