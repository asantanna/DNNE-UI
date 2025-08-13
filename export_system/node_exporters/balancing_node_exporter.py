#!/usr/bin/env python3
"""
Exporter for BalancingNode node using queue-based template
"""

from ..graph_exporter import ExportableNode

class BalancingNodeExporter(ExportableNode):
    """Exporter for Balancing Node (active passthrough)"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/balancing_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        """Prepare template variables for Balancing Node"""
        # Define parameter specifications - FAIL-FAST: no defaults
        param_specs = [
            {'name': 'item_name', 'widget_index': 0},
            {'name': 'enabled', 'widget_index': 1},
            {'name': 'min_hz', 'widget_index': 2},
            {'name': 'max_hz', 'widget_index': 3},
            {'name': 'target_hz', 'widget_index': 4},
            {'name': 'target_percentage', 'widget_index': 5},
            {'name': 'priority', 'widget_index': 6},
            {'name': 'guaranteed', 'widget_index': 7},
            {'name': 'max_latency_ms', 'widget_index': 8},
            {'name': 'window_size', 'widget_index': 9},
            {'name': 'log_violations', 'widget_index': 10},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = ['item_name', 'enabled', 'min_hz', 'max_hz', 'target_hz', 
                          'target_percentage', 'priority', 'guaranteed', 'max_latency_ms',
                          'window_size', 'log_violations']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"BalancingNode {node_id} missing required parameters: {missing_params}. "
                f"The UI must provide all balancing configuration parameters."
            )
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BalancingNode",
            "ITEM_NAME": params['item_name'],
            "ENABLED": params['enabled'],
            "MIN_HZ": params['min_hz'],
            "MAX_HZ": params['max_hz'],
            "TARGET_HZ": params['target_hz'],
            "TARGET_PERCENTAGE": params['target_percentage'],
            "PRIORITY": params['priority'],
            "GUARANTEED": params['guaranteed'],
            "MAX_LATENCY_MS": params['max_latency_ms'],
            "WINDOW_SIZE": params['window_size'],
            "LOG_VIOLATIONS": params['log_violations'],
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import time", 
            "import asyncio",
            "from collections import deque",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # BalancingNode is a passthrough - it outputs whatever it receives as input
        # The actual schema will be resolved from the input connection
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
                return input_schema["input"].get("type")
                
        return None