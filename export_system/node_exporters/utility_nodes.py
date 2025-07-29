#!/usr/bin/env python3
"""
Exporters for utility nodes using queue-based templates
"""

from ..graph_exporter import ExportableNode


class ORNodeExporter(ExportableNode):
    """Exporter for the OR/ANY routing node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/or_node_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "ORNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        return ["input_a", "input_b", "input_c"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]


class BalancingNodeExporter(ExportableNode):
    """Exporter for Balancing Node (active passthrough)"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/balancing_node_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        """Prepare template variables for Balancing Node"""
        # Define parameter specifications matching the node's widget order
        param_specs = [
            {'name': 'item_name', 'widget_index': 0, 'default': 'items'},
            {'name': 'enabled', 'widget_index': 1, 'default': True},
            {'name': 'min_hz', 'widget_index': 2, 'default': -1.0},
            {'name': 'max_hz', 'widget_index': 3, 'default': -1.0},
            {'name': 'target_hz', 'widget_index': 4, 'default': -1.0},
            {'name': 'target_percentage', 'widget_index': 5, 'default': -1.0},
            {'name': 'priority', 'widget_index': 6, 'default': 0},
            {'name': 'guaranteed', 'widget_index': 7, 'default': False},
            {'name': 'max_latency_ms', 'widget_index': 8, 'default': -1.0},
            {'name': 'window_size', 'widget_index': 9, 'default': 100},
            {'name': 'log_violations', 'widget_index': 10, 'default': True},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
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


class BalancingConfigExporter(ExportableNode):
    """Exporter for Balancing Config (virtual node)"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/balancing_config_queue.py"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        """Prepare template variables for Balancing Config"""
        # Virtual nodes pass configuration to connected nodes
        # The actual configuration is handled during graph export
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "BalancingConfig",
        }
    
    @classmethod
    def get_imports(cls):
        # Virtual nodes don't need imports
        return []
    
    @classmethod
    def is_virtual(cls):
        """Mark this as a virtual node"""
        return True
    
    @classmethod
    def get_input_names(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["config"]


# Registration function
def register_utility_exporters(exporter):
    """Register all utility node exporters"""
    exporter.register_node("ORNode", ORNodeExporter)
    exporter.register_node("BalancingNode", BalancingNodeExporter)
    exporter.register_node("BalancingConfig", BalancingConfigExporter)