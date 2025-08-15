#!/usr/bin/env python3
"""
Exporter for SplitNode using queue-based template
"""

from ..graph_exporter import ExportableNode

class SplitNodeExporter(ExportableNode):
    """Exporter for the Split tensor splitting node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/split_node_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract parameters using helper functions
        param_specs = [
            {'name': 'dimension', 'widget_index': 0},
            {'name': 'split_mode', 'widget_index': 1},
            {'name': 'split_pos', 'widget_index': 2}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Direct access - will fail fast if missing
        dimension = params['dimension']
        split_mode = params['split_mode']
        split_pos = params['split_pos']
        
        # Parse split_pos string into list of integers
        try:
            split_values = [int(x.strip()) for x in split_pos.split(',') if x.strip()]
        except ValueError as e:
            raise ValueError(
                f"SplitNode {node_id}: Failed to parse split_pos '{split_pos}' as comma-separated integers: {e}"
            )
        
        if not split_values:
            raise ValueError(
                f"SplitNode {node_id}: split_pos '{split_pos}' resulted in empty list"
            )
            
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "SplitNode",
            "DIMENSION": dimension,
            "SPLIT_MODE": split_mode,
            "SPLIT_VALUES": split_values
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
        return ["input"]
    
    @classmethod
    def get_output_names(cls):
        return ["output_a", "output_b", "output_c", "output_d"]