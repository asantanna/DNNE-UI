#!/usr/bin/env python3
"""
Exporter for BalancingConfig node - Virtual configuration node for PPOAgent
"""

from ..graph_exporter import ExportableNode

class BalancingConfigExporter(ExportableNode):
    """Exporter for BalancingConfig virtual node - provides balancing configuration to PPOAgent"""
    # BalancingConfig is a virtual node - only provides configuration
    # Virtual status is handled by @dnne_node decorator
    
    @classmethod
    def get_template_name(cls):
        # Virtual nodes don't need templates
        return None
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Virtual nodes don't generate code
        return {}
    
    @classmethod
    def get_imports(cls):
        # Virtual nodes don't generate files
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["config"]
    
    @classmethod
    def get_input_names(cls):
        return []
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        """Provide balancing configuration schema"""
        # Extract parameters from node data
        param_specs = [
            {'name': 'enabled', 'widget_index': 0},
            {'name': 'target_yield_rate', 'widget_index': 1},
            {'name': 'balance_interval', 'widget_index': 2},
            {'name': 'min_batch_size', 'widget_index': 3},
            {'name': 'max_batch_size', 'widget_index': 4},
            {'name': 'adjustment_factor', 'widget_index': 5},
        ]
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Build schema with configuration
        return {
            "outputs": {
                "config": {
                    "type": "balancing_config",
                    "enabled": params.get('enabled', False),
                    "target_yield_rate": params.get('target_yield_rate', 0.95),
                    "balance_interval": params.get('balance_interval', 100),
                    "min_batch_size": params.get('min_batch_size', 16),
                    "max_batch_size": params.get('max_batch_size', 256),
                    "adjustment_factor": params.get('adjustment_factor', 1.1),
                }
            }
        }