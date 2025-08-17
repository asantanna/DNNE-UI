#!/usr/bin/env python3
"""
Exporter for BalancerConfig node - Virtual configuration node for PPOAgent
"""

from ..graph_exporter import ExportableNode

class BalancerConfigExporter(ExportableNode):
    """Exporter for BalancerConfig virtual node - provides balancing configuration to PPOAgent"""
    # BalancerConfig is a virtual node - only provides configuration
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
        
        # Build schema with configuration - fail if required params missing
        required_params = ['enabled', 'target_yield_rate', 'balance_interval', 
                          'min_batch_size', 'max_batch_size', 'adjustment_factor']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"BalancerConfig node missing required parameters: {missing_params}. "
                f"All balancing configuration parameters must be provided."
            )
        
        return {
            "outputs": {
                "config": {
                    "type": "balancing_config",
                    "enabled": params['enabled'],
                    "target_yield_rate": params['target_yield_rate'],
                    "balance_interval": params['balance_interval'],
                    "min_batch_size": params['min_batch_size'],
                    "max_batch_size": params['max_batch_size'],
                    "adjustment_factor": params['adjustment_factor'],
                }
            }
        }
    
    @classmethod
    def get_balancing_config(cls, node_id, node_data):
        """Query method to get balancing configuration from this virtual node.
        
        This method is called by non-virtual nodes (like PPOAgent) to retrieve
        balancing configuration without directly accessing this node's widgets.
        
        Args:
            node_id: The ID of this BalancerConfig node
            node_data: The node data dictionary containing widget values
            
        Returns:
            Dictionary with balancing configuration parameters
        """
        # Use parameter specs matching BalancerConfig INPUT_TYPES order
        param_specs = [
            {'name': 'enabled', 'widget_index': 0},
            {'name': 'min_hz', 'widget_index': 1},
            {'name': 'max_hz', 'widget_index': 2},
            {'name': 'target_hz', 'widget_index': 3},
            {'name': 'target_percentage', 'widget_index': 4},
            {'name': 'priority', 'widget_index': 5},
            {'name': 'guaranteed', 'widget_index': 6},
            {'name': 'max_latency_ms', 'widget_index': 7},
        ]
        
        # Get parameters using the helper that checks both inputs and widgets_values
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Check if config is enabled - fail if missing
        if 'enabled' not in params:
            raise ValueError(
                f"BalancerConfig node {node_id} missing 'enabled' parameter. "
                f"This indicates the UI is not sending widget values correctly."
            )
        enabled = params['enabled']
        if enabled is None:
            raise ValueError(
                f"BalancerConfig node {node_id} missing 'enabled' parameter. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        if not enabled:
            return {"enabled": False, "type": "balancing_config"}
            
        # Build configuration structure matching what BalancerConfig.create_config() returns
        config = {
            'enabled': True,
            'frequency': {},
            'throughput': {},
            'scheduling': {},
            'latency': {},
        }
        
        # Add scheduling settings - fail if missing when enabled
        if 'priority' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: priority parameter missing")
        if 'guaranteed' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: guaranteed parameter missing")
        
        config['scheduling']['priority'] = params['priority']
        config['scheduling']['guaranteed'] = params['guaranteed']
        
        # Add frequency settings - fail if params missing, -1 means don't care
        if 'min_hz' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: min_hz parameter missing")
        if 'max_hz' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: max_hz parameter missing")
        if 'target_hz' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: target_hz parameter missing")
        
        # Only add to config if >= 0 (negative means don't care)
        if params['min_hz'] >= 0:
            config['frequency']['min_hz'] = params['min_hz']
        if params['max_hz'] >= 0:
            config['frequency']['max_hz'] = params['max_hz']
        if params['target_hz'] >= 0:
            config['frequency']['target_hz'] = params['target_hz']
            
        # Add throughput settings - fail if param missing
        if 'target_percentage' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: target_percentage parameter missing")
        
        # Only add to config if >= 0 (negative means don't care)
        if params['target_percentage'] >= 0:
            config['throughput']['target_percentage'] = params['target_percentage']
            
        # Add latency settings - fail if param missing
        if 'max_latency_ms' not in params:
            raise ValueError(f"BalancerConfig node {node_id}: max_latency_ms parameter missing")
        
        # Only add to config if >= 0 (negative means don't care)
        if params['max_latency_ms'] >= 0:
            config['latency']['max_latency_ms'] = params['max_latency_ms']
        
        return config