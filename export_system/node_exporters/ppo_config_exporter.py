#!/usr/bin/env python3
"""
Exporter for PPOConfig node using queue-based template
"""

from ..graph_exporter import ExportableNode

class PPOConfigExporter(ExportableNode):
    """Exporter for PPO configuration virtual node"""
    # PPOConfig is a virtual node - only provides configuration
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
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["config"]
    
    @classmethod
    def get_input_names(cls):
        return []
    
    @classmethod
    def get_ppo_config(cls, node_id, node_data):
        """Query method to get PPO configuration from this virtual node.
        
        This method is called by non-virtual nodes (like PPOAgent) to retrieve
        configuration without directly accessing this node's widgets.
        
        Args:
            node_id: The ID of this PPOConfig node
            node_data: The node data dictionary containing widget values
            
        Returns:
            Dictionary with PPO configuration parameters properly mapped
        """
        # Use parameter specs matching PPOConfig INPUT_TYPES order
        param_specs = [
            {'name': 'learning_rate', 'widget_index': 0},
            {'name': 'num_epochs', 'widget_index': 1},  # maps to mini_epochs
            {'name': 'minibatch_size', 'widget_index': 2},
            {'name': 'clip_param', 'widget_index': 3},  # maps to e_clip
            {'name': 'value_loss_coef', 'widget_index': 4},  # maps to critic_coef
            {'name': 'entropy_coef', 'widget_index': 5},
            {'name': 'gamma', 'widget_index': 6},
            {'name': 'gae_lambda', 'widget_index': 7},  # maps to tau
            {'name': 'max_grad_norm', 'widget_index': 8},  # maps to grad_norm
            {'name': 'horizon_length', 'widget_index': 9},
            {'name': 'max_iterations', 'widget_index': 10},  # maps to max_epochs
            {'name': 'lr_schedule', 'widget_index': 11},  # maps to schedule_type
            {'name': 'lr_schedule_kl_threshold', 'widget_index': 12},
            {'name': 'use_clipped_value_loss', 'widget_index': 13},  # maps to clip_value
            {'name': 'normalize_advantage', 'widget_index': 14},
            {'name': 'normalize_input', 'widget_index': 15},
            {'name': 'normalize_value', 'widget_index': 16},
            {'name': 'reward_shaper_scale', 'widget_index': 17},  # not used but in node
            {'name': 'e_clip', 'widget_index': 18},  # duplicate param
            {'name': 'truncate_grads', 'widget_index': 19},  # not used
            {'name': 'bounds_loss_coef', 'widget_index': 20},
        ]
        
        # Get parameters using the helper that checks both inputs and widgets_values
        raw_params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        required_params = [
            'learning_rate', 'num_epochs', 'minibatch_size', 'clip_param',
            'value_loss_coef', 'entropy_coef', 'gamma', 'gae_lambda',
            'max_grad_norm', 'horizon_length', 'max_iterations', 'lr_schedule',
            'lr_schedule_kl_threshold', 'use_clipped_value_loss', 'normalize_advantage',
            'normalize_input', 'normalize_value', 'bounds_loss_coef'
        ]
        missing_params = [p for p in required_params if raw_params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"PPOConfig node {node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Map to the expected output format
        return {
            'learning_rate': raw_params['learning_rate'],
            'mini_epochs': raw_params['num_epochs'],
            'minibatch_size': raw_params['minibatch_size'],
            'e_clip': raw_params['clip_param'],
            'critic_coef': raw_params['value_loss_coef'],
            'entropy_coef': raw_params['entropy_coef'],
            'gamma': raw_params['gamma'],
            'tau': raw_params['gae_lambda'],
            'grad_norm': raw_params['max_grad_norm'],
            'horizon_length': raw_params['horizon_length'],
            'max_epochs': raw_params['max_iterations'],
            'schedule_type': raw_params['lr_schedule'],
            'lr_schedule_kl_threshold': raw_params['lr_schedule_kl_threshold'],
            'clip_value': raw_params['use_clipped_value_loss'],
            'normalize_advantage': raw_params['normalize_advantage'],
            'normalize_input': raw_params['normalize_input'],
            'normalize_value': raw_params['normalize_value'],
            'bounds_loss_coef': raw_params['bounds_loss_coef'],
        }