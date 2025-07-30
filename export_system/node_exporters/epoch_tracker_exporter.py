#!/usr/bin/env python3
"""
Exporter for EpochTracker node using queue-based template
"""

from ..graph_exporter import ExportableNode

class EpochTrackerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/epoch_tracker_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        # The widgets_values array has [false, 0, 0, 5, 10] so max_epochs is at index 3
        param_specs = [
            {'name': 'max_epochs', 'widget_index': 3, 'default': 100},
            {'name': 'early_stop_patience', 'widget_index': 4, 'default': 10}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        max_epochs = params['max_epochs']
        early_stop_patience = params['early_stop_patience']
        
        if not isinstance(max_epochs, (int, float)) or max_epochs <= 0:
            raise ValueError(f"EpochTracker node {node_id}: max_epochs must be a positive number, got: {max_epochs}")
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "EpochTrackerNode",
            "MAX_EPOCHS": int(max_epochs),
            "EARLY_STOP_PATIENCE": int(early_stop_patience)
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["training_summary"]
    
    @classmethod
    def get_input_names(cls):
        return ["epoch_stats", "loss", "accuracy"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "training_summary": {
                    "type": "dict",
                    "dtype": "dict"
                }
            }
        }