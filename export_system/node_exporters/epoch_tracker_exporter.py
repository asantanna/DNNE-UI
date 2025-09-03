#!/usr/bin/env python3
"""
Exporter for EpochTracker node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_TRAINING

class EpochTrackerExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/epoch_tracker_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader - FAIL-FAST: no defaults
        # The widgets_values array contains max_epochs at index 0 and telemetry_level at index 1
        param_specs = [
            {'name': 'max_epochs', 'widget_index': 0},
            {'name': 'telemetry_level', 'widget_index': 1}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Validate required parameters are present
        if 'max_epochs' not in params or params['max_epochs'] is None:
            raise ValueError(
                f"EpochTracker node {node_id} missing required parameter: max_epochs. "
                f"The UI must provide the maximum number of epochs."
            )
        
        max_epochs = params['max_epochs']
        
        if not isinstance(max_epochs, (int, float)) or max_epochs <= 0:
            raise ValueError(f"EpochTracker node {node_id}: max_epochs must be a positive number, got: {max_epochs}")
        
        # Default telemetry_level to "off" if not present (for compatibility)
        telemetry_level = params.get('telemetry_level', 'off')
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "EpochTrackerNode",
            "MAX_EPOCHS": int(max_epochs),
            "TELEMETRY_LEVEL": f'"{telemetry_level}"'  # String needs quotes
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
    
    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_TRAINING