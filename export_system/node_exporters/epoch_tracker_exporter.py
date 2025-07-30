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
        max_epochs = cls.get_node_parameter(node_data, 'max_epochs', default_value=None, widget_index=0)
        
        if max_epochs is None:
            raise ValueError(f"EpochTracker node {node_id}: missing max_epochs parameter. "
                           f"Available in node_data: inputs={node_data.get('inputs', {}).keys()}, "
                           f"widgets_values={node_data.get('widgets_values', [])}")
        
        if not isinstance(max_epochs, (int, float)) or max_epochs <= 0:
            raise ValueError(f"EpochTracker node {node_id}: max_epochs must be a positive number, got: {max_epochs}")
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "EpochTrackerNode",
            "MAX_EPOCHS": int(max_epochs)
        }
    
    @classmethod
    def get_imports(cls):
        return []
    
    @classmethod
    def get_output_names(cls):
        return ["training_summary"]
    
    @classmethod
    def get_input_names(cls):
        return ["epoch_stats", "loss", "accuracy", "max_epochs"]
    
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