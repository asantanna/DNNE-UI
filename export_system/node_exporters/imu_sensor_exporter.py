#!/usr/bin/env python3
"""
Exporter for IMUSensor node using queue-based template
"""

from ..graph_exporter import ExportableNode

class IMUSensorExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/imu_sensor_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'sample_rate', 'default': 100.0},
            {'name': 'add_noise', 'default': True}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "IMUSensorNode",
            "SAMPLE_RATE": params['sample_rate'],
            "ADD_NOISE": params['add_noise']
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]