#!/usr/bin/env python3
"""
Exporter for CameraSensor node using queue-based template
"""

from ..graph_exporter import ExportableNode

class CameraSensorExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/camera_sensor_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'resolution', 'default': '640x480'},
            {'name': 'fps', 'default': 30.0},
            {'name': 'use_real_camera', 'default': False},
            {'name': 'camera_index', 'default': 0}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # Parse resolution
        resolution = params['resolution']
        width, height = map(int, resolution.split('x'))
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "CameraSensorNode",
            "FPS": params['fps'],
            "WIDTH": width,
            "HEIGHT": height,
            "USE_REAL_CAMERA": params['use_real_camera'],
            "CAMERA_INDEX": params['camera_index']
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import numpy as np",
            "import torch",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["image"]
    
    @classmethod
    def get_input_names(cls):
        return []
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        # Parse resolution from node data
        resolution = node_data.get("widgets_values", ["640x480"])[0] if node_data.get("widgets_values") else "640x480"
        width, height = map(int, resolution.split('x'))
        
        return {
            "outputs": {
                "image": {
                    "type": "tensor",
                    "shape": [3, height, width],  # CHW format
                    "dtype": "float32"
                }
            }
        }