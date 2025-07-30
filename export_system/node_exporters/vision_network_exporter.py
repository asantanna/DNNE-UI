#!/usr/bin/env python3
"""
Exporter for VisionNetwork node using queue-based template
"""

from ..graph_exporter import ExportableNode

class VisionNetworkExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/vision_network_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Use universal parameter reader for consistent data access
        param_specs = [
            {'name': 'model', 'default': 'resnet18'},
            {'name': 'pretrained', 'default': True},
            {'name': 'output_dim', 'default': 512},
            {'name': 'device', 'default': 'cuda'}
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "VisionNetworkNode",
            "MODEL_TYPE": params['model'],
            "PRETRAINED": params['pretrained'],
            "OUTPUT_DIM": params['output_dim'],
            "DEVICE": params['device']
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
            "import torchvision.transforms as transforms",
        ]