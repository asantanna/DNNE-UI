#!/usr/bin/env python3
"""
Exporter for CrossEntropyLoss node using queue-based template
"""

from ..graph_exporter import ExportableNode

class CrossEntropyLossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/cross_entropy_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LossNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["loss", "accuracy"]
    
    @classmethod
    def get_input_names(cls):
        return ["predictions", "labels"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "loss": {
                    "type": "scalar",
                    "dtype": "float32"
                },
                "accuracy": {
                    "type": "scalar", 
                    "dtype": "float32"
                }
            }
        }