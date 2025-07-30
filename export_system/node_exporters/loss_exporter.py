#!/usr/bin/env python3
"""
Exporter for Loss node using queue-based template
"""

from ..graph_exporter import ExportableNode

class LossExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/loss_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        params = node_data.get("inputs", {})
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "LossNode",
            "LOSS_TYPE": params.get("loss_type", "cross_entropy")
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn as nn",
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["loss", "loss_value"]
    
    @classmethod
    def get_input_names(cls):
        return ["predictions", "labels"]