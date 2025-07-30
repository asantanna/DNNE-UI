#!/usr/bin/env python3
"""
Exporter for TrainingStep node using queue-based template
"""

from ..graph_exporter import ExportableNode

class TrainingStepExporter(ExportableNode):
    @classmethod
    def get_template_name(cls):
        return "nodes/training_step_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "TrainingStepNode"
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio"
        ]
    
    @classmethod
    def get_output_names(cls):
        return ["ready", "step_complete"]
    
    @classmethod
    def get_input_names(cls):
        return ["loss", "optimizer"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "ready": {
                    "type": "trigger",
                    "dtype": "bool"
                },
                "step_complete": {
                    "type": "trigger",
                    "dtype": "bool"
                }
            }
        }