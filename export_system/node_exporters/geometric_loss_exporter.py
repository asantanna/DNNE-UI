#!/usr/bin/env python3
"""
Exporter for GeometricLoss node using queue-based template
"""

from ..graph_exporter import ExportableNode
from ..subsystems import SUBSYSTEM_TRAINING

class GeometricLossExporter(ExportableNode):
    """Exporter for the Geometric Loss node"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/geometric_loss_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget values
        widgets = node_data.get("widgets_values", [])
        
        # Default value if not provided
        error_metric = "Euclidean Dist"
        
        # Extract from widgets array based on expected order
        if len(widgets) >= 1:
            error_metric = widgets[0]
            
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "GeometricLossNode",
            "ERROR_METRIC": error_metric
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import torch.nn.functional as F",
            "from typing import Dict, Any",
        ]
    
    @classmethod
    def get_export_files(cls, node_id, node_data):
        """Return list of files to copy during export."""
        # Copy math_utils.py from templates to the framework directory
        import os
        templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates", "framework", "math_utils.py")
        return [(templates_dir, "framework")]
    
    @classmethod
    def get_input_names(cls):
        return ["predictions", "estimates"]
    
    @classmethod
    def get_output_names(cls):
        return ["output"]
    
    @classmethod
    def get_initial_output_schema(cls, node_data):
        return {
            "outputs": {
                "output": {
                    "type": "scalar",
                    "dtype": "float32"
                }
            }
        }
    
    @classmethod
    def get_subsystem(cls):
        return SUBSYSTEM_TRAINING