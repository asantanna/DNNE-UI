#!/usr/bin/env python3
"""
Exporter for DataStreamer node using queue-based template
"""

from ..graph_exporter import ExportableNode

class DataStreamerExporter(ExportableNode):
    """Exporter for the DataStreamer node that streams CSV data"""
    
    @classmethod
    def get_template_name(cls):
        return "nodes/data_streamer_queue.tpl"
    
    @classmethod
    def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
        # Extract widget values
        widgets = node_data.get("widgets_values", [])
        
        # Default values matching the node definition
        file_path = "./data/trajectory.csv"
        sync_mode = "none"
        frequency_hz = 100.0
        auto_first_row = True
        loop_data = False
        eof_mode = "stop"
        delimiter = ","
        skip_header = True
        
        # Extract values from widgets array
        # Order must match the order in INPUT_TYPES
        if len(widgets) >= 8:
            file_path = widgets[0]
            sync_mode = widgets[1]
            frequency_hz = float(widgets[2])
            auto_first_row = bool(widgets[3])
            loop_data = bool(widgets[4])
            eof_mode = widgets[5]
            delimiter = widgets[6]
            skip_header = bool(widgets[7])
        
        return {
            "NODE_ID": node_id,
            "CLASS_NAME": "DataStreamerNode",
            "FILE_PATH": file_path,
            "SYNC_MODE": sync_mode,
            "FREQUENCY_HZ": frequency_hz,
            "AUTO_FIRST_ROW": auto_first_row,
            "LOOP_DATA": loop_data,
            "EOF_MODE": eof_mode,
            "DELIMITER": delimiter,
            "SKIP_HEADER": skip_header,
        }
    
    @classmethod
    def get_imports(cls):
        return [
            "import torch",
            "import asyncio",
            "import pandas as pd",
            "import numpy as np",
            "import json",
            "import os",
            "import time",
            "from typing import Dict, Any, Optional",
        ]
    
    @classmethod
    def get_input_names(cls):
        # Optional inputs
        return ["sync", "reset"]
    
    @classmethod
    def get_output_names(cls):
        return ["data", "done", "metadata"]