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
        # Use get_node_parameters_batch to handle both UI and programmatic export
        param_specs = [
            {'name': 'file_path', 'widget_index': 0},
            {'name': 'sync_mode', 'widget_index': 1},
            {'name': 'frequency_hz', 'widget_index': 2},
            {'name': 'auto_first_row', 'widget_index': 3},
            {'name': 'loop_data', 'widget_index': 4},
            {'name': 'eof_mode', 'widget_index': 5},
            {'name': 'delimiter', 'widget_index': 6},
            {'name': 'skip_header', 'widget_index': 7},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # FAIL-FAST: Validate required parameters
        required_params = ['file_path', 'sync_mode', 'frequency_hz', 'auto_first_row',
                          'loop_data', 'eof_mode', 'delimiter', 'skip_header']
        missing_params = [p for p in required_params if params.get(p) is None]
        if missing_params:
            raise ValueError(
                f"DataStreamer node {node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Extract values with proper type conversion
        file_path = params['file_path']
        sync_mode = params['sync_mode']
        frequency_hz = float(params['frequency_hz'])
        auto_first_row = bool(params['auto_first_row'])
        loop_data = bool(params['loop_data'])
        eof_mode = params['eof_mode']
        delimiter = params['delimiter']
        skip_header = bool(params['skip_header'])
        
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