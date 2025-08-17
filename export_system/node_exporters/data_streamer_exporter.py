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
            {'name': 'src_path', 'widget_index': 0},
            {'name': 'dest_dir', 'widget_index': 1},
            {'name': 'sync_mode', 'widget_index': 2},
            {'name': 'frequency_hz', 'widget_index': 3},
            {'name': 'auto_first_row', 'widget_index': 4},
            {'name': 'loop_data', 'widget_index': 5},
            {'name': 'eof_mode', 'widget_index': 6},
            {'name': 'delimiter', 'widget_index': 7},
            {'name': 'skip_header', 'widget_index': 8},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        
        # FAIL-FAST: Validate required parameters
        required_params = ['src_path', 'dest_dir', 'sync_mode', 'frequency_hz', 'auto_first_row',
                          'loop_data', 'eof_mode', 'delimiter', 'skip_header']
        missing_params = [p for p in required_params if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"DataStreamer node {node_id} missing required parameters: {missing_params}. "
                f"This may indicate the UI is not sending widget values correctly."
            )
        
        # Extract values with proper type conversion
        # Strip whitespace from paths to handle user input
        src_path = params['src_path'].strip() if params['src_path'] else ''
        dest_dir = params['dest_dir'].strip() if params['dest_dir'] else ''
        sync_mode = params['sync_mode']
        frequency_hz = float(params['frequency_hz'])
        auto_first_row = bool(params['auto_first_row'])
        loop_data = bool(params['loop_data'])
        eof_mode = params['eof_mode']
        delimiter = params['delimiter']
        skip_header = bool(params['skip_header'])
        
        # Validate src_path and dest_dir combinations
        if not src_path and dest_dir:
            raise ValueError(
                f"DataStreamer node {node_id}: dest_dir is specified but src_path is empty. "
                f"Cannot determine source file to copy."
            )
        
        # Validate dest_dir is relative when provided
        import os
        if dest_dir and os.path.isabs(dest_dir):
            raise ValueError(
                f"DataStreamer node {node_id}: dest_dir must be a relative path, got absolute path: {dest_dir}"
            )
        
        # Determine the file path to use in the exported code
        if src_path:
            # File will be copied to the export package
            file_name = os.path.basename(src_path)
            
            # Handle dest_dir cases
            if not dest_dir:
                # Empty dest_dir means use package root (".")
                dest_dir = "."
            # else: use dest_dir as-is (must be relative path, already validated)
            
            # Construct the path that will be used in the exported code
            # Always use forward slashes for cross-platform compatibility
            if dest_dir == ".":
                file_path = file_name
            else:
                # Use forward slash explicitly to avoid Windows backslash issues
                file_path = f"{dest_dir}/{file_name}"
        else:
            # Both empty - no file to copy, use a placeholder
            file_path = "data.csv"  # Default placeholder for runtime configuration
        
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
    
    @classmethod
    def get_export_files(cls, node_id, node_data):
        """Return list of files/directories to copy during export."""
        # Get the src_path and dest_dir parameters
        param_specs = [
            {'name': 'src_path', 'widget_index': 0},
            {'name': 'dest_dir', 'widget_index': 1},
        ]
        
        params = cls.get_node_parameters_batch(node_data, param_specs)
        # Strip whitespace from paths to handle user input - fail if missing
        if 'src_path' not in params or 'dest_dir' not in params:
            raise ValueError(f"DataStreamer node {node_id}: src_path and dest_dir parameters must be present")
        src_path = params['src_path'].strip() if params['src_path'] else ''
        dest_dir = params['dest_dir'].strip() if params['dest_dir'] else ''
        
        # If no source path, nothing to copy
        if not src_path:
            return []
        
        # If dest_dir is empty, use package root
        if not dest_dir:
            dest_dir = "."
        
        # Return the file/directory to copy
        # The graph exporter will handle whether it's a file or directory
        return [(src_path, dest_dir)]