"""
Data Streamer Node
Streams data from CSV files row-by-row with configurable synchronization modes for real-time robotics control.
"""

import os
import torch
import json
from typing import Dict, Any, Optional, Tuple
from inspect import cleandoc
from custom_nodes.base import RoboticsNodeBase
from custom_nodes.node_colors import get_node_colors


class DataStreamerNode(RoboticsNodeBase):
    """Data Streamer Node
    Streams data from CSV files row-by-row with configurable synchronization modes for real-time robotics control."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("utility")["color"]
    BGCOLOR = get_node_colors("utility")["bgcolor"]
    CATEGORY = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "./data/trajectory.csv",
                    "tooltip": "Path to CSV file containing data to stream. Supports relative and absolute paths."
                }),
                "sync_mode": (["none", "external", "timed"], {
                    "default": "none",
                    "tooltip": "Synchronization mode: 'none' streams as fast as possible, 'external' waits for sync input, 'timed' uses frequency_hz"
                }),
                "frequency_hz": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.1,
                    "max": 10000.0,
                    "tooltip": "Output frequency in Hz when sync_mode is 'timed'. Ignored for other modes."
                }),
                "auto_first_row": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Send first row immediately without waiting for sync (useful for initialization in external mode)"
                }),
                "loop_data": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Loop back to first row after reaching end of file"
                }),
                "eof_mode": (["stop", "pulse_done", "hold_last"], {
                    "default": "stop",
                    "tooltip": "End-of-file behavior: 'stop' stops streaming, 'pulse_done' sends done signal, 'hold_last' keeps sending last row"
                }),
                "delimiter": ("STRING", {
                    "default": ",",
                    "tooltip": "CSV delimiter character (default: comma)"
                }),
                "skip_header": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip first row if it contains column headers"
                }),
            },
            "optional": {
                "sync": ("TRIGGER", {
                    "tooltip": "Synchronization input (required when sync_mode is 'external')"
                }),
                "reset": ("TRIGGER", {
                    "tooltip": "Reset stream to beginning of file"
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "TRIGGER", "DICT")
    RETURN_NAMES = ("data", "done", "metadata")
    FUNCTION = "stream_data"

    def stream_data(self, file_path: str, sync_mode: str, frequency_hz: float,
                   auto_first_row: bool, loop_data: bool, eof_mode: str,
                   delimiter: str, skip_header: bool,
                   sync: Optional[Any] = None, reset: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[Any], Dict]:
        """
        In UI mode, just validate the file exists and return dummy outputs.
        Actual streaming happens in the exported queue-based code.
        """
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Check for metadata file
        base_name = os.path.splitext(file_path)[0]
        metadata_path = f"{base_name}_metadata.json"
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata from {metadata_path}: {e}")
        
        # Try to determine data shape by reading first data row
        try:
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f, delimiter=delimiter)
                
                # Skip header if requested
                if skip_header:
                    next(reader, None)
                
                # Read first data row to get shape
                first_row = next(reader, None)
                if first_row:
                    num_columns = len(first_row)
                    # Create dummy tensor with correct shape
                    dummy_data = torch.zeros(num_columns, dtype=torch.float32)
                else:
                    dummy_data = torch.zeros(1, dtype=torch.float32)
        except Exception as e:
            print(f"Warning: Could not read CSV file {file_path}: {e}")
            dummy_data = torch.zeros(1, dtype=torch.float32)
        
        # Done signal is None in UI mode
        done_signal = None
        
        # Add file info to metadata
        metadata["file_path"] = file_path
        metadata["sync_mode"] = sync_mode
        if sync_mode == "timed":
            metadata["frequency_hz"] = frequency_hz
        
        return (dummy_data, done_signal, metadata)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DataStreamer": DataStreamerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DataStreamer": "Data Streamer"
}