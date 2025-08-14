"""
Data Streamer Node
Streams data from CSV files row-by-row with configurable synchronization modes for real-time robotics control.
"""

import os
import torch
import json
from typing import Dict, Any, Optional, Tuple
from inspect import cleandoc
from custom_nodes.utils.visnode_base import RoboticsNodeBase
from custom_nodes.utils.node_colors import get_node_colors


class DataStreamerNode(RoboticsNodeBase):
    """Data Streamer Node
    Streams data from CSV files row-by-row with configurable synchronization modes for real-time robotics control."""
    
    DESCRIPTION = cleandoc(__doc__)
    COLOR = get_node_colors("data")["color"]
    BGCOLOR = get_node_colors("data")["bgcolor"]
    CATEGORY = "utility"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_path": ("STRING", {
                    "default": "",
                    "tooltip": "Source path to CSV file or directory to copy to export package. Leave blank to skip copying."
                }),
                "dest_dir": ("STRING", {
                    "default": "data",
                    "tooltip": "Destination directory name in export package (relative to package root). Used only if src_path is provided."
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
                "sync": ("*TRIGGER", {
                    "tooltip": "Synchronization input (required when sync_mode is 'external')"
                }),
                "reset": ("*TRIGGER", {
                    "tooltip": "Reset stream to beginning of file"
                }),
            }
        }

    RETURN_TYPES = ("DATASTREAMER_DATA_TENSOR", "DATASTREAMER_DONE_TRIGGER", "DATASTREAMER_METADATA")
    RETURN_NAMES = ("data", "done", "metadata")
    FUNCTION = None  # DNNE nodes don't execute in UI, only export


# Node registration
NODE_CLASS_MAPPINGS = {
    "DataStreamer": DataStreamerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DataStreamer": "Data Streamer"
}