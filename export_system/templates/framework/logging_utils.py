"""
Logging utilities for DNNE exported workflows.
"""

import logging
import time
from datetime import datetime


class RelativeTimeFormatter(logging.Formatter):
    """Custom formatter that shows elapsed time since program start"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.start_logged = False
        
    def formatTime(self, record, datefmt=None):
        # Log the absolute start time once
        if not self.start_logged:
            self.start_logged = True
            abs_time = datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # This will be included in the first log message
            return f"Program started at {abs_time}\n00:00:00.000"
        
        # Calculate elapsed time
        elapsed_seconds = record.created - self.start_time
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        milliseconds = int((elapsed_seconds % 1) * 1000)
        
        # Format as HH:MM:SS.mmm (hours can be > 24)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def configure_relative_logging(format_str='%(asctime)s - %(name)s - %(message)s'):
    """Configure logging with relative time formatter"""
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(RelativeTimeFormatter(format_str))
    
    # Configure root logger
    logging.basicConfig(
        handlers=[console_handler],
        force=True  # Override any existing configuration
    )
    
    return console_handler