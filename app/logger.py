from collections import deque
from datetime import datetime
import io
import logging
import sys
import threading
import time

logs = None
stdout_interceptor = None
stderr_interceptor = None


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


class LogInterceptor(io.TextIOWrapper):
    def __init__(self, stream,  *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        super().__init__(buffer, *args, **kwargs, encoding=encoding, line_buffering=stream.line_buffering)
        self._lock = threading.Lock()
        self._flush_callbacks = []
        self._logs_since_flush = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}
        with self._lock:
            self._logs_since_flush.append(entry)

            # Simple handling for cr to overwrite the last output if it isnt a full line
            # else logs just get full of progress messages
            if isinstance(data, str) and data.startswith("\r") and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        super().write(data)

    def flush(self):
        super().flush()
        for cb in self._flush_callbacks:
            cb(self._logs_since_flush)
            self._logs_since_flush = []

    def on_flush(self, callback):
        self._flush_callbacks.append(callback)


def get_logs():
    return logs


def on_flush(callback):
    if stdout_interceptor is not None:
        stdout_interceptor.on_flush(callback)
    if stderr_interceptor is not None:
        stderr_interceptor.on_flush(callback)

def setup_logger(log_level: str = 'INFO', capacity: int = 300, use_stdout: bool = False):
    global logs
    if logs:
        return

    # Override output streams and log to buffer
    logs = deque(maxlen=capacity)

    global stdout_interceptor
    global stderr_interceptor
    stdout_interceptor = sys.stdout = LogInterceptor(sys.stdout)
    stderr_interceptor = sys.stderr = LogInterceptor(sys.stderr)

    # Setup default global logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Add file handler for DNNE.log in dnne_logs directory
    import os
    log_dir = 'dnne_logs'
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'DNNE.log'), mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Create UTF-8 wrapped stream for stderr to handle emojis on Windows
    import codecs
    utf8_stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')
    stream_handler = logging.StreamHandler(utf8_stderr)
    # Use RelativeTimeFormatter for console output
    stream_handler.setFormatter(RelativeTimeFormatter("%(asctime)s - %(levelname)s - %(message)s"))

    if use_stdout:
        # Only errors and critical to stderr
        stream_handler.addFilter(lambda record: not record.levelno < logging.ERROR)

        # Lesser to stdout - also wrap in UTF-8
        utf8_stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
        stdout_handler = logging.StreamHandler(utf8_stdout)
        stdout_handler.setFormatter(RelativeTimeFormatter("%(asctime)s - %(levelname)s - %(message)s"))
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        logger.addHandler(stdout_handler)

    logger.addHandler(stream_handler)


STARTUP_WARNINGS = []


def log_startup_warning(msg):
    logging.warning(msg)
    STARTUP_WARNINGS.append(msg)


def print_startup_warnings():
    for s in STARTUP_WARNINGS:
        logging.warning(s)
    STARTUP_WARNINGS.clear()
