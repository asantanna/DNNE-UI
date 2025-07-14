#!/usr/bin/env python3
"""
Isaac Gym Utilities - Common functionality for Isaac Gym integration

Provides utilities for common Isaac Gym operations and configuration.
Note: Monkey patching has been removed as it doesn't work with C++ bindings.
"""

from typing import Dict, Any, Optional
import json
import time
from pathlib import Path


def save_timing_data(timing_data: Dict[str, Any], filepath: str = '/tmp/isaacgym_cpp_timings.json'):
    """
    Save timing data to a JSON file for the profiler to read.
    
    Args:
        timing_data: Dictionary of timing measurements
        filepath: Where to save the JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(timing_data, f, indent=2)


def create_timing_context(name: str, timing_dict: Dict[str, Any]):
    """
    Context manager for timing operations.
    
    Usage:
        timing_data = {}
        with create_timing_context('gym.simulate', timing_data):
            gym.simulate(sim)
    
    Args:
        name: Name of the operation being timed
        timing_dict: Dictionary to store timing results
    """
    class TimingContext:
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.perf_counter() - self.start
            if name not in timing_dict:
                timing_dict[name] = {'count': 0, 'total_ms': 0.0}
            timing_dict[name]['count'] += 1
            timing_dict[name]['total_ms'] += elapsed * 1000
    
    return TimingContext()


def load_isaac_gym_config(config_path: Path) -> Dict[str, Any]:
    """
    Load Isaac Gym configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        # If PyYAML not available, return empty config
        return {}


def setup_isaac_gym_logging(verbose: bool = False):
    """
    Configure logging for Isaac Gym operations.
    
    Args:
        verbose: Enable verbose logging
    """
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(name)s: %(message)s'
    )
    
    # Set Isaac Gym specific loggers
    logger = logging.getLogger('isaacgym')
    logger.setLevel(level)