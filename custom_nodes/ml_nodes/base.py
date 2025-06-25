"""
Base utilities for ML nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import base types
from custom_nodes.robotics_nodes.robotics_types import TensorData, Context
from custom_nodes.robotics_nodes import RoboticsNodeBase

# Global context instance
context = None

def get_context():
    """Get or create global context"""
    global context
    if context is None:
        context = Context()
    return context